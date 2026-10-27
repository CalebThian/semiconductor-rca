# -*- coding: utf-8 -*-
"""
reduce_params.py  ——  訓練前的參數篩選（解決 MT-NGC 在高維下的 O(N²) 爆炸）

為什麼需要這一步
----------------
  MT-NGC 對每個 target 各建一個 MLP，第一層輸入是「所有參數 × lookback」，
  所以總參數量約為 N² × lookback × hidden_dim：

      N=200   →  0.09 GB（含 Adam 動量）
      N=500   →  0.54 GB
      N=1000  →  2.15 GB
      N=5000  → 53.64 GB   ← 不可能

  而且 48 億個模型參數超過 int32 上限（21.47 億）兩倍以上，大小計算
  一旦溢位就會配置到 NULL，Windows 顯示「指令位置 0x00000000」。
  這不是 bug，是架構的本質限制。

篩選順序（由無爭議到有取捨）
----------------------------
  1. 丟常數欄          —— 變異為 0，對任何方法都沒有資訊
  2. 丟近常數欄        —— 變異極小，實際上是常數加量化雜訊
  3. 丟重複欄          —— 與其他欄完全相關，進圖只會製造假邊
  4. 篩到 top-K        —— 真正的取捨，兩種模式見下

  前三步是免費的，不需要任何假設。第四步才需要你決定。

兩種 top-K 模式
---------------
  --mode variance（預設，不使用標籤）
      依 Normal 資料的變異數排序。圖是從 Normal 學的，所以用 Normal
      的變異數篩選不會把標籤資訊漏進圖裡。

  --mode welch（使用標籤）
      依 |Welch t| 排序，命中率高，但**選欄這一步用到了標籤**。
      之後若要主張「圖是在不看標籤的情況下學到的」，這個模式會讓
      該主張失效。只有在不需要這個主張時才用。

  兩種模式都會「無條件保留候選參數」（--candidates），因為圖的用途是
  把候選參數分組，候選不在圖裡的話整個下游都做不了。

執行範例
--------
  # 先看會剩多少，不輸出（建議第一次這樣跑）
  python scripts\\reduce_params.py --config config\\pipeline.json --dry-run

  # 實際輸出，保留 300 個
  python scripts\\reduce_params.py --config config\\pipeline.json --topk 300

  # 帶候選清單（候選一定保留）
  python scripts\\reduce_params.py --config config\\pipeline.json ^
    --topk 300 --candidates candidates.txt

  # 不用 config
  python scripts\\reduce_params.py ^
    --csv data\\raw\\big.csv --wafer-col LOT_ID,WAFER_ID ^
    --label-col "Symbol Define" --abnormal-value Abnormal --topk 300

輸出
----
  <原檔名>_reduced.csv    篩選後的資料（給 MT-NGC 訓練用）
  <原檔名>_reduced.txt    保留的參數名稱清單，一行一個
  <原檔名>_reduced.log    完整篩選紀錄

環境：Python 3.6+ / pandas / numpy
"""

from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ENCODINGS = ('utf-8-sig', 'utf-8', 'cp950', 'big5hkscs', 'gbk', 'latin-1')

DEFAULT_META = ['WFR_ID', 'WAFER_ID', 'LOT_ID', 'START_DATETIME', 'END_DATETIME',
                'STEP', 'STEP_ID', 'RECIPE', 'CHAMBER', 'EQP_ID', 'TOOL_ID',
                'TIME', 'TIMESTAMP', 'label', 'LABEL']


class Log(object):
    def __init__(self):
        self.lines = []

    def __call__(self, s=''):
        print(s, flush=True)
        self.lines.append(s)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.lines) + '\n')


def read_csv_any(path, emit):
    last = None
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc)
            if enc not in ('utf-8-sig', 'utf-8'):
                emit('  （以 %s 編碼讀取）' % enc)
            return df
        except UnicodeDecodeError as e:
            last = e
        except Exception:
            raise
    sys.exit('無法以 %s 任一編碼讀取 %s：%s'
             % ('/'.join(ENCODINGS), path, last))


def mem_gb(n, lookback, hidden):
    """MT-NGC 的權重量估計（GB），含 Adam 的兩份動量。"""
    p = float(n) * n * lookback * hidden
    return p * 4 / 1024.0 ** 3, p * 4 * 3 / 1024.0 ** 3, p


def parse_args():
    p = argparse.ArgumentParser(
        description='訓練前參數篩選，把 N 降到 MT-NGC 能負荷的範圍')
    p.add_argument('--config', default=None, help='沿用 run_all 的設定檔')
    p.add_argument('--csv', default=None)
    p.add_argument('--wafer-col', default=None)
    p.add_argument('--label-col', default=None)
    p.add_argument('--abnormal-value', default=None)
    p.add_argument('--exclude', default='')
    p.add_argument('--topk', type=int, default=300,
                   help='保留幾個參數（預設 300；超過 500 不建議）')
    p.add_argument('--mode', choices=('variance', 'welch'),
                   default='variance',
                   help='top-K 的排序依據。variance 不用標籤（預設）')
    p.add_argument('--candidates', default=None,
                   help='候選參數清單（txt 一行一個，或 csv）；無條件保留')
    p.add_argument('--cand-col', default=None,
                   help='candidates 為 csv 時，參數名稱所在的欄名')
    p.add_argument('--near-const-frac', type=float, default=1e-8,
                   help='變異數小於「全體最大變異數 × 此值」視為近常數')
    p.add_argument('--dup-corr', type=float, default=0.999,
                   help='相關係數絕對值超過此值視為重複欄（0 = 不檢查）')
    p.add_argument('--lookback', type=int, default=None,
                   help='只用於記憶體估計，預設取 config 的 mtngc.lookback')
    p.add_argument('--hidden', type=int, default=None,
                   help='只用於記憶體估計，預設取 config 的 mtngc.hidden_dim')
    p.add_argument('--out', default=None)
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def resolve(args):
    cfg, mc = {}, {}
    if args.config:
        if not os.path.exists(args.config):
            sys.exit('找不到設定檔：%s' % args.config)
        with open(args.config, 'r', encoding='utf-8') as f:
            full = json.load(f)
        cfg = full.get('data', {})
        mc = full.get('mtngc', {})
    out = {
        'csv': args.csv or cfg.get('train_csv') or cfg.get('csv'),
        'wafer_col': args.wafer_col or cfg.get('wafer_col'),
        'label_col': args.label_col or cfg.get('label_col'),
        'abnormal_value': args.abnormal_value or cfg.get('abnormal_value'),
        'exclude': list(cfg.get('exclude', [])),
        'lookback': args.lookback or mc.get('lookback', 3),
        'hidden': args.hidden or mc.get('hidden_dim', 64),
    }
    out['exclude'] += [c.strip() for c in args.exclude.split(',') if c.strip()]
    miss = [k for k in ('csv', 'wafer_col', 'label_col') if not out[k]]
    if miss:
        sys.exit('缺少必要參數：%s\n請用 --config，或逐一指定 --csv / '
                 '--wafer-col / --label-col。' % ', '.join(miss))
    return out


def load_candidates(path, col, emit):
    """讀候選清單，支援 txt（一行一個）與 csv。"""
    if not path:
        return []
    if not os.path.exists(path):
        sys.exit('找不到候選清單：%s' % path)
    if path.lower().endswith('.csv'):
        df = read_csv_any(path, emit)
        if col and col in df.columns:
            names = df[col].astype(str).tolist()
        else:
            # 沒指定欄名時取第一個看起來像名稱的欄（非數值欄）
            objs = [c for c in df.columns if df[c].dtype == object]
            if not objs:
                sys.exit('候選 csv 裡找不到文字欄，請用 --cand-col 指定。')
            names = df[objs[0]].astype(str).tolist()
            emit('  （未指定 --cand-col，取第一個文字欄「%s」）' % objs[0])
    else:
        for enc in ENCODINGS:
            try:
                with open(path, 'r', encoding=enc) as f:
                    names = [ln.strip() for ln in f]
                break
            except UnicodeDecodeError:
                continue
        else:
            sys.exit('無法解碼候選清單：%s' % path)
    seen, out = set(), []
    for n in names:
        n = n.strip()
        if n and not n.startswith('#') and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def main():
    args = parse_args()
    dc = resolve(args)
    emit = Log()

    emit('=' * 70)
    emit('訓練前參數篩選')
    emit('=' * 70)
    emit('原始檔：%s（唯讀）' % dc['csv'])
    emit('模式  ：%s%s'
         % (args.mode,
            '（不使用標籤）' if args.mode == 'variance' else '（使用標籤）'))

    if not os.path.exists(dc['csv']):
        sys.exit('找不到資料檔：%s' % dc['csv'])
    raw = read_csv_any(dc['csv'], emit)
    emit('  原始 shape：%s' % (raw.shape,))

    key_cols = [c.strip() for c in dc['wafer_col'].split(',') if c.strip()]
    miss = [c for c in key_cols + [dc['label_col']] if c not in raw.columns]
    if miss:
        sys.exit('找不到欄位 %s。前 40 個實際欄位：\n%s'
                 % (miss, list(raw.columns)[:40]))

    exclude = set(DEFAULT_META) | set(key_cols) | {dc['label_col']}
    exclude |= set(dc['exclude'])
    cols = [c for c in raw.select_dtypes(include=[np.number]).columns
            if c not in exclude]
    if not cols:
        sys.exit('沒有可用的參數欄位。')

    n_orig = len(cols)
    lb, hd = int(dc['lookback']), int(dc['hidden'])
    g1, g3, p1 = mem_gb(len(cols), lb, hd)
    emit('  參數欄位：%d 個' % len(cols))
    emit('')
    emit('[0] 篩選前的記憶體估計（lookback=%d, hidden_dim=%d）' % (lb, hd))
    emit('    模型參數量 %s → 權重 %.2f GB，含 Adam 動量 %.2f GB'
         % (format(int(p1), ','), g1, g3))
    if p1 > 2 ** 31:
        emit('    ⚠️ 模型參數量超過 int32 上限（2,147,483,647）—— 大小計算')
        emit('       溢位會配置到 NULL，Windows 顯示「指令位置 0x00000000」。')

    lab = raw[dc['label_col']].astype(str)
    av = str(dc['abnormal_value']) if dc['abnormal_value'] else None
    is_ab = (lab == av) if av else pd.Series(False, index=raw.index)
    emit('')
    emit('  標籤分布：%s'
         % '、'.join('%s %d' % (k, v) for k, v in lab.value_counts().items()))

    # 圖是從 Normal 學的，所有篩選統計都在 Normal 上算
    normal = raw.loc[~is_ab, cols]
    if len(normal) == 0:
        emit('  ⚠️ 沒有 Normal 資料，改用全部資料計算統計量。')
        normal = raw[cols]
    emit('  篩選統計以 %d 列 Normal 資料計算' % len(normal))

    cand = load_candidates(args.candidates, args.cand_col, emit)
    if cand:
        hit = [c for c in cand if c in cols]
        missc = [c for c in cand if c not in cols]
        emit('  候選清單：%d 個，其中 %d 個在資料中' % (len(cand), len(hit)))
        if missc:
            emit('    ⚠️ 以下候選不在參數欄位裡（拼法不同或被 exclude）：')
            emit('       %s%s' % (', '.join(map(str, missc[:10])),
                                  ' ...' if len(missc) > 10 else ''))
        cand = hit

    # ---- 1 常數欄 ----
    emit('')
    emit('[1] 常數欄（變異數為 0）')
    var = normal.var(axis=0, ddof=0)
    const = [c for c in cols if not np.isfinite(var[c]) or var[c] == 0]
    emit('    丟棄 %d 個' % len(const))
    if const:
        emit('       %s%s' % (', '.join(map(str, const[:8])),
                              ' ...' if len(const) > 8 else ''))
    # 候選若是常數，留著也學不到任何邊，但使用者必須知道
    cand_const = [c for c in cand if c in set(const)]
    if cand_const:
        emit('    ⚠️ 以下候選參數在 Normal 資料裡是常數，無法保留：')
        emit('       %s' % ', '.join(map(str, cand_const)))
        emit('       （常數欄無法學到任何邊，進圖只會是孤立節點）')
    cols = [c for c in cols if c not in const]
    cand = [c for c in cand if c not in set(const)]
    if not cols:
        sys.exit('所有參數都是常數，沒有可用欄位。')

    # ---- 2 近常數欄 ----
    emit('')
    emit('[2] 近常數欄（變異數 < 最大變異數 × %g）' % args.near_const_frac)
    var = normal[cols].var(axis=0, ddof=0)
    thr = var.max() * args.near_const_frac
    cand_set = set(cand)
    nearc = [c for c in cols if var[c] < thr and c not in cand_set]
    protected = [c for c in cols if var[c] < thr and c in cand_set]
    emit('    門檻 %.6g，丟棄 %d 個' % (thr, len(nearc)))
    if protected:
        emit('    %d 個候選參數低於門檻但予以保留：%s'
             % (len(protected), ', '.join(map(str, protected[:8]))))
    cols = [c for c in cols if c not in set(nearc)]

    # ---- 3 重複欄 ----
    emit('')
    emit('[3] 重複欄（|相關係數| > %.4f）' % args.dup_corr)
    dup = []
    if args.dup_corr > 0 and len(cols) > 1:
        if len(cols) > 1500:
            emit('    參數仍有 %d 個，完整相關矩陣過大；' % len(cols))
            emit('    先依變異數取前 1500 個再做重複檢查（候選一律保留）。')
            pre = var[cols].sort_values(ascending=False).index[:1500].tolist()
            keep_pre = set(pre) | cand_set
            n_add = len(keep_pre) - len(set(pre))
            if n_add:
                emit('    其中 %d 個候選本來在 1500 名之外，已補回。' % n_add)
            cols = [c for c in cols if c in keep_pre]
        sub = normal[cols]
        # 標準化後用內積算相關，比 DataFrame.corr 省記憶體
        X = sub.values.astype(np.float64)
        X = X - X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, ddof=0)
        sd[sd == 0] = 1.0
        X = X / sd
        C = np.abs(X.T @ X) / len(X)
        np.fill_diagonal(C, 0.0)
        kept_idx, drop_set = [], set()
        # 候選先走，確保一組重複欄裡留下的是候選而不是它的複本；
        # 其餘依變異數由大到小。
        by_var = list(np.argsort(-var[cols].values))
        ci = [k for k, c in enumerate(cols) if c in cand_set]
        order = ci + [i for i in by_var if i not in set(ci)]
        for i in order:
            if i in drop_set:
                continue
            kept_idx.append(i)
            same = np.where(C[i] > args.dup_corr)[0]
            for j in same:
                if j != i:
                    drop_set.add(j)
        dup = [cols[j] for j in sorted(drop_set)]
        emit('    丟棄 %d 個' % len(dup))
        if dup:
            emit('       %s%s' % (', '.join(map(str, dup[:8])),
                                  ' ...' if len(dup) > 8 else ''))
        cols = [c for c in cols if c not in set(dup)]
    else:
        emit('    已跳過。')

    emit('')
    emit('  免費篩選後剩餘：%d 個參數' % len(cols))
    g1, g3, p1 = mem_gb(len(cols), lb, hd)
    emit('  此時記憶體估計：權重 %.2f GB，含動量 %.2f GB' % (g1, g3))

    # ---- 4 top-K ----
    emit('')
    emit('[4] top-K 篩選（保留 %d 個，依 %s）' % (args.topk, args.mode))
    if len(cols) <= args.topk:
        emit('    剩餘數已不超過 %d，無需再篩。' % args.topk)
        keep = list(cols)
    else:
        if args.mode == 'welch':
            if not av or not is_ab.any():
                sys.exit('welch 模式需要異常樣本，但找不到標籤值「%s」。'
                         % dc['abnormal_value'])
            a = raw.loc[is_ab, cols]
            b = raw.loc[~is_ab, cols]
            ma, mb = a.mean(), b.mean()
            va, vb = a.var(ddof=1), b.var(ddof=1)
            se = np.sqrt(va / max(len(a), 1) + vb / max(len(b), 1))
            se = se.replace(0, np.nan)
            score = (ma - mb).abs() / se
            score = score.fillna(0.0)
            emit('    ⚠️ 此模式用到標籤。之後不可主張「圖未看過標籤」。')
        else:
            score = normal[cols].var(axis=0, ddof=0)

        ranked = score.sort_values(ascending=False)
        # 候選無條件保留，再用排名補滿
        keep = [c for c in cand if c in cols]
        n_cand = len(keep)
        for c in ranked.index:
            if len(keep) >= args.topk:
                break
            if c not in set(keep):
                keep.append(c)
        if n_cand:
            emit('    候選無條件保留 %d 個，排名補滿至 %d 個'
                 % (n_cand, len(keep)))
            # 候選若排名很後面，值得讓使用者知道
            rk = {c: i + 1 for i, c in enumerate(ranked.index)}
            low = [(c, rk[c]) for c in cand if c in rk and rk[c] > args.topk]
            if low:
                emit('      以下候選本來排在 %d 名之外，是靠保留機制進來的：'
                     % args.topk)
                for c, r in low[:10]:
                    emit('        %-24s 第 %d 名' % (str(c)[:24], r))
        else:
            emit('    依排名取前 %d 個' % len(keep))
        # 維持原始欄序，讓輸出檔與原檔的欄位順序一致
        kset = set(keep)
        keep = [c for c in cols if c in kset]

    g1, g3, p1 = mem_gb(len(keep), lb, hd)
    emit('')
    emit('[5] 結果')
    emit('    最終保留 %d 個參數（原始 %d 個，共丟棄 %d 個）'
         % (len(keep), n_orig, n_orig - len(keep)))
    emit('    模型參數量 %s → 權重 %.2f GB，含 Adam 動量 %.2f GB'
         % (format(int(p1), ','), g1, g3))
    if p1 > 2 ** 31:
        emit('    ⚠️ 仍超過 int32 上限，請再調小 --topk。')
    elif g3 > 4:
        emit('    ⚠️ 含動量超過 4 GB，在一般機器上仍可能失敗，'
             '建議 --topk 再減半。')
    else:
        emit('    在一般機器上可以執行。')

    # ---- 輸出 ----
    emit('')
    emit('[6] 輸出')
    if args.out:
        out_csv = args.out
    else:
        base, ext = os.path.splitext(dc['csv'])
        out_csv = base + '_reduced' + (ext or '.csv')
    out_txt = os.path.splitext(out_csv)[0] + '.txt'
    out_log = os.path.splitext(out_csv)[0] + '.log'

    if args.dry_run:
        emit('    --dry-run 已指定，不寫出檔案。將產生：')
        emit('      %s' % out_csv)
        emit('      %s' % out_txt)
        emit('      %s' % out_log)
    else:
        d = os.path.dirname(os.path.abspath(out_csv))
        if d and not os.path.isdir(d):
            os.makedirs(d)
        meta = [c for c in raw.columns
                if c in set(key_cols) | {dc['label_col']}
                or c in set(DEFAULT_META)]
        meta = [c for c in raw.columns if c in set(meta)]  # 保持原欄序
        keep_set = set(keep)
        out_cols = [c for c in raw.columns if c in set(meta) or c in keep_set]
        raw[out_cols].to_csv(out_csv, index=False, encoding='utf-8-sig')
        with open(out_txt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(map(str, keep)) + '\n')
        emit('    已存：%s（shape %s）'
             % (out_csv, (len(raw), len(out_cols))))
        emit('    已存：%s（%d 個參數名稱）' % (out_txt, len(keep)))

    emit('')
    emit('=' * 70)
    emit('下一步：把 run_all 設定檔的 data.train_csv 指向篩選後的檔')
    emit('  "train_csv": "%s"' % out_csv.replace('\\', '/'))
    emit('  data.csv 維持原樣（分析階段仍可看全部參數）')
    emit('=' * 70)
    emit('注意：圖只會包含這 %d 個參數。下游的分組與規則萃取若用到'
         % len(keep))
    emit('      不在圖裡的參數，那些參數會被當成孤立節點處理。')

    if not args.dry_run:
        emit.save(out_log)
        print('\n紀錄已存：%s' % out_log)


if __name__ == '__main__':
    main()