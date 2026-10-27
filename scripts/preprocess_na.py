# -*- coding: utf-8 -*-
"""
preprocess_na.py  ——  NA 前處理（獨立步驟，不修改原始檔）

用途
----
  在跑任何分析之前，先把含缺值的資料清理成一份新檔，原始檔完全不動。
  這樣同事的 pipeline 可以繼續讀原始檔，不受影響。

為什麼需要這一步
----------------
  pandas 的 groupby().mean() 會「靜默跳過」NaN。含缺值的 wafer 照樣會
  產生數值，只是那個平均是用較少的點算出來的，而且在後續統計檢定裡
  和其他 wafer 等權重比較。這比直接報錯更危險，因為結果看起來正常。

處理策略（兩層）
----------------
  1. 缺值比例超過門檻的參數 → 丟「欄」。
     若某 sensor 有 95% 缺值，丟列會把 95% 的資料砍光，所以先丟欄。
  2. 其餘含缺值的列 → 丟「列」。

  兩層都會回報前後的類別組成與 wafer 存活情況，因為丟列可能不是均勻的：
  若異常批的缺值較多（感測器在異常時斷線、取樣失敗），丟列等於系統性
  移除訊號。

執行範例
--------
  # 沿用 pipeline 的 config（欄位定義保證一致，建議這樣用）
  python scripts\\preprocess_na.py --config config\\case1.json

  # 自行指定
  python scripts\\preprocess_na.py ^
    --csv data\\raw\\semiconductor\\case1.csv ^
    --wafer-col LOT_ID,WAFER_ID ^
    --label-col "Symbol Define" ^
    --abnormal-value Abnormal

  # 只檢查不輸出（先看會砍掉多少）
  python scripts\\preprocess_na.py --config config\\case1.json --dry-run

輸出
----
  <原檔名>_clean.csv   清理後的資料（utf-8-sig，Excel 可直接開）
  <原檔名>_clean.log   這次處理的完整紀錄（可附在報告附錄）

環境：Python 3.6+ / pandas / numpy
"""

from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# 與 run_pipeline.py 保持一致
ENCODINGS = ('utf-8-sig', 'utf-8', 'cp950', 'big5hkscs', 'gbk', 'latin-1')

DEFAULT_META = ['WFR_ID', 'WAFER_ID', 'LOT_ID', 'START_DATETIME', 'END_DATETIME',
                'STEP', 'STEP_ID', 'RECIPE', 'CHAMBER', 'EQP_ID', 'TOOL_ID',
                'TIME', 'TIMESTAMP', 'label', 'LABEL']


# ---------------------------------------------------------------- 工具

class Log(object):
    """同時印到螢幕與收集成檔。"""

    def __init__(self):
        self.lines = []

    def __call__(self, s=''):
        print(s)
        self.lines.append(s)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.lines) + '\n')


def read_csv_any(path, emit):
    """依序嘗試常見編碼讀 csv。台灣環境的 csv 常為 CP950/Big5。"""
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


def parse_args():
    p = argparse.ArgumentParser(
        description='NA 前處理：清理缺值並另存新檔，不修改原始檔')
    p.add_argument('--config', default=None,
                   help='沿用 run_pipeline 的 config（取其 data 區塊）')
    p.add_argument('--csv', default=None, help='原始 csv 路徑')
    p.add_argument('--wafer-col', default=None,
                   help='wafer 識別欄，複合鍵用逗號分隔，如 LOT_ID,WAFER_ID')
    p.add_argument('--label-col', default=None, help='標籤欄名')
    p.add_argument('--abnormal-value', default=None, help='異常標籤的值')
    p.add_argument('--exclude', default='',
                   help='額外排除的欄位，逗號分隔')
    p.add_argument('--out', default=None,
                   help='輸出路徑（預設 <原檔名>_clean.csv）')
    p.add_argument('--drop-col-na-frac', type=float, default=0.5,
                   help='缺值比例超過此值的參數直接丟欄（0 = 關閉，預設 0.5）')
    p.add_argument('--keep-dropped-cols', action='store_true',
                   help='高缺值參數保留在輸出檔中（僅回報不丟）')
    p.add_argument('--min-rows-per-wafer', type=int, default=0,
                   help='清理後列數少於此值的 wafer 整片丟棄（0 = 不丟）')
    p.add_argument('--dry-run', action='store_true',
                   help='只檢查並回報，不寫出任何檔案')
    return p.parse_args()


def resolve_config(args):
    """config 提供底值，命令列參數覆蓋之。"""
    cfg = {}
    if args.config:
        if not os.path.exists(args.config):
            sys.exit('找不到設定檔：%s' % args.config)
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = json.load(f).get('data', {})

    out = {
        'csv': args.csv or cfg.get('csv'),
        'wafer_col': args.wafer_col or cfg.get('wafer_col'),
        'label_col': args.label_col or cfg.get('label_col'),
        'abnormal_value': args.abnormal_value or cfg.get('abnormal_value'),
        'exclude': list(cfg.get('exclude', [])),
    }
    out['exclude'] += [c.strip() for c in args.exclude.split(',') if c.strip()]

    miss = [k for k in ('csv', 'wafer_col', 'label_col') if not out[k]]
    if miss:
        sys.exit('缺少必要參數：%s\n請用 --config 指定設定檔，或用 --csv / '
                 '--wafer-col / --label-col 逐一指定。' % ', '.join(miss))
    return out


# ---------------------------------------------------------------- 主流程

def main():
    args = parse_args()
    dc = resolve_config(args)
    emit = Log()

    emit('=' * 70)
    emit('NA 前處理')
    emit('=' * 70)
    emit('原始檔：%s' % dc['csv'])
    if args.config:
        emit('設定檔：%s' % args.config)
    emit('※ 原始檔為唯讀，本程式只會另存新檔')

    if not os.path.exists(dc['csv']):
        sys.exit('找不到資料檔：%s' % dc['csv'])
    raw = read_csv_any(dc['csv'], emit)
    emit('  原始 shape：%s' % (raw.shape,))

    # ---- 欄位定位（與 run_pipeline 同一套規則）----
    key_cols = [c.strip() for c in dc['wafer_col'].split(',') if c.strip()]
    miss = [c for c in key_cols + [dc['label_col']] if c not in raw.columns]
    if miss:
        sys.exit('找不到欄位 %s。實際欄位：\n%s'
                 % (miss, list(raw.columns)[:40]))

    if len(key_cols) == 1:
        wkey_series = raw[key_cols[0]].astype(str)
    else:
        wkey_series = raw[key_cols].astype(str).agg('_'.join, axis=1)
    emit('  wafer 鍵：%s → %d 個唯一值'
         % (' + '.join(key_cols), wkey_series.nunique()))

    exclude = set(DEFAULT_META) | set(key_cols) | {dc['label_col']}
    exclude |= set(dc['exclude'])
    sensor_cols = [c for c in raw.select_dtypes(include=[np.number]).columns
                   if c not in exclude]
    if not sensor_cols:
        sys.exit('沒有可用的參數欄位，請檢查 --exclude。')
    emit('  參數欄位：%d 個' % len(sensor_cols))

    lab = raw[dc['label_col']].astype(str)
    av = str(dc['abnormal_value']) if dc['abnormal_value'] else None

    # ---- wafer 層級的基準線（分析單位是 wafer，不是 row）----
    waf_lab_b = pd.DataFrame({'w': wkey_series, 'l': lab}) \
        .drop_duplicates().set_index('w')['l']
    n_waf_b = len(waf_lab_b)

    # ---- 缺值盤點 ----
    emit('')
    emit('[1] 缺值盤點')
    n_cell = int(raw[sensor_cols].isna().sum().sum())
    if n_cell == 0:
        emit('  參數欄位無缺值，無需處理。')
        if not args.dry_run:
            emit('  （仍會另存一份，讓後續流程統一讀 _clean 檔）')
    else:
        na_frac = raw[sensor_cols].isna().mean()
        emit('  共 %d 個缺值（佔 %.2f%%），分布於 %d / %d 個參數'
             % (n_cell, 100.0 * n_cell / (len(raw) * len(sensor_cols)),
                int((na_frac > 0).sum()), len(sensor_cols)))
        worst = na_frac[na_frac > 0].sort_values(ascending=False)
        emit('  缺值最多的參數：')
        for c, v in worst.head(8).items():
            emit('    %-24s %6.1f%%' % (str(c)[:24], 100.0 * v))
        if len(worst) > 8:
            emit('    （其餘 %d 個略）' % (len(worst) - 8))

    # ---- 丟欄 ----
    dropped_cols = []
    thr = args.drop_col_na_frac
    emit('')
    emit('[2] 高缺值參數（門檻 %s）'
         % ('已關閉' if thr <= 0 else '%.0f%%' % (100.0 * thr)))
    if n_cell == 0:
        emit('  無缺值，無需丟欄。')
    elif thr <= 0:
        emit('  --drop-col-na-frac 為 0，不丟欄；所有缺值都走丟列處理。')
    else:
        na_frac = raw[sensor_cols].isna().mean()
        dropped_cols = [c for c in sensor_cols if na_frac[c] > thr]
        if not dropped_cols:
            emit('  無參數超過門檻。')
        elif args.keep_dropped_cols:
            emit('  %d 個參數超過門檻，但 --keep-dropped-cols 已指定，'
                 '保留在輸出檔中：' % len(dropped_cols))
            for c in dropped_cols:
                emit('    %-24s %6.1f%%' % (str(c)[:24], 100.0 * na_frac[c]))
            emit('  ⚠️ 這些欄仍含缺值，丟列階段會砍掉大量資料。')
            dropped_cols = []
        else:
            emit('  丟棄 %d 個參數（丟欄而非丟列，否則資料會被砍光）：'
                 % len(dropped_cols))
            for c in dropped_cols:
                emit('    %-24s %6.1f%%' % (str(c)[:24], 100.0 * na_frac[c]))
            sensor_cols = [c for c in sensor_cols if c not in dropped_cols]
            if not sensor_cols:
                sys.exit('所有參數的缺值都超過 %.0f%%，無可用欄位。\n'
                         '請調高 --drop-col-na-frac 或檢查資料來源。'
                         % (100.0 * thr))
            emit('  剩餘參數：%d 個' % len(sensor_cols))

    # ---- 丟列 ----
    emit('')
    emit('[3] 丟棄含缺值的列')
    # 被判定要丟的欄，必須真的從輸出中移除，否則清理後的檔仍帶缺值
    out_base = raw.drop(columns=dropped_cols) if dropped_cols else raw

    keep = raw[sensor_cols + [dc['label_col']]].notna().all(axis=1)
    n_drop = int((~keep).sum())
    n_before = len(raw)
    if n_drop == 0:
        emit('  無列需要丟棄。')
        clean = out_base.copy()
    else:
        lab_b = lab.value_counts()
        clean = out_base[keep].copy()
        lab_a = clean[dc['label_col']].astype(str).value_counts()
        emit('  丟棄 %d / %d 列（%.1f%%），剩餘 %d 列'
             % (n_drop, n_before, 100.0 * n_drop / n_before, len(clean)))
        emit('  各標籤的列數變化：')
        for k in lab_b.index:
            b, a = int(lab_b.get(k, 0)), int(lab_a.get(k, 0))
            emit('    %-14s %6d → %6d（丟棄 %.1f%%）'
                 % (str(k)[:14], b, a, 100.0 * (b - a) / max(b, 1)))
        if av and av in lab_b.index:
            r_ab = (lab_b.get(av, 0) - lab_a.get(av, 0)) \
                / max(lab_b.get(av, 1), 1)
            r_all = n_drop / float(max(n_before, 1))
            if r_ab > max(2 * r_all, r_all + 0.1):
                emit('  ⚠️ 異常批被丟棄的比例（%.1f%%）明顯高於整體'
                     '（%.1f%%）。' % (100.0 * r_ab, 100.0 * r_all))
                emit('     缺值可能與異常相關，丟列會系統性地移除訊號。'
                     '建議另跑一次保留缺值的版本對照。')
        if len(clean) == 0:
            sys.exit('丟棄缺值後沒有資料剩下。\n'
                     '請調低 --drop-col-na-frac（讓更多高缺值欄被丟掉），'
                     '或檢查資料來源。')

    # ---- wafer 存活檢查（這是真正要看的，分析單位是 wafer）----
    emit('')
    emit('[4] wafer 存活檢查')
    if len(key_cols) == 1:
        wk_a = clean[key_cols[0]].astype(str)
    else:
        wk_a = clean[key_cols].astype(str).agg('_'.join, axis=1)
    rows_per = wk_a.value_counts()
    alive = set(rows_per.index)
    lost = [w for w in waf_lab_b.index if w not in alive]

    emit('  wafer 數：%d → %d' % (n_waf_b, len(alive)))
    if lost:
        lost_lab = waf_lab_b.loc[lost].value_counts()
        emit('  ❌ 有 %d 片 wafer 的所有列都被丟掉，整片消失：' % len(lost))
        for k, v in lost_lab.items():
            tot = int((waf_lab_b == k).sum())
            emit('    %-14s %d / %d 片（%.1f%%）'
                 % (str(k)[:14], int(v), tot, 100.0 * v / max(tot, 1)))
        emit('    消失的 wafer：%s%s'
             % (', '.join(map(str, lost[:10])),
                ' ...' if len(lost) > 10 else ''))
        if av and av in lost_lab.index:
            emit('    ⚠️ 其中含異常 wafer。異常樣本數本來就少，'
                 '少一片對統計檢定的影響遠大於少幾列。')
    else:
        emit('  無 wafer 整片消失。')

    # 每片剩幾列：列數太少的 wafer，其平均值雜訊大
    if len(rows_per):
        q = rows_per.quantile([0, 0.05, 0.5, 1.0])
        emit('  每片剩餘列數：最少 %d / 5%% 分位 %d / 中位 %d / 最多 %d'
             % (int(q.iloc[0]), int(q.iloc[1]), int(q.iloc[2]),
                int(q.iloc[3])))
        if args.min_rows_per_wafer > 0:
            thin = [w for w, n in rows_per.items()
                    if n < args.min_rows_per_wafer]
            if thin:
                thin_lab = waf_lab_b.loc[[w for w in thin
                                          if w in waf_lab_b.index]]
                emit('  丟棄列數 < %d 的 wafer：%d 片（%s）'
                     % (args.min_rows_per_wafer, len(thin),
                        ', '.join('%s %d' % (k, v) for k, v
                                  in thin_lab.value_counts().items())))
                clean = clean[~wk_a.isin(thin)].copy()
                emit('  剩餘 %d 列 / %d 片'
                     % (len(clean), len(alive) - len(thin)))
            else:
                emit('  無 wafer 列數低於 %d。' % args.min_rows_per_wafer)
        elif int(q.iloc[0]) < 3:
            emit('  ⚠️ 有 wafer 只剩不到 3 列，其平均值雜訊大。'
                 '可用 --min-rows-per-wafer 丟掉這些片。')

    # ---- 輸出前自我檢查 ----
    emit('')
    emit('[5] 輸出前檢查')
    left_sensor = int(clean[sensor_cols].isna().sum().sum())
    if left_sensor:
        sys.exit('內部錯誤：清理後參數欄仍有 %d 個缺值，請回報。' % left_sensor)
    emit('  參數欄位：%d 個，無缺值 ✓' % len(sensor_cols))
    other = [c for c in clean.columns if c not in sensor_cols]
    left_other = int(clean[other].isna().sum().sum()) if other else 0
    if left_other:
        oc = clean[other].isna().sum()
        emit('  非參數欄位仍有 %d 個缺值（未處理，因為丟列只看參數與標籤）：'
             % left_other)
        for c, v in oc[oc > 0].items():
            emit('    %-24s %d' % (str(c)[:24], int(v)))
        emit('  這些欄不進入分析，但若你的其他程式會用到，需自行處理。')
    else:
        emit('  其餘欄位：無缺值 ✓')

    # ---- 輸出 ----
    emit('')
    emit('[6] 輸出')
    if args.out:
        out_csv = args.out
    else:
        base, ext = os.path.splitext(dc['csv'])
        out_csv = base + '_clean' + (ext or '.csv')
    out_log = os.path.splitext(out_csv)[0] + '.log'

    if args.dry_run:
        emit('  --dry-run 已指定，不寫出檔案。')
        emit('  若要實際輸出，移除 --dry-run 後重跑，將產生：')
        emit('    %s' % out_csv)
        emit('    %s' % out_log)
    else:
        out_dir = os.path.dirname(os.path.abspath(out_csv))
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        # utf-8-sig：Windows 的 Excel 直接雙擊開啟不會亂碼
        clean.to_csv(out_csv, index=False, encoding='utf-8-sig')
        emit('  已存：%s' % out_csv)
        emit('  shape：%s（原始 %s）' % ((clean.shape,), (raw.shape,)))
        if dropped_cols:
            emit('')
            emit('  ⚠️ 輸出檔已移除 %d 個高缺值參數。若同事的 pipeline 讀的是'
                 '原始檔，' % len(dropped_cols))
            emit('     兩邊看到的參數集合會不一致。需要一致時，請他在他的'
                 '設定中排除：')
            emit('     %s' % ', '.join(map(str, dropped_cols)))

    emit('')
    emit('=' * 70)
    emit('下一步：把 config 的 data.csv 指向清理後的檔，再跑分析')
    emit('  "csv": "%s"' % out_csv.replace('\\', '/'))
    emit('  python scripts\\run_pipeline.py --config config\\<你的設定>.json')
    emit('（清理後已無缺值，run_pipeline 的 NA 檢查會顯示「無缺值」）')
    emit('=' * 70)

    if not args.dry_run:
        emit.save(out_log)
        print('\n紀錄已存：%s' % out_log)


if __name__ == '__main__':
    main()