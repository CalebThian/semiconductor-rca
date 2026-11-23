# -*- coding: utf-8 -*-
"""stability_sweep.py —— 可調參數的穩定度實驗

問題：縮減到不同的參數量、edge_k 設不同的值，根因分析的結論會不會變？

做法是掃描設定、收集每次的結論、量它們彼此有多一致。重點不是哪一組設定
最好（沒有驗證集外的標準可以判斷），而是**結論對設定有多敏感**。設定換一
個數字就換一組根因，那份報告就不能拿去跟產線講。

兩層結構，因為兩類旋鈕的成本差兩個數量級：

  第一層（貴）：reduce.max_params 改變訓練的參數集，必須重新訓練。
  第二層（便宜）：edge_k / min_edge / min_gain 只影響步驟 3，一張圖可以
                  重複使用。

所以每個 max_params 只訓練一次，再對那張圖掃所有步驟 3 的組合。

執行
----
  # 預設：3 個參數量 × 3 個 edge_k = 9 次分析、3 次訓練
  python scripts/stability_sweep.py --config config/case2.json \\
      --out results/stability

  # 自訂網格
  python scripts/stability_sweep.py --config config/case2.json \\
      --n-params 30,50,100,200 --edge-k 1,2,3 \\
      --min-edge 0.4,0.6,0.8 --min-gain 0.01,0.02 \\
      --out results/stability

  # 已經訓練過，只想重掃步驟 3
  python scripts/stability_sweep.py --config config/case2.json \\
      --out results/stability --skip-train

候選集
------
預設讓分析自己依效果量挑候選（--candidates auto），流程自足、不需要同事的
pipeline。要把候選集固定成同事某一輪的輸出，用 --candidates <checklist.csv>
—— 那樣候選不隨設定變動，量到的就純粹是圖與規則萃取的敏感度。

環境：Python 3.8+、numpy、pandas；訓練需要 torch
"""

from __future__ import print_function

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# =====================================================================
# 工具
# =====================================================================

def nums(s, typ=float):
    """"1,2,3" → [1.0, 2.0, 3.0]；空字串 → []"""
    return [typ(x) for x in str(s).split(',') if str(x).strip() != '']


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def dump_json(obj, path):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def deep_merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def run(cmd, cwd=None, quiet=True):
    """回傳 (returncode, 輸出)。失敗不中止，讓掃描繼續。"""
    r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    out = r.stdout.decode('utf-8', 'replace')
    if r.returncode != 0 and not quiet:
        print(out[-2000:])
    return r.returncode, out


# =====================================================================
# 第一層：每個參數量訓練一次
# =====================================================================

def train_one(base_cfg, base_path, n_param, work, py):
    """以 reduce.max_params=n_param 跑步驟 0+1，回傳圖的路徑。"""
    gdir = os.path.join(work, 'graphs', 'N%d' % n_param)
    cfg = json.loads(json.dumps(base_cfg))
    deep_merge(cfg, {
        'reduce': {
            'enabled': True,
            'max_params': int(n_param),
            # 0a 的輸出與 max_params 無關，各個 N 共用一份即可
            'clean_out': os.path.join(work, 'data', 'clean.csv'),
            'out': os.path.join(work, 'data', 'N%d_reduced.csv' % n_param),
        },
        'mtngc': {'graph_dir': gdir},
    })
    cpath = os.path.join(work, 'cfg', 'train_N%d.json' % n_param)
    dump_json(cfg, cpath)
    rc, out = run([py, os.path.join(HERE, 'run_all.py'),
                   '--config', cpath, '--only', '1'], cwd=ROOT)
    if rc != 0:
        print('  ❌ N=%d 訓練失敗' % n_param)
        print('\n'.join(out.strip().splitlines()[-12:]))
        return None, out
    # run_all 依 mtngc.mode 決定檔名，直接找產物
    cands = [os.path.join(gdir, f) for f in sorted(os.listdir(gdir))
             if f.endswith('.npy')] if os.path.isdir(gdir) else []
    return (cands[0] if cands else None), out


# =====================================================================
# 第二層：每張圖掃步驟 3 的旋鈕
# =====================================================================

def analyse_one(base_cfg, gpath, knobs, work, py, cand_file, tag):
    """以指定的圖與旋鈕跑步驟 3，回傳 result.json 的路徑。"""
    dc = base_cfg['data']
    ac = base_cfg.get('analysis', {}) or {}
    prefix = os.path.join(work, 'runs', tag)
    sub = {
        'data': {
            'csv': dc['csv'],
            'wafer_col': dc.get('wafer_col', 'LOT_ID,WFR_ID,START_DATETIME'),
            'label_col': dc.get('label_col', 'Symbol Define'),
            'abnormal_value': dc.get('abnormal_value', 'Abnormal'),
            'exclude': dc.get('exclude', []),
            'dropna': dc.get('dropna', 'row'),
            'drop_col_na_frac': dc.get('drop_col_na_frac', 0.5),
            'max_row_loss': dc.get('max_row_loss', 0.3),
            'min_row_keep': dc.get('min_row_keep', 0.8),
        },
        'candidates': {
            'file': cand_file or '',
            'sensor_col': ac.get('sensor_col', 'Sensor'),
            'priority_col': ac.get('priority_col', 'Priority'),
            'topk': ac.get('topk', 5),
            'min_effect': ac.get('min_effect', 0.5),
            'n_boot': ac.get('n_boot', 200),
            'n_perm': ac.get('n_perm', 300),
            'add_statistical': ac.get('add_statistical', True),
            'expansion': ac.get('expansion', {}),
            'true_roots': ac.get('true_roots', []),
        },
        'graph': dict(ac.get('graph', {}), path=gpath,
                      edge_k=int(knobs['edge_k']),
                      # 掃描時關掉自動升級，否則 edge_k 這個變因會被蓋掉
                      edge_k_large=None,
                      min_edge=knobs['min_edge']),
        'rules': dict(ac.get('rules', {}),
                      min_gain=float(knobs['min_gain'])),
        'output': {'prefix': prefix, 'brief': True, 'show_welch': False},
        'seed': base_cfg.get('seed', 0),
    }
    cpath = os.path.join(work, 'cfg', 'an_%s.json' % tag)
    dump_json(sub, cpath)
    rc, out = run([py, os.path.join(HERE, 'run_pipeline.py'),
                   '--config', cpath], cwd=ROOT)
    rj = prefix + '_result.json'
    if rc != 0 or not os.path.exists(rj):
        return None, out
    return rj, out


# =====================================================================
# 指標
# =====================================================================

def rule_params(res):
    """最終規則用到的參數名稱集合。"""
    r = (res.get('rule') or {})
    txt = r.get('final_rule') or ''
    out = set()
    for part in str(txt).split(' 且 '):
        part = part.strip()
        if not part:
            continue
        # "名稱 > 1.234"，名稱本身可能含空白，所以由右邊切
        for op in (' > ', ' < '):
            if op in part:
                out.add(part.rsplit(op, 1)[0].strip())
                break
    return out


def jac(a, b):
    a, b = set(a), set(b)
    u = len(a | b)
    return 1.0 if u == 0 else len(a & b) / float(u)


def topk_overlap(a, b, k):
    a, b = list(a)[:k], list(b)[:k]
    if not a or not b:
        return float('nan')
    return len(set(a) & set(b)) / float(min(len(a), len(b)))


def co_pairs(groups):
    """同組的參數配對集合，用來比較分組結構。"""
    out = set()
    for g in groups or []:
        ms = sorted(m['name'] for m in g.get('members', []))
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                out.add((ms[i], ms[j]))
    return out


# =====================================================================
# 主流程
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description='掃描可調參數，量根因分析結論的穩定度')
    ap.add_argument('--config', required=True, help='run_all 的設定檔')
    ap.add_argument('--out', default='results/stability', help='工作目錄')
    ap.add_argument('--n-params', default='',
                    help='reduce.max_params 的網格，逗號分隔。'
                         '空字串 = 只用設定檔的值（不重訓）')
    ap.add_argument('--edge-k', default='1,2,3')
    ap.add_argument('--min-edge', default='',
                    help='空 = 用設定檔的值（通常是自動 0.6）')
    ap.add_argument('--min-gain', default='')
    ap.add_argument('--candidates', default='auto',
                    help='auto = 分析自行挑選；或給一個 checklist.csv '
                         '把候選集固定住')
    ap.add_argument('--skip-train', action='store_true',
                    help='沿用 --out 下既有的圖，只重跑步驟 3')
    ap.add_argument('--python', default=sys.executable)
    args = ap.parse_args()

    base = load_json(args.config)
    work = os.path.abspath(args.out)
    for d in ('cfg', 'data', 'graphs', 'runs'):
        p = os.path.join(work, d)
        if not os.path.isdir(p):
            os.makedirs(p)

    ac = base.get('analysis', {}) or {}
    gc0 = ac.get('graph', {}) or {}
    rc0 = ac.get('rules', {}) or {}
    n_grid = [int(x) for x in nums(args.n_params, float)] or \
        [int((base.get('reduce', {}) or {}).get('max_params', 300))]
    k_grid = [int(x) for x in nums(args.edge_k, float)]
    e_grid = nums(args.min_edge) or [gc0.get('min_edge', None)]
    g_grid = nums(args.min_gain) or [rc0.get('min_gain', 0.02)]
    cand_file = '' if args.candidates == 'auto' else \
        os.path.abspath(args.candidates)

    print('=' * 70)
    print('穩定度掃描')
    print('=' * 70)
    print('  參數量   %s' % n_grid)
    print('  edge_k   %s' % k_grid)
    print('  min_edge %s' % ['自動' if v is None else v for v in e_grid])
    print('  min_gain %s' % g_grid)
    print('  候選集   %s' % ('分析自行挑選' if not cand_file else cand_file))
    print('  訓練 %d 次、分析 %d 次'
          % (0 if args.skip_train else len(n_grid),
             len(n_grid) * len(k_grid) * len(e_grid) * len(g_grid)))
    print('')

    # ---- 第一層：訓練 ----
    graphs = {}
    for n in n_grid:
        gdir = os.path.join(work, 'graphs', 'N%d' % n)
        existing = [os.path.join(gdir, f) for f in sorted(os.listdir(gdir))
                    if f.endswith('.npy')] if os.path.isdir(gdir) else []
        if args.skip_train and existing:
            graphs[n] = existing[0]
            print('[訓練] N=%-4d 沿用既有的圖' % n)
            continue
        print('[訓練] N=%-4d ...' % n, flush=True)
        gp, out = train_one(base, args.config, n, work, args.python)
        if gp is None:
            continue
        graphs[n] = gp
        nodes = os.path.splitext(gp)[0] + '_nodes.txt'
        n_node = (len(open(nodes, encoding='utf-8').read().split())
                  if os.path.exists(nodes) else -1)
        print('         圖 %d 個節點' % n_node)
    if not graphs:
        sys.exit('沒有任何圖訓練成功，無法繼續。')

    # ---- 第二層：分析 ----
    rows = []
    results = {}
    combos = list(itertools.product(sorted(graphs), k_grid, e_grid, g_grid))
    for idx, (n, k, e, g) in enumerate(combos, 1):
        tag = 'N%d_k%d_e%s_g%s' % (
            n, k, 'auto' if e is None else ('%.2f' % e), '%.3f' % g)
        print('[分析 %d/%d] %s' % (idx, len(combos), tag), flush=True)
        rj, out = analyse_one(base, graphs[n],
                              {'edge_k': k, 'min_edge': e, 'min_gain': g},
                              work, args.python, cand_file, tag)
        if rj is None:
            print('         ❌ 失敗：%s'
                  % ' / '.join(out.strip().splitlines()[-2:]))
            continue
        res = load_json(rj)
        results[tag] = res
        rule = res.get('rule') or {}
        test = rule.get('test') or {}
        data = res.get('data') or {}
        groups = res.get('groups') or []
        rp = rule_params(res)
        roots = set(res.get('true_roots') or [])
        order = res.get('order') or []
        rows.append({
            'tag': tag, 'max_params': n, 'edge_k': k,
            'min_edge': '自動' if e is None else e, 'min_gain': g,
            '圖節點': data.get('n_graph', -1),
            '分析參數': data.get('n_param', -1),
            '候選數': len(res.get('candidates') or []),
            '組數': len(groups),
            '落單組': sum(1 for x in groups
                          if len(x.get('members', [])) == 1),
            '規則': rule.get('final_rule', ''),
            '來源': rule.get('final_source', ''),
            '條件數': len(rp),
            '驗證F1': test.get('F1', float('nan')),
            '解釋比例': test.get('解釋比例', float('nan')),
            '誤報率': test.get('誤報率', float('nan')),
            '真根因在規則內': (len(roots & rp) if roots else -1),
            '真根因最差名次': (max((order.index(r) + 1 for r in roots
                                    if r in order), default=-1)
                               if roots else -1),
        })

    if not rows:
        sys.exit('所有分析都失敗了。')
    df = pd.DataFrame(rows)
    csv_path = os.path.join(work, 'sweep.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # ---- 報告 ----
    print('')
    print('=' * 70)
    print('結果（%d 組設定）' % len(df))
    print('=' * 70)
    show = ['tag', '圖節點', '候選數', '組數', '落單組', '條件數',
            '驗證F1', '解釋比例', '誤報率', '規則']
    with pd.option_context('display.max_colwidth', 46,
                           'display.width', 200):
        print(df[show].to_string(index=False))

    # 參數的出現率 —— 這是最該看的一段。
    # 三個欄位要分開看，因為它們的穩定度差很多：候選集是完整清單、規則是
    # 最小充分集。真根因彼此相關時，規則放了一個之後再加第二個幾乎沒有
    # 增益，於是它不會進規則 —— 那不代表它不是根因。實測模擬資料：三個
    # 真根因 100% 都在候選集裡，但其中一個只有 3% 的設定讓它進入規則。
    print('')
    print('=' * 70)
    print('各參數的出現率（%d 組設定）' % len(df))
    print('=' * 70)
    n_run = float(len(df))
    cnt_rule, cnt_cand, ranks = {}, {}, {}
    for tag, res in results.items():
        for p in rule_params(res):
            cnt_rule[p] = cnt_rule.get(p, 0) + 1
        for c in (res.get('candidates') or []):
            cnt_cand[c['name']] = cnt_cand.get(c['name'], 0) + 1
        for i, p in enumerate(res.get('order') or []):
            ranks.setdefault(p, []).append(i + 1)
    if cnt_cand:
        roots = set(next(iter(results.values())).get('true_roots') or [])
        allp = sorted(set(cnt_cand) | set(cnt_rule),
                      key=lambda p: (-cnt_cand.get(p, 0),
                                     np.mean(ranks.get(p, [999]))))
        print('  %-26s %9s %9s %12s' % ('參數', '進候選', '進規則', '排查名次'))
        for p in allp[:20]:
            rr = ranks.get(p, [])
            print('  %-26s %5d %3.0f%% %5d %3.0f%% %6s%s'
                  % (str(p)[:26],
                     cnt_cand.get(p, 0), 100 * cnt_cand.get(p, 0) / n_run,
                     cnt_rule.get(p, 0), 100 * cnt_rule.get(p, 0) / n_run,
                     ('%d~%d' % (min(rr), max(rr))) if rr else '-',
                     '  ★真根因' if p in roots else ''))
        if len(allp) > 20:
            print('  …另有 %d 個' % (len(allp) - 20))
        print('')
        print('  「進候選」100% 的參數不受設定影響，是可以寫進報告的結論。')
        print('  「進規則」偏低不代表不是根因 —— 規則是最小充分集，彼此')
        print('  相關的參數只會留下一個，其餘的增益不足而被排除。')

    # 兩兩一致性
    tags = list(results)
    if len(tags) > 1:
        rj, ro, rg = [], [], []
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                a, b = results[tags[i]], results[tags[j]]
                rj.append(jac(rule_params(a), rule_params(b)))
                ro.append(topk_overlap(a.get('order') or [],
                                       b.get('order') or [], 5))
                rg.append(jac(co_pairs(a.get('groups')),
                              co_pairs(b.get('groups'))))
        print('')
        print('=' * 70)
        print('兩兩一致性（%d 對）' % len(rj))
        print('=' * 70)
        rc_ = []
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                a_ = {c['name'] for c in (results[tags[i]].get('candidates') or [])}
                b_ = {c['name'] for c in (results[tags[j]].get('candidates') or [])}
                rc_.append(jac(a_, b_))
        for lab, v in (('候選集 Jaccard', rc_),
                       ('規則參數 Jaccard', rj),
                       ('排查順序 top-5 重疊', ro),
                       ('分組結構 Jaccard', rg)):
            v = [x for x in v if x == x]
            if v:
                print('  %-22s 平均 %.3f　最低 %.3f　完全一致 %.0f%%'
                      % (lab, float(np.mean(v)), float(np.min(v)),
                         100.0 * sum(1 for x in v if x >= 0.999) / len(v)))

    # 單一旋鈕的敏感度
    print('')
    print('=' * 70)
    print('單一旋鈕的敏感度（固定其他旋鈕，只變動該欄）')
    print('=' * 70)
    for knob in ('max_params', 'edge_k', 'min_edge', 'min_gain'):
        others = [c for c in ('max_params', 'edge_k', 'min_edge', 'min_gain')
                  if c != knob]
        vals = []
        for _key, sub in df.groupby(others):
            if len(sub) < 2:
                continue
            ps = [rule_params(results[t]) for t in sub['tag']]
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    vals.append(jac(ps[i], ps[j]))
        if vals:
            print('  %-12s 規則參數 Jaccard 平均 %.3f（%d 對）%s'
                  % (knob, float(np.mean(vals)), len(vals),
                     '' if np.mean(vals) >= 0.8 else '   ← 敏感'))
        else:
            print('  %-12s （只有一個值，未掃描）' % knob)

    print('')
    print('明細已存：%s' % csv_path)
    print('各次執行的完整輸出在：%s' % os.path.join(work, 'runs'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
