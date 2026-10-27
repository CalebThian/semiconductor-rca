# -*- coding: utf-8 -*-
"""
make_sim_data.py  ——  產生模擬的 FD 資料，格式與公司資料一致

欄位格式
--------
  LOT_ID, WFR_ID, START_DATETIME, Symbol Define, 其餘皆為參數
  一列一片 wafer，依 START_DATETIME 遞增（與公司資料相同：
  Normal 175 列對應 175 片）。

為什麼不是純隨機
----------------
  若參數之間沒有真實的依賴結構，MT-NGC 學到的圖只是雜訊，整條流程會在
  「沒有訊號的資料」上被驗證，看不出對錯。所以這裡從一個已知的結構生成：

    x_t = (I - B)^-1 · (A1 · x_{t-1} + e_t)

  B 是同期的稀疏 DAG（依隨機拓樸序，嚴格下三角），A1 是稀疏的滯後矩陣
  （縮放到譜半徑 < 0.9 以保證穩定）。真值鄰接矩陣會一併存檔，所以可以
  評估圖的回復率，不只看根因排名。

  異常 wafer 的作法是在**根因參數**上注入位移，位移會沿 B 與 A1 傳播到
  下游參數 —— 所以下游也會偏移，根因定位不會變成一眼可見的題目。
  `--propagation` 控制傳播強度。

  另外刻意摻入三類「垃圾欄」來測前處理：常數欄、近常數欄、重複欄，
  以及可控比例的缺值。

執行
----
  # 預設：300 參數、14 lot × 25 片、約 45% 異常
  python scripts/make_sim_data.py --out data/raw/sim/case_sim.csv

  # 大資料集，用來測參數縮減（5000 參數會觸發 int32 溢位那條路徑）
  python scripts/make_sim_data.py --n-params 5000 --out data/raw/sim/big.csv

  # 乾淨資料（不摻垃圾欄與缺值），用於驗證演算法本身
  python scripts/make_sim_data.py --const-frac 0 --dup-frac 0 \\
      --near-const-frac 0 --nan-frac 0 --out data/raw/sim/clean.csv

產出
----
  <out>.csv            資料本身
  <out>_truth.json     真值：根因、鄰接矩陣、垃圾欄分類
  <out>_config.json    可直接餵給 run_pipeline.py 的設定檔

環境：Python 3.6+ / numpy / pandas
"""

from __future__ import print_function

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

# 讓參數名稱看起來像真的 FD 感測器，報告給主管看時比 P0001 可信
STEMS = [
    'CHAMBER_PRESSURE', 'RF_FWD_PWR', 'RF_REF_PWR', 'BIAS_VOLTAGE',
    'GAS_FLOW_AR', 'GAS_FLOW_CF4', 'GAS_FLOW_O2', 'GAS_FLOW_N2',
    'THROTTLE_POS', 'CHAMBER_TEMP', 'CHUCK_TEMP', 'HE_BACKSIDE_PRES',
    'ENDPOINT_INTENSITY', 'MATCH_C1_POS', 'MATCH_C2_POS', 'DC_BIAS',
    'APC_VALVE_ANGLE', 'TURBO_PUMP_SPEED', 'FORELINE_PRESSURE',
    'COIL_CURRENT', 'ESC_LEAKAGE', 'OES_PEAK_440', 'OES_PEAK_520',
    'PLASMA_IMPEDANCE', 'PHASE_ANGLE', 'WAFER_TEMP_EST', 'EXHAUST_FLOW',
    'COOLANT_TEMP_IN', 'COOLANT_TEMP_OUT', 'LINER_TEMP',
]


def make_names(n):
    out = []
    k = 0
    while len(out) < n:
        k += 1
        for s in STEMS:
            out.append('%s_%02d' % (s, k))
            if len(out) == n:
                break
    return out


def random_dag(n, density, rng):
    """同期效應：依隨機拓樸序產生嚴格下三角的稀疏矩陣。

    回傳 (B, order)。B[i, j] != 0 表示 j → i 的同期效應。
    """
    order = rng.permutation(n)
    pos = np.empty(n, dtype=int)
    pos[order] = np.arange(n)
    B = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if pos[j] < pos[i] and rng.random() < density:
                B[i, j] = rng.uniform(0.25, 0.65) * rng.choice([-1.0, 1.0])
    return B, order


def random_lagged(n, density, rng, max_radius=0.9):
    """滯後效應：稀疏矩陣，縮放到譜半徑 < max_radius 以保證穩定。"""
    A = np.zeros((n, n))
    mask = rng.random((n, n)) < density
    A[mask] = rng.uniform(0.2, 0.6, mask.sum()) * rng.choice(
        [-1.0, 1.0], mask.sum())
    # 自回歸項：讓每個參數有一定的時間連續性（真實感測器都有）
    np.fill_diagonal(A, rng.uniform(0.3, 0.6, n))
    r = np.max(np.abs(np.linalg.eigvals(A)))
    if r > max_radius:
        A *= max_radius / r
    return A


def main():
    p = argparse.ArgumentParser(description='產生模擬 FD 資料')
    p.add_argument('--out', default='data/raw/sim/case_sim.csv')
    p.add_argument('--n-params', type=int, default=300, help='參數欄位總數')
    p.add_argument('--n-lots', type=int, default=14)
    p.add_argument('--wafers-per-lot', type=int, default=25)
    p.add_argument('--abnormal-frac', type=float, default=0.45,
                   help='異常 wafer 比例（公司資料約 162/337）')
    p.add_argument('--n-roots', type=int, default=3, help='真根因數')
    p.add_argument('--shift', type=float, default=3.0,
                   help='根因在異常 wafer 的位移（標準差倍數）')
    p.add_argument('--propagation', type=float, default=1.0,
                   help='傳播強度倍率。0 = 只有根因偏移（題目最簡單）')
    p.add_argument('--inst-density', type=float, default=0.004,
                   help='同期 DAG 的邊密度')
    p.add_argument('--lag-density', type=float, default=0.003,
                   help='滯後矩陣的邊密度')
    p.add_argument('--const-frac', type=float, default=0.10,
                   help='常數欄比例')
    p.add_argument('--near-const-frac', type=float, default=0.05)
    p.add_argument('--dup-frac', type=float, default=0.08,
                   help='重複欄比例（複製其他欄並加極小雜訊）')
    p.add_argument('--nan-frac', type=float, default=0.01,
                   help='缺值比例（分布於部分參數）')
    p.add_argument('--nan-abnormal-bias', type=float, default=0.0,
                   help='>0 時讓缺值偏向異常 wafer，用來測那條警告')
    p.add_argument('--anomaly-mode', choices=('propagate', 'saturated'),
                   default='saturated',
                   help='異常注入方式。saturated（預設）：位移只加在根因的'
                        '最終值上，根因必然是最大邊際位移者 —— 與公司三個'
                        'case 的型態一致，適合當迴歸測試資料。'
                        'propagate：位移注入根因的擾動項並沿因果結構傳播，'
                        '下游也會偏移且可能蓋過根因，較真實但難度不可控。')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    n_waf = args.n_lots * args.wafers_per_lot
    n_sig = args.n_params - int(args.n_params * (
        args.const_frac + args.near_const_frac + args.dup_frac))
    if n_sig < args.n_roots + 5:
        sys.exit('有訊號的參數只剩 %d 個，太少。請調低垃圾欄比例或調高 '
                 '--n-params。' % n_sig)

    print('=' * 64)
    print('產生模擬 FD 資料')
    print('=' * 64)
    print('wafer %d（%d lot × %d 片）' % (n_waf, args.n_lots,
                                          args.wafers_per_lot))
    print('參數 %d（有結構 %d，垃圾欄 %d）'
          % (args.n_params, n_sig, args.n_params - n_sig))

    # ---- 結構 ----
    print('')
    print('[1] 產生因果結構')
    B, order = random_dag(n_sig, args.inst_density, rng)
    A1 = random_lagged(n_sig, args.lag_density, rng)
    I_B_inv = np.linalg.inv(np.eye(n_sig) - B)
    print('    同期邊 %d，滯後邊 %d（含自回歸 %d）'
          % (int((B != 0).sum()), int((A1 != 0).sum()), n_sig))

    # 根因取拓樸序靠前者 —— 上游參數才有東西可傳播
    roots = sorted(order[:args.n_roots].tolist())
    print('    根因（拓樸序最前 %d 個）：索引 %s' % (args.n_roots, roots))

    # ---- 異常區間 ----
    # 真實的異常是「製程偏移」，會連續影響一段 wafer，不是隨機散布
    n_ab = int(round(n_waf * args.abnormal_frac))
    start_ab = n_waf - n_ab
    is_ab = np.zeros(n_waf, dtype=bool)
    is_ab[start_ab:] = True
    print('')
    print('[2] 異常區間：第 %d 片起連續 %d 片（偏移型異常）'
          % (start_ab + 1, n_ab))

    # ---- 生成時序 ----
    print('')
    print('[3] 生成時序')

    def simulate(shift, seed, mode):
        r = np.random.default_rng(seed)
        Z = np.zeros((n_waf, n_sig))
        prev = np.zeros(n_sig)
        for t in range(n_waf):
            e = r.normal(0, 1, n_sig)
            if mode == 'propagate' and is_ab[t]:
                # 位移注入根因的擾動項，由 (I-B)^-1 與 A1 傳播到下游
                ramp = min(1.0, (t - start_ab + 1) / max(3.0, 0.1 * n_ab))
                for rt in roots:
                    e[rt] += shift * ramp
            x = I_B_inv @ (A1 @ prev * args.propagation + e)
            Z[t] = x
            prev = x
        if mode == 'saturated':
            # 位移加在根因的最終值上，以各根因自己的標準差為單位。
            # 不經傳播，所以根因必然是最大的邊際位移者。
            sd = Z.std(axis=0, ddof=1) + 1e-12
            for i, t in enumerate(np.where(is_ab)[0]):
                ramp = min(1.0, (i + 1) / max(3.0, 0.1 * n_ab))
                for rt in roots:
                    Z[t, rt] += shift * sd[rt] * ramp
        return Z

    def root_ranks(Z):
        """各參數的 |Cohen's d| 排名，回傳根因的名次（1 起算）。"""
        a, b = Z[is_ab], Z[~is_ab]
        sd = np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2.0) + 1e-12
        d = np.abs(a.mean(0) - b.mean(0)) / sd
        order = np.argsort(-d)
        rank = np.empty(n_sig, dtype=int)
        rank[order] = np.arange(1, n_sig + 1)
        return [int(rank[r]) for r in roots], d

    X = simulate(args.shift, args.seed, args.anomaly_mode)
    rk, _d = root_ranks(X)
    print('    異常注入模式：%s' % args.anomaly_mode)
    print('    根因的 |d| 名次：%s（共 %d 個有結構的參數）'
          % (', '.join('idx%d=%d' % (r, k) for r, k in zip(roots, rk)),
             n_sig))
    if max(rk) > 20:
        print('    ⚠️ 有根因排在 20 名之外。')
        if args.anomaly_mode == 'propagate':
            print('       這是傳播的真實後果：下游參數會累積多個根因的'
                  '位移，標準化後可能大於根因本身。')
            print('       提高 --shift 無法改變名次（|d| 對 shift 近似'
                  '線性，名次在大位移下趨於固定）。')
            print('       要可控難度請改用 --anomaly-mode saturated，'
                  '或調低 --inst-density。')
        else:
            print('       saturated 模式下不該出現，請回報。')

    # 每個參數給不同的量級與偏置，貼近真實感測器（流程會自行標準化）
    scale = np.exp(rng.normal(0, 1.2, n_sig))
    offset = rng.normal(0, 50, n_sig)
    X = X * scale + offset

    # ---- 垃圾欄 ----
    print('')
    print('[4] 摻入垃圾欄')
    cols = [X[:, j] for j in range(n_sig)]
    kinds = ['signal'] * n_sig
    n_const = int(args.n_params * args.const_frac)
    n_near = int(args.n_params * args.near_const_frac)
    n_dup = args.n_params - n_sig - n_const - n_near
    for _ in range(n_const):
        cols.append(np.full(n_waf, float(rng.integers(1, 100))))
        kinds.append('const')
    for _ in range(n_near):
        cols.append(rng.normal(float(rng.integers(1, 100)), 1e-7, n_waf))
        kinds.append('near_const')
    dup_src = []
    for _ in range(max(0, n_dup)):
        j = int(rng.integers(0, n_sig))
        dup_src.append(j)
        cols.append(cols[j] + rng.normal(0, 1e-9, n_waf))
        kinds.append('dup')
    print('    常數 %d／近常數 %d／重複 %d' % (n_const, n_near, max(0, n_dup)))

    # 打散欄位順序，避免真根因剛好都排在最前面
    perm = rng.permutation(len(cols))
    names_all = make_names(len(cols))
    data = {}
    kind_of, root_names = {}, []
    new_index_of = {}
    for new_i, old_i in enumerate(perm):
        nm = names_all[new_i]
        data[nm] = cols[old_i]
        kind_of[nm] = kinds[old_i]
        new_index_of[old_i] = nm
        if old_i < n_sig and old_i in roots:
            root_names.append(nm)

    # ---- meta 欄 ----
    print('')
    print('[5] meta 欄位')
    lot_ids, wfr_ids, times = [], [], []
    t0 = dt.datetime(2026, 3, 2, 8, 0, 0)
    cur = t0
    for l in range(args.n_lots):
        for w in range(args.wafers_per_lot):
            lot_ids.append('L%s%03d' % (chr(65 + l % 26), 1000 + l))
            wfr_ids.append(w + 1)
            times.append(cur)
            cur += dt.timedelta(minutes=int(rng.integers(3, 7)))
        # lot 之間有較長的間隔（換片、換 recipe），對應 segment 邊界
        cur += dt.timedelta(hours=int(rng.integers(2, 9)))
    print('    時間範圍 %s ~ %s'
          % (times[0].strftime('%Y-%m-%d %H:%M'),
             times[-1].strftime('%Y-%m-%d %H:%M')))

    # 一次 concat，不要逐欄 insert —— 300 欄逐欄加會讓 DataFrame 碎片化
    meta = pd.DataFrame({
        'LOT_ID': lot_ids,
        'WFR_ID': wfr_ids,
        'START_DATETIME': [t.strftime('%Y-%m-%d %H:%M:%S') for t in times],
        'Symbol Define': np.where(is_ab, 'Abnormal', 'Normal'),
    })
    df = pd.concat([meta, pd.DataFrame(data)], axis=1)

    # ---- 缺值 ----
    print('')
    print('[6] 缺值')
    param_cols = list(data.keys())
    n_cell = 0
    if args.nan_frac > 0:
        # 只讓一部分參數有缺值，比全面散布更貼近真實。
        # 根因一律排除：否則缺值處理可能把真根因整欄丟掉，測試資料的
        # 難度就取決於一個與方法無關的巧合。
        pool = [c for c in param_cols if c not in set(root_names)]
        n_bad_col = max(1, int(0.15 * len(pool)))
        bad_cols = list(rng.choice(pool, n_bad_col, replace=False))
        for c in bad_cols:
            pr = np.full(n_waf, args.nan_frac / 0.15)
            if args.nan_abnormal_bias > 0:
                pr = np.where(is_ab, pr * (1 + args.nan_abnormal_bias), pr)
            m = rng.random(n_waf) < np.clip(pr, 0, 0.95)
            df.loc[m, c] = np.nan
            n_cell += int(m.sum())
        # 另外放一個幾乎全空的參數，測「丟欄而非丟列」那條路
        victim = str(rng.choice([c for c in pool if c not in bad_cols]))
        df.loc[rng.random(n_waf) < 0.93, victim] = np.nan
        n_cell += int(df[victim].isna().sum())
        print('    %d 個參數含缺值（共 %d 格），其中 %s 缺 93%%'
              % (n_bad_col + 1, n_cell, victim))
    else:
        print('    無缺值')

    # ---- 輸出 ----
    out = args.out
    d = os.path.dirname(os.path.abspath(out))
    if d and not os.path.isdir(d):
        os.makedirs(d)
    df.to_csv(out, index=False, encoding='utf-8-sig')
    base = os.path.splitext(out)[0]

    # 真值：鄰接矩陣用「打散後的欄名」表示，才能直接和流程的輸出對照
    sig_names = [new_index_of[j] for j in range(n_sig)]
    truth = {
        'csv': out,
        'n_wafers': int(n_waf),
        'n_params': int(args.n_params),
        'n_signal_params': int(n_sig),
        'roots': sorted(root_names),
        'shift_sd': args.shift,
        'propagation': args.propagation,
        'abnormal_start_row': int(start_ab),
        'signal_param_names_in_structural_order': sig_names,
        'instantaneous_edges': [
            {'source': sig_names[j], 'target': sig_names[i],
             'weight': round(float(B[i, j]), 4)}
            for i in range(n_sig) for j in range(n_sig) if B[i, j] != 0],
        'lagged_edges': [
            {'source': sig_names[j], 'target': sig_names[i],
             'weight': round(float(A1[i, j]), 4)}
            for i in range(n_sig) for j in range(n_sig)
            if A1[i, j] != 0 and i != j],
        'column_kind': kind_of,
        'seed': args.seed,
    }
    with open(base + '_truth.json', 'w', encoding='utf-8') as f:
        json.dump(truth, f, ensure_ascii=False, indent=2)

    cfg = {
        'data': {
            'csv': out,
            'train_csv': '',
            'wafer_col': 'LOT_ID,WFR_ID',
            'label_col': 'Symbol Define',
            'abnormal_value': 'Abnormal',
            'exclude': [],
            'dropna': 'row',
            'drop_col_na_frac': 0.5,
        },
        'candidates': {
            'file': '', 'topk': 6, 'min_effect': 0.2, 'n_boot': 200,
            'n_perm': 100, 'true_roots': sorted(root_names),
            'sensor_col': 'Sensor', 'priority_col': 'Priority',
        },
        'graph': {'path': '', 'edge_k': 2, 'min_concentration': 0.3,
                  'max_hop': 2},
        'rules': {'enabled': True, 'min_gain': 0.02, 'gain_metric': 'F1',
                  'extend': False, 'extend_when': 1, 'extend_max_add': 2,
                  'enum_truncate': True, 'enum_prune': True,
                  'enum_max_rows': 20000, 'max_per_group': 1,
                  'max_combo': 200000},
        'output': {'prefix': base + '_out', 'show_welch': True,
                   'brief': False},
    }
    with open(base + '_config.json', 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print('')
    print('=' * 64)
    print('已存:')
    print('  %s          （shape %s）' % (out, (df.shape,)))
    print('  %s_truth.json   真值' % base)
    print('  %s_config.json  可直接執行的設定檔' % base)
    print('')
    print('真根因：%s' % '、'.join(sorted(root_names)))
    print('')
    print('下一步:')
    print('  python scripts/run_pipeline.py --config %s_config.json' % base)
    print('（graph.path 留空 → 不分組，每個候選自成一組；'
          '要測分組請先訓練出圖再填入）')
    print('=' * 64)
    print('')
    print('DIGEST waf=%d/%d p=%d sig=%d roots=%s na=%d inst_e=%d lag_e=%d'
          % (int((~is_ab).sum()), int(is_ab.sum()), args.n_params, n_sig,
             len(root_names), n_cell, len(truth['instantaneous_edges']),
             len(truth['lagged_edges'])))


if __name__ == '__main__':
    main()
