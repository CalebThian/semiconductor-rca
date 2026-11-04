# -*- coding: utf-8 -*-
r"""
run_all.py  ——  三階段統籌：MT-NGC 訓練 → 同事 pipeline → 分析流程

流程
----
  1. 訓練 MT-NGC，輸出：
       - <graph_dir>/<train_dataset>_graph_<task>.npy   （供後續分析）
       - datasets/<train_dataset>_edges.csv             （供同事的模型）
         格式：source,score,target（一行一條邊）
  2. 以 subprocess 執行同事的 pipeline（其 config 不需調整）
  3. 以同事輸出的候選 csv 為輸入，執行分析流程並產生 HTML 報告

用法
----
  python run_all.py --init run_all_config.json     # 產生設定檔範本
  python run_all.py --config run_all_config.json   # 執行
  python run_all.py --config ... --from-step 2     # 從第 2 步開始（跳過訓練）
  python run_all.py --config ... --only 1          # 只跑第 1 步

⚠️ 檔名衝突
-----------
你與同事的腳本同名（皆為 run_pipeline.py）。本檔以完整路徑呼叫，
但仍建議將其中一支改名，避免 import 或執行時取到錯的檔案。

⚠️ 邊的方向
-----------
圖矩陣定義為 W[i, j] = 預測 target i 時 source j 的重要度，故邊為 j → i：
    source = 因，destination = 果
若同事的模型採相反約定，需請對方轉置，或在 config 設 edges.transpose = true。

環境：Python 3.8+ / numpy / pandas / torch（步驟 1 需要）
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

TEMPLATE = {
    "train_dataset": "case1",

    "data": {
        "csv": "data/raw/semiconductor/case1.csv",
        "train_csv": "",
        "wafer_col": "LOT_ID,WFR_ID,START_DATETIME",
        "label_col": "Symbol Define",
        "abnormal_value": "Abnormal",
        "exclude": [],
        "dropna": "row",
        "drop_col_na_frac": 0.5,
        "min_row_keep": 0.8,
        "max_row_loss": 0.3
    },

    "reduce": {
        "enabled": True,
        "mode": "welch",
        "max_params": 300,
        "mem_budget_gb": 2.0,
        "train_na_frac": 0.0,
        "clean_out": "",
        "candidates": "",
        "out": "",
        "script": "scripts/reduce_params.py"
    },

    "mtngc": {
        "task": "task2b",
        "lookback": 3,
        "hidden_dim": 64,
        "lambda_sparse": 0.1,
        "max_epochs": 100,
        "patience": 20,
        "n_blocks": 10,
        "mode": "stability",
        "n_seed": 20,
        "subsample": 0.8,
        "stability_edge_k": 2,
        "seed": 0,
        "graph_dir": "results"
    },

    "edges": {
        "out": "datasets/{train_dataset}_edges.csv",
        "edge_k": 0,
        "min_confidence": 0.0,
        "normalize": True,
        "transpose": False,
        "precision": 6
    },

    "colleague": {
        "cwd": "fd_analysis/gnn-tam-main",
        "script": "run_pipeline.py",
        "config": "pipeline_config.yaml",
        "python": "",
        "group_col": "",
        "use_config": False,
        "source": "reduced",
        "train_dataset": "",
        "dataset_suffix": "",
        "csv_arg": "",
        "copy_clean_to": "data/{train_dataset}.csv",
        "output_csv": "experiment_results/checklist/checklist_{train_dataset}_{group_col}.csv"
    },

    "analysis": {
        "script": "scripts/run_pipeline.py",
        "config_out": "config/{train_dataset}_auto.json",
        "sensor_col": "Sensor",
        "priority_col": "Priority",
        "prefix": "results/{train_dataset}_pipeline",
        "brief": True,
        "show_welch": False,
        "n_perm": 300,
        "n_boot": 200,
        "add_statistical": True,
        "expansion": {"enabled": True, "max_hop": 2, "min_effect": None},
        "graph": {"edge_k": 2, "min_concentration": 0.0, "max_hop": 3},
        "rules": {"enabled": True, "test_frac": 0.4, "n_random": 200,
                  "min_gain": 0.02, "gain_metric": "F1", "extend": False},
        "true_roots": []
    },

    "report": {
        "script": "scripts/make_html_report.py",
        "out": "results/{train_dataset}_report.html",
        "enabled": True
    }
}


# =====================================================================

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        txt = re.sub(r'^\s*//.*$', '', f.read(), flags=re.M)
    cfg = json.loads(txt)

    def merge(dst, src):
        for k, v in src.items():
            if k not in dst:
                dst[k] = v
            elif isinstance(v, dict) and isinstance(dst[k], dict):
                merge(dst[k], v)
    merge(cfg, TEMPLATE)
    return cfg


def subst(s, ds, gc=''):
    """替換設定檔路徑裡的佔位符。

    {train_dataset} 是正式名稱，{ds} 是別名 —— 兩者都接受，因為
    設定檔裡兩種寫法都出現過，而沒被替換的佔位符會變成檔名的一部分
    （實測產生了一個真的叫 "{ds}_clean.csv" 的檔案），而且不會報錯。
    """
    if not isinstance(s, str):
        return s
    return (s.replace('{train_dataset}', ds)
             .replace('{ds}', ds)
             .replace('{group_col}', gc))


ENCODINGS = ['utf-8-sig', 'utf-8', 'cp950', 'big5hkscs', 'gbk', 'latin-1']


def read_text_any(path):
    """依序嘗試常見編碼讀純文字（節點名稱側車檔可能含中文欄名）。"""
    last = None
    for enc in ENCODINGS:
        try:
            with open(path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError as e:
            last = e
    raise UnicodeDecodeError('utf-8', b'', 0, 1,
                             '無法讀取 %s：%s' % (path, last))


def read_csv_any(path, **kw):
    """依序嘗試常見編碼讀 csv（台灣環境的 csv 常為 CP950/Big5）。"""
    last = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError as e:
            last = e
    raise UnicodeDecodeError('utf-8', b'', 0, 1,
                             '無法讀取 %s：%s' % (path, last))


def ensure_dir(path):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d)


def banner(n, title):
    print('\n' + '=' * 70)
    print('步驟 %s｜%s' % (n, title))
    print('=' * 70)


# =====================================================================
# 步驟 1：訓練 MT-NGC 並輸出邊清單
# =====================================================================

def interleaved_blocks(arr, n_blocks):
    """切成連續區塊，回傳 (資料, segment_lengths)。保時序，供 lookback 使用。"""
    n = len(arr)
    n_blocks = max(1, min(int(n_blocks), n))
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    seg = [int(edges[i + 1] - edges[i]) for i in range(n_blocks)
           if edges[i + 1] > edges[i]]
    return arr, seg


def graph_path(cfg):
    """依模式決定圖檔名，使步驟 1 與步驟 3 指向同一個檔。"""
    mc = cfg['mtngc']
    gdir = mc.get('graph_dir', 'results')
    ds = cfg['train_dataset']
    if mc.get('mode', 'average') == 'stability':
        return os.path.join(gdir, '%s_stable_graph.npy' % ds)
    return os.path.join(gdir, '%s_graph_%s.npy' % (ds, mc['task']))


def block_subsample(arr, frac, n_blocks, rng):
    """保時序的區塊子抽樣：切成區塊後隨機取一部分。"""
    n = len(arr)
    n_blocks = max(2, min(int(n_blocks), n))
    e = np.linspace(0, n, n_blocks + 1).astype(int)
    blocks = [(int(e[i]), int(e[i + 1])) for i in range(n_blocks)
              if e[i + 1] > e[i]]
    k = max(2, int(round(len(blocks) * frac)))
    pick = sorted(rng.choice(len(blocks), size=min(k, len(blocks)),
                             replace=False))
    sel = [blocks[i] for i in pick]
    return (np.concatenate([arr[a:b] for a, b in sel], axis=0),
            [b - a for a, b in sel])


def top_edges(W, k):
    """每個節點取最強的 k 條入邊，回傳布林矩陣。"""
    W = np.abs(np.asarray(W, dtype=float)).copy()
    np.fill_diagonal(W, 0.0)
    n = W.shape[0]
    A = np.zeros((n, n), dtype=bool)
    for i in range(n):
        if W[i].max() > 0:
            A[i, np.argsort(-W[i])[:k]] = True
    np.fill_diagonal(A, False)
    return A


def _run_reduce(cfg, src, out, topk, na_thr, cand, label):
    """呼叫 reduce_params.py。step0a 與 step0b 共用。

    reduce_params 的輸出只含被選中的欄位，所以 --max-na-frac 0 會產出
    一個完全無缺值的檔：任何含缺值的欄位都不被選中，因此不必丟列，
    時間軸保持完整。這一點對 Granger 類方法很關鍵 —— 丟列會在時間軸上
    打洞，而 ffill 會製造虛假的滯後自相關（實測約等於缺值比例）。
    """
    dc = cfg['data']
    mc = cfg['mtngc']
    ensure_dir(out)
    script = cfg.get('reduce', {}).get('script', 'scripts/reduce_params.py')
    if not os.path.exists(script):
        sys.exit('找不到縮減腳本：%s' % script)
    cmd = [sys.executable, os.path.abspath(script),
           '--csv', os.path.abspath(src),
           '--wafer-col', dc['wafer_col'],
           '--label-col', dc['label_col'],
           '--abnormal-value', str(dc.get('abnormal_value', 'Abnormal')),
           '--topk', str(topk),
           '--mode', cfg.get('reduce', {}).get('mode', 'welch'),
           '--lookback', str(int(mc['lookback'])),
           '--hidden', str(int(mc['hidden_dim'])),
           '--max-na-frac', str(na_thr),
           '--out', os.path.abspath(out)]
    if dc.get('exclude'):
        cmd += ['--exclude', ','.join(map(str, dc['exclude']))]
    if cand and os.path.exists(cand):
        cmd += ['--candidates', os.path.abspath(cand)]
        # 欄名由 analysis.sensor_col 指定，不要讓 reduce_params 去猜
        sc = cfg.get('analysis', {}).get('sensor_col', '') or 'Sensor'
        cmd += ['--cand-col', sc]
        print('候選清單：%s（欄位 %s，無條件保留）' % (cand, sc))
    print('[%s] 執行：%s\n' % (label, ' '.join(cmd)))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit('%s 回傳錯誤碼 %d。' % (label, r.returncode))
    if not os.path.exists(out):
        sys.exit('%s 未產生 %s。' % (label, out))
    return out


def step0_prepare(cfg):
    """步驟 0：產出兩個檔案，回傳 (無缺值全參數檔, 訓練用縮減檔)。

    三個階段需要不同的輸入，不可共用一個檔：

      步驟 1 訓練   無缺值 + 縮減   —— 參數量隨 N 平方，且一格 NaN
                                      會讓標準化把整欄變 NaN
      步驟 2 候選集 無缺值 + 全參數 —— 候選要涵蓋全部參數，否則它就
                                      不再是獨立於 Welch 的第二個意見；
                                      但同事的 pipeline 不處理缺值
      步驟 3 分析   原始檔          —— Welch 是 O(N)，且 run_pipeline
                                      自己有 NA 處理

    0a 產出的無缺值全參數檔同時供步驟 2 使用（見 colleague 區塊的說明）。

    每條返回路徑都把實際路徑記到 cfg['_clean_csv'] / cfg['_train_csv']，
    步驟 2 才知道真正該餵哪個檔。推算路徑在走捷徑時會指到從沒寫出來的
    檔案。
    """
    def done(clean_, train_):
        cfg['_clean_csv'], cfg['_train_csv'] = clean_, train_
        return clean_, train_

    rc = cfg.get('reduce', {}) or {}
    dc = cfg['data']
    src = dc['csv']
    if not os.path.exists(src):
        sys.exit('找不到資料檔：%s' % src)
    if not rc.get('enabled', True):
        print('（reduce.enabled 為 false，步驟 0 整段略過）')
        return done(src, (dc.get('train_csv') or src))

    banner(0, '訓練與候選集的資料前處理')

    head = read_csv_any(src, nrows=5)
    meta = {'LOT_ID', 'WFR_ID', 'WAFER_ID', 'START_DATETIME',
            'END_DATETIME', dc['label_col']}
    meta |= set(c.strip() for c in dc['wafer_col'].split(',') if c.strip())
    meta |= set(dc.get('exclude', []))
    n_param = len([c for c in head.columns if c not in meta])

    mc = cfg['mtngc']
    lb, hd = int(mc['lookback']), int(mc['hidden_dim'])
    budget = float(rc.get('mem_budget_gb', 2.0)) * 1024 ** 3
    n_mem = int((budget / (12.0 * lb * hd)) ** 0.5)
    topk = min(int(rc.get('max_params', 300)), n_mem)
    na_thr = float(rc.get('train_na_frac', 0.0))
    print('原始參數 %d 個' % n_param)
    print('缺值門檻 %.2f（0 = 任何含缺值的欄位都排除）' % na_thr)
    print('記憶體預算 %.1f GB（lookback=%d, hidden=%d）→ 上限 %d 個'
          % (rc.get('mem_budget_gb', 2.0), lb, hd, n_mem))
    print('訓練保留 min(max_params=%d, %d) = %d 個'
          % (int(rc.get('max_params', 300)), n_mem, topk))

    # ---- 0a 無缺值、保留全部參數 ----
    print('')
    print('--- 0a 清除缺值（保留全部參數，供步驟 2 與步驟 0b 使用）---')
    clean = derived_path(cfg, 'clean_out', '_clean')
    _run_reduce(cfg, src, clean, n_param, na_thr, '', '0a')

    # ---- 0b 由 0a 的輸出再縮減給訓練 ----
    print('')
    print('--- 0b 縮減參數（訓練用）---')
    if dc.get('train_csv'):
        print('（data.train_csv 已指定為 %s，略過 0b）' % dc['train_csv'])
        return done(clean, dc['train_csv'])
    ch = read_csv_any(clean, nrows=5)
    n_clean = len([c for c in ch.columns if c not in meta])
    if n_clean <= topk:
        print('清除缺值後只剩 %d 個參數，不超過 %d，直接用 0a 的輸出訓練。'
              % (n_clean, topk))
        return done(clean, clean)
    # 0a 已清乾淨，這一步不必再設缺值門檻
    train = derived_path(cfg, 'out', '_reduced')
    # 候選清單：reduce.candidates 未指定時，自動沿用同事上一輪的輸出。
    # 候選集由步驟 2 產生，第一輪跑 0b 時還不存在，所以第一輪沒有可釘的
    # 候選；第二輪起就有了，釘住它們可讓「候選不在圖裡」的落差歸零，
    # 而且不必手動填路徑。
    cand = rc.get('candidates', '')
    if not cand:
        cc = cfg.get('colleague', {}) or {}
        gc = cc.get('group_col', '') or ''
        cwd = os.path.abspath(subst(cc.get('cwd', '.') or '.',
                                    cfg['train_dataset'], gc))
        # 同事的輸出檔名用的是他被呼叫時的資料集名稱
        rel = subst(cc.get('output_csv', '') or '',
                    colleague_ds(cfg), gc)
        if rel:
            auto = rel if os.path.isabs(rel) else os.path.join(cwd, rel)
            if os.path.exists(auto):
                cand = auto
                print('候選清單自動沿用同事上一輪的輸出：%s' % auto)
            else:
                print('（尚無候選清單 —— 第一輪必然如此。'
                      '步驟 2 跑完後，下一輪會自動釘住候選。）')
    _run_reduce(cfg, clean, train, topk, 1.0, cand, '0b')
    print('')
    print('步驟 1 將用：%s（%d 個參數）' % (train, topk))
    csrc, _s2, lbl2 = colleague_source(cfg)
    print('步驟 2 將用：%s（%s，由 colleague.source 決定）' % (csrc, lbl2))
    if lbl2 == '縮減檔':
        print('  註：候選集因此只能從 Welch 選出的參數裡挑 —— '
              '覆蓋率落差消失，')
        print('      但候選集不再獨立於 Welch 排序，方法說明需據此調整。')
    return done(clean, train)


def step1_train(cfg, train_csv=None):
    ds = cfg['train_dataset']
    mc, ec, dc = cfg['mtngc'], cfg['edges'], cfg['data']
    banner(1, 'MT-NGC 訓練與邊清單輸出')

    sys.path.insert(0, os.getcwd())
    try:
        import torch
        from src.models.mt_ngc_method import MTNeuralGrangerMethod
        from scripts.residual_anomaly import split_by_symbol
    except ImportError as e:
        sys.exit('步驟 1 需要 torch 與專案模組（src/、scripts/），'
                 '但匯入失敗：%s\n'
                 '請確認在專案根目錄執行；若圖已訓練好，'
                 '可用 --from-step 2 跳過此步。' % e)

    # 訓練用原始檔，分析用（可能已 NA 清理的）csv。
    # MT-NGC 學的是 Normal 的時序依賴；丟掉含缺值的列會在時間軸上打出
    # 破洞，而 lookback 視窗會跨過破洞、把不相鄰的時點當成相鄰 —— 這正是
    # 當初要傳 segment_lengths 去避免的污染。所以訓練不吃清理後的檔。
    train_csv = train_csv or dc.get('train_csv') or dc['csv']
    if not os.path.exists(train_csv):
        sys.exit('找不到訓練用資料檔：%s' % train_csv)
    if train_csv != dc['csv']:
        print('訓練資料：%s' % train_csv)
        print('（分析階段另用 %s）' % dc['csv'])
    normal, abnormal, names = split_by_symbol(train_csv)
    print('Normal %d／Abnormal %d／參數 %d'
          % (len(normal), len(abnormal), len(names)))
    if len(normal) == 0:
        sys.exit('Normal 資料為 0 列，無法訓練。\n'
                 '若 %s 是 NA 清理後的檔，請把 data.train_csv 指向原始檔。'
                 % train_csv)

    # 區塊短於 lookback 時可用樣本為 0。空張量餵進 torch 會在 C 層崩潰
    # （Windows 顯示「記憶體讀取失敗」，沒有 Python traceback），
    # 所以這裡先攔下來並說明原因。
    _n = len(normal)
    _nb = max(1, min(int(mc['n_blocks']), _n))
    _e = np.linspace(0, _n, _nb + 1).astype(int)
    _seg = [int(_e[i + 1] - _e[i]) for i in range(_nb) if _e[i + 1] > _e[i]]
    _usable = sum(max(L - int(mc['lookback']), 0) for L in _seg)
    print('可用訓練樣本：%d（%d 列切 %d 塊，最短 %d 列，lookback %d）'
          % (_usable, _n, len(_seg), min(_seg), mc['lookback']))
    if _usable <= 0:
        sys.exit('可用訓練樣本為 0，無法訓練。\n'
                 '請調小 mtngc.n_blocks，或確認訓練資料的列數是否'
                 '因 NA 清理而大幅減少。')

    mode = mc.get('mode', 'average')
    n_seed = max(1, int(mc.get('n_seed', 1)))
    stab = mode == 'stability'
    if stab:
        print('模式：stability selection（%d 次，子抽樣 %.0f%%，'
              'edge_k=%d）—— 輸出為邊的出現頻率'
              % (n_seed, 100 * mc.get('subsample', 0.8),
                 mc.get('stability_edge_k', 2)))
    else:
        print('模式：權重平均（%d 個 seed）' % n_seed)

    rng = np.random.RandomState(mc['seed'])
    data0, seg0 = interleaved_blocks(normal, mc['n_blocks'])
    mats = []
    t_all = time.time()
    for r in range(n_seed):
        t_one = time.time()
        print('  [%d/%d] 訓練中 ...' % (r + 1, n_seed), flush=True)
        if stab:
            data, seg = block_subsample(normal, mc.get('subsample', 0.8),
                                        mc['n_blocks'], rng)
        else:
            data, seg = data0, seg0
        sd = mc['seed'] + r * 97
        np.random.seed(sd)
        torch.manual_seed(sd)
        m = MTNeuralGrangerMethod(
            lookback=mc['lookback'], hidden_dim=mc['hidden_dim'],
            lambda_sparse=mc['lambda_sparse'], max_epochs=mc['max_epochs'],
            patience=mc['patience'], output_graph=mc['task'], verbose=False)
        m.fit(data, segment_lengths=seg)
        Wr = np.abs(m.model.get_graph(mc['task']).astype(float))
        if stab:
            mats.append(top_edges(Wr, mc.get('stability_edge_k', 2))
                        .astype(float))
        else:
            mx = Wr.max()
            mats.append(Wr / mx if mx > 0 else Wr)
        el = time.time() - t_one
        done = r + 1
        msg = '  [%d/%d] 完成（%.1f 秒）' % (done, n_seed, el)
        if done < n_seed:
            # 以目前平均速度估剩餘時間，讓長時間執行時看得出進度
            eta = (time.time() - t_all) / done * (n_seed - done)
            msg += '，預估剩餘 %s' % (
                '%.0f 秒' % eta if eta < 90 else '%.1f 分' % (eta / 60.0))
        print(msg, flush=True)

    W = np.mean(mats, axis=0)
    if stab:
        # 兩兩 Jaccard：衡量不同次訓練選出的邊有多一致
        js = []
        for a in range(len(mats)):
            for b in range(a + 1, len(mats)):
                A_, B_ = mats[a] > 0, mats[b] > 0
                js.append((A_ & B_).sum() / max((A_ | B_).sum(), 1))
        off = W[~np.eye(W.shape[0], dtype=bool)]
        print('  邊集合兩兩 Jaccard：平均 %.3f（最低 %.3f）'
              % (np.mean(js) if js else 1.0, np.min(js) if js else 1.0))
        print('  邊出現頻率：>=0.8 有 %d 條、0.6~0.8 有 %d 條、<0.2 有 %d 條'
              % (int((off >= 0.8).sum()),
                 int(((off >= 0.6) & (off < 0.8)).sum()),
                 int((off < 0.2).sum())))
        if js and np.mean(js) < 0.5:
            print('  ⚠️ Jaccard < 0.5，圖不穩定；'
                  '可提高 lambda_sparse 或縮小 hidden_dim。')
    elif n_seed > 1:
        print('  已對 %d 個 seed 取平均（圖的訓練具隨機性，平均可降低變異）'
              % n_seed)

    # 存圖之前先確認它不是退化的。訓練輸入只要有一格 NaN，標準化就會把
    # 整欄變 NaN，進而讓所有 predictor 的權重變 NaN —— 實測 2/6000 格
    # 缺值即可讓 20 個 predictor 全毀。那樣存出來的圖沒有邊，二值化後
    # 每個候選自成一組，看起來完全像「圖沒找到結構」而不是「訓練失敗」。
    n_bad = int((~np.isfinite(W)).sum())
    off = W[~np.eye(W.shape[0], dtype=bool)]
    if n_bad:
        sys.exit('❌ 圖有 %d / %d 格非有限值，訓練已發散。\n'
                 '   最常見的原因是訓練資料含缺值：標準化會把含 NaN 的欄'
                 '整欄變成 NaN。\n'
                 '   請確認步驟 0 已產生無缺值的訓練檔'
                 '（reduce.train_na_frac），或檢查資料來源。'
                 % (n_bad, W.size))
    if np.abs(off).max() == 0:
        sys.exit('❌ 圖的非對角元素全為 0，沒有任何邊。\n'
                 '   這不是「找不到結構」而是訓練沒有學到東西 —— '
                 '請檢查訓練資料是否含缺值、\n'
                 '   或 lambda_sparse（目前 %s）是否過大。'
                 % mc.get('lambda_sparse'))

    gpath = graph_path(cfg)
    ensure_dir(gpath)
    np.save(gpath, W)
    print('圖已存：%s（非零邊 %d，最大值 %.4g）'
          % (gpath, int((off != 0).sum()), float(np.abs(off).max())))

    # 節點名稱側車檔。沒有它的話，步驟 3 只能比對矩陣大小 —— 而縮減參數
    # 或 NA 丟欄都會改變欄位集合，大小相同但順序不同時索引會錯而無人知曉。
    # 有了它就能按名稱對齊，訓練與分析也不必用同一組參數。
    npath = os.path.splitext(gpath)[0] + '_nodes.txt'
    with open(npath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(map(str, names)) + '\n')
    print('節點名稱已存：%s（%d 個）' % (npath, len(names)))

    # ---- 邊清單 ----
    Wm = np.abs(W).copy()
    np.fill_diagonal(Wm, 0.0)
    # stability 模式的值已是「出現頻率」(0~1)，再正規化會扭曲語意
    if not stab and ec.get('normalize', True) and Wm.max() > 0:
        Wm = Wm / Wm.max()
    elif stab:
        print('（stability 模式：score 為邊的出現頻率，不再正規化）')
    n = Wm.shape[0]
    keep = np.ones((n, n), dtype=bool)
    if ec.get('edge_k', 0) > 0:
        keep[:] = False
        for i in range(n):
            keep[i, np.argsort(-Wm[i])[:ec['edge_k']]] = True
    keep &= Wm > ec.get('min_confidence', 0.0)
    np.fill_diagonal(keep, False)

    rows = []
    for i in range(n):                 # i = 果
        for j in range(n):             # j = 因
            if keep[i, j]:
                src, dst = names[j], names[i]
                if ec.get('transpose', False):
                    src, dst = dst, src
                rows.append((src, dst, Wm[i, j]))
    rows.sort(key=lambda r: -r[2])

    # 邊清單是同事 pipeline 的**輸入**，所以檔名要用他被呼叫時的資料集
    # 名稱，不是原始名稱。他以 --train_dataset case2_clean 被呼叫時會去找
    # datasets/case2_clean_edges.csv；用原始名稱寫出 case2_edges.csv 他就
    # 找不到。這和 colleague.output_csv 是同一類錯誤。
    epath = subst(ec['out'], colleague_ds(cfg))
    ensure_dir(epath)
    fmt = '%%.%df' % ec.get('precision', 6)
    # 用 csv.writer 而非手寫字串：參數名稱含逗號時（真實感測器名常帶
    # 單位，例如 "Flow (sccm), Ar"）手寫會讓欄位錯位，讀回來的名稱變成
    # 碎片，於是與資料集的交集為 0 —— 症狀是「edges 涵蓋 sensor 數為 0」。
    # 引號處理交給標準庫。
    n_comma = sum(1 for s_, d_, _w in rows if ',' in s_ or ',' in d_)
    with open(epath, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['source', 'score', 'target'])
        for s_, d_, w_ in rows:
            w.writerow([s_, fmt % w_, d_])
    if n_comma:
        print('  註：%d 條邊的參數名稱含逗號，已加引號。'
              '若對方以手寫方式切欄（split(",")）會錯位。' % n_comma)
    print('邊清單已存：%s（%d 條邊，密度 %.3f）'
          % (epath, len(rows), len(rows) / float(max(n * (n - 1), 1))))
    print('方向：source 為因、target 為果%s'
          % ('（已依設定轉置）' if ec.get('transpose') else ''))
    return gpath, epath


# =====================================================================
# 步驟 2：執行同事的 pipeline
# =====================================================================

def derived_path(cfg, key, suffix):
    """決定衍生檔（清理檔／縮減檔）的路徑。

    設定值為空時，放在 data.csv 的**同一個目錄**，檔名是原檔名加後綴：
        data/raw/semiconductor/case2.csv
          → data/raw/semiconductor/case2_clean.csv
          → data/raw/semiconductor/case2_reduced.csv
    這樣衍生檔和原始資料放在一起，不會散到另一個 data/clean/ 目錄去。
    設定值非空時以它為準（仍支援 {train_dataset} 佔位符）。
    """
    rc = cfg.get('reduce', {}) or {}
    v = (rc.get(key, '') or '').strip()
    if v:
        return subst(v, cfg['train_dataset'])
    src = cfg['data']['csv']
    d = os.path.dirname(src)
    stem, ext = os.path.splitext(os.path.basename(src))
    out = os.path.join(d, stem + suffix + (ext or '.csv'))
    if os.path.abspath(out) == os.path.abspath(src):
        sys.exit('衍生檔路徑與原始資料相同（%s），拒絕覆寫。' % src)
    return out


def colleague_source(cfg):
    """同事的 pipeline 要吃哪一份資料，回傳 (路徑, 後綴, 說明)。

    colleague.source:
      "reduced"（預設）步驟 0b 的縮減檔。候選集因此只能從 Welch 選出的
                      參數裡挑 —— 覆蓋率落差消失（候選必然都是圖的節點），
                      但候選集不再獨立於 Welch 排序。方法說明要據此調整。
      "clean"         步驟 0a 的無缺值全參數檔。候選涵蓋全部可用參數，
                      仍是獨立於 Welch 的第二個意見，代價是部分候選可能
                      不在圖裡（由 check_candidate_coverage 回報）。
      "original"      原始檔。他的程式不處理缺值時不要用。

    步驟 0 有三條捷徑會「不產生縮減檔」就返回（reduce.enabled 為 false、
    data.train_csv 已指定、清除缺值後參數已不超過上限），這時步驟 1 訓練
    用的是別的檔，但 derived_path 算出來的 _reduced 路徑從沒被寫出來
    —— 於是步驟 2 報「找不到 case2_reduced」。所以優先採用步驟 0 實際
    記錄的路徑（cfg['_train_csv'] / cfg['_clean_csv']），只有在單獨跑
    步驟 2（--only 2）而沒有那筆記錄時才回頭用推算的路徑。
    """
    cc = cfg.get('colleague', {}) or {}
    src = (cc.get('source', '') or 'reduced').lower()
    if src == 'reduced':
        p = cfg.get('_train_csv') or derived_path(cfg, 'out', '_reduced')
    elif src == 'clean':
        p = cfg.get('_clean_csv') or derived_path(cfg, 'clean_out', '_clean')
    elif src == 'original':
        return cfg['data']['csv'], '', '原始檔'
    else:
        sys.exit('colleague.source 只能是 reduced / clean / original，'
                 '收到 "%s"。' % src)
    # 說明文字按實際檔名給，不按 source 設定給 —— 捷徑路徑下 source 是
    # "reduced" 而檔案其實是 _clean，印成「縮減檔」會把人引到錯的地方找。
    stem = os.path.splitext(os.path.basename(p))[0]
    orig = os.path.splitext(os.path.basename(cfg['data']['csv']))[0]
    sfx = stem[len(orig):] if stem.startswith(orig) else ''
    lbl = {'_reduced': '縮減檔', '_clean': '無缺值檔', '': '原始檔'}.get(
        sfx, '衍生檔（%s）' % stem)
    return p, sfx, lbl


def colleague_ds(cfg):
    """同事的 pipeline 實際被呼叫時用的資料集名稱。

    他自己從 --train_dataset 解析資料路徑，所以要讓他讀步驟 0a 的
    無缺值檔，就得餵他一個不同的名稱。優先序：

      colleague.train_dataset  明確指定（完全覆寫）
      colleague.dataset_suffix 在原名後加後綴
      兩者皆空                  用「實際要餵給他的那個檔的主幹」

    這個名稱必須同時用在三個地方，否則就會錯開：
      1. 傳給他的 --train_dataset
      2. 他的 output_csv 檔名裡的 {train_dataset}
      3. 複製無缺值檔的目的檔名
    先前第 2 項用的是原始名稱，於是他寫出 checklist_case2_clean_.csv
    而 run_all 去找 checklist_case2_.csv，找不到。
    """
    cc = cfg.get('colleague', {}) or {}
    ds = cfg['train_dataset']
    explicit = cc.get('train_dataset', '') or ''
    if explicit:
        return explicit
    sfx = cc.get('dataset_suffix', None)
    if sfx:
        return ds + sfx
    # 未指定時直接用實際要餵給他的檔案主幹。推導後綴（"_reduced"）看似
    # 等價，但步驟 0 走捷徑時餵出去的其實是 case2_clean.csv，名稱卻會變
    # case2_reduced —— 名字和檔案錯開，他就讀不到。用檔名主幹則由構造
    # 保證三處一致：--train_dataset、copy_clean_to、output_csv。
    return os.path.splitext(os.path.basename(colleague_source(cfg)[0]))[0]


def check_colleague_input(cfg, cand_csv):
    """由同事的輸出反推他實際讀了哪個檔。

    run_all 印出「已複製無缺值檔」只證明複製成功，不證明他的程式讀了它
    —— 他從 --train_dataset 自己解析路徑，中間任何一步不合就會回去讀
    原本那份。所以這裡用他的輸出來反推：

      checklist 裡出現「只存在於原始檔、不存在於清理檔」的參數
        → 他讀的是原始檔（那些欄位在清理檔裡已被移除）

    這是正向證據，不是推測。反之若全部命中清理檔，只能說「與清理檔
    一致」—— 因為清理檔的欄位是原始檔的子集，兩邊都讀得到。
    """
    clean, _sfx, lbl = colleague_source(cfg)
    if not (clean and os.path.exists(clean) and os.path.exists(cand_csv)):
        return
    if os.path.abspath(clean) == os.path.abspath(cfg['data']['csv']):
        return          # source=original 時沒有可比的差集
    try:
        from pandas.api.types import is_numeric_dtype
        ch = set(read_csv_any(clean, nrows=1).columns)
        og = set(read_csv_any(cfg['data']['csv'], nrows=1).columns)
        cdf = read_csv_any(cand_csv)
    except Exception as e:
        print('（來源檔檢查略過：%s）' % e)
        return
    col = cfg.get('analysis', {}).get('sensor_col', 'Sensor')
    if col not in cdf.columns:
        objs = [c for c in cdf.columns if not is_numeric_dtype(cdf[c])]
        if not objs:
            return
        col = objs[0]
    cands = [str(v).strip() for v in cdf[col].dropna()]
    if not cands:
        return
    removed = og - ch               # 清理時被移除的欄位
    smoking = [c for c in cands if c in removed]
    print('')
    print('同事讀到的資料來源檢查（應為%s）' % lbl)
    print('  %s %d 欄／原始檔 %d 欄／差集 %d 欄'
          % (lbl, len(ch), len(og), len(removed)))
    print('  候選 %d 個，其中 %d 個落在「只存在於原始檔」的欄位裡'
          % (len(cands), len(smoking)))
    if smoking:
        print('  ❌ 他讀的不是%s。證據：以下候選在%s裡不存在 ——'
              % (lbl, lbl))
        print('     %s%s' % (', '.join(smoking[:10]),
                             ' …' if len(smoking) > 10 else ''))
        print('  請檢查 colleague.copy_clean_to 的路徑是否符合他的程式'
              '從 train_dataset 解析出的位置，')
        print('  以及 colleague.train_dataset 是否已設為那個名稱。')
    else:
        print('  ✓ 所有候選都在%s中，與讀%s一致。' % (lbl, lbl))
        print('    （%s是原始檔的子集，所以這是必要條件而非充分條件；'
              '上面的 ❌ 才是決定性證據。）' % lbl)


def check_candidate_coverage(cfg, gpath, cand_csv):
    """回報有多少候選參數不在圖的節點集合裡。

    候選集由步驟 2 產生，而參數縮減在步驟 0 就做了 —— 所以縮減時無法
    保留候選。不在圖裡的候選會被當成孤立節點，分組對它們沒有作用，
    而這件事在報告裡看不出來（它們照樣會出現在排查順序中）。
    所以這裡明確回報，不讓它靜默發生。
    """
    npath = os.path.splitext(gpath)[0] + '_nodes.txt'
    if not (os.path.exists(npath) and os.path.exists(cand_csv)):
        return
    try:
        gnames = set(ln.strip() for ln in
                     read_text_any(npath).splitlines() if ln.strip())
        cdf = read_csv_any(cand_csv)
    except Exception as e:
        print('（候選覆蓋率檢查略過：%s）' % e)
        return

    col = cfg.get('analysis', {}).get('sensor_col', 'Sensor')
    if col not in cdf.columns:
        # 同上：pandas 3 的字串欄不是 object dtype
        from pandas.api.types import is_numeric_dtype
        objs = [c for c in cdf.columns if not is_numeric_dtype(cdf[c])]
        if not objs:
            return
        col = objs[0]
    cands = [str(v).strip() for v in cdf[col].dropna()]
    if not cands:
        return
    miss = [c for c in cands if c not in gnames]
    print('')
    print('候選與圖的覆蓋率：%d / %d 個候選在圖裡'
          % (len(cands) - len(miss), len(cands)))
    if miss:
        print('  ⚠️ %d 個候選不在圖的節點集合中，將被當成孤立節點：'
              % len(miss))
        print('     %s%s' % (', '.join(miss[:10]),
                             ' …' if len(miss) > 10 else ''))
        print('  這些參數在步驟 0 的縮減中被丟棄。若它們重要，兩個做法：')
        print('    1. 把候選檔存起來，下次用 reduce.candidates 指向它'
              '（無條件保留）')
        print('    2. 調高 reduce.max_params，或改以 --from-step 2 '
              '沿用既有的圖')


def step2_colleague(cfg):
    ds = cfg['train_dataset']
    cc = cfg['colleague']
    gc = cc.get('group_col', '') or ''
    banner(2, '執行同事的 pipeline')

    # script / config / output 皆相對於同事的 cwd 解析
    cwd = os.path.abspath(subst(cc.get('cwd', '') or '.', ds, gc))
    if not os.path.isdir(cwd):
        sys.exit('找不到同事的工作目錄：%s' % cwd)

    def at_cwd(rel):
        # 用同事實際被呼叫的名稱替換，而不是原始名稱 —— 他的 output_csv
        # 檔名裡的 {train_dataset} 會是那個名稱
        rel = subst(rel, colleague_ds(cfg), gc)
        return rel if os.path.isabs(rel) else os.path.join(cwd, rel)

    script = at_cwd(cc['script'])
    if not os.path.exists(script):
        sys.exit('找不到同事的腳本：%s' % script)

    # 邊清單是他的輸入。檔名由 edges.out 決定（已用 colleague_ds 的名稱
    # 替換），但那個路徑是你設的 —— 若與他的程式從 train_dataset 推出來
    # 的位置不符，他會找不到而以自己的方式失敗。先攔下來說清楚。
    ep = subst(cfg.get('edges', {}).get('out', '') or '',
               colleague_ds(cfg))
    if ep:
        epa = ep if os.path.isabs(ep) else os.path.join(os.getcwd(), ep)
        if os.path.exists(epa):
            print('邊清單：%s' % epa)
        else:
            print('⚠️ 找不到邊清單 %s' % epa)
            print('   同事的 pipeline 以 --train_dataset %s 被呼叫，'
                  '會去找帶該名稱的 edges 檔。' % colleague_ds(cfg))
            print('   請先執行步驟 1，或確認 edges.out 的路徑與他的程式'
                  '預期的位置一致。')

    py = cc.get('python') or sys.executable
    cmd = [py, script]
    # 已指定資料集時，只傳 --train_dataset；config 由他的腳本自行取預設，
    # 避免 config 裡的 dataset 清單覆蓋我們指定的那一個。
    use_cfg = cc.get('use_config', False)
    if use_cfg:
        conf = at_cwd(cc['config'])
        if not os.path.exists(conf):
            sys.exit('找不到同事的 config：%s' % conf)
        cmd += ['--config', conf]
    # 同事的 pipeline 自己從 --train_dataset 解析資料路徑，run_all 無法
    # 指定它讀哪個檔。但他的程式不處理缺值，而原始檔有缺值 —— 所以要
    # 讓它改讀步驟 0a 的無缺值全參數檔。兩種做法：
    #
    #   1. colleague.train_dataset：把 0a 的輸出複製到他的資料目錄、
    #      取一個新的資料集名稱，再以那個名稱呼叫。他不必改任何程式。
    #   2. colleague.csv_arg：他的程式若支援指定 csv（例如 --csv），
    #      填入那個參數名即可。
    #
    # 不要用縮減後的檔 —— 候選集只能從縮減後的參數裡挑，而縮減用的是
    # Welch，候選集就不再是獨立於 Welch 的第二個意見了。
    # 自動把 0a 的無缺值檔放到同事的資料目錄，省掉手動 cp。
    # 路徑相對於 colleague.cwd 解析，搭配 colleague.train_dataset 使用。
    # 注意：copy_clean_to 裡的 {train_dataset} 會被替換成「同事被呼叫時
    # 用的名稱」（含後綴），不是原始名稱。所以預設值寫
    # data/{train_dataset}.csv 就會變成 data/case2_clean.csv，
    # 不要再自己加一次 _clean。
    dst_rel = cc.get('copy_clean_to', '') or ''
    if dst_rel:
        clean, _sfx, _lbl = colleague_source(cfg)
        if not os.path.exists(clean):
            # 單獨跑步驟 2 時 cfg['_train_csv'] 是空的，路徑只能推算，
            # 而推算的 _reduced 檔在步驟 0 走捷徑的那幾種情況下根本沒被
            # 寫出來。所以除了報路徑，還要說出同名的檔在哪裡。
            stem = os.path.splitext(os.path.basename(clean))[0]
            near = []
            for root, _d, files in os.walk(os.path.dirname(
                    os.path.abspath(cfg['data']['csv'])) or '.'):
                for fn in files:
                    if fn.startswith(os.path.splitext(os.path.basename(
                            cfg['data']['csv']))[0]) and fn.endswith('.csv'):
                        near.append(os.path.join(root, fn))
            msg = ['colleague.copy_clean_to 已設定，但找不到要餵給同事的'
                   '%s：%s' % (_lbl, clean)]
            msg.append('   這個檔由步驟 0 產生，而步驟 0 只在執行步驟 1 '
                       '時才會跑。')
            if near:
                msg.append('   同一個目錄下現有的檔：')
                for f in sorted(set(near))[:8]:
                    msg.append('     %s' % f)
                msg.append('   若其中有 %s_clean.csv 而沒有 %s.csv，'
                           '就是步驟 0 走了捷徑'
                           % (cfg['train_dataset'], stem))
                msg.append('   （清除缺值後參數已不超過上限，'
                           '於是沒再做 0b 縮減）。')
                msg.append('   把 colleague.source 改成 "clean"，'
                           '或連步驟 1 一起跑（--from-step 1）。')
            else:
                msg.append('   請執行 --from-step 1。')
            sys.exit('\n'.join(msg))
        dst = at_cwd(dst_rel)
        # 防止覆蓋同事的原始資料。最危險的情況是把 copy_clean_to 指到
        # 他原本那份檔案 —— 那會在他不知情的情況下改掉他的輸入。
        orig = os.path.abspath(cfg['data']['csv'])
        if os.path.abspath(dst) == orig:
            sys.exit('colleague.copy_clean_to 指向 data.csv 本身（%s），'
                     '拒絕覆寫原始資料。' % orig)
        stem = os.path.splitext(os.path.basename(dst))[0]
        if stem == ds and not cc.get('train_dataset', ''):
            sys.exit('colleague.copy_clean_to 的檔名主幹是 "%s"，與'
                     'train_dataset 同名，\n   這會覆蓋同事原本讀的那份'
                     '資料。請改用別的名稱（例如 %s_clean），\n'
                     '   並把 colleague.train_dataset 設為該名稱。'
                     % (ds, ds))
        if os.path.abspath(dst) == os.path.abspath(clean):
            # 同事的程式本來就讀那個檔所在的目錄 —— 不需要複製
            print('同事的資料路徑就是該檔本身，無需複製：%s' % clean)
        else:
            d = os.path.dirname(dst)
            if d and not os.path.isdir(d):
                os.makedirs(d)
            shutil.copyfile(clean, dst)
            print('已複製%s → %s' % (_lbl, dst))
        # BOM 檢查：我們的 read_csv_any 以 utf-8-sig 開頭所以讀得動，
        # 但他若用 pandas 預設編碼，第一個欄名會變 "﻿LOT_ID"，
        # 於是 LOT_ID 不被認作中繼欄位而被當成一個感測器參數。
        with open(dst, 'rb') as f:
            if f.read(3) == b'\xef\xbb\xbf':
                print('⚠️ 餵給同事的檔開頭有 BOM。他若以 pandas 預設編碼'
                      '讀，第一個欄名會是 "\\ufeffLOT_ID"，')
                print('   LOT_ID 會被當成感測器參數（參數數目多 1）。'
                      '請確認他讀檔時用 encoding="utf-8-sig"。')

    ds_use = colleague_ds(cfg)
    if ds_use != ds:
        if cc.get('train_dataset', '') or '':
            why = 'colleague.train_dataset 指定'
        elif cc.get('dataset_suffix', None):
            why = 'colleague.dataset_suffix="%s"' % cc['dataset_suffix']
        else:
            why = '取自實際餵給他的檔名'
        print('餵給同事的資料集名稱：%s（%s；原為 %s）' % (ds_use, why, ds))
    cmd += ['--train_dataset', ds_use]
    ca = cc.get('csv_arg', '')
    if ca:
        clean = colleague_source(cfg)[0]
        if clean and os.path.exists(clean):
            cmd += [ca, os.path.abspath(clean)]
            print('以 %s 指定%s：%s'
                  % (ca, colleague_source(cfg)[2], clean))
        else:
            print('⚠️ colleague.csv_arg 已設定，但找不到步驟 0a 的輸出，'
                  '同事的 pipeline 將讀它自己的資料（可能含缺值）。')
    elif ds_use == ds:
        print('⚠️ 餵給同事的資料集名稱與原始相同，他會讀自己原本那份'
              '（可能含缺值）。')
        print('   請設 colleague.dataset_suffix（預設 "_clean"）或 '
              'colleague.train_dataset。')
    print('工作目錄：%s' % cwd)
    print('執行：%s' % ' '.join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        sys.exit('同事的 pipeline 回傳錯誤碼 %d，流程中止。' % r.returncode)
    print('完成（%.1f 秒）' % (time.time() - t0))

    out = at_cwd(cc['output_csv'])
    if not os.path.exists(out):
        hint = ''
        if '{group_col}' in cc['output_csv'] and not gc:
            hint = ('\n   colleague.group_col 未填，檔名中的 {group_col} '
                    '被替換成空字串。')
        # 列出同目錄的候選檔，方便對照
        dd = os.path.dirname(out)
        if os.path.isdir(dd):
            got = sorted(os.listdir(dd))[:10]
            hint += '\n   該目錄現有檔案：%s' % (', '.join(got) if got else '（空）')
        sys.exit('找不到同事的輸出檔：%s%s' % (out, hint))

    df = read_csv_any(out)
    print('候選輸出：%s' % out)
    print('  %d 列，欄位：%s' % (len(df), ', '.join(map(str, df.columns))))
    return out


# =====================================================================
# 步驟 3：執行分析流程
# =====================================================================

def step3_analysis(cfg, graph_path, cand_csv):
    ds = cfg['train_dataset']
    ac, dc = cfg['analysis'], cfg['data']
    banner(3, '執行分析流程')

    sub = {
        # 步驟 3 讀原始檔（不是步驟 0 的輸出）—— 分析階段要看得到全部
        # 參數，被縮減丟掉的那些也要能出現在報告裡。Welch 是 O(N)，
        # 5000 個參數算得動。代價是缺值在這裡重新出現，所以缺值設定
        # 必須跟著傳下去：原本只傳 5 個鍵，dropna / drop_col_na_frac /
        # max_row_loss 都被靜默丟掉，於是 run_pipeline 用自己的預設值，
        # 而它中止時印的「請改 data.dropna」在 run_all 這條路上無效。
        "data": {
            "csv": dc['csv'], "wafer_col": dc['wafer_col'],
            "label_col": dc['label_col'],
            "abnormal_value": dc['abnormal_value'],
            "exclude": dc.get('exclude', []),
            "dropna": dc.get('dropna', 'row'),
            "drop_col_na_frac": dc.get('drop_col_na_frac', 0.5),
            "max_row_loss": dc.get('max_row_loss', 0.3),
            "min_row_keep": dc.get('min_row_keep', 0.8)
        },
        "candidates": {
            "file": cand_csv,
            "sensor_col": ac.get('sensor_col', 'Sensor'),
            "priority_col": ac.get('priority_col', 'Priority'),
            "topk": ac.get('topk', 5),
            "min_effect": ac.get('min_effect', 0.5),
            "n_boot": ac.get('n_boot', 200),
            "n_perm": ac.get('n_perm', 300),
            "add_statistical": ac.get('add_statistical', True),
            "expansion": ac.get('expansion', {}),
            "true_roots": ac.get('true_roots', [])
        },
        "graph": dict(ac.get('graph', {}), path=graph_path),
        "rules": ac.get('rules', {}),
        "output": {
            "prefix": subst(ac['prefix'], ds),
            "brief": ac.get('brief', True),
            "show_welch": ac.get('show_welch', False)
        },
        "seed": cfg.get('seed', 0)
    }
    cpath = subst(ac['config_out'], ds)
    # analysis.config_out 是「產生」的檔，每跑一次步驟 3 就整個覆寫。
    # 它若跟 --config 指向同一個檔，使用者對設定的每一次修改都會在下一
    # 次執行時被抹掉 —— 而症狀只是「改了沒有用」，看不出檔案被改回去。
    own = cfg.get('_config_path', '')
    if own and os.path.abspath(cpath) == os.path.abspath(own):
        sys.exit('analysis.config_out 與 --config 是同一個檔（%s）。\n'
                 '   這個檔由步驟 3 產生並覆寫，你對它做的修改會被抹掉。\n'
                 '   請把 analysis.config_out 改成別的名字，例如\n'
                 '     "config_out": "config/{train_dataset}_pipeline.json"'
                 % cpath)
    print('缺值設定（轉給分析流程）：dropna=%s　drop_col_na_frac=%s　'
          'max_row_loss=%s'
          % (sub['data']['dropna'], sub['data']['drop_col_na_frac'],
             sub['data']['max_row_loss']))
    print('  min_row_keep=%s（>0 時自動覆寫 drop_col_na_frac）'
          % sub['data']['min_row_keep'])
    ensure_dir(cpath)
    with open(cpath, 'w', encoding='utf-8') as f:
        json.dump(sub, f, ensure_ascii=False, indent=2)
    print('已產生分析 config：%s' % cpath)

    script = ac['script']
    if not os.path.exists(script):
        sys.exit('找不到分析腳本：%s' % script)
    cmd = [sys.executable, os.path.abspath(script), '--config',
           os.path.abspath(cpath)]
    print('執行：%s\n' % ' '.join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit('分析流程回傳錯誤碼 %d。' % r.returncode)
    return subst(ac['prefix'], ds) + '_result.json'


def step4_report(cfg, result_json):
    rc = cfg.get('report', {})
    if not rc.get('enabled', True):
        return
    ds = cfg['train_dataset']
    script = rc.get('script', 'scripts/make_html_report.py')
    if not os.path.exists(script):
        print('（找不到 %s，略過 HTML 報告）' % script)
        return
    if not os.path.exists(result_json):
        print('（找不到 %s，略過 HTML 報告）' % result_json)
        return
    out = subst(rc['out'], ds)
    ensure_dir(out)
    cmd = [sys.executable, os.path.abspath(script), '--json',
           os.path.abspath(result_json), '--out', os.path.abspath(out)]
    subprocess.run(cmd)


# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description='三階段統籌：MT-NGC 訓練 → 同事 pipeline → 分析流程')
    ap.add_argument('--config', help='設定檔路徑')
    ap.add_argument('--init', help='產生設定檔範本至指定路徑')
    ap.add_argument('--from-step', type=int, default=1, choices=[1, 2, 3],
                    help='從第幾步開始（跳過前面的步驟，沿用既有產出）')
    ap.add_argument('--only', type=int, default=0, choices=[0, 1, 2, 3],
                    help='只執行某一步；0 表示不限制')
    args = ap.parse_args()

    if args.init:
        if os.path.exists(args.init):
            sys.exit('檔案已存在：%s' % args.init)
        ensure_dir(args.init)
        with open(args.init, 'w', encoding='utf-8') as f:
            json.dump(TEMPLATE, f, ensure_ascii=False, indent=2)
        print('設定檔範本已產生：%s' % args.init)
        print('請先填入以下欄位再執行：')
        print('  data.csv            你的資料檔')
        print('  data.wafer_col      wafer 識別欄（不唯一時用複合鍵）')
        print('  colleague.script    同事的 run_pipeline.py 完整路徑')
        print('  colleague.config    同事的 pipeline_config.yaml')
        print('  colleague.output_csv 同事輸出的候選 csv 路徑')
        return

    if not args.config:
        sys.exit('請指定 --config，或以 --init 產生範本。')
    if not os.path.exists(args.config):
        sys.exit('找不到設定檔：%s' % args.config)
    cfg = load_config(args.config)
    cfg['_config_path'] = args.config
    ds = cfg['train_dataset']

    lo = args.only or args.from_step
    hi = args.only or 3
    print('資料集：%s　執行步驟：%d ~ %d' % (ds, lo, hi))

    gpath = graph_path(cfg)
    _gc = cfg['colleague'].get('group_col', '') or ''
    _ccwd = os.path.abspath(subst(cfg['colleague'].get('cwd', '.') or '.',
                                  ds, _gc))
    # 候選檔名用的是同事實際被呼叫的資料集名稱，不是原始名稱
    _co = subst(cfg['colleague']['output_csv'],
                colleague_ds(cfg), _gc)
    cand = _co if os.path.isabs(_co) else os.path.join(_ccwd, _co)
    result_json = subst(cfg['analysis']['prefix'], ds) + '_result.json'

    # 0a 的輸出路徑先備好 —— --from-step 2 時步驟 0 不會跑，但步驟 2
    # 仍需要指向上一輪產生的無缺值檔
    cfg['_clean_csv'] = derived_path(cfg, 'clean_out', '_clean')

    if lo <= 1 <= hi:
        # 0a 的輸出（無缺值、全參數）步驟 2 也要用，所以步驟 0 不能
        # 只在要訓練時才跑
        clean_csv, train_csv = step0_prepare(cfg)
        cfg['_clean_csv'] = clean_csv
        gpath, _ = step1_train(cfg, train_csv)
    elif not os.path.exists(gpath) and lo <= 3 <= hi:
        sys.exit('跳過步驟 1，但找不到既有的圖檔：%s\n'
                 '請先執行步驟 1，或確認 mtngc.graph_dir 設定。' % gpath)

    if lo <= 2 <= hi:
        cand = step2_colleague(cfg)
    elif lo <= 3 and not os.path.exists(cand):
        sys.exit('跳過步驟 2，但找不到候選檔：%s' % cand)

    # 來源檔檢查要跟著「步驟 2 跑過」而不是「步驟 3 要跑」。
    # 原本綁在 lo <= 3 <= hi，所以 --only 2（最常用來單測步驟 2 的指令）
    # 反而不會檢查。
    if os.path.exists(cand):
        check_colleague_input(cfg, cand)

    # 覆蓋率要比對圖的節點，所以需要圖存在；步驟 3 的前提也是如此
    if lo <= 3 <= hi:
        check_candidate_coverage(cfg, gpath, cand)

    if lo <= 3 <= hi:
        result_json = step3_analysis(cfg, gpath, cand)
        step4_report(cfg, result_json)

    print('\n' + '=' * 70)
    print('完成')
    print('=' * 70)
    if os.path.exists(result_json):
        print('  結果 JSON：%s' % result_json)
    rp = subst(cfg.get('report', {}).get('out', ''), ds)
    if rp and os.path.exists(rp):
        print('  HTML 報告：%s' % rp)


if __name__ == '__main__':
    main()