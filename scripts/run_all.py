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
import json
import os
import re
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
        "wafer_col": "LOT_ID,WAFER_ID",
        "label_col": "Symbol Define",
        "abnormal_value": "Abnormal",
        "exclude": [],
        "dropna": "row",
        "drop_col_na_frac": 0.5
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
        "cwd": "../fd_analysis/gnn-tam-main",
        "script": "run_pipeline.py",
        "config": "pipeline_config.yaml",
        "python": "",
        "group_col": "",
        "use_config": False,
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
    if not isinstance(s, str):
        return s
    return s.replace('{train_dataset}', ds).replace('{group_col}', gc)


ENCODINGS = ['utf-8-sig', 'utf-8', 'cp950', 'big5hkscs', 'gbk', 'latin-1']


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


def step1_train(cfg):
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
    train_csv = dc.get('train_csv') or dc['csv']
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

    gpath = graph_path(cfg)
    ensure_dir(gpath)
    np.save(gpath, W)
    print('圖已存：%s' % gpath)

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

    epath = subst(ec['out'], ds)
    ensure_dir(epath)
    fmt = '%%.%df' % ec.get('precision', 6)
    with open(epath, 'w', encoding='utf-8', newline='') as f:
        f.write('source,score,target\n')
        for s_, d_, w_ in rows:
            f.write('%s,%s,%s\n' % (s_, fmt % w_, d_))
    print('邊清單已存：%s（%d 條邊，密度 %.3f）'
          % (epath, len(rows), len(rows) / float(max(n * (n - 1), 1))))
    print('方向：source 為因、target 為果%s'
          % ('（已依設定轉置）' if ec.get('transpose') else ''))
    return gpath, epath


# =====================================================================
# 步驟 2：執行同事的 pipeline
# =====================================================================

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
        rel = subst(rel, ds, gc)
        return rel if os.path.isabs(rel) else os.path.join(cwd, rel)

    script = at_cwd(cc['script'])
    if not os.path.exists(script):
        sys.exit('找不到同事的腳本：%s' % script)

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
    cmd += ['--train_dataset', ds]
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
        "data": {
            "csv": dc['csv'], "wafer_col": dc['wafer_col'],
            "label_col": dc['label_col'],
            "abnormal_value": dc['abnormal_value'],
            "exclude": dc.get('exclude', [])
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
    ds = cfg['train_dataset']

    lo = args.only or args.from_step
    hi = args.only or 3
    print('資料集：%s　執行步驟：%d ~ %d' % (ds, lo, hi))

    gpath = graph_path(cfg)
    _gc = cfg['colleague'].get('group_col', '') or ''
    _ccwd = os.path.abspath(subst(cfg['colleague'].get('cwd', '.') or '.',
                                  ds, _gc))
    _co = subst(cfg['colleague']['output_csv'], ds, _gc)
    cand = _co if os.path.isabs(_co) else os.path.join(_ccwd, _co)
    result_json = subst(cfg['analysis']['prefix'], ds) + '_result.json'

    if lo <= 1 <= hi:
        gpath, _ = step1_train(cfg)
    elif not os.path.exists(gpath) and lo <= 3 <= hi:
        sys.exit('跳過步驟 1，但找不到既有的圖檔：%s\n'
                 '請先執行步驟 1，或確認 mtngc.graph_dir 設定。' % gpath)

    if lo <= 2 <= hi:
        cand = step2_colleague(cfg)
    elif lo <= 3 and not os.path.exists(cand):
        sys.exit('跳過步驟 2，但找不到候選檔：%s' % cand)

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