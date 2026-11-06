# -*- coding: utf-8 -*-
r"""
run_pipeline.py  ——  半導體根因分析整合流程

一個指令跑完全部：資料檢查 → 參數排序 → 穩定度驗證 → 分組 → 規則萃取。
所有調參集中在 config.json，候選參數可用純文字檔指定。

用法
----
  1. 產生設定檔範本：
       python run_pipeline.py --init config_case1.json

  2. 編輯 config_case1.json，填入資料路徑與欄位名稱

  3. 執行：
       python run_pipeline.py --config config_case1.json

候選參數檔（選用）
------------------
一行一個參數名稱，空行與 # 開頭的行會被忽略：

    # case1 候選
    TEPO_IV
    TEOS_IV
    CHAMBER_PRESSURE

未提供時，自動以 Welch 效果量取 top-k。

本檔為自足式（不 import 其他腳本），以避免離線環境的路徑問題。
探索性分析仍請使用各別腳本。

環境：Python 3.8+ / numpy / pandas
"""

import argparse
import itertools
import json
import os
import re
import sys
import time
import unicodedata

import numpy as np
import pandas as pd


def _w(s):
    """字串的顯示寬度（中文字算 2）。"""
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1
               for c in str(s))


def fmt_table(df, left_cols=()):
    """對齊中文欄位的表格輸出。pandas 的 to_string 以字元數計寬，
    中文為雙寬字元故會錯位，此處以顯示寬度重新對齊。"""
    cols = list(df.columns)
    widths = []
    for c in cols:
        w = _w(c)
        for v in df[c]:
            w = max(w, _w(v))
        widths.append(w)
    out = []
    head = []
    for c, w in zip(cols, widths):
        head.append(str(c) + ' ' * (w - _w(c)) if c in left_cols
                    else ' ' * (w - _w(c)) + str(c))
    out.append('  '.join(head))
    for _, row in df.iterrows():
        cells = []
        for c, w in zip(cols, widths):
            v = str(row[c])
            cells.append(v + ' ' * (w - _w(v)) if c in left_cols
                         else ' ' * (w - _w(v)) + v)
        out.append('  '.join(cells))
    return '\n'.join(out)

DEFAULT_META = ['WFR_ID', 'WAFER_ID', 'LOT_ID', 'START_DATETIME', 'END_DATETIME',
                'STEP', 'STEP_ID', 'RECIPE', 'CHAMBER', 'EQP_ID', 'TOOL_ID',
                'TIME', 'TIMESTAMP', 'label', 'LABEL']

TEMPLATE = {
    "data": {
        "csv": "data/raw/semiconductor/case1.csv",
        "wafer_col": "LOT_ID,WFR_ID,START_DATETIME",
        "label_col": "Symbol Define",
        "abnormal_value": "Abnormal",
        "exclude": [],
        "dropna": "row",
        "drop_col_na_frac": 0.5,
        "min_row_keep": 0.8
    },
    "candidates": {
        "file": "",
        "topk": 5,
        "min_effect": 0.5,
        "n_boot": 200,
        "true_roots": [],
        "sensor_col": "sensor",
        "priority_col": "priority",
        "n_perm": 300,
        "add_statistical": True,
        "expansion": {
            "enabled": True,
            "max_hop": 2,
            "min_effect": None
        }
    },
    "graph": {
        "path": "results/case1_real_graph_task2b.npy",
        "edge_k": 1,
        "edge_k_large": 2,
        "large_nodes": 50,
        "min_concentration": 0.0,
        "min_edge": None,
        "max_hop": 3
    },
    "rules": {
        "enabled": True,
        "test_frac": 0.4,
        "n_quantile": 39,
        "n_random": 200,
        # null = 不設限。2000 是列舉改寫成剪枝 DFS 之前的遺物：那時成本
        # 正比於名目組合數，現在不是（實測 4300 萬組合只走 3 個節點），
        # 用它當閥只會在不必要時整段略過。工作量改由 enum_max_nodes 直接
        # 限制。設一個數字仍可擋掉列舉，序貫不受影響。
        # 注意 max_per_group > 1 時這個值另有用途（決定是否退回每組一個）,
        # 那裡用的是展開後的組合數，與這裡同名但語意不同。
        "max_combo": None,
        "min_gain": 0.02,
        "gain_metric": "F1",
        "enum_truncate": True,
        "enum_prune": True,
        "enum_max_rows": 20000,
        # 工作量的真正上限：走訪的節點數。max_rows 只限制輸出列數，
        # max_combo 用的名目組合數則與工作量無關。
        "enum_max_nodes": 2000000,
        # 整張表只剩條件數 1 時，強制補幾筆多參數組合供對照（標 *）。
        # 0 = 不補。
        "force_multi": 2,
        "max_per_group": 1,
        "extend": False,
        "extend_when": 1,
        "extend_max_add": 2
    },
    "output": {
        "prefix": "results/case1_pipeline",
        "show_welch": True,
        "brief": False
    },
    "seed": 0
}


# =====================================================================
# 統計
# =====================================================================

def cohen_d(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) /
                 max(len(a) + len(b) - 2, 1)) + 1e-12
    return (b.mean() - a.mean()) / sp


def welch_t(Xn, Xa):
    mn, ma = np.nanmean(Xn, axis=0), np.nanmean(Xa, axis=0)
    vn = np.nanvar(Xn, axis=0, ddof=1)
    va = np.nanvar(Xa, axis=0, ddof=1)
    se = np.sqrt(vn / max(len(Xn), 1) + va / max(len(Xa), 1)) + 1e-12
    return np.nan_to_num(np.abs(ma - mn) / se, nan=0.0, posinf=0.0)


def var_ratio(Xn, Xa):
    vn = np.nanvar(Xn, axis=0) + 1e-12
    va = np.nanvar(Xa, axis=0) + 1e-12
    return np.nan_to_num(np.maximum(va / vn, vn / va), nan=1.0, posinf=1.0)


def ranks_from(s):
    o = np.argsort(-np.asarray(s, dtype=float), kind='mergesort')
    r = np.empty(len(s), dtype=int)
    r[o] = np.arange(1, len(s) + 1)
    return r


ENCODINGS = ['utf-8-sig', 'utf-8', 'cp950', 'big5hkscs', 'gbk', 'latin-1']


def read_csv_any(path, **kw):
    """依序嘗試常見編碼讀 csv。

    台灣環境的 csv 常為 CP950/Big5，pandas 預設 utf-8 會在遇到中文或
    全形符號時丟 UnicodeDecodeError。latin-1 永不失敗，作為最後退路
    （可能出現亂碼，但不會中斷流程）。
    """
    last = None
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc, **kw)
            if enc not in ('utf-8-sig', 'utf-8'):
                print('  （%s 以 %s 編碼讀取）'
                      % (os.path.basename(path), enc))
            return df
        except UnicodeDecodeError as e:
            last = e
        except Exception:
            raise
    raise UnicodeDecodeError(
        'utf-8', b'', 0, 1,
        '無法以 %s 任一編碼讀取 %s：%s'
        % ('/'.join(ENCODINGS), path, last))


def read_text_any(path):
    last = None
    for enc in ENCODINGS:
        try:
            with open(path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError as e:
            last = e
    raise UnicodeDecodeError('utf-8', b'', 0, 1,
                             '無法讀取 %s：%s' % (path, last))


def calibrate_threshold(X, y, n_perm, seed):
    """以標籤置換校準效果量門檻。

    打亂 Normal/Abnormal 標籤後重算所有參數的效果量，重複 n_perm 次，
    得到「完全沒有真實差異時，效果量會有多大」的分布。

    回傳 (全域門檻, 單參數門檻, 置換得到的所有效果量)：
      全域門檻   每次置換取全參數最大值，再取 95% 分位。已內含多重比較
                 校正，較嚴格，適合判定「這個參數確實異常」。
      單參數門檻 所有置換效果量混合後的 95% 分位，未校正，較寬鬆，
                 適合篩選擴展結果 —— 擴展本就是要找邊際訊號弱的參數。
    """
    rng = np.random.RandomState(seed)
    maxes, pooled = [], []
    yb = y.astype(bool)
    for _ in range(n_perm):
        yp = rng.permutation(yb)
        e = np.array([abs(cohen_d(X[~yp, j], X[yp, j]))
                      for j in range(X.shape[1])])
        maxes.append(e.max())
        pooled.append(e)
    pooled = np.concatenate(pooled)
    return (float(np.percentile(maxes, 95)),
            float(np.percentile(pooled, 95)), pooled)


# =====================================================================
# 圖
# =====================================================================

def binarize(W, k, min_conc=0.0, min_edge=0.0, stats=None):
    """每個節點取最強的 k 條入邊；低於 min_edge 的邊不算。

    這裡真正造成不穩定的是**同分**，不是弱邊。以模擬資料量測（N=20、
    真實因果圖已知、n_seed=10）：

      W 是對每個 seed 的 top-k 二值遮罩取平均，所以值被量化成 1/n_seed
      的倍數，而且大量飽和在 1.0（task2a：20 列裡有 25 個非對角格是
      1.0）。於是「該列最大值有同分」的列佔了將近一半 —— task1 4/20、
      task2a 8/20、task2b 9/20。那些列在 k=1 之下取誰，完全由 np.argsort
      的欄索引順序決定。

      隨機重排欄位再選一次、對回原標號，與原選擇的 Jaccard：
        k=1  task1 0.815  task2a 0.660  task2b 0.630
        k=2  task1 0.909  task2a 1.000  task2b 0.969
      也就是 k=1 的邊集合有多達 37% 是欄位順序決定的。注意方向：k=2 比
      k=1 更穩，所以把 edge_k 從 2 降到 1 會讓圖更隨意而不是更穩定。

    同分的一整組一起收，就把這個依賴完全消掉（上述兩個 k 都變成 1.000）。
    這是本函式最重要的一處修正。

    min_edge 則是另一回事，不要高估它：它讓結果對 k 比較不敏感，但**消
    不掉** k 的影響（量測到 t=0.8 時 k=1 與 k=2 的 Jaccard 是 0.92 /
    0.85 / 0.86，始終不到 1.0），因為門檻只縮小候選池，不會取消 k 這個
    上限。它的用處是把低頻率的邊擋在分組之外：t=0.5~0.6 對 task2b 幾乎
    免費（union F1 .385 → .393），t=0.8 就開始明顯損失 recall
    （task2b .309 → .235）。

    門檻在前、k 在後：k 只是上限。過了門檻卻超出 k 的邊都是真的穩定邊，
    不是硬湊出來的。

    另外，原本的寫法會用零值欄位補滿 k（argsort 對整列排序），那不是弱
    邊而是沒有訊號。實測在 k=1、k=2 不會發生（每列至少有 2 個非零），
    k=3 則 task2a 有 8/60 條是這樣來的。這裡只取非零者。

    集中度 = 該列最大權重 / 該列權重總和，尺度無關，可跨列比較。
    外生節點的權重分散，集中度接近 1/(n-1)。min_conc 預設 0 時這個
    判斷是關閉的（任何非零列都過），要用必須明確給一個正值。

    stats 給一個 dict 就會填入診斷數字（n_root 無入邊的節點數、n_tie 因
    同分而超出 k 的列數、n_cut 被 min_edge 擋掉的邊數）。
    """
    W = np.abs(np.asarray(W, dtype=float))
    np.fill_diagonal(W, 0.0)
    n = W.shape[0]
    A = np.zeros((n, n), dtype=bool)
    n_root = n_tie = n_cut = 0
    for i in range(n):
        row, s = W[i], W[i].sum()
        if s <= 0 or row.max() / s < min_conc:
            n_root += 1
            continue
        cand = np.flatnonzero(row >= min_edge) if min_edge > 0 \
            else np.flatnonzero(row > 0)
        n_cut += int((row > 0).sum()) - len(cand)
        if len(cand) == 0:
            n_root += 1
            continue
        vals = row[cand]
        order = cand[np.argsort(-vals, kind='stable')]
        if len(order) > k:
            # 同分的一整組一起收，否則取誰就看欄位順序
            cut = row[order[k - 1]]
            sel = [j for j in order if row[j] >= cut]
            if len(sel) > k:
                n_tie += 1
        else:
            sel = list(order)
        A[i, sel] = True
    np.fill_diagonal(A, False)
    if stats is not None:
        stats.update(n_root=n_root, n_tie=n_tie, n_cut=n_cut)
    return A, W


def order_robustness(W, k, min_conc=0.0, min_edge=0.0, n_perm=50, seed=0):
    """隨機重排欄位、重選、對回原標號，量「這張鄰接矩陣有多少是欄位順序
    決定的」。回傳平均 Jaccard，1.0 表示與欄位順序無關。

    這是比「k 與 k±1 的 Jaccard」有意義得多的診斷。後者在無門檻的 top-k
    之下是恆等式：k=1 的選擇必然是 k=2 選擇的子集，所以 Jaccard 恆等於
    n_k1/n_k2 ≈ k1/k2 = 0.5 —— 不論矩陣內容是什麼。純雜訊和完美還原都會
    讀到 0.5，那個數字不帶任何資訊。重排欄位則直接問到要問的事。
    """
    rng = np.random.RandomState(seed)
    n = W.shape[0]
    base = set(zip(*np.nonzero(binarize(W, k, min_conc, min_edge)[0])))
    js = []
    for _ in range(n_perm):
        p = rng.permutation(n)
        Ap = binarize(W[np.ix_(p, p)], k, min_conc, min_edge)[0]
        back = set((int(p[i]), int(p[j])) for i, j in zip(*np.nonzero(Ap)))
        un = len(base | back)
        js.append(len(base & back) / float(un) if un else 1.0)
    return float(np.mean(js)) if js else 1.0


def break_cycles(A, W):
    A = A.copy()
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] and A[j, i]:
                if W[i, j] >= W[j, i]:
                    A[j, i] = False
                else:
                    A[i, j] = False
    return A


def reachable(A, src, dst, max_hop):
    if src == dst:
        return False
    seen, frontier = {src}, [src]
    for _ in range(max_hop):
        nxt = []
        for u in frontier:
            for v in np.where(A[:, u])[0]:
                v = int(v)
                if v == dst:
                    return True
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
        if not frontier:
            break
    return False


def hops_to(A, target, max_hop):
    """回傳 {u: 跳數}，u 在 max_hop 內可到達 target（u 為 target 的上游）。"""
    seen, frontier = {}, [target]
    for h in range(1, max_hop + 1):
        nxt = []
        for v in frontier:
            for u in np.where(A[v])[0]:          # u 是 v 的父 → u → v
                u = int(u)
                if u not in seen and u != target:
                    seen[u] = h
                    nxt.append(u)
        frontier = nxt
        if not frontier:
            break
    return seen


def hops_from(A, src, max_hop):
    """回傳 {v: 跳數}，src 在 max_hop 內可到達 v（v 為 src 的下游）。"""
    seen, frontier = {}, [src]
    for h in range(1, max_hop + 1):
        nxt = []
        for u in frontier:
            for v in np.where(A[:, u])[0]:
                v = int(v)
                if v not in seen and v != src:
                    seen[v] = h
                    nxt.append(v)
        frontier = nxt
        if not frontier:
            break
    return seen


def expand_from(A, base, max_hop, eff, min_eff):
    """由 base 沿圖擴展，回傳 {參數: 統計}。

    只保留自身效果量超過 min_eff 者 —— 否則只是結構上的鄰居，
    與本次異常無關。
    """
    bs = set(base)
    up, down, hop = {}, {}, {}
    for c in base:
        for u, h in hops_to(A, c, max_hop).items():
            if u in bs:
                continue
            up[u] = up.get(u, 0) + 1
            hop[u] = min(hop.get(u, 99), h)
        for v, h in hops_from(A, c, max_hop).items():
            if v in bs:
                continue
            down[v] = down.get(v, 0) + 1
            hop[v] = min(hop.get(v, 99), h)
    out = {}
    for u in set(list(up) + list(down)):
        if min_eff > 0 and eff[u] <= min_eff:
            continue
        out[u] = {'up_links': up.get(u, 0), 'down_links': down.get(u, 0),
                  'links': up.get(u, 0) + down.get(u, 0),
                  'min_hop': hop.get(u, 99)}
    return out


def group_candidates(A, cand, priority, max_hop):
    """連通分量分組。組內：錨點 → 直接關係（依主順序）→ 間接關聯。"""
    rel = {}
    for u in cand:
        for v in cand:
            if u != v:
                rel[(u, v)] = bool(A[v, u]) or reachable(A, u, v, max_hop)

    parent = {c: c for c in cand}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u in cand:
        for v in cand:
            if u != v and (rel.get((u, v)) or rel.get((v, u))):
                ru, rv = find(u), find(v)
                if ru != rv:
                    parent[rv] = ru

    comps = {}
    for c in cand:
        comps.setdefault(find(c), []).append(c)

    TIER = {'上游': 0, '下游': 0, '間接關聯': 1}
    groups = []
    for members in comps.values():
        members = sorted(members, key=lambda x: priority.get(x, 1e9))
        anchor = members[0]
        rest = []
        for m in members[1:]:
            if rel.get((m, anchor)):
                tag = '上游'
            elif rel.get((anchor, m)):
                tag = '下游'
            else:
                tag = '間接關聯'
            rest.append((m, tag))
        rest.sort(key=lambda x: (TIER[x[1]], priority.get(x[0], 1e9)))
        groups.append([(anchor, '錨點')] + rest)
    groups.sort(key=lambda g: priority.get(g[0][0], 1e9))
    return groups, rel


# =====================================================================
# 規則
# =====================================================================

def best_stump(x, y, qs):
    ok = np.isfinite(x)
    if ok.sum() < 10 or len(np.unique(x[ok])) < 2:
        return None
    grid = np.unique(np.nanpercentile(x[ok], np.linspace(2.5, 97.5, qs)))
    pos, neg = (y == 1) & ok, (y == 0) & ok
    npos, nneg = pos.sum(), neg.sum()
    if npos == 0 or nneg == 0:
        return None
    best = None
    for t in grid:
        for d in ('>', '<'):
            hit = (x > t) if d == '>' else (x < t)
            j = (hit & pos).sum() / npos - (hit & neg).sum() / nneg
            if best is None or j > best[2]:
                best = (d, float(t), float(j))
    return best


def apply_rule(X, names, conds):
    m = np.ones(len(X), dtype=bool)
    for nm, d, t in conds:
        col = X[:, names.index(nm)]
        hit = (col > t) if d == '>' else (col < t)
        m &= np.nan_to_num(hit, nan=False)
    return m


def truncate_combo(combo, stump, names, X, y, gm, min_gain):
    """依序加入 combo 中各組選出的參數，增益低於門檻即截斷。
    與序貫選擇用同一套停止規則，使兩張表可直接比較。"""
    conds, cur = [], None
    for nm in combo:
        if nm not in stump:
            continue
        cd = conds + [(nm, stump[nm][0], stump[nm][1])]
        ev = eval_rule(apply_rule(X, names, cd), y)
        gain = ev[gm] if cur is None else ev[gm] - cur[gm]
        if cur is not None and gain < min_gain:
            break
        conds, cur = cd, ev
    return conds


def enumerate_rules(opts, stump, names, Xtr, ytr, Xte, yte, gm, min_gain,
                    enum_trunc, prune=True, max_rows=20000, emit=print,
                    stats=None, max_nodes=2000000, force_multi=2):
    """全組合列舉，以前綴共享的深度優先搜尋取代 itertools.product。

    與原本的「product + 每個組合跑一次 truncate_combo」結果相同，但避免
    三類重複工作：

      1. 每個參數的 stump 門檻固定，其布林遮罩只算一次（原本每個組合、
         每個條件都重算一次 col > t，還附帶 names.index 的線性搜尋）。
      2. 共同前綴只算一次。(a,b,c) 與 (a,b,c2) 的 (a,b) 原本各算一遍。
      3. 增益不足時 truncate_combo 會 break，使 (a,b,*) 的所有後續組合
         都退化成同一條規則 [a]。原本全部算完才被 seen_rule 丟掉；
         這裡直接不展開該子樹。

    prune=True 時另加一道可容許上界剪枝：加條件只會讓遮罩變小，所以
    解釋比例 R 單調不增，而 F1 = 2PR/(P+R) 在 P=1 時最大，故整個子樹的
    F1 上界為 2R/(1+R)。該上界低於目前最佳訓練 F1 時，子樹不可能產生
    更好的規則，直接跳過。最佳解不會被剪掉，但表格中被支配的列會減少
    —— 需要完整表格時設 rules.enum_prune = false。
    """
    idx = {nm: k for k, nm in enumerate(names)}        # 取代 names.index
    pos_tr, neg_tr = ytr == 1, ytr == 0
    npos_tr = max(int(pos_tr.sum()), 1)
    nneg_tr = max(int(neg_tr.sum()), 1)

    # 每個 stump 的遮罩預先算好，之後只做位元 AND。
    # 訓練集再拆成正/負兩段：tp 只需看正例那段（異常片通常很少），
    # 可先算 tp 做剪枝，確定要留才去算 fp。
    pos_te, neg_te = yte == 1, yte == 0
    npos_te, nneg_te = int(pos_te.sum()), int(neg_te.sum())

    mp, mn, ep, en, seg = {}, {}, {}, {}, {}
    for nm, st_ in stump.items():
        if nm not in idx:
            continue
        d, t = st_[0], st_[1]      # stump 存 (方向, 門檻)，best_stump 回傳 3 元組
        col = Xtr[:, idx[nm]]
        h = (col > t) if d == '>' else (col < t)
        h = np.nan_to_num(h, nan=False).astype(bool)
        mp[nm], mn[nm] = h[pos_tr], h[neg_tr]
        cte = Xte[:, idx[nm]]
        hte = (cte > t) if d == '>' else (cte < t)
        hte = np.nan_to_num(hte, nan=False).astype(bool)
        ep[nm], en[nm] = hte[pos_te], hte[neg_te]
        # 規則字串的片段也預先格式化：trunc=False 時葉節點數等於組合數，
        # 每個葉再把所有條件格式化一次，%.4g 的成本會變成主要開銷。
        seg[nm] = '%s %s %.4g' % (nm, d, t)

    def metric(tp, fp):
        """搜尋階段只需要純量，不建 dict（原本每次建 8 個 round 的 dict）。

        必須與 eval_rule 一樣四捨五入到小數 3 位 —— 序貫選擇用的是
        eval_rule 的值，增益落在門檻邊界（如 0.0196 對 0.020）時，
        捨入與否會讓停止判斷相反，兩張表就不再可比。
        """
        rec = tp / npos_tr
        if gm == '解釋比例':
            return round(rec, 3)
        if gm == 'J':
            return round(rec - fp / nneg_tr, 3)
        prec = tp / max(tp + fp, 1)
        return round(2 * prec * rec / max(prec + rec, 1e-12), 3)

    rows, seen = [], set()
    best = [-1.0]          # 目前最佳訓練指標，用 list 以便在內層函式更新
    best_st = [None]       # 對應的狀態，供事後強制延伸用
    n_node = [0]
    n_prune = [0]

    def record(st, tag=''):
        """st = (conds, cur_m, key, ap, an, bp, bn)，全部沿 DFS 增量累積。"""
        conds, cur_m, key, ap, an, bp, bn = st
        if not conds or key in seen:
            return
        seen.add(key)
        if cur_m > best[0]:
            best[0] = cur_m
            best_st[0] = st
        if len(rows) >= max_rows:
            return
        # 四個計數都直接取自累積遮罩，不再重建任何全長遮罩
        tp = int(ap.sum())
        tr = counts_to_eval(tp, npos_tr - tp, int(an.sum()),
                            npos_tr, nneg_tr)
        tpe = int(bp.sum())
        te = counts_to_eval(tpe, npos_te - tpe, int(bn.sum()),
                            npos_te, nneg_te)
        rows.append({'組合': ' + '.join(nm for nm, _, _ in conds),
                     '條件數': '%d%s' % (len(conds), tag),
                     '規則': key,
                     '訓練F1': tr['F1'],
                     '驗證解釋數': te['解釋數'],
                     '驗證未解釋數': te['未解釋數'],
                     '驗證解釋比例': te['解釋比例'],
                     '驗證誤報數': te['誤報數'],
                     '驗證誤報率': te['誤報率'],
                     '驗證精確率': te['精確率'], '驗證F1': te['F1']})

    t0 = [time.time(), 0, False]   # [起始時間, 上次回報的節點數, 已報上限]
    # tail[i] = 第 i 組以後的組合數，用來換算一個子樹代表多少組合
    tail = [1] * (len(opts) + 1)
    for i in range(len(opts) - 1, -1, -1):
        tail[i] = tail[i + 1] * max(len(opts[i]), 1)
    n_bound, n_cut = [0], [0]

    def dfs(gi, st):
        n_node[0] += 1
        # 真正的工作量上限架在這裡。max_rows 只停止輸出列（emit_row 的
        # return），不會停止展開，所以剪枝失效時 DFS 仍可能無止境地走。
        # 而「名目組合數」完全不預測工作量 —— 實測 4300 萬組合只走 3 個
        # 節點 —— 用它當閥會在不必要時整段略過。走訪節點數是直接量。
        if max_nodes and n_node[0] > max_nodes:
            if not t0[2]:
                t0[2] = True
                emit('    ⚠️ 已走訪 %s 個節點，達 enum_max_nodes 上限，'
                     '停止展開。' % format(max_nodes, ','))
                emit('       目前為止的規則仍然有效，但不保證是最佳的。'
                     '要跑完請調高 rules.enum_max_nodes。')
            return
        # 節點數可能很大時給個進度，否則長時間沒有輸出看不出是否還在跑
        if n_node[0] - t0[1] >= 500000:
            t0[1] = n_node[0]
            emit('    ...已走訪 %s 個節點，%d 條規則（%.0f 秒）'
                 % (format(n_node[0], ','), len(seen), time.time() - t0[0]))
        if gi == len(opts):
            if st[3] is not None:
                record(st)
            return
        conds, cur_m, key, ap, an, bp, bn = st
        for part in opts[gi]:
            c2, m2, k2, p2, q2, r2, s2 = st
            broke = False
            for nm in part:
                if nm not in mp:
                    continue
                np2 = mp[nm] if p2 is None else (p2 & mp[nm])
                tp = int(np2.sum())
                nq2 = mn[nm] if q2 is None else (q2 & mn[nm])
                nm_m = metric(tp, int(nq2.sum()))
                if enum_trunc and m2 is not None and nm_m - m2 < min_gain:
                    broke = True
                    break
                c2 = c2 + [(nm, stump[nm][0], stump[nm][1])]
                k2 = seg[nm] if not k2 else (k2 + ' 且 ' + seg[nm])
                m2, p2, q2 = nm_m, np2, nq2
                r2 = ep[nm] if r2 is None else (r2 & ep[nm])
                s2 = en[nm] if s2 is None else (s2 & en[nm])
            nst = (c2, m2, k2, p2, q2, r2, s2)
            if broke:
                # 與 truncate_combo 的 break 等價：該組合截斷於此，
                # 且後續各組不再參與 —— 整個子樹都會得到同一條規則。
                if p2 is not None:
                    record(nst)
                n_cut[0] += tail[gi + 1]
                continue
            if prune and p2 is not None:
                rc_ = int(p2.sum()) / npos_tr
                if 2 * rc_ / (1 + rc_) <= best[0]:
                    n_prune[0] += 1
                    # 這個子樹底下所有組合都被上界證明不可能更好。
                    # 「走訪 N 個節點」看起來像沒試，其實是試了並證明
                    # 不必展開 —— 把被排除的組合數記下來，那是個保證，
                    # 不是略過。
                    n_bound[0] += tail[gi + 1]
                    record(nst)
                    continue
            dfs(gi + 1, nst)

    sys.setrecursionlimit(max(1000, 50 * len(opts) + 100))
    dfs(0, ([], None, '', None, None, None, None))

    # ---- 保底：至少留一組多參數組合供比較 ----
    # 截斷加剪枝之下，整張表可能只剩條件數 1 的規則 —— 那時沒有任何
    # 多參數組合可以對照，看不出「多加一個條件會怎樣」。這裡從最佳的那
    # 條往下延伸，**忽略增益門檻**，強制產生幾筆。
    # 它們不參與最終採用（訓練 F1 本來就比較差，比較時自然落敗），只是
    # 把被截斷掉的那一步的數字補回來，讓截斷這個決定看得見。
    def _gidx(nm_):
        for gi_, grp in enumerate(opts):
            if any(nm_ == x for part in grp for x in part):
                return gi_
        return -1

    if force_multi > 0 and best_st[0] is not None:
        have = max((len(r['組合'].split(' + ')) for r in rows), default=0)
        if have < 2:
            conds, cur_m, key, ap, an, bp, bn = best_st[0]
            used = {_gidx(nm_) for nm_, _, _ in conds}
            for _ in range(force_multi):
                cand = None
                for gi_, grp in enumerate(opts):
                    if gi_ in used:
                        continue
                    for part in grp:
                        for nm_ in part:
                            if nm_ not in mp:
                                continue
                            p2 = mp[nm_] if ap is None else (ap & mp[nm_])
                            q2 = mn[nm_] if an is None else (an & mn[nm_])
                            m2 = metric(int(p2.sum()), int(q2.sum()))
                            if cand is None or m2 > cand[0]:
                                cand = (m2, gi_, nm_, p2, q2)
                if cand is None:
                    break
                m2, gi_, nm_, p2, q2 = cand
                used.add(gi_)
                conds = conds + [(nm_, stump[nm_][0], stump[nm_][1])]
                key = seg[nm_] if not key else (key + ' 且 ' + seg[nm_])
                ap, an = p2, q2
                bp = ep[nm_] if bp is None else (bp & ep[nm_])
                bn = en[nm_] if bn is None else (bn & en[nm_])
                cur_m = m2
                record((conds, cur_m, key, ap, an, bp, bn), tag='*')
            emit('    已補上 %d 筆多參數組合（標 *）：忽略增益門檻往下延伸，'
                 '供對照用。' % force_multi)
    _total = tail[0]
    _acc = n_bound[0] + n_cut[0]
    emit('  列舉：走訪 %d 個節點，產生 %d 條不重複規則%s'
         % (n_node[0], len(seen),
            '，剪枝 %d 個子樹' % n_prune[0] if n_prune[0] else ''))
    # 「走訪 3 個節點」會讓人以為它根本沒嘗試組合。實際上整個空間都被
    # 涵蓋了：一部分展開、一部分由上界證明不可能更好、一部分因增益停止
    # 而截斷（截斷的子樹全部得到同一條規則）。把帳算清楚。
    if _total > 1:
        emit('    搜尋空間 %s 個組合：%s 個由上界 F1<=2R/(1+R) 證明'
             '不可能更好，%s 個因增益不足而截斷。'
             % (format(_total, ','), format(n_bound[0], ','),
                format(n_cut[0], ',')))
        if n_bound[0] and not t0[2]:
            emit('    被排除的不是「沒試」—— 合取只會讓解釋比例下降，'
                 '所以那些分支的 F1 上限已低於目前最佳。')
    if len(rows) < len(seen):
        emit('  （表格上限 %d 列，已達上限；完整規則數 %d）'
             % (max_rows, len(seen)))
    if stats is not None:
        stats.update(nodes=n_node[0], rules=len(seen), pruned=n_prune[0])
    return rows


def counts_to_eval(tp, fn, fp, npos, nneg):
    """由計數直接組出指標表。列舉時已有 tp/fp，不必再走一次遮罩。"""
    npos, nneg = max(npos, 1), max(nneg, 1)
    prec, rec = tp / max(tp + fp, 1), tp / npos
    return {'解釋數': tp, '未解釋數': fn, '解釋比例': round(rec, 3),
            '誤報數': fp, '誤報率': round(fp / nneg, 3),
            '精確率': round(prec, 3),
            'F1': round(2 * prec * rec / max(prec + rec, 1e-12), 3),
            'J': round(rec - fp / nneg, 3)}


def eval_rule(mask, y):
    pos, neg = y == 1, y == 0
    return counts_to_eval(int((mask & pos).sum()),
                          int((~mask & pos).sum()),
                          int((mask & neg).sum()),
                          int(pos.sum()), int(neg.sum()))


# =====================================================================

def load_config(path):
    txt = re.sub(r'^\s*//.*$', '', read_text_any(path), flags=re.M)
    cfg = json.loads(txt)
    for k, v in TEMPLATE.items():                       # 補上缺漏的預設值
        if k not in cfg:
            cfg[k] = v
        elif isinstance(v, dict):
            for kk, vv in v.items():
                cfg[k].setdefault(kk, vv)
    return cfg


def read_candidate_file(path, sensor_col='sensor', priority_col='priority'):
    """讀候選參數清單。

    .csv  取 sensor_col 欄；若有 priority_col 則依其升冪排序（數字小者優先），
          以保留上游模型給的優先序。其餘欄位忽略。
    其他  當純文字，一行一個名稱，空行與 # 之後的內容忽略。
    """
    if path.lower().endswith('.csv'):
        df = read_csv_any(path)
        cols = {c.strip().lower(): c for c in df.columns}
        sc = cols.get(sensor_col.strip().lower())
        if sc is None:
            raise KeyError('候選 csv 缺少欄位 "%s"。實際欄位：%s'
                           % (sensor_col, list(df.columns)))
        pc = cols.get(priority_col.strip().lower())
        if pc is not None:
            df = df.sort_values(pc, kind='mergesort')
        out, seen = [], set()
        for v in df[sc].astype(str):
            v = v.strip()
            if v and v not in seen:
                seen.add(v)
                out.append(v)
        return out
    out = []
    for ln in read_text_any(path).splitlines():
        ln = ln.split('#')[0].strip()
        if ln:
            out.append(ln)
    return out


def main():
    ap = argparse.ArgumentParser(description='半導體根因分析整合流程')
    ap.add_argument('--config', help='設定檔路徑')
    ap.add_argument('--init', help='產生設定檔範本至指定路徑')
    args = ap.parse_args()

    if args.init:
        if os.path.exists(args.init):
            sys.exit('檔案已存在，請換一個路徑：%s' % args.init)
        d = os.path.dirname(os.path.abspath(args.init))
        if d and not os.path.isdir(d):
            os.makedirs(d)
        with open(args.init, 'w', encoding='utf-8') as f:
            json.dump(TEMPLATE, f, ensure_ascii=False, indent=2)
        print('設定檔範本已產生：%s' % args.init)
        print('請編輯 data.csv、data.wafer_col、graph.path 等欄位後執行：')
        print('  python run_pipeline.py --config %s' % args.init)
        return

    if not args.config:
        sys.exit('請指定 --config，或以 --init 產生範本。')
    if not os.path.exists(args.config):
        sys.exit('找不到設定檔：%s' % args.config)
    cfg = load_config(args.config)

    dc, cc = cfg['data'], cfg['candidates']
    gc, rc = cfg['graph'], cfg['rules']
    prefix = cfg['output']['prefix']
    show_w = cfg['output'].get('show_welch', True)
    d = os.path.dirname(os.path.abspath(prefix))
    if d and not os.path.isdir(d):
        os.makedirs(d)

    L = []
    RESULT = {'case': os.path.basename(cfg['data']['csv'])}
    _on = [True]          # brief 模式下，前段（步驟 1-5）不輸出

    # 各步驟耗時：由 emit 偵測步驟標題自動分段，不必在每個步驟插計時碼。
    # brief 模式不輸出前段，但計時照記，否則看不出時間花在哪。
    _T = {'t0': time.time(), 'cur': '載入', 'marks': []}
    _ENUM = {}               # 列舉統計，由 enumerate_rules 填入
    _SHAPE = {}              # 資料規模，供摘要行使用

    def _mark(name):
        now = time.time()
        _T['marks'].append((_T['cur'], now - _T['t0']))
        _T['t0'], _T['cur'] = now, name

    def emit(s=''):
        if isinstance(s, str) and s.startswith('\n[步驟 '):
            _mark(s.strip().split(']')[0].lstrip('[') or s.strip())
        if not _on[0]:
            return
        print(s)
        L.append(s)

    def warn(s=''):
        """一律輸出，brief 模式也不抑制。

        brief 是為了不印步驟 1-5 的細節，但缺值處理的診斷是使用者必須
        據以做決定的東西。被抑制之後，中止時畫面上只剩最後那一行「丟列
        損失超過 max_row_loss」—— 看不到是哪些欄位造成、也看不到三個
        選項，於是訊息等於只說了「不行」而沒說「為什麼」和「怎麼辦」。
        """
        print(s)
        L.append(s)

    if cfg['output'].get('brief', False):
        _on[0] = False

    emit('=' * 70)
    emit('半導體根因分析流程')
    emit('=' * 70)
    emit('設定檔：%s' % args.config)

    # ---- 步驟 1：資料檢查與分組 ----
    if not os.path.exists(dc['csv']):
        sys.exit('找不到資料檔：%s' % dc['csv'])
    raw = read_csv_any(dc['csv'])
    emit('\n[步驟 1] 資料檢查')
    emit('  原始 shape：%s' % (raw.shape,))

    key_cols = [c.strip() for c in dc['wafer_col'].split(',') if c.strip()]
    miss = [c for c in key_cols + [dc['label_col']] if c not in raw.columns]
    if miss:
        sys.exit('找不到欄位 %s。實際欄位：\n%s' % (miss, list(raw.columns)[:40]))

    if len(key_cols) == 1:
        wkey = key_cols[0]
    else:
        wkey = '__WKEY__'
        # concat 而非 raw[wkey] = ...：欄數多時逐欄 insert 會讓 DataFrame
        # 碎片化，pandas 會為此發出 PerformanceWarning
        raw = pd.concat(
            [raw, raw[key_cols].astype(str).agg('_'.join, axis=1)
             .rename(wkey)], axis=1)
    n_uni = raw[wkey].nunique()
    emit('  wafer 鍵：%s → %d 個唯一值（共 %d 列）'
         % (' + '.join(key_cols), n_uni, len(raw)))

    mixed = int((raw.groupby(wkey)[dc['label_col']].nunique() > 1).sum())
    if mixed:
        sys.exit('❌ 有 %d 個 wafer 分組含多種標籤，wafer 鍵不唯一。\n'
                 '   請於 config 的 data.wafer_col 改用複合鍵，'
                 '例如 "LOT_ID,WFR_ID,START_DATETIME"' % mixed)
    emit('  標籤一致性：通過')

    # 鍵太唯一和鍵不唯一一樣危險，但不會被上面的標籤檢查攔到：
    # 每個分組只有一列時每組都只有一個標籤，檢查照樣通過。
    # 一列一片的資料（本專案的 FD 資料）本來就該如此；
    # 但一片多列的原始資料若把 START_DATETIME 放進鍵裡，
    # 每個時間點會變成一片獨立的「wafer」，聚合就失去意義。
    rows_per = raw.groupby(wkey).size()
    if n_uni == len(raw):
        emit('  每個鍵只對應 1 列 → 聚合為恆等運算（一列一片的資料即如此）')
        dtlike = [c for c in key_cols
                  if 'TIME' in c.upper() or 'DATE' in c.upper()]
        if dtlike and len(key_cols) > 1:
            without = raw[[c for c in key_cols if c not in dtlike]] \
                .astype(str).agg('_'.join, axis=1).nunique()
            if without < n_uni:
                emit('    註：移除 %s 後只剩 %d 個唯一值。'
                     % ('、'.join(dtlike), without))
                emit('    若這份資料本來是一片多列（原始 FD 軌跡），'
                     '請把時間欄從 wafer_col 移除，')
                emit('    否則每個時間點會被當成一片獨立的 wafer。'
                     '若是一列一片則無需處理。')
    else:
        emit('  每個鍵對應的列數：最少 %d／中位 %d／最多 %d'
             % (int(rows_per.min()), int(rows_per.median()),
                int(rows_per.max())))

    exclude = set(DEFAULT_META) | set(key_cols) | {wkey, dc['label_col']}
    exclude |= set(dc.get('exclude', []))
    sensor_cols = [c for c in raw.select_dtypes(include=[np.number]).columns
                   if c not in exclude]
    if not sensor_cols:
        sys.exit('沒有可用的參數欄位，請檢查 data.exclude。')

    # ---- NA 前處理 ----
    # 注意：groupby().mean() 會靜默跳過 NaN，含缺值的 wafer 仍會產生數值，
    # 只是那個平均是用較少的點算的。此處改為明確處理。
    na_mode = str(dc.get('dropna', 'row')).lower()
    n_before = len(raw)
    n_na_cell = int(raw[sensor_cols].isna().sum().sum())
    if n_na_cell == 0:
        warn('  NA 檢查：參數欄位無缺值')
    elif na_mode == 'none':
        warn('  NA 檢查：共 %d 個缺值（dropna=none，不處理；'
             '聚合時會自動略過）' % n_na_cell)
    else:
        na_frac = raw[sensor_cols].isna().mean()
        warn('  NA 檢查：共 %d 個缺值，分布於 %d / %d 個參數'
             % (n_na_cell, int((na_frac > 0).sum()), len(sensor_cols)))
        # 把實際生效的值印出來。中止訊息叫人去改這三個設定，但若設定沒
        # 傳到（改了 run_all 產生的 config、或 run_all 沒把鍵轉下來），
        # 改完重跑會停在一模一樣的地方，而畫面上看不出是哪一種。
        _mrk = float(dc.get('min_row_keep', 0.0))
        warn('    生效設定：dropna=%s　drop_col_na_frac=%s　'
             'max_row_loss=%.2f'
             % (na_mode,
                ('自動（min_row_keep=%.2f）' % _mrk) if _mrk > 0
                else '%.2f' % float(dc.get('drop_col_na_frac', 0.5)),
                float(dc.get('max_row_loss', 0.3))))

        # ---- 自動調門檻 ----
        # data.min_row_keep 設了就改成求解：在「列存活率 ≥ 目標」的條件
        # 下保留最多的參數。存活率對門檻單調不增（門檻高 → 留的欄多 →
        # 能存活的列少），所以二分搜尋即可，不必逐一試門檻。
        #
        # 搜尋的候選只取資料中實際出現的缺值比例值。以欄數 k 搜尋會踩到
        # 同分問題（邊界上數個欄位缺值比例相同，全留會讓存活率掉到目標
        # 以下，全丟又浪費），而門檻的語意本來就是「≤ t 的全留」，所以
        # 用門檻值搜尋既無同分問題，寫回設定檔也能重現同樣的結果。
        #
        # 注意這在門檻這個參數化之下是最佳的，但不是全域最佳：同樣的欄
        # 數換一個子集可能留下更多列（取決於缺值位置的重疊）。設定檔只
        # 給得出一個門檻，所以就在門檻的範圍內求最佳。
        # thr 為 None 表示「不丟欄」。不能用 0 表示關閉 —— 自動搜尋
        # 會合法地選出 0.0（只留完全無缺值的參數），若 0 仍當成關閉，
        # 門檻算對了卻不會套用，丟列照樣把資料砍光。
        thr = float(dc.get('drop_col_na_frac', 0.5))
        if thr <= 0:
            thr = None
        target = float(dc.get('min_row_keep', 0.0))
        if target > 0 and n_na_cell:
            if not 0.0 < target <= 1.0:
                sys.exit('data.min_row_keep 必須在 0 與 1 之間，收到 %s。'
                         % target)
            lab_c = dc['label_col']

            def _surv(t):
                cols = [c for c in sensor_cols if na_frac[c] <= t]
                if not cols:
                    return 1.0, []
                s = float(raw[cols + [lab_c]].notna().all(axis=1).mean())
                return s, cols

            levels = sorted(set(float(v) for v in na_frac.values))
            s_all, _c = _surv(levels[-1])
            warn('    自動調門檻：目標保留 %.0f%% 的列' % (100 * target))
            if s_all >= target:
                warn('      全部參數都留著就有 %.1f%% —— 不必丟欄。'
                     % (100 * s_all))
                thr = None
            else:
                ok = [t for t in levels if _surv(t)[0] >= target]
                if not ok:
                    warn('      ❌ 達不到：每個參數都有缺值，連只留最乾淨'
                         '的那一批，存活率也只有 %.1f%%。'
                         % (100 * _surv(levels[0])[0]))
                    warn('      門檻調不出來 —— 用 data.dropna = "none"。')
                    sys.exit('min_row_keep=%.2f 以丟欄的方式達不到，'
                             '請改用 dropna="none"。' % target)
                # ok 是 levels 的前綴（單調性），取最大的那個
                lo, hi = 0, len(levels) - 1
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if _surv(levels[mid])[0] >= target:
                        lo = mid
                    else:
                        hi = mid - 1
                thr = levels[lo]
                s_sel, cols_sel = _surv(thr)
                warn('      門檻 %.4f → 留 %d / %d 參數，列存活 %.1f%%'
                     % (thr, len(cols_sel), len(sensor_cols), 100 * s_sel))
                if lo + 1 < len(levels):
                    s_nx, c_nx = _surv(levels[lo + 1])
                    warn('      （再放寬到 %.4f 會多留 %d 個參數，'
                         '但列存活掉到 %.1f%%）'
                         % (levels[lo + 1], len(c_nx) - len(cols_sel),
                            100 * s_nx))
                if thr > 0:
                    warn('      寫回設定檔：'
                         '"drop_col_na_frac": %.4f' % thr)
                else:
                    warn('      這一檔只有完全無缺值的參數能留。設定檔裡 '
                         'drop_col_na_frac=0 的意思是「不丟欄」，')
                    warn('      所以寫不回去 —— 保留 min_row_keep 這個'
                         '設定即可。')
        if thr is not None:
            bad = [c for c in sensor_cols if na_frac[c] > thr]
            if bad:
                warn('    %d 個參數%s，改丟參數而非丟列：%s'
                     % (len(bad),
                        '有缺值' if thr == 0 else '缺值超過 %.0f%%' % (100 * thr),
                        ', '.join('%s(%.0f%%)' % (c, 100 * na_frac[c])
                                  for c in bad[:6])
                        + ('…' if len(bad) > 6 else '')))
                sensor_cols = [c for c in sensor_cols if c not in bad]
                if not sensor_cols:
                    sys.exit('所有參數的缺值都超過 %.0f%%，無可用欄位。'
                             '請調低 data.drop_col_na_frac 或檢查資料。' % (100 * thr))

        # 列刪除在高維下會崩潰：每欄只缺少數百分比，但「整列都沒有缺值」
        # 的機率是各欄的乘積。299 欄中有 46 欄各缺 6.7%，存活率就只剩
        # (1-0.067)^46 ≈ 4%。drop_col_na_frac 只攔單欄嚴重的，攔不住這種
        # 累積效應，所以要先算存活率再決定。
        keep = raw[sensor_cols + [dc['label_col']]].notna().all(axis=1)
        n_drop = int((~keep).sum())
        surv = 1.0 - n_drop / float(max(n_before, 1))
        max_loss = float(dc.get('max_row_loss', 0.3))
        if n_drop and surv < 1.0 - max_loss:
            na_frac2 = raw[sensor_cols].isna().mean()
            culprits = na_frac2[na_frac2 > 0].sort_values(ascending=False)
            warn('  ❌ 丟列會移除 %d / %d 列（%.1f%%），只剩 %d 列。'
                 % (n_drop, n_before, 100.0 * n_drop / max(n_before, 1),
                    n_before - n_drop))
            # 兩種型態的處置完全不同，所以診斷要分開講，不能一律說成
            # 「許多欄位各缺一點」—— 五欄各缺 35% 也會把列砍光，但那種
            # 情況丟欄就解決了。以缺值最高的那一欄區分。
            n_bad = int((na_frac2 > 0).sum())
            worst = float(culprits.iloc[0]) if len(culprits) else 0.0
            if worst >= 0.15:
                warn('     原因是 %d 個欄位缺值偏高（最高 %.0f%%），'
                     '而丟列的條件是「整列都沒有缺值」。'
                     % (n_bad, 100.0 * worst))
            else:
                warn('     原因不是單一欄位嚴重 —— 最高也只缺 %.0f%% ——'
                     '，而是 %d 個欄位各缺一點，' % (100.0 * worst, n_bad))
            warn('     「整列無缺值」的機率是各欄存活率的乘積。')
            warn('     缺值最多的幾個欄位：')
            for c, v in culprits.head(8).items():
                warn('       %-28s %5.1f%%' % (str(c)[:28], 100.0 * v))
            # 把「丟欄換回幾列」算成實數，而不是讓人去試。缺值分散在許多
            # 欄位時，丟欄其實救不了：門檻必須低到幾乎丟掉全部欄位，列才
            # 回得來。這張表會直接顯示那件事，省掉逐一試門檻的來回。
            warn('')
            warn('     若改以丟欄為主，各門檻實際換回多少列：')
            lab_c = dc['label_col']
            shown, best = set(), (0.0, None)
            for t in (0.30, 0.20, 0.10, 0.05, 0.02, 0.01, 0.0):
                cols = [c for c in sensor_cols if na_frac2[c] <= t]
                if len(cols) in shown:
                    continue
                shown.add(len(cols))
                if not cols:
                    warn('       %.2f → 留 0 個參數（門檻以下沒有欄位）'
                         % t)
                    continue
                k = int(raw[cols + [lab_c]].notna().all(axis=1).sum())
                warn('       %.2f → 留 %d / %d 參數，剩 %d / %d 列（%.0f%%）'
                     % (t, len(cols), len(sensor_cols), k, n_before,
                        100.0 * k / max(n_before, 1)))
                r = k / float(max(n_before, 1))
                if r > best[0]:
                    best = (r, (t, len(cols)))
            if best[1] is None or best[0] < 1.0 - max_loss:
                warn('     → 丟欄救不了這份資料：缺值分散在太多欄位，'
                     '門檻再低也換不回足夠的列。')
                warn('       選項 1 是死路，用選項 2（dropna="none"）。')
            else:
                warn('     → 門檻 %.2f 可行：留 %d 個參數、%.0f%% 的列。'
                     % (best[1][0], best[1][1], 100.0 * best[0]))
            warn('')
            warn('     三個選項：')
            warn('       1. 調低 data.drop_col_na_frac（目前 %.2f），'
                 '改以丟欄為主' % thr)
            warn('       2. data.dropna = "none"，保留缺值'
                 '（聚合時自動略過，但各 wafer 的平均用的點數不同）')
            warn('       3. 調高 data.max_row_loss（目前 %.2f），'
                 '明確接受這個損失' % max_loss)
            sys.exit('丟列損失超過 max_row_loss，已中止以免在殘餘資料上'
                     '產生看似正常但無意義的結果。')
        if n_drop:
            # 丟列可能不均勻地砍掉某一類，須回報前後的類別組成
            lab_b = raw[dc['label_col']].astype(str).value_counts()
            raw = raw[keep].reset_index(drop=True)
            lab_a = raw[dc['label_col']].astype(str).value_counts()
            warn('    丟棄含缺值的列：%d / %d（%.1f%%），剩餘 %d 列'
                 % (n_drop, n_before, 100.0 * n_drop / max(n_before, 1),
                    len(raw)))
            for k in lab_b.index:
                b, a = int(lab_b[k]), int(lab_a.get(k, 0))
                warn('      %-12s %d → %d（丟棄 %.1f%%）'
                     % (k, b, a, 100.0 * (b - a) / max(b, 1)))
            av = str(dc['abnormal_value'])
            r_ab = (lab_b.get(av, 0) - lab_a.get(av, 0)) / max(lab_b.get(av, 1), 1)
            r_all = n_drop / max(n_before, 1)
            if r_ab > max(2 * r_all, r_all + 0.1):
                warn('      ⚠️ 異常批被丟棄的比例明顯高於整體 —— '
                     '缺值可能與異常相關，丟列會系統性地移除訊號。')
            if len(raw) == 0:
                sys.exit('丟棄缺值後沒有資料剩下。請改用 dropna=none，'
                         '或調整 data.drop_col_na_frac。')

    g = raw.groupby(wkey, sort=False)
    feat = g[sensor_cols].mean()
    lab = g[dc['label_col']].first().reindex(feat.index)
    names = list(feat.columns)
    X = feat.values.astype(float)
    y = (lab.astype(str) == dc['abnormal_value']).values.astype(int)
    emit('  wafer %d（正常 %d／異常 %d），參數 %d'
         % (len(X), int((y == 0).sum()), int(y.sum()), len(names)))
    _SHAPE.update(n=int((y == 0).sum()), a=int(y.sum()), p=len(names))
    if y.sum() == 0:
        sys.exit('沒有標為 %s 的 wafer。實際標籤：\n%s'
                 % (dc['abnormal_value'], lab.value_counts().to_string()))

    RESULT['data'] = {'n_wafer': int(len(X)), 'n_normal': int((y == 0).sum()),
                      'n_abnormal': int(y.sum()), 'n_param': len(names)}
    Xn, Xa = X[y == 0], X[y == 1]

    # ---- 步驟 2：Welch 排序 ----
    emit('\n[步驟 2] Welch t 檢定')
    t = welch_t(Xn, Xa)
    eff = np.array([abs(cohen_d(Xn[:, j], Xa[:, j])) for j in range(len(names))])
    t_rank = ranks_from(t)
    v_rank = ranks_from(var_ratio(Xn, Xa))
    emit('  完成，%d 個參數已排序' % len(names))

    # ---- 步驟 3：bootstrap ----
    emit('\n[步驟 3] bootstrap 穩定度（%d 次）' % cc['n_boot'])
    rng = np.random.RandomState(cfg['seed'])
    no, ab = np.where(y == 0)[0], np.where(y == 1)[0]
    boot = np.zeros((cc['n_boot'], len(names)), dtype=int)
    for b in range(cc['n_boot']):
        bn = rng.choice(no, size=len(no), replace=True)
        ba = rng.choice(ab, size=len(ab), replace=True)
        boot[b] = ranks_from(welch_t(X[bn], X[ba]))
    p95 = np.percentile(boot, 95, axis=0).astype(int)
    emit('  完成')

    # ---- 門檻校準 ----
    thr_fw = thr_pw = None
    n_perm = cc.get('n_perm', 0)
    if n_perm and n_perm > 0:
        emit('\n[步驟 4] 效果量門檻校準（標籤置換 %d 次）' % n_perm)
        thr_fw, thr_pw, pooled = calibrate_threshold(X, y, n_perm,
                                                     cfg['seed'])
        emit('  全域門檻   |d| > %.3f（取全參數最大值的 95%% 分位，'
             '含多重比較校正）' % thr_fw)
        emit('  單參數門檻 |d| > %.3f（混合分布的 95%% 分位，未校正）'
             % thr_pw)
        emit('  實際資料：超過全域門檻 %d 個、超過單參數門檻 %d 個／共 %d 個'
             % (int((eff > thr_fw).sum()), int((eff > thr_pw).sum()),
                len(names)))
        RESULT['thresholds'] = {'global': thr_fw, 'per_param': thr_pw}
    else:
        emit('\n[步驟 4] 未啟用門檻校準（candidates.n_perm = 0）')

    # ---- 候選集組裝：指定 ∪ 統計 ∪ 擴展 ----
    emit('\n[步驟 5] 候選集組裝')
    source = {}          # 參數 index → 來源標籤
    order = [int(c) for c in np.argsort(-eff, kind='mergesort')]

    cand_file = cc.get('file', '').strip()
    base = []
    given_pos = {}        # 指定候選在檔案中的位置，用來保持使用者給的順序
    if cand_file:
        if not os.path.exists(cand_file):
            sys.exit('找不到候選參數檔：%s' % cand_file)
        try:
            req = read_candidate_file(cand_file,
                                      cc.get('sensor_col', 'sensor'),
                                      cc.get('priority_col', 'priority'))
        except KeyError as e:
            sys.exit(str(e))
        if cand_file.lower().endswith('.csv'):
            # 保留上游模型給的原始欄位，報告直接呈現，不另加工
            _cdf = read_csv_any(cand_file)
            _cols = {c.strip().lower(): c for c in _cdf.columns}
            _sc = _cols.get(cc.get('sensor_col', 'sensor').strip().lower())
            _pc = _cols.get(cc.get('priority_col', 'priority').strip().lower())
            if _pc is not None:
                _cdf = _cdf.sort_values(_pc, kind='mergesort')
            RESULT['candidate_table'] = {
                'columns': list(_cdf.columns),
                'sensor_col': _sc,
                'rows': _cdf.where(pd.notnull(_cdf), None)
                            .to_dict(orient='records')}
        unknown = [c for c in req if c not in names]
        if unknown:
            emit('  ⚠️ 候選檔中這些名稱不在參數欄位裡，已略過：%s' % unknown)
        base = [names.index(c) for c in req if c in names]
        given_pos = {c: k for k, c in enumerate(base)}
        emit('  指定候選：%s，%d 個（不套用 min_effect，順序依檔案）'
             % (cand_file, len(base)))
    else:
        me = cc.get('min_effect', 0.5)
        if isinstance(me, str) and me.lower() == 'auto':
            me = thr_fw if thr_fw is not None else 0.0
            emit('  min_effect = auto → 採用全域門檻 %.3f' % me)
        pool = [c for c in order if eff[c] >= me]
        base = pool[:cc['topk']]
        emit('  自動選取：|d| >= %.3f 者 %d 個，取前 %d'
             % (me, len(pool), len(base)))
    for c in base:
        source[c] = '指定' if cand_file else '統計(top-k)'

    # 統計方法找到的異常：超過全域門檻者一律納入
    if cc.get('add_statistical', True) and thr_fw is not None:
        add = [c for c in order if eff[c] > thr_fw and c not in source]
        for c in add:
            source[c] = '統計(門檻)'
        emit('  統計補入：超過全域門檻%s者 %d 個'
             % (' |d| > %.3f ' % thr_fw if show_w else '', len(add)))
    elif cc.get('add_statistical', True):
        emit('  統計補入：略過（未啟用門檻校準）')

    if not source:
        sys.exit('沒有有效候選。')

    # 擴展：由目前候選沿圖向外找結構上相連、且自身有偏移的參數
    exp_stat = {}
    ec = cc.get('expansion', {}) or {}
    gpath = gc.get('path', '').strip()
    W = None
    if gpath and os.path.exists(gpath):
        W = np.load(gpath)
        # 有節點名稱側車檔時按名稱對齊，而不是靠矩陣大小。
        # 訓練常在縮減後的參數集上進行（參數數隨 N 平方影響記憶體），
        # 分析則保留完整參數集，兩邊的欄位集合本來就不同。
        # 只比大小的話，大小相同而順序不同會造成無聲的索引錯位。
        npath = os.path.splitext(gpath)[0] + '_nodes.txt'
        if os.path.exists(npath):
            gnames = [ln.strip() for ln in
                      read_text_any(npath).splitlines() if ln.strip()]
            if len(gnames) != W.shape[0]:
                emit('  ⚠️ 節點名稱檔 %d 筆與圖維度 %d 不符，圖已停用。'
                     % (len(gnames), W.shape[0]))
                W = None
            else:
                pos = {nm: k for k, nm in enumerate(names)}
                hit = [(k, pos[nm]) for k, nm in enumerate(gnames)
                       if nm in pos]
                miss = [nm for nm in gnames if nm not in pos]
                extra = [nm for nm in names if nm not in set(gnames)]
                if not hit:
                    emit('  ⚠️ 圖的節點名稱與分析的參數完全不重疊，'
                         '圖已停用。')
                    emit('     圖的前幾個節點：%s'
                         % ', '.join(map(str, gnames[:5])))
                    W = None
                else:
                    Wf = np.zeros((len(names), len(names)), dtype=float)
                    gi = np.array([a for a, _ in hit])
                    ai = np.array([b for _, b in hit])
                    Wf[np.ix_(ai, ai)] = W[np.ix_(gi, gi)]
                    emit('  圖按名稱對齊：%d / %d 個節點對上'
                         % (len(hit), len(gnames)))
                    if miss:
                        emit('    圖有但分析沒有的參數 %d 個（已忽略）：%s%s'
                             % (len(miss), ', '.join(map(str, miss[:5])),
                                ' …' if len(miss) > 5 else ''))
                    if extra:
                        emit('    分析有但圖沒有的參數 %d 個'
                             '（視為孤立節點）：%s%s'
                             % (len(extra),
                                ', '.join(map(str, extra[:5])),
                                ' …' if len(extra) > 5 else ''))
                    W = Wf
        elif W.shape[0] != len(names):
            emit('  ⚠️ 圖維度 %d 與參數數 %d 不符，且無節點名稱檔，'
                 '擴展與分組皆略過。' % (W.shape[0], len(names)))
            emit('     訓練時若用了縮減後的參數，請確認已產生 '
                 '%s' % os.path.basename(npath))
            W = None
    # ---- 邊的門檻 ----
    # 門檻該設多少取決於 W 的值是什麼意思，而這件事寫在步驟 1 產生的
    # _meta.json 裡。stability 模式下值是「邊的出現頻率」，門檻就是
    # stability selection 的 π（文獻常取 0.6~0.8）；average 模式下值是
    # 正規化權重，沒有同等的理論依據，所以不自己亂給，維持 0 並說明。
    # ---- edge_k：小圖用 1，大圖自動升到 2 ----
    # k=1 每個節點只留一條入邊，圖是一片森林，結論保守、報告也短。參數
    # 一多就不夠用：節點 N 個、邊也只有 N 條，候選之間連得上的機率隨 N
    # 下降，分組退化成一堆單人組 —— 那不是「沒有結構」而是預算不夠。
    #
    # large_nodes 的值是量出來的，不是猜的。隨機圖、8 個候選、max_hop=3，
    # 各 150 次，候選落單的比例：
    #
    #     節點數      k=1     k=2
    #        20       14%      0%
    #        50       47%      4%
    #       100       66%     17%
    #       200       82%     38%
    #       500       91%     69%
    #      1000       96%     83%
    #
    # k=1 在 N=50 就有近半候選落單，所以門檻設在 50。注意 k=2 也只是把
    # 問題往後推（N=500 時 69%）—— 真正的指標是下面印出的落單比例，不是
    # 節點數。edge_k_large 設成 null 可關掉自動切換。
    #
    # 代價照實說：以模擬資料量測，k=1 的還原力低於 k=2（union F1
    # task2a .205 vs .352、task1 .295 vs .407）。欄序隨意性在兩個 k 都
    # 已經是 1.00，不再是選 k 的考量。
    if W is not None:
        _klarge = gc.get('edge_k_large', None)
        _nlarge = int(gc.get('large_nodes', 50) or 0)
        _n = W.shape[0]
        if _klarge and _nlarge and _n > _nlarge and gc['edge_k'] < _klarge:
            warn('  edge_k：%d → %d（節點 %d 個，超過 large_nodes=%d）'
                 % (gc['edge_k'], int(_klarge), _n, _nlarge))
            gc = dict(gc, edge_k=int(_klarge))
        else:
            warn('  edge_k：%d（節點 %d 個）' % (gc['edge_k'], _n))

    min_edge = gc.get('min_edge', None)
    gmeta = {}
    # n_seed=1 時 W 只有 0 與 1 兩種值，每一列的非零全部同分 —— 同分全收
    # 的規則於是把 stability_edge_k 條邊照單全收，edge_k 完全不起作用。
    # 這不是錯誤而是沒有資訊可以分辨：一次訓練不足以說哪條邊比較可信。
    # 但它會讓人以為調 edge_k 有效而其實沒有，所以要說出來。
    if W is not None and len(np.unique(np.abs(W))) <= 2:
        warn('  註：圖的值只有兩種（n_seed=1），每列的邊全部同分，'
             'edge_k 不起作用。')
        warn('     每個節點會拿到 mtngc.stability_edge_k 條入邊。'
             '要讓 edge_k 有意義請提高 mtngc.n_seed。')
    if W is not None:
        mpath = os.path.splitext(gpath)[0] + '_meta.json'
        if os.path.exists(mpath):
            try:
                gmeta = json.loads(read_text_any(mpath))
            except Exception as e:
                warn('  （圖的描述檔讀不到：%s）' % e)
    if min_edge is None:
        if gmeta.get('value') == 'frequency':
            min_edge = 0.6
            warn('  邊門檻：%.2f（圖的值是 %d 次訓練的出現頻率，'
                 '採 stability selection 的慣用門檻）'
                 % (min_edge, gmeta.get('n_seed', 0)))
        else:
            min_edge = 0.0
            if W is not None:
                warn('  邊門檻：0（圖的值不是出現頻率，無從設定頻率門檻）')
                warn('     mtngc.mode 設成 "stability" 後，每列會只留下'
                     '穩定出現的邊，結論才不會隨 edge_k 改變。')
    else:
        min_edge = float(min_edge)
        warn('  邊門檻：%.2f（analysis.graph.min_edge 指定）' % min_edge)

    if ec.get('enabled', True) and W is not None:
        A_e, Wa_e = binarize(W, gc['edge_k'], gc['min_concentration'],
                             min_edge)
        A_e = break_cycles(A_e, Wa_e)
        me_exp = ec.get('min_effect', None)
        if me_exp is None:
            me_exp = thr_pw if thr_pw is not None else 0.0
        exp_stat = expand_from(A_e, list(source), ec.get('max_hop', 2),
                               eff, me_exp)
        exp_stat = {u: v for u, v in exp_stat.items() if u not in source}
        for u in exp_stat:
            source[u] = '擴展'
        emit('  圖擴展：max_hop=%d%s，新增 %d 個'
             % (ec.get('max_hop', 2),
                '，自身 |d| > %.3f' % me_exp if show_w else '',
                len(exp_stat)))
    elif ec.get('enabled', True):
        emit('  圖擴展：略過（無可用的因果圖）')

    # 候選集的主順序：來源層級優先，同層級內依 Welch 名次。
    # 指定候選排在最前，確保它們優先成為分組的錨點與規則的起始條件；
    # 統計補入次之，圖擴展最後（後者自身的邊際訊號最弱，證據較間接）。
    TIER = {'指定': 0, '統計(top-k)': 0, '統計(門檻)': 1, '擴展': 2}
    cand = sorted(source,
                  key=lambda c: (TIER.get(source[c], 9),
                                 given_pos.get(c, t_rank[c])))
    rows_c = []
    for k, c in enumerate(cand, 1):
        row = {'順序': k, '參數': names[c], '來源': source[c]}
        if show_w:
            row.update({'|d|': round(float(eff[c]), 3),
                        'Welch名次': t_rank[c], '最壞名次': p95[c]})
        row.update({'上游連結': exp_stat.get(c, {}).get('up_links', ''),
                    '下游連結': exp_stat.get(c, {}).get('down_links', ''),
                    '最近距離': exp_stat.get(c, {}).get('min_hop', '')})
        rows_c.append(row)
    emit('')
    emit(fmt_table(pd.DataFrame(rows_c), left_cols=('參數', '來源')))
    emit('  合計 %d 個候選（順序：指定 → 統計補入 → 圖擴展；'
         '指定者依檔案順序，其餘依 Welch 名次）' % len(cand))

    roots = [r for r in cc.get('true_roots', []) if r in names]
    if roots:
        if show_w:
            worst = max(p95[names.index(r)] for r in roots)
            emit('  真根因最壞名次：%d（%s）'
                 % (worst, ', '.join('%s=%d' % (r, p95[names.index(r)])
                                     for r in roots)))
        miss = [r for r in roots if names.index(r) not in source]
        if miss:
            emit('  ⚠️ 這些真根因未進入候選集：%s' % ', '.join(miss))
        else:
            pos = {names[c]: k + 1 for k, c in enumerate(cand)}
            emit('  真根因在候選集中的位置：%s'
                 % ', '.join('%s=%d(%s)' % (r, pos[r],
                                            source[names.index(r)])
                             for r in roots))

    res = pd.DataFrame({
        '參數': names, 'welch_t': np.round(t, 3),
        '效果量d': np.round(eff, 3), '名次': t_rank,
        'rank_p05': np.percentile(boot, 5, axis=0).astype(int),
        'rank_median': np.median(boot, axis=0).astype(int),
        'rank_p95': p95, '變異數比名次': v_rank,
        '是候選': [i in source for i in range(len(names))],
        '候選來源': [source.get(i, '') for i in range(len(names))],
    }).sort_values('名次')
    res.to_csv('%s_ranking.csv' % prefix, index=False, encoding='utf-8-sig')
    RESULT['true_roots'] = roots
    RESULT['candidates'] = [
        {'order': k + 1, 'name': names[c], 'source': source[c],
         'd': float(eff[c]), 'welch_rank': int(t_rank[c]),
         'rank_p95': int(p95[c]),
         'up_links': exp_stat.get(c, {}).get('up_links', None),
         'down_links': exp_stat.get(c, {}).get('down_links', None),
         'min_hop': exp_stat.get(c, {}).get('min_hop', None),
         'is_root': names[c] in roots}
        for k, c in enumerate(cand)]
    RESULT['candidate_names'] = [names[c] for c in cand]

    # ---- 步驟 6：分組 ----
    _on[0] = True          # 後段一律輸出（brief 模式只抑制前段）
    groups_named = [[names[c]] for c in cand]      # 無圖時每個自成一組
    if cfg['output'].get('brief', False):
        emit('=' * 70)
        emit('根因分析結果  |  %s' % os.path.basename(dc['csv']))
        emit('=' * 70)
        emit('候選 %d 個：%s' % (len(cand), ', '.join(names[c] for c in cand)))
    if True:
        if W is None:
            emit('\n[步驟 6] 無可用的因果圖，略過分組（每個候選自成一組）。')
        else:
            bstat = {}
            A, Wa = binarize(W, gc['edge_k'], gc['min_concentration'],
                             min_edge, bstat)
            # 報「欄位順序決定了多少」而不是「k 改了差多少」。後者在無門檻
            # 的 top-k 之下是恆等式（k=1 必為 k=2 的子集 → Jaccard 恆為
            # k1/k2），純雜訊也會讀到 0.5，不帶資訊。
            rob = order_robustness(W, gc['edge_k'],
                                   gc['min_concentration'], min_edge)
            if bstat.get('n_cut'):
                emit('  邊門檻擋掉 %d 條邊；%d 個節點因此無入邊（視為源頭）'
                     % (bstat['n_cut'], bstat['n_root']))
            if bstat.get('n_tie'):
                emit('  %d 列在第 %d／%d 名同分，同分的一組全收'
                     '（否則取誰取決於欄位順序）'
                     % (bstat['n_tie'], gc['edge_k'], gc['edge_k'] + 1))
            emit('  鄰接矩陣 %d 條邊（edge_k=%d）；'
                 '欄序穩健度 %.2f' % (int(A.sum()), gc['edge_k'], rob))
            if rob < 0.99:
                emit('    ⚠️ 有 %.0f%% 的邊是欄位順序決定的 —— '
                     '同分的邊比 edge_k 容得下的多。' % (100 * (1 - rob)))
                emit('       提高 edge_k 或 mtngc.n_seed（降低同分）。'
                     '降低 edge_k 會更糟，不要往那個方向調。')
            A = break_cycles(A, Wa)
            pri = {c: k for k, c in enumerate(cand)}
            groups, _rel = group_candidates(A, cand, pri, gc['max_hop'])
            # 落單比例才是「圖的邊夠不夠」的直接指標，節點數只是它的代理。
            # 候選各自成組時，報告看起來像「這些參數彼此無關」，但更常見
            # 的原因是每個節點只有 edge_k 條入邊、連不到彼此。兩者在報告
            # 上長得一模一樣，所以要把數字說出來。
            _alone = sum(1 for g in groups if len(g) == 1)
            if cand:
                _fr = 100.0 * _alone / len(cand)
                emit('  分組：%d 個候選 → %d 組，其中 %d 組只有 1 個'
                     '（%.0f%%）' % (len(cand), len(groups), _alone, _fr))
                if _fr >= 60:
                    emit('    ⚠️ 多數候選落單。這可能是真的沒有關聯，也'
                         '可能是圖的邊不夠 —— 節點 %d 個、每個節點只留'
                         ' %d 條入邊。' % (len(names), gc['edge_k']))
                    emit('       把 analysis.graph.edge_k 調高一級再比較'
                         '分組結果；若落單比例大幅下降，先前那版就是邊'
                         '不夠而不是無關聯。')
            conc = np.array([np.abs(W[i]).max() / (np.abs(W[i]).sum() + 1e-12)
                             for i in range(len(names))])
            emit('\n[步驟 6] 分組（edge_k=%d, min_concentration=%.3f）'
                 % (gc['edge_k'], gc['min_concentration']))
            emit('  邊數 %d；無父節點 %d/%d；集中度中位數 %.3f（均勻基準 %.3f）'
                 % (A.sum(), int((A.sum(axis=1) == 0).sum()), len(names),
                    np.median(conc), 1.0 / max(len(names) - 1, 1)))
            groups_named = []
            flat = []
            for gi, grp in enumerate(groups, 1):
                anchor = grp[0][0]
                emit('  第 %d 組｜錨點 %s%s'
                     % (gi, names[anchor],
                        '（順序 %d）' % (pri[anchor] + 1) if show_w else ''))
                flat.append(anchor)
                for m, tag in grp[1:]:
                    emit('        └ %s %s%s'
                         % (tag, names[m],
                            '（順序 %d）' % (pri[m] + 1) if show_w else ''))
                    flat.append(m)
                if len(grp) == 1:
                    emit('        （候選集內無相關參數）')
                groups_named.append([names[m] for m, _ in grp])
            emit('  展開後排查順序：%s' % ' → '.join(names[c] for c in flat))
            RESULT['groups'] = [
                {'index': gi,
                 'members': [{'name': names[m], 'tag': tag,
                              'order': pri[m] + 1,
                              'is_root': names[m] in roots}
                             for m, tag in grp]}
                for gi, grp in enumerate(groups, 1)]
            RESULT['order'] = [names[c] for c in flat]
            RESULT['graph_edges'] = [
                {'source': names[int(j)], 'item': names[int(i)]}
                for i in cand for j in np.where(A[i])[0]
                if int(j) in set(cand)]
            if roots and show_w:
                pos = [k + 1 for k, c in enumerate(flat) if names[c] in roots]
                if len(pos) == len(roots):
                    emit('  真根因分組後最壞名次：%d（Welch 主順序為 %d）'
                         % (max(pos),
                            max(cand.index(names.index(r)) + 1 for r in roots
                                if names.index(r) in cand)))
            if len(groups) == len(cand):
                emit('  ※ 每個候選自成一組：候選之間在圖上互不相關。')

    # ---- 步驟 7：規則萃取 ----
    if rc.get('enabled', True):
        # 組別順序沿用步驟 6 的分組順序，使規則萃取的「組 N」與分組結構圖
        # 的「第 N 組」一致。組內各參數的優先度相同（逐一試過取最佳），
        # 不因來源或 Welch 名次而有先後。
        n_combo = int(np.prod([len(g) for g in groups_named]))
        _SHAPE['g'] = len(groups_named)
        emit('\n[步驟 7] 合取規則萃取（組合數 %d）' % n_combo)
        emit('  組別順序沿用分組結果：%s'
             % ' → '.join('第%d組[%s]' % (k, ','.join(g))
                          for k, g in enumerate(groups_named, 1)))
        # 這裡原本是 `if n_combo > max_combo: 略過`，而那個 else 底下裝著
        # 整個步驟 7 —— 包括逐組序貫選擇。序貫是貪心的，成本是
        # 組數 × 組員數，跟各組大小的乘積毫無關係，卻一起被擋掉，於是
        # 組數一多就連一條規則都產不出來。
        #
        # 而且乘積也不再預測列舉的成本：全組合列舉改成前綴共享 DFS 加上
        # 可採上界 F1 <= 2R/(1+R) 的剪枝之後，實測 16 組各 3 個
        # （名目 4300 萬組合）只走 3 個節點、0.00 秒；刻意構造剪枝最不利
        # 的資料（異常佔 80%、每個參數幾乎全涵蓋）也一樣。真正的成本上限
        # 是 enum_max_rows。
        #
        # 所以上限移到方案 B 之前，只擋列舉那一段。
        if True:
            r2 = np.random.RandomState(cfg['seed'])
            itr, ite = [], []
            for lb in (0, 1):
                ids = np.where(y == lb)[0]
                r2.shuffle(ids)
                cut = int(len(ids) * (1 - rc['test_frac']))
                itr.extend(ids[:cut])
                ite.extend(ids[cut:])
            itr, ite = np.array(sorted(itr)), np.array(sorted(ite))
            Xtr, ytr, Xte, yte = X[itr], y[itr], X[ite], y[ite]
            emit('  切分：訓練 %d（異常 %d）／驗證 %d（異常 %d）'
                 % (len(itr), int(ytr.sum()), len(ite), int(yte.sum())))

            stump = {}
            n_cand_nm = 0
            for grp in groups_named:
                for nm in grp:
                    n_cand_nm += 1
                    b = best_stump(Xtr[:, names.index(nm)], ytr,
                                   rc['n_quantile'])
                    if b:
                        stump[nm] = (b[0], b[1])

            # 沒有任何可用的切點時必須停在這裡。先前會繼續走完，印出
            # 「採用 0 個條件：當  時判定為異常」並算出一個 F1 ——
            # 那條規則沒有任何條件，等於把全部 wafer 判為異常。
            if not stump:
                emit('')
                emit('  ❌ %d 個候選參數都無法產生切點，規則萃取中止。'
                     % n_cand_nm)
                emit('     best_stump 需要訓練集至少 10 個有效值、且該參數'
                     '至少有 2 個不同值、')
                emit('     正負兩類都存在。目前訓練集 %d 片（異常 %d）。'
                     % (len(itr), int(ytr.sum())))
                emit('     wafer 數過少時請檢查步驟 1 的 NA 處理是否'
                     '砍掉了大量資料。')
                RESULT.setdefault('rule', {})['aborted'] = \
                    'no usable stump (train=%d, abnormal=%d)' % (
                        len(itr), int(ytr.sum()))
            else:

                # --- 逐組序貫選擇（增益不足即停止） ---
                gm = rc.get('gain_metric', 'F1')
                min_gain = rc.get('min_gain', 0.02)
                # 同組組合搜尋：實測增益有限，預設關閉；程式碼保留供日後評估
                do_ext = rc.get('extend', False)
                emit('  逐組加入條件（下表數字皆為訓練集；'
                     '以 %s 判斷增益，門檻 %.3f）%s'
                     % (gm, min_gain,
                        '；條件數 <= %d 時於同組內枚舉組合'
                        % rc.get('extend_when', 1) if do_ext else ''))
                emit('  註：合取條件只會使解釋比例下降、精確率上升，'
                     '故不以解釋比例當增益指標。')
                g_of = {nm: gi for gi, grp in enumerate(groups_named, 1)
                        for nm in grp}
                # 每一步在**所有**尚未使用的組裡挑最好的，而不是依組序取
                # 下一組。原本的寫法有兩個問題疊在一起：依固定組序前進，
                # 且第一個增益不足的組就 break —— 於是後面的組從來沒被
                # 看過。實測（真因放在第 1 組與第 7 組、中間都是雜訊）：
                # 依組序在第 2 組停住，F1 0.547、只找到一個真因；
                # 改成最佳優先後 F1 0.959，兩個真因都找到。
                # 現在的「停止」才是真的停止條件：所有剩下的組裡，沒有任何
                # 參數能帶來足夠增益。
                conds, path, cur, used = [], [], None, set()
                while len(used) < len(groups_named):
                    trial = []
                    for gi, grp in enumerate(groups_named, 1):
                        if gi in used:
                            continue
                        for nm in grp:
                            if nm not in stump:
                                continue
                            cd = conds + [(nm, stump[nm][0], stump[nm][1])]
                            trial.append((nm, gi,
                                          eval_rule(apply_rule(Xtr, names, cd),
                                                    ytr), cd))
                    if not trial:
                        break
                    nm, gi, ev, cd = max(trial, key=lambda x: x[2][gm])
                    gain = ev[gm] if cur is None else ev[gm] - cur[gm]
                    ok = cur is None or gain >= min_gain
                    path.append({'階段': '序貫', '步驟': len(path) + 1, '組': gi,
                                 '選用參數': nm,
                                 '條件': '%s %.4g' % (stump[nm][0], stump[nm][1]),
                                 '訓練解釋比例': ev['解釋比例'],
                                 '訓練誤報率': ev['誤報率'],
                                 '訓練' + gm: ev[gm], '增益': round(gain, 3),
                                 '採納': '是' if ok
                                 else '否（已無組能帶來足夠增益，停止）'})
                    if not ok:
                        break
                    conds, cur = cd, ev
                    used.add(gi)

                # 同組組合搜尋：序貫只採納少數條件時，於該組內枚舉組合。
                # 用於分辨真根因是集中在同一組，還是分散在不同組。
                ext_when = rc.get('extend_when', 1)
                ext_max = rc.get('extend_max_add', 2)
                if do_ext and cur is not None and len(conds) > ext_when:
                    emit('  （同組組合搜尋未執行：序貫已採納 %d 個條件，'
                         '超過 extend_when=%d）' % (len(conds), ext_when))
                if do_ext and cur is not None and len(conds) <= ext_when:
                    base_nm = conds[-1][0]
                    gi = g_of[base_nm]
                    others = [nm for nm in groups_named[gi - 1]
                              if nm in stump and nm not in {c[0] for c in conds}]
                    if not others:
                        emit('  （同組組合搜尋未執行：條件 %s 所屬的第 %d 組 [%s] '
                             '沒有其他可用參數）'
                             % (base_nm, gi, ','.join(groups_named[gi - 1])))
                    else:
                        rows_e = []
                        for r in range(1, min(ext_max, len(others)) + 1):
                            for add in itertools.combinations(others, r):
                                cd = conds + [(nm, stump[nm][0], stump[nm][1])
                                              for nm in add]
                                ev = eval_rule(apply_rule(Xtr, names, cd), ytr)
                                evt = eval_rule(apply_rule(Xte, names, cd), yte)
                                rows_e.append({'加入參數': ' + '.join(add),
                                               '條件數': len(cd),
                                               '訓練解釋比例': ev['解釋比例'],
                                               '訓練誤報率': ev['誤報率'],
                                               '訓練' + gm: ev[gm],
                                               '增益': round(ev[gm] - cur[gm], 3),
                                               '驗證' + gm: evt[gm],
                                               '_cd': cd, '_ev': ev})
                        et = pd.DataFrame(rows_e).sort_values('增益',
                                                              ascending=False)
                        emit('')
                        emit('  同組組合搜尋（條件數 %d <= %d，於第 %d 組內枚舉）：'
                             % (len(conds), ext_when, gi))
                        emit(fmt_table(et.drop(columns=['_cd', '_ev']),
                                       left_cols=('加入參數',)))
                        top = et.iloc[0]
                        if top['增益'] >= min_gain:
                            path.append({'階段': '同組組合',
                                         '步驟': len(path) + 1, '組': gi,
                                         '選用參數': top['加入參數'],
                                         '條件': '(見上表)',
                                         '訓練解釋比例': top['_ev']['解釋比例'],
                                         '訓練誤報率': top['_ev']['誤報率'],
                                         '訓練' + gm: top['_ev'][gm],
                                         '增益': top['增益'], '採納': '是'})
                            conds, cur = top['_cd'], top['_ev']
                            emit('  → 採納：加入 %s。真根因可能集中於第 %d 組。'
                                 % (top['加入參數'], gi))
                        else:
                            emit('  → 最佳組合增益 %.3f 未達門檻 %.3f，維持原規則。'
                                 % (top['增益'], min_gain))
                            emit('  → 真根因可能不集中於第 %d 組。' % gi)
                emit('')
                emit(fmt_table(pd.DataFrame(path),
                               left_cols=('階段', '選用參數', '條件', '採納')))
                seq_conds = list(conds)      # 另存：下方列舉迴圈會重用 conds
                seq_tr = eval_rule(apply_rule(Xtr, names, seq_conds), ytr)
                seq_te = eval_rule(apply_rule(Xte, names, seq_conds), yte)
                emit('')
                emit('  採用 %d 個條件：當 %s 時判定為異常'
                     % (len(seq_conds),
                        ' 且 '.join('%s %s %.4g' % c for c in seq_conds)))
                emit('  驗證集：解釋 %d／未解釋 %d（%.1f%%），誤報 %d（%.1f%%），'
                     'F1 %.3f'
                     % (seq_te['解釋數'], seq_te['未解釋數'],
                        100 * seq_te['解釋比例'], seq_te['誤報數'],
                        100 * seq_te['誤報率'], seq_te['F1']))

                # enum_truncate=False 時，方案 B 評估完整長度的組合（不早停），
                # 使其真正補足方案 A —— 兩者都截斷時結果會大量重複。
                enum_trunc = rc.get('enum_truncate', True)
                # max_per_group > 1 時，每組可取 1~m 個參數，使同組配對納入
                # 正規枚舉。一組一個原是為避免冗餘（同組參數彼此相關），
                # 但真根因有時集中在同一組，該情形下一組一個的空間有缺口。
                mpg = max(1, int(rc.get('max_per_group', 1)))
                opts = []
                for g in groups_named:
                    gm_ = [nm for nm in g if nm in stump]
                    sub = []
                    for r in range(1, min(mpg, len(gm_)) + 1):
                        sub.extend(itertools.combinations(gm_, r))
                    opts.append(sub or [()])
                n_combo = int(np.prod([len(o) for o in opts]))
                if mpg > 1:
                    emit('  每組可取 1~%d 個參數，組合數 %d' % (mpg, n_combo))
                    # max_combo 可為 None（不設限）
                    if rc.get('max_combo') and n_combo > rc['max_combo']:
                        emit('  ⚠️ 超過 max_combo %d，退回每組一個。'
                             % rc['max_combo'])
                        opts = [[(nm,) for nm in g if nm in stump] or [()]
                                for g in groups_named]
                        # 退回之後組合數變小了，要重算 —— 否則下面會拿
                        # 退回前的大值去判斷，退了等於沒退。
                        n_combo = int(np.prod([len(o) for o in opts]))
                        emit('    退回後組合數 %d' % n_combo)
                _mc = rc.get('max_combo', None)
                if _mc and n_combo > _mc:
                    # 只擋列舉，序貫的結果照樣採用 —— 這兩者的成本無關
                    emit('  全組合列舉略過：組合數 %d 超過 max_combo %d。'
                         % (n_combo, _mc))
                    emit('    上面的序貫結果仍然有效，最終規則就用它。')
                    emit('    （預設已不設這個上限，工作量由 '
                         'enum_max_nodes 直接限制。）')
                    rows = []
                else:
                    rows = enumerate_rules(
                        opts, stump, names, Xtr, ytr, Xte, yte, gm, min_gain,
                        enum_trunc, prune=rc.get('enum_prune', True),
                        max_rows=int(rc.get('enum_max_rows', 20000)),
                        emit=emit, stats=_ENUM,
                        max_nodes=int(rc.get('enum_max_nodes', 2000000)),
                        force_multi=int(rc.get('force_multi', 2)))
                _ENUM['combos'] = n_combo

                # 序貫的結果先無條件寫進 RESULT。原本整段 RESULT['rule']
                # 只寫在 `if rows:` 裡面 —— 列舉沒跑或沒產生列時，規則就
                # 完全不進結果檔，HTML 報告於是一片空白，而序貫明明已經
                # 算出一條可用的規則。列舉有結果時下面再覆蓋。
                _seq_rule = ' 且 '.join('%s %s %.4g' % c for c in seq_conds)
                RESULT['rule'] = {
                    'path': path,
                    'seq_rule': _seq_rule,
                    'final_rule': _seq_rule,
                    'final_source': '序貫',
                    'metric': gm,
                    'seq': {'train': seq_tr[gm], 'test': seq_te[gm]},
                    'test': {k: (float(v) if isinstance(v, (int, float))
                                 else v) for k, v in seq_te.items()},
                    'n_test_abnormal': int(yte.sum()),
                    'n_test_normal': int((yte == 0).sum()),
                    'max_per_group': mpg,
                    'enum_truncate': bool(enum_trunc)}
                if rows:
                    # 依**訓練** F1 排序。原本排驗證 F1，而下面
                    # enum_best = rt.iloc[0] 取的就是這個排序的第一名 ——
                    # 於是「用驗證集挑出候選、再用訓練集去比」，兩個準則
                    # 打架。本程式自己的註解也寫著用驗證集挑等於對驗證集
                    # 過擬合。實測（1593 條規則）：排驗證得到訓練 F1
                    # 0.094 的那條，而列舉裡訓練 F1 最高的是 0.152 ——
                    # 真正該拿去比的那條根本沒被看到。
                    # 表格也跟著排訓練 F1：排驗證會把過擬合驗證集的規則
                    # 擺在最前面給人看。
                    rt = pd.DataFrame(rows).sort_values(
                        ['訓練F1', '驗證F1'], ascending=False)
                    rt.to_csv('%s_rules.csv' % prefix, index=False,
                              encoding='utf-8-sig')
                    emit('')
                    # --- 最終採用：以訓練集指標決定，驗證集僅用於報告 ---
                    # 兩種搜尋的空間互不包含：列舉窮舉「一組一個」的所有組合，
                    # 序貫在同一空間走貪心路徑但另可走同組配對。故各有盲點，
                    # 任一方都可能較好。
                    # 以訓練集決定採用者 —— 若改看驗證集挑，等於對驗證集過擬合，
                    # 其數字將不再是無偏的效能估計。
                    # 標 * 的是保底補上的 —— 它們忽略增益門檻才產生，
                    # 序貫正是因為增益不足而拒絕了它們。讓它們回頭被採用
                    # 會自相矛盾，所以只當對照，不進最終比較。
                    _real = rt[~rt['條件數'].astype(str).str.endswith('*')]
                    enum_best = (_real if len(_real) else rt).iloc[0]
                    emit('')
                    emit('  最終採用（依訓練集 %s 決定）：' % gm)
                    emit('    序貫      訓練 %.3f ／ 驗證 %.3f'
                         % (seq_tr[gm], seq_te[gm]))
                    emit('    全組合最佳 訓練 %.3f ／ 驗證 %.3f'
                         % (enum_best['訓練F1'], enum_best['驗證F1']))
                    if enum_best['訓練F1'] > seq_tr[gm]:
                        final_rule = enum_best['規則']
                        final_te = {'F1': enum_best['驗證F1'],
                                    '解釋數': enum_best['驗證解釋數'],
                                    '未解釋數': enum_best['驗證未解釋數'],
                                    '解釋比例': enum_best['驗證解釋比例'],
                                    '誤報數': enum_best['驗證誤報數'],
                                    '誤報率': enum_best['驗證誤報率']}
                        emit('    → 採用全組合最佳')
                    else:
                        final_rule = ' 且 '.join('%s %s %.4g' % c
                                                 for c in seq_conds)
                        final_te = seq_te
                        emit('    → 採用序貫')
                    emit('    規則：當 %s 時判定為異常' % final_rule)
                    emit('    驗證集：解釋 %d／未解釋 %d（%.1f%%），'
                         '誤報 %d（%.1f%%），F1 %.3f'
                         % (final_te['解釋數'], final_te['未解釋數'],
                            100 * final_te['解釋比例'], final_te['誤報數'],
                            100 * final_te['誤報率'], final_te['F1']))

                    RESULT['rule'].update({
                        'enum_top': rt.drop(columns=['規則']).head(5)
                                      .to_dict(orient='records'),
                        'enum_rule': str(enum_best['規則']),
                        'final_rule': final_rule,
                        'final_source': ('全組合' if enum_best['訓練F1'] >
                                         seq_tr[gm] else '序貫'),
                        'enum': {'train': float(enum_best['訓練F1']),
                                 'test': float(enum_best['驗證F1'])},
                        'test': {k: (float(v) if isinstance(v, (int, float))
                                     else v)
                                 for k, v in final_te.items()}})
                    emit('')
                    emit('  全組合列舉（每組取 %s 參數%s）：'
                         % ('一個' if mpg <= 1 else '1~%d 個' % mpg,
                            '，同樣套用增益停止' if enum_trunc else '，不截斷'))
                    emit(fmt_table(rt.drop(columns=['規則']).head(5),
                                   left_cols=('組合', '條件數')))
                    emit('  「條件數」為該規則實際採納的條件數%s'
                         % ('，截斷後重複的規則已合併。' if enum_trunc else '。'))
                    # 條件數 1 的規則看起來像「沒在做組合」，要說明它是
                    # 怎麼來的，否則這張表讀起來就是序貫的劣化重複。
                    _n1 = int((rt['條件數'].astype(str) == '1').sum())
                    if rt['條件數'].astype(str).str.endswith('*').any():
                        emit('  標 * 的是保底補上的多參數組合：忽略增益'
                             '門檻往下延伸，只作對照，不參與最終採用。')
                    if enum_trunc and _n1:
                        emit('  其中 %d 條只有 1 個條件：枚舉從不同的第一個'
                             '參數出發，若加入第二個條件的增益未達 %.3f'
                             % (_n1, min_gain))
                        emit('    就立刻截斷，於是停在長度 1。'
                             '那不是「沒做組合」，是「這條路一開始就走不下去」。')
                    if _n1 == len(rt) - 1 or len(rt) <= 3:
                        emit('  ⚠️ 列舉幾乎只產出序貫已經走過的路徑。'
                             'enum_truncate 與序貫用同一套增益停止，')
                        emit('     兩者都截斷時結果會大量重複 —— '
                             '設 rules.enum_truncate=false 可讓列舉評估'
                             '完整長度的組合，')
                        emit('     但合取越多條件解釋比例越低，實測那些'
                             '長規則的 F1 反而更差。這個比較本身就是結論：'
                             '序貫已經夠了。')
                    gap = float((rt['訓練F1'] - rt['驗證F1']).mean())
                    emit('  訓練與驗證 F1 平均落差：%.3f' % gap)
                    if gap > 0.15:
                        emit('  ⚠️ 落差偏大，門檻對訓練集過擬合，以驗證集為準。')

                    best = rt.iloc[0]
                    emit('')
                    emit('  最佳規則：當 %s 時判定為異常' % best['規則'])
                    emit('  驗證集：解釋 %d／未解釋 %d（%.1f%%），誤報 %d（%.1f%%）'
                         % (best['驗證解釋數'], best['驗證未解釋數'],
                            100 * best['驗證解釋比例'], best['驗證誤報數'],
                            100 * best['驗證誤報率']))

                    # 隨機對照
                    if rc['n_random'] > 0:
                        r3 = np.random.RandomState(cfg['seed'] + 1)
                        br = []
                        for _ in range(rc['n_random']):
                            pick = r3.choice(len(names), size=len(groups_named),
                                             replace=False)
                            cds = []
                            for p in pick:
                                b = best_stump(Xtr[:, p], ytr, rc['n_quantile'])
                                if b:
                                    cds.append((names[p], b[0], b[1]))
                            if len(cds) == len(groups_named):
                                br.append(eval_rule(apply_rule(Xte, names, cds),
                                                    yte)['F1'])
                        if br:
                            br = np.array(br)
                            pv = float((br >= final_te['F1']).mean())
                            emit('  隨機對照（挑 %d 個參數，%d 次）：'
                                 '平均 F1 %.3f，95%%分位 %.3f'
                                 % (len(groups_named), len(br), br.mean(),
                                    np.percentile(br, 95)))
                            emit('  最終規則 F1 %.3f，p = %.3f'
                                 % (final_te['F1'], pv))
                            emit('  %s' % ('✅ 顯著優於隨機挑參數。' if pv < 0.05
                                           else '❌ 未顯著優於隨機，說服力有限。'))
                            RESULT.setdefault('rule', {})['random'] = {
                                'mean': float(br.mean()),
                                'p95': float(np.percentile(br, 95)),
                                'p': pv}
    else:
        emit('\n[步驟 7] 規則萃取已關閉。')

    _mark('收尾')
    _on[0] = True            # 計時表一律輸出，brief 模式也要看得到
    tot = sum(d for _n, d in _T['marks'])
    emit('')
    emit('各步驟耗時（總計 %.1f 秒）' % tot)
    for nm, d in _T['marks']:
        if d < 0.05 and d / max(tot, 1e-9) < 0.01:
            continue
        bar = '█' * int(round(28 * d / max(tot, 1e-9)))
        emit('  %-12s %7.2f 秒  %5.1f%%  %s'
             % (nm, d, 100 * d / max(tot, 1e-9), bar))
    RESULT['timing'] = {nm: round(d, 3) for nm, d in _T['marks']}

    # ---- 一行摘要 ----
    # 無法把檔案帶出公司機器時，照抄這一行即可，不必謄整張表。
    # 刻意只用 ASCII 與短鍵，便於手打或拍照。
    def _si(v):
        v = float(v)
        for u, s in ((1e9, 'G'), (1e6, 'M'), (1e3, 'k')):
            if v >= u:
                return '%.2f%s' % (v / u, s)
        return '%.0f' % v

    slow = sorted(_T['marks'], key=lambda kv: -kv[1])[:3]
    parts = ['DIGEST tot=%.1fs' % tot,
             'slow: ' + ' '.join('%s=%.1f' % (n.replace('步驟 ', 's'), d)
                                 for n, d in slow)]
    if _SHAPE:
        parts.append('w=%d/%d p=%d g=%s'
                     % (_SHAPE.get('n', 0), _SHAPE.get('a', 0),
                        _SHAPE.get('p', 0), _SHAPE.get('g', '-')))
    if _ENUM:
        parts.append('enum nd=%s ru=%s pr=%s cb=%s'
                     % (_si(_ENUM.get('nodes', 0)),
                        _si(_ENUM.get('rules', 0)),
                        _si(_ENUM.get('pruned', 0)),
                        _si(_ENUM.get('combos', 0))))
    digest = ' | '.join(parts)
    emit('')
    emit(digest)
    RESULT['digest'] = digest

    with open('%s_report.txt' % prefix, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    with open('%s_result.json' % prefix, 'w', encoding='utf-8') as f:
        json.dump(RESULT, f, ensure_ascii=False, indent=2, default=str)
    emit('')
    emit('輸出：%s_report.txt / _ranking.csv / _rules.csv / _result.json'
         % prefix)


if __name__ == '__main__':
    main()