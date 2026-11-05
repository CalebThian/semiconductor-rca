# -*- coding: utf-8 -*-
"""驗證列舉的剪枝不會漏掉更好的規則。

「走訪 3 個節點」容易被讀成「根本沒嘗試組合」。實際上整個空間都被涵蓋：
一部分展開，一部分由上界 F1 <= 2R/(1+R) 證明不可能更好。這支腳本在小規模
上把 prune 關掉重跑，確認兩邊找到的最佳訓練 F1 相同。

注意不能拿「窮舉所有前綴」當基準：enum_truncate 開啟時 DFS 會拒絕加入
增益不足的條件，它的搜尋空間本來就小於全部前綴，兩者不可比。要隔離剪枝
的影響，就得固定截斷規則、只切換 prune。

執行：python scripts/verify_prune.py
"""
from __future__ import print_function
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_pipeline as RP                                    # noqa: E402


def one(ng, gs, T, seed):
    rng = np.random.default_rng(seed)
    nm = ['P%02d' % i for i in range(ng * gs)]
    X = rng.normal(size=(T, len(nm)))
    y = ((X[:, 0] > 0.3) & (X[:, gs] > 0.2)).astype(int)
    if y.sum() < 10 or (y == 0).sum() < 10:
        return None
    h = T // 2
    Xtr, ytr, Xte, yte = X[:h], y[:h], X[h:], y[h:]
    stump = {}
    for k, n_ in enumerate(nm):
        s = RP.best_stump(Xtr[:, k], ytr, 20)
        if s:
            stump[n_] = s
    if not stump:
        return None
    groups = [nm[i * gs:(i + 1) * gs] for i in range(ng)]
    opts = [[(x,) for x in g if x in stump] or [()] for g in groups]
    pos = int(ytr.sum())

    def gm(tp, fp):
        r = round(tp / max(pos, 1), 3)
        p = round(tp / max(tp + fp, 1), 3)
        return 0.0 if r + p == 0 else round(2 * r * p / (r + p), 3)

    def ev(cols):
        m = np.ones(h, dtype=bool)
        for c in cols:
            op, th = stump[c][0], stump[c][1]
            xi = Xtr[:, nm.index(c)]
            m &= (xi > th) if op == '>' else (xi < th)
        return gm(int((m & (ytr == 1)).sum()), int((m & (ytr == 0)).sum()))

    # 隔離剪枝：截斷規則兩邊相同，只切換 prune。
    # （不能拿「窮舉所有前綴」當基準 —— 截斷時 DFS 會拒絕加入增益不足的
    #   條件，所以它的搜尋空間本來就比全部前綴小，兩者不可比。）
    out = {}
    for pr in (True, False):
        st = {}
        rows = RP.enumerate_rules(opts, stump, nm, Xtr, ytr, Xte, yte, gm,
                                  0.02, True, prune=pr, max_rows=100000,
                                  emit=lambda *_a, **_k: None, stats=st)
        if not rows:
            return None
        out[pr] = (max(r['訓練F1'] for r in rows), len(rows), st['nodes'])
    return out[True], out[False]


def main():
    print('剪枝正確性驗證：開剪枝與不開剪枝找到的最佳訓練 F1 必須相同')
    print('%-14s %-10s %-10s %-12s %s'
          % ('規模', 'prune=True', 'prune=False', '節點 剪/不剪', '結論'))
    bad = 0
    for ng, gs, T, seed in ((4, 3, 300, 0), (5, 3, 400, 1), (6, 3, 400, 2),
                            (6, 4, 400, 3), (7, 3, 500, 4), (5, 5, 400, 5)):
        r = one(ng, gs, T, seed)
        if r is None:
            continue
        (f1p, np_, ndp), (f1f, nf, ndf) = r
        ok = abs(f1p - f1f) < 1e-9
        bad += 0 if ok else 1
        print('%-14s %-10.3f %-10.3f %-12s %s'
              % ('%d 組 x %d' % (ng, gs), f1p, f1f, '%d / %d' % (ndp, ndf),
                 '✓ 相同' if ok else '❌ 剪枝漏解'))
    print('')
    print('')
    print('結論：%s' % ('全部通過 —— 被剪掉的分支確實不可能更好。'
                       if not bad else '有 %d 個規模漏解，剪枝有誤。' % bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
