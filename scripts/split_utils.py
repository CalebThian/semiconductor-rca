# -*- coding: utf-8 -*-
"""
split_utils.py  ——  Normal 資料的區塊切分(block split)

問題背景
--------
  隨機打散    : 覆蓋全時段 ✓ / 視窗內時序連續 ✗（lookback 視窗是隨機 wafer 拼的）
  連續尾段切分: 覆蓋全時段 ✗ / 視窗內時序連續 ✓（模型沒看過後段，漂移未學到）

兩者都有缺陷。區塊切分同時滿足兩個條件：
  把序列切成 n_blocks 個連續區塊，間隔取出當 held-out。
  每個區塊內部時序完整，held-out 均勻分散在整段時間。

代價：區塊邊界是真正的斷點，必須把 segment_lengths 傳進 make_dataset，
      否則視窗會跨越區塊，等於重新引入污染。

環境：Python 3.6+ / numpy
"""

import numpy as np


def block_split_indices(n, heldout_frac=0.2, n_blocks=10, lookback=None):
    """把 0..n-1 切成 n_blocks 個連續區塊，間隔取出當 held-out。

    回傳 (train_slices, held_slices)，各為 [(start, end), ...]。

    每隔 round(1/heldout_frac) 個區塊取一個當 held-out，
    所以 heldout_frac=0.2 → 每 5 個區塊取 1 個。
    """
    if n <= 0:
        raise ValueError('n 必須為正')
    if not 0 < heldout_frac < 1:
        raise ValueError('heldout_frac 必須介於 0 與 1')
    n_blocks = max(2, min(int(n_blocks), n))

    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    blocks = [(int(edges[i]), int(edges[i + 1])) for i in range(n_blocks)]
    blocks = [b for b in blocks if b[1] > b[0]]

    if lookback is not None:
        short = [b for b in blocks if (b[1] - b[0]) <= lookback]
        if short:
            raise ValueError(
                '有 %d 個區塊短於 lookback(%d)，整段無法產生樣本。'
                '請把 n_blocks 調小（目前 %d，資料長度 %d，平均區塊 %.1f）。'
                % (len(short), lookback, n_blocks, n, n / float(len(blocks))))

    every = max(2, int(round(1.0 / heldout_frac)))
    # 從第 1 個區塊(而非第 0 個)開始取，讓 held-out 不落在最頭尾
    held_ids = set(range(1, len(blocks), every))
    if not held_ids:
        held_ids = {len(blocks) // 2}

    train_slices = [b for k, b in enumerate(blocks) if k not in held_ids]
    held_slices = [b for k, b in enumerate(blocks) if k in held_ids]
    if not train_slices or not held_slices:
        raise ValueError('切分後某一邊為空，請調整 n_blocks 或 heldout_frac')
    return train_slices, held_slices


def apply_slices(arr, slices):
    """依 slices 取出資料並串接，同時回傳每段長度。

    回傳 (sub_array, segment_lengths)。
    segment_lengths 對應 sub_array 的列順序，可直接傳給 make_dataset。
    """
    parts = [arr[a:b] for a, b in slices]
    seg = [int(b - a) for a, b in slices]
    return np.concatenate(parts, axis=0), seg


def block_split(arr, heldout_frac=0.2, n_blocks=10, lookback=None, verbose=True):
    """一次做完。回傳 (train, train_seg, held, held_seg)。"""
    n = len(arr)
    tr_s, hd_s = block_split_indices(n, heldout_frac, n_blocks, lookback)
    train, train_seg = apply_slices(arr, tr_s)
    held, held_seg = apply_slices(arr, hd_s)
    if verbose:
        print('  區塊切分: 共 %d 塊，train %d 塊 / held-out %d 塊'
              % (len(tr_s) + len(hd_s), len(tr_s), len(hd_s)))
        print('  train %d 列(%d 段, 最短 %d)，held-out %d 列(%d 段, 最短 %d)'
              % (len(train), len(train_seg), min(train_seg),
                 len(held), len(held_seg), min(held_seg)))
        if lookback:
            usable_tr = sum(max(L - lookback, 0) for L in train_seg)
            usable_hd = sum(max(L - lookback, 0) for L in held_seg)
            print('  lookback=%d 下可用樣本: train %d / held-out %d'
                  % (lookback, usable_tr, usable_hd))
            print('  (若不傳 segment_lengths，train 會多出 %d 個跨界污染樣本)'
                  % (len(train) - lookback - usable_tr))
    return train, train_seg, held, held_seg


if __name__ == '__main__':
    tr_s, hd_s = block_split_indices(100, 0.2, 10)
    assert len(tr_s) + len(hd_s) == 10
    assert len(hd_s) == 2, hd_s
    cover = sorted([x for a, b in tr_s + hd_s for x in range(a, b)])
    assert cover == list(range(100)), '切分必須完整覆蓋且不重疊'

    a = np.arange(100).reshape(100, 1)
    tr, trs, hd, hds = block_split(a, 0.2, 10, lookback=3)
    assert len(tr) + len(hd) == 100
    assert sum(trs) == len(tr) and sum(hds) == len(hd)
    # held-out 應分散在時間軸上，而非集中一端
    hd_vals = hd.ravel()
    assert hd_vals.min() < 30 and hd_vals.max() > 50, hd_vals

    try:
        block_split_indices(20, 0.2, 10, lookback=5)
        raise AssertionError('應該要因區塊過短而報錯')
    except ValueError:
        pass

    print('\n全部測試通過。')
