# -*- coding: utf-8 -*-
"""
diag_edges.py  ——  為什麼 edges 檔涵蓋的 sensor 數是 0

比對三組名稱，找出不一致的原因：
  A. edges 檔裡的 source / target
  B. 同事實際讀的資料集的欄名
  C. 候選清單（checklist）裡的 sensor 名

「涵蓋 0 個」幾乎都不是數量問題，而是字串對不起來。常見原因依發生率：
  1. edges 檔是舊的（上一輪用不同資料集名稱寫的，名稱來自不同的參數集）
  2. 參數名稱含逗號，而 edges 是手寫 CSV 沒做引號處理 → 欄位錯位
  3. 編碼不同（寫 utf-8、讀 cp950），非 ASCII 名稱變亂碼
  4. 前後空白：一邊有、一邊被 strip 掉
  5. 大小寫或全半角差異

執行
----
  python scripts/diag_edges.py \\
      --edges datasets/case2_reduced_edges.csv \\
      --data  data/raw/semiconductor/case2_reduced.csv \\
      --checklist ../fd_analysis/gnn-tam-main/experiment_results/checklist/checklist_case2_reduced_.csv

  --checklist 可省略。

環境：Python 3.6+（只用標準庫）
"""

from __future__ import print_function

import argparse
import csv
import os
import sys
import unicodedata

ENCODINGS = ('utf-8-sig', 'utf-8', 'cp950', 'big5hkscs', 'gbk', 'latin-1')

META = {'LOT_ID', 'WFR_ID', 'WAFER_ID', 'START_DATETIME', 'END_DATETIME',
        'Symbol Define'}


def read_rows(path):
    """回傳 (rows, 使用的編碼)。rows 是 list of list。"""
    last = None
    for enc in ENCODINGS:
        try:
            with open(path, 'r', encoding=enc, newline='') as f:
                return list(csv.reader(f)), enc
        except UnicodeDecodeError as e:
            last = e
    sys.exit('無法解碼 %s：%s' % (path, last))


def show(name, s):
    """把一個字串的異常之處顯示出來。"""
    bits = []
    if s != s.strip():
        bits.append('前後有空白')
    if any(ord(c) > 127 for c in s):
        bits.append('含非 ASCII')
    if any(unicodedata.east_asian_width(c) in 'WF' for c in s):
        bits.append('含全角')
    if ',' in s:
        bits.append('含逗號')
    tag = ('  ← ' + '、'.join(bits)) if bits else ''
    print('    %-34s len=%-3d %s%s' % (repr(s)[:34], len(s), name, tag))


def main():
    p = argparse.ArgumentParser(description='診斷 edges 與資料集的名稱不一致')
    p.add_argument('--edges', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--checklist', default=None)
    p.add_argument('--sensor-col', default='Sensor')
    args = p.parse_args()

    for f in (args.edges, args.data):
        if not os.path.exists(f):
            sys.exit('找不到 %s' % f)

    print('=' * 70)
    print('edges 名稱診斷')
    print('=' * 70)

    # ---- A edges ----
    erows, eenc = read_rows(args.edges)
    print('')
    print('[A] edges 檔：%s' % args.edges)
    print('    編碼 %s｜共 %d 列（含標頭）' % (eenc, len(erows)))
    if not erows:
        sys.exit('edges 檔是空的。')
    hdr = erows[0]
    print('    標頭 %s（%d 欄）' % (hdr, len(hdr)))
    bad_width = [i for i, r in enumerate(erows[1:], 2) if len(r) != len(hdr)]
    if bad_width:
        print('    ❌ 有 %d 列的欄數與標頭不符（前幾列：%s）'
              % (len(bad_width), bad_width[:5]))
        print('       這是手寫 CSV 未做引號處理的典型症狀 —— '
              '參數名稱含逗號時欄位會錯位。')
        for k in bad_width[:3]:
            print('       第 %d 列（%d 欄）：%s' % (k, len(erows[k - 1]),
                                                   erows[k - 1][:6]))
    try:
        i_src = hdr.index('source')
        i_tgt = hdr.index('target')
    except ValueError:
        print('    ❌ 標頭沒有 source / target。實際標頭：%s' % hdr)
        i_src, i_tgt = 0, len(hdr) - 1
        print('       改用第 0 欄與最後一欄繼續。')
    enames = set()
    for r in erows[1:]:
        if len(r) > max(i_src, i_tgt):
            enames.add(r[i_src])
            enames.add(r[i_tgt])
    print('    節點名稱 %d 個，樣本：' % len(enames))
    for s in sorted(enames)[:5]:
        show('edges', s)

    # ---- B 資料集 ----
    drows, denc = read_rows(args.data)
    print('')
    print('[B] 資料集：%s' % args.data)
    print('    編碼 %s｜%d 列' % (denc, len(drows)))
    dcols = [c for c in (drows[0] if drows else []) if c not in META]
    print('    參數欄位 %d 個，樣本：' % len(dcols))
    for s in dcols[:5]:
        show('data', s)

    # ---- 比對 ----
    print('')
    print('=' * 70)
    print('比對')
    print('=' * 70)
    dset = set(dcols)
    hit = enames & dset
    print('  交集 %d／edges %d／資料集 %d' % (len(hit), len(enames),
                                             len(dset)))
    if not hit:
        print('  ❌ 完全沒有交集。')
        # 逐步放寬，找出差在哪一層
        for label, f in (('去前後空白', lambda s: s.strip()),
                         ('轉小寫', lambda s: s.strip().lower()),
                         ('去空白與底線', lambda s: s.strip().lower()
                          .replace('_', '').replace(' ', '')),
                         ('NFKC 正規化', lambda s: unicodedata.normalize(
                             'NFKC', s).strip().lower())):
            n = len({f(x) for x in enames} & {f(x) for x in dset})
            print('    %s 後交集：%d' % (label, n))
            if n:
                print('      → 差異就在這一層，對症處理即可。')
                break
        else:
            print('    所有正規化都無交集 —— 兩邊根本是不同的參數集合。')
            print('      最可能是 edges 檔是舊的：它的名稱來自上一輪'
                  '（不同的資料集名稱或不同的縮減結果）。')
            print('      請重跑步驟 1 產生與這個資料集對應的 edges。')
            print('      edges 專有的名稱樣本：%s'
                  % sorted(enames - dset)[:5])
            print('      資料集專有的名稱樣本：%s'
                  % sorted(dset - enames)[:5])
    else:
        print('  ✓ edges 的節點有 %d 個在資料集中（%.0f%%）'
              % (len(hit), 100.0 * len(hit) / max(len(enames), 1)))
        miss = enames - dset
        if miss:
            print('  edges 有但資料集沒有的 %d 個：%s'
                  % (len(miss), sorted(miss)[:5]))

    # ---- C checklist ----
    if args.checklist and os.path.exists(args.checklist):
        crows, cenc = read_rows(args.checklist)
        print('')
        print('[C] checklist：%s' % args.checklist)
        print('    編碼 %s' % cenc)
        ch = crows[0] if crows else []
        ci = ch.index(args.sensor_col) if args.sensor_col in ch else 0
        cnames = {r[ci] for r in crows[1:] if len(r) > ci}
        print('    sensor 名稱 %d 個（欄 "%s"）' % (len(cnames), ch[ci]))
        print('    與 edges 的交集：%d' % len(cnames & enames))
        print('    與資料集的交集：%d' % len(cnames & dset))
        if cnames and not (cnames & enames):
            print('    ❌ checklist 的 sensor 全部不在 edges 裡 —— '
                  '他讀到的 edges 可能不是這一份。')
    print('')


if __name__ == '__main__':
    main()
