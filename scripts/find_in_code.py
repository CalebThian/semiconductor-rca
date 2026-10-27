# -*- coding: utf-8 -*-
"""
find_in_code.py  ——  findstr 的替代品，純 Python，跨平台

用途：在程式碼檔案中搜尋關鍵字，印出行號與內容（可帶前後文）。
      公司機器不能用 findstr 時用這支。

範例：
    python find_in_code.py scripts\residual_anomaly.py lookback window lag
    python find_in_code.py scripts\residual_anomaly.py --context 3 "for t in"
    python find_in_code.py scripts --recursive --ext .py segment_lengths

環境：Python 3.6+（只用標準庫）
"""

import argparse
import os
import re
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description='在程式碼中搜尋關鍵字（findstr 替代品）')
    p.add_argument('path', help='檔案或資料夾路徑')
    p.add_argument('patterns', nargs='+', help='要搜尋的關鍵字（可多個）')
    p.add_argument('--context', '-C', type=int, default=0,
                   help='顯示命中行的前後幾行')
    p.add_argument('--case-sensitive', action='store_true',
                   help='區分大小寫（預設不分）')
    p.add_argument('--regex', action='store_true',
                   help='把關鍵字當正規表示式處理')
    p.add_argument('--recursive', '-r', action='store_true',
                   help='path 是資料夾時遞迴搜尋')
    p.add_argument('--ext', default='.py',
                   help='遞迴時只搜這些副檔名，逗號分隔')
    p.add_argument('--out', default=None,
                   help='把結果同時寫進這個檔（方便複製貼上）')
    return p.parse_args()


def read_lines(path):
    for enc in ('utf-8', 'utf-8-sig', 'cp950', 'latin-1'):
        try:
            with open(path, 'r', encoding=enc) as f:
                return f.read().splitlines()
        except (UnicodeDecodeError, LookupError):
            continue
    return None


def collect_files(args):
    if os.path.isfile(args.path):
        return [args.path]
    if not os.path.isdir(args.path):
        sys.exit('找不到路徑：%s' % args.path)
    exts = tuple(e.strip() for e in args.ext.split(',') if e.strip())
    files = []
    if args.recursive:
        for root, _dirs, names in os.walk(args.path):
            if any(p in root for p in ('.git', '__pycache__', 'node_modules')):
                continue
            for n in names:
                if n.endswith(exts):
                    files.append(os.path.join(root, n))
    else:
        for n in sorted(os.listdir(args.path)):
            fp = os.path.join(args.path, n)
            if os.path.isfile(fp) and n.endswith(exts):
                files.append(fp)
    return files


def main():
    args = parse_args()
    flags = 0 if args.case_sensitive else re.IGNORECASE
    pats = []
    for p in args.patterns:
        pats.append(re.compile(p if args.regex else re.escape(p), flags))

    files = collect_files(args)
    if not files:
        sys.exit('沒有符合的檔案。')

    out_lines = []

    def emit(s):
        print(s)
        out_lines.append(s)

    total = 0
    for fp in files:
        lines = read_lines(fp)
        if lines is None:
            emit('!! 無法解碼：%s' % fp)
            continue
        hits = []
        for i, ln in enumerate(lines):
            matched = [p.pattern for p in pats if p.search(ln)]
            if matched:
                hits.append(i)
        if not hits:
            continue
        total += len(hits)
        emit('')
        emit('=' * 70)
        emit('%s   （命中 %d 行 / 全檔 %d 行）' % (fp, len(hits), len(lines)))
        emit('=' * 70)
        shown = set()
        for i in hits:
            lo = max(0, i - args.context)
            hi = min(len(lines), i + args.context + 1)
            if args.context and shown and lo > max(shown) + 1:
                emit('   ...')
            for j in range(lo, hi):
                if j in shown:
                    continue
                shown.add(j)
                mark = '>' if j in hits else ' '
                emit('%s %5d | %s' % (mark, j + 1, lines[j]))

    emit('')
    emit('總計命中 %d 行，搜尋 %d 個檔案。' % (total, len(files)))
    if total == 0:
        emit('沒有命中。可能原因：關鍵字拼法不同、或搜錯檔案。')
        emit('建議先用 --recursive 在整個 scripts 資料夾找一次。')

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out_lines))
        print('\n結果已存：%s' % args.out)


if __name__ == '__main__':
    main()