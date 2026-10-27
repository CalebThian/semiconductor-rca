# -*- coding: utf-8 -*-
"""
collect_deps.py  ——  一次算出完整的模組依賴閉包

問題
----
  從舊 repo 搬程式到新 repo 時，漏一個模組就是一次 ModuleNotFoundError，
  一個一個補要跑很多輪。這支程式從入口檔出發，沿 import 一路追到底，
  一次列出所有需要的檔案，以及哪些還沒搬過來。

做法
----
  用 ast 解析每個檔案的 import（不執行程式碼，所以不會因為缺套件而失敗），
  只追蹤屬於這個專案的模組（預設 src / scripts），對照兩個 repo 判斷
  每個檔案的狀態。

執行
----
  python scripts/collect_deps.py \
      --src ~/causal-sensor-graph \
      --dst ~/semiconductor-rca \
      --entry scripts/run_all.py scripts/run_pipeline.py

  只看缺哪些、順便印出 cp 指令：
  python scripts/collect_deps.py --src ~/causal-sensor-graph \
      --dst ~/semiconductor-rca --entry scripts/run_all.py --emit-cp

環境：Python 3.6+（只用標準庫）
"""

from __future__ import print_function

import argparse
import ast
import os
import sys

ENCODINGS = ('utf-8-sig', 'utf-8', 'cp950', 'big5hkscs', 'latin-1')


def read_text(path):
    for enc in ENCODINGS:
        try:
            with open(path, 'r', encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
    return None


def module_to_paths(mod):
    """把 a.b.c 轉成可能的檔案路徑（模組檔或套件目錄）。"""
    base = mod.replace('.', '/')
    return [base + '.py', base + '/__init__.py']


def imports_of(path, roots):
    """回傳這個檔案 import 的專案內模組名稱。

    用 ast 解析，不執行程式碼 —— 所以缺 torch 之類的第三方套件也能分析。
    函式內的 import 也會抓到（run_all.py 的 torch 匯入就在函式裡）。
    """
    txt = read_text(path)
    if txt is None:
        return set(), '無法解碼'
    try:
        tree = ast.parse(txt)
    except SyntaxError as e:
        return set(), '語法錯誤 line %s' % e.lineno
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split('.')[0] in roots:
                    out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:          # 相對匯入，這些 repo 沒用到，略過
                continue
            if node.module and node.module.split('.')[0] in roots:
                out.add(node.module)
                # from src.models import x —— x 可能是子模組也可能是名稱，
                # 兩者都當候選，解析不到的那個會被忽略
                for a in node.names:
                    out.add(node.module + '.' + a.name)
    return out, None


def resolve(mod, repos):
    """在數個 repo 裡找這個模組的檔案，回傳 (相對路徑, 所在repo) 或 None。"""
    for cand in module_to_paths(mod):
        for name, root in repos:
            if os.path.isfile(os.path.join(root, cand)):
                return cand, name
    return None, None


def main():
    p = argparse.ArgumentParser(description='算出模組依賴閉包')
    p.add_argument('--src', required=True, help='舊 repo（來源）')
    p.add_argument('--dst', required=True, help='新 repo（目的）')
    p.add_argument('--entry', nargs='+', required=True,
                   help='入口檔，相對於 repo 根目錄')
    p.add_argument('--roots', default='src,scripts',
                   help='視為專案內模組的頂層名稱，逗號分隔')
    p.add_argument('--emit-cp', action='store_true',
                   help='印出可直接執行的 cp 指令')
    args = p.parse_args()

    src = os.path.abspath(os.path.expanduser(args.src))
    dst = os.path.abspath(os.path.expanduser(args.dst))
    roots = set(r.strip() for r in args.roots.split(',') if r.strip())
    for d, n in ((src, '--src'), (dst, '--dst')):
        if not os.path.isdir(d):
            sys.exit('%s 不是目錄：%s' % (n, d))

    # 解析順序：先看新 repo（已搬的），再看舊 repo（待搬的）
    repos = [('dst', dst), ('src', src)]

    seen = set()
    queue = []
    for e in args.entry:
        if os.path.isfile(os.path.join(dst, e)):
            queue.append((e, 'dst'))
        elif os.path.isfile(os.path.join(src, e)):
            queue.append((e, 'src'))
        else:
            sys.exit('找不到入口檔 %s（兩個 repo 都沒有）' % e)

    in_dst, to_copy, missing, bad = [], [], [], []

    while queue:
        rel, where = queue.pop(0)
        if rel in seen:
            continue
        seen.add(rel)
        (in_dst if where == 'dst' else to_copy).append(rel)
        full = os.path.join(dst if where == 'dst' else src, rel)
        mods, err = imports_of(full, roots)
        if err:
            bad.append((rel, err))
            continue
        for m in sorted(mods):
            cand, w = resolve(m, repos)
            if cand is None:
                # from src.x import foo 會產生候選 src.x.foo，而 foo 是
                # 函式不是模組。若父模組找得到，這個候選就只是名稱，
                # 不是缺少的檔案 —— 不可回報，否則整份清單都是誤判。
                parent = m.rsplit('.', 1)[0] if '.' in m else None
                if parent and resolve(parent, repos)[0] is not None:
                    continue
                missing.append(m)
                continue
            if cand not in seen:
                queue.append((cand, w))

    print('=' * 66)
    print('依賴閉包')
    print('=' * 66)
    print('來源：%s' % src)
    print('目的：%s' % dst)
    print('入口：%s' % ', '.join(args.entry))

    print('')
    print('[1] 已在新 repo（%d 個）' % len(in_dst))
    for f in sorted(in_dst):
        print('    %s' % f)

    print('')
    print('[2] 還要從舊 repo 搬過來（%d 個）' % len(to_copy))
    for f in sorted(to_copy):
        print('    %s' % f)

    miss = sorted(set(missing))
    print('')
    print('[3] 兩個 repo 都找不到（%d 個）' % len(miss))
    for m in miss:
        print('    %s' % m)
    if miss:
        print('    ↑ 這些可能是外部套件、已刪除的模組，或名稱拼錯。')

    if bad:
        print('')
        print('[4] 無法解析（%d 個）' % len(bad))
        for f, e in bad:
            print('    %-50s %s' % (f, e))

    if args.emit_cp and to_copy:
        print('')
        print('=' * 66)
        print('複製指令（直接貼進終端）')
        print('=' * 66)
        print('SRC=%s' % args.src)
        print('DST=%s' % args.dst)
        dirs = sorted(set(os.path.dirname(f) for f in to_copy if
                          os.path.dirname(f)))
        if dirs:
            print('mkdir -p %s' % ' '.join('"$DST"/%s' % d for d in dirs))
        for f in sorted(to_copy):
            print('cp "$SRC"/%s "$DST"/%s' % (f, f))
        pkgs = sorted(set(d for f in to_copy
                          for d in [os.path.dirname(f)] if d))
        if pkgs:
            print('for d in %s; do touch "$DST"/$d/__init__.py; done'
                  % ' '.join(pkgs))

    print('')
    print('DIGEST in_dst=%d to_copy=%d unresolved=%d unparsed=%d'
          % (len(in_dst), len(to_copy), len(miss), len(bad)))


if __name__ == '__main__':
    main()
