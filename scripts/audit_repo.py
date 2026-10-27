# -*- coding: utf-8 -*-
"""
audit_repo.py  ——  分 repo 或首次 push 之前的稽核

為什麼需要
----------
  從工作目錄刪掉檔案，不會把它從 git 歷史裡刪掉。投稿 repo 一旦公開，
  歷史中的每個 blob 都跟著公開 —— 包含曾經 commit 過又刪掉的公司資料。

  這支程式只讀不寫，不會改動任何東西。

檢查項目
--------
  1. 目前被追蹤的資料類檔案（csv/npy/xlsx/parquet/pkl/...）
  2. **git 歷史中**曾出現過的資料類檔案（即使現在已刪除）
  3. 歷史中的大 blob（GitHub 超過 100MB 直接拒收，50MB 警告）
  4. .ipynb_checkpoints 等應被忽略卻被追蹤的路徑
  5. 疑似憑證的字串（粗篩，只看被追蹤的文字檔）
  6. remote 設定（避免把公司 repo 推到投稿用的 remote）

執行
----
  在 repo 根目錄：
    python scripts/audit_repo.py

  指定其他 repo：
    python scripts/audit_repo.py --repo ../paper-repo

  只要摘要那一行：
    python scripts/audit_repo.py --quiet

環境：Python 3.6+（只用標準庫）＋ 系統上的 git
"""

from __future__ import print_function

import argparse
import os
import subprocess
import sys

DATA_EXT = ('.csv', '.tsv', '.npy', '.npz', '.xlsx', '.xls', '.parquet',
            '.pkl', '.pickle', '.h5', '.hdf5', '.feather', '.mat',
            '.db', '.sqlite', '.zip', '.7z', '.tar', '.gz', '.rar')
MODEL_EXT = ('.pt', '.pth', '.ckpt', '.onnx', '.safetensors', '.joblib')
IGNORE_PATH = ('.ipynb_checkpoints', '__pycache__', '.DS_Store',
               '.idea', '.vscode')
SECRET_HINT = ('AKIA', 'BEGIN RSA PRIVATE KEY', 'BEGIN OPENSSH PRIVATE KEY',
               'BEGIN PRIVATE KEY', 'ghp_', 'github_pat_', 'xoxb-',
               'password =', 'password=', 'api_key =', 'api_key=',
               'secret_key =', 'secret_key=', 'AIza')
TEXT_EXT = ('.py', '.ipynb', '.json', '.yaml', '.yml', '.toml', '.cfg',
            '.ini', '.md', '.txt', '.sh', '.bat', '.ps1', '.env')


def git(args, repo, ok_fail=False):
    """跑 git 並回傳 stdout 的行。失敗時回傳空 list（ok_fail）或中止。"""
    try:
        p = subprocess.Popen(['git'] + args, cwd=repo,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
    except OSError:
        sys.exit('找不到 git，請確認已安裝且在 PATH 中。')
    if p.returncode != 0:
        if ok_fail:
            return []
        sys.exit('git %s 失敗：%s'
                 % (' '.join(args), err.decode('utf-8', 'replace').strip()))
    return out.decode('utf-8', 'replace').splitlines()


def human(n):
    n = float(n)
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or u == 'GB':
            return '%.1f%s' % (n, u)
        n /= 1024


def main():
    ap = argparse.ArgumentParser(description='分 repo / 首次 push 前的稽核')
    ap.add_argument('--repo', default='.', help='repo 路徑（預設目前目錄）')
    ap.add_argument('--quiet', action='store_true', help='只印摘要那一行')
    ap.add_argument('--top', type=int, default=10, help='各清單顯示幾筆')
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)

    def say(s=''):
        if not args.quiet:
            print(s, flush=True)

    def hr(t):
        say('')
        say('-' * 64)
        say(t)
        say('-' * 64)

    if not os.path.isdir(os.path.join(repo, '.git')):
        inside = git(['rev-parse', '--is-inside-work-tree'], repo, True)
        if not inside or inside[0].strip() != 'true':
            sys.exit('%s 不是 git repo。' % repo)

    say('=' * 64)
    say('repo 稽核：%s' % repo)
    say('=' * 64)
    say('※ 只讀，不會修改任何東西')

    flags = {}

    # ---- 1 remote ----
    hr('[1] remote')
    rem = git(['remote', '-v'], repo, True)
    if not rem:
        say('  沒有設定 remote。')
    for r in rem:
        say('  %s' % r)
    flags['remotes'] = len(set(r.split()[0] for r in rem if r.split()))

    br = git(['rev-parse', '--abbrev-ref', 'HEAD'], repo, True)
    say('  目前分支：%s' % (br[0] if br else '?'))

    # ---- 2 目前追蹤的檔案 ----
    hr('[2] 目前被追蹤的檔案')
    files = [f for f in git(['ls-files'], repo) if f.strip()]
    say('  共 %d 個' % len(files))

    def lower_ends(p, exts):
        return p.lower().endswith(exts)

    cur_data = [f for f in files if lower_ends(f, DATA_EXT)]
    cur_model = [f for f in files if lower_ends(f, MODEL_EXT)]
    cur_junk = [f for f in files
                if any(('/' + s + '/') in ('/' + f + '/') or
                       f.endswith(s) for s in IGNORE_PATH)]
    for tag, lst in (('資料類', cur_data), ('模型權重', cur_model),
                     ('應忽略的路徑', cur_junk)):
        say('  %s：%d 個' % (tag, len(lst)))
        for f in lst[:args.top]:
            p = os.path.join(repo, f)
            sz = os.path.getsize(p) if os.path.exists(p) else 0
            say('    %-52s %s' % (f[:52], human(sz)))
        if len(lst) > args.top:
            say('    …其餘 %d 個' % (len(lst) - args.top))
    flags['cur_data'] = len(cur_data)
    flags['cur_model'] = len(cur_model)
    flags['cur_junk'] = len(cur_junk)

    # ---- 3 歷史中的資料類路徑 ----
    hr('[3] git 歷史中曾出現的資料類路徑（即使現在已刪除）')
    hist = set()
    for ln in git(['log', '--all', '--pretty=format:', '--name-only',
                   '--diff-filter=A'], repo, True):
        f = ln.strip()
        if f and (lower_ends(f, DATA_EXT) or lower_ends(f, MODEL_EXT)):
            hist.add(f)
    gone = sorted(h for h in hist if h not in set(files))
    say('  歷史中出現過：%d 個；其中已不在目前版本：%d 個'
        % (len(hist), len(gone)))
    for f in gone[:args.top]:
        say('    %s' % f)
    if len(gone) > args.top:
        say('    …其餘 %d 個' % (len(gone) - args.top))
    if gone:
        say('')
        say('  ⚠️ 這些檔案已從工作目錄刪除，但仍留在歷史裡。')
        say('     clone 或公開這個 repo 的人都能取回它們。')
    flags['hist_data'] = len(hist)
    flags['hist_gone'] = len(gone)

    # ---- 4 歷史中的大 blob ----
    hr('[4] 歷史中最大的 blob')
    objs = git(['rev-list', '--objects', '--all'], repo, True)
    idx = {}
    for ln in objs:
        parts = ln.split(' ', 1)
        if len(parts) == 2:
            idx[parts[0]] = parts[1]
    big = []
    if idx:
        try:
            p = subprocess.Popen(
                ['git', 'cat-file', '--batch-check=%(objectname) '
                 '%(objecttype) %(objectsize)'],
                cwd=repo, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            out, _e = p.communicate(
                ('\n'.join(idx.keys()) + '\n').encode('utf-8'))
            for ln in out.decode('utf-8', 'replace').splitlines():
                f = ln.split()
                if len(f) == 3 and f[1] == 'blob':
                    big.append((int(f[2]), idx.get(f[0], '?')))
        except OSError:
            pass
    big.sort(reverse=True)
    over50 = sum(1 for s, _ in big if s > 50 * 1024 ** 2)
    over100 = sum(1 for s, _ in big if s > 100 * 1024 ** 2)
    say('  blob 總數 %d｜>50MB %d 個｜>100MB %d 個'
        % (len(big), over50, over100))
    for s, pth in big[:args.top]:
        say('    %-10s %s' % (human(s), pth[:50]))
    if over100:
        say('')
        say('  ⚠️ GitHub 拒收超過 100MB 的 blob，push 會直接失敗。')
    flags['over50'] = over50
    flags['over100'] = over100
    flags['repo_mb'] = round(sum(s for s, _ in big) / 1024.0 ** 2, 1)

    # ---- 5 疑似憑證 ----
    hr('[5] 疑似憑證（粗篩，被追蹤的文字檔）')
    hits = []
    me = os.path.realpath(__file__)
    skipped_self = False
    for f in files:
        if not lower_ends(f, TEXT_EXT):
            continue
        p = os.path.join(repo, f)
        # 這支程式自己就含那些樣式字串，會命中自己。略過，並說明。
        if os.path.exists(p) and os.path.realpath(p) == me:
            skipped_self = True
            continue
        if not os.path.exists(p) or os.path.getsize(p) > 2 * 1024 ** 2:
            continue
        try:
            with open(p, 'r', encoding='utf-8', errors='replace') as fh:
                for n, line in enumerate(fh, 1):
                    for k in SECRET_HINT:
                        if k in line:
                            hits.append((f, n, k))
                            break
        except (IOError, OSError):
            continue
    say('  命中 %d 處%s' % (len(hits),
                            '（已略過本程式自身）' if skipped_self else ''))
    for f, n, k in hits[:args.top]:
        say('    %s:%d  含「%s」' % (f, n, k))
    if hits:
        say('')
        say('  （粗篩會有誤判，請自行確認；真的有憑證就必須換掉，')
        say('    因為改 commit 不等於憑證失效。）')
    flags['secrets'] = len(hits)

    # ---- 6 .gitignore ----
    hr('[6] .gitignore')
    gi = os.path.join(repo, '.gitignore')
    if os.path.exists(gi):
        with open(gi, 'r', encoding='utf-8', errors='replace') as fh:
            pats = [l.strip() for l in fh if l.strip()
                    and not l.strip().startswith('#')]
        say('  有 .gitignore，%d 條規則' % len(pats))
        for need in ('data/', '*.csv', '*.npy', '.ipynb_checkpoints',
                     'results/', '__pycache__/'):
            say('    %-20s %s' % (need,
                                  '已含' if any(need.strip('/') in p
                                                for p in pats) else '缺'))
    else:
        say('  沒有 .gitignore。')
    flags['gitignore'] = int(os.path.exists(gi))

    # ---- 摘要 ----
    digest = ('DIGEST files=%d cur_data=%d cur_model=%d junk=%d | '
              'hist_data=%d hist_gone=%d | blob>50M=%d >100M=%d size=%sMB | '
              'secrets=%d gitignore=%d remotes=%d'
              % (len(files), flags['cur_data'], flags['cur_model'],
                 flags['cur_junk'], flags['hist_data'], flags['hist_gone'],
                 flags['over50'], flags['over100'], flags['repo_mb'],
                 flags['secrets'], flags['gitignore'], flags['remotes']))
    say('')
    say('=' * 64)
    print(digest, flush=True)

    if not args.quiet:
        say('')
        say('判讀：')
        say('  hist_data 或 hist_gone > 0 → 歷史裡有資料，投稿 repo 請')
        say('    「不帶歷史」重新開始（見 git 注意事項），不要只 rm 再 commit')
        say('  >100M > 0 → push 會被 GitHub 拒絕')
        say('  secrets > 0 → 先確認，真有憑證要換掉而非只改 commit')


if __name__ == '__main__':
    main()
