# -*- coding: utf-8 -*-
"""讓終端輸出在 cp950 主控台不會中斷。

Windows 繁體中文的主控台編碼是 cp950，編不了 emoji。實測：

    print('✅ ...')
    UnicodeEncodeError: 'cp950' codec can't encode character '\\u2705'

訊息本身是附帶說明，不該有能力終止一支已經跑了十分鐘的訓練。

為什麼不直接把符號改成 ASCII：有些符號是寫進 HTML 報告的，那個檔以
UTF-8 寫出，符號在瀏覽器裡正常。同一個字元在兩個地方的命運不同，逐字串
改會改錯邊，而且以後任何人多加一個符號又會再踩一次。所以改在輸出端：

  - UTF-8 環境：完全不包裝，行為不變。
  - 其他編碼：已知符號換成 ASCII（保留語意），其餘編不了的字元以 ?
    代替而不丟例外。

只影響 sys.stdout / sys.stderr。寫檔的地方各自指定 encoding，不經過這裡。

用法（放在 import 之後）：

    try:                      # cp950 主控台編不了 emoji
        import _console
        _console.safe_console()
    except Exception:
        pass

包在 try 裡是因為同事的整合版可能沒有這個檔 —— 少了它只是少一層保護，
不該讓流程起不來。
"""

from __future__ import print_function

import sys

# 換成 ASCII 而不是刪掉：這些符號標示的是成敗，刪掉等於把訊號丟了
SUBS = {
    '✅': '[OK]',    # ✅
    '✓': 'v',       # ✓
    '✔': 'v',       # ✔
    '✗': 'x',       # ✗
    '✘': 'x',       # ✘
    '❌': '[X]',     # ❌
    '⚠': '[!]',     # ⚠
    '️': '',        # 變體選擇符，常跟在 ⚠ 後面
    '⭐': '*',       # ⭐
    '★': '*',       # ★
    '☆': '*',       # ☆
    '✓': 'v',
    '█': '#',       # █ 長條圖
    '░': '.',       # ░
}


class _Safe(object):
    """包一層的 stdout/stderr：先換符號，再把編不了的字元換成 ?。"""

    def __init__(self, stream):
        self._s = stream

    def write(self, t):
        if isinstance(t, str):
            for k, v in SUBS.items():
                if k in t:
                    t = t.replace(k, v)
            try:
                return self._s.write(t)
            except UnicodeEncodeError:
                enc = getattr(self._s, 'encoding', None) or 'ascii'
                return self._s.write(
                    t.encode(enc, 'replace').decode(enc, 'replace'))
        return self._s.write(t)

    def __getattr__(self, name):
        return getattr(self._s, name)


def _needs_wrap(stream):
    enc = (getattr(stream, 'encoding', '') or '').lower().replace('-', '')
    if enc in ('utf8', 'utf'):
        return False
    for ch in SUBS:
        if not ch:
            continue
        try:
            ch.encode(enc or 'ascii')
        except (UnicodeEncodeError, LookupError):
            return True
    return False


def safe_console():
    """需要時才包裝；重複呼叫不會疊加。"""
    for name in ('stdout', 'stderr'):
        st = getattr(sys, name, None)
        if st is None or isinstance(st, _Safe):
            continue
        try:
            if _needs_wrap(st):
                setattr(sys, name, _Safe(st))
        except Exception:
            pass


if __name__ == '__main__':
    # 自測：強制以 cp950 輸出，確認不會中斷
    import io
    buf = io.TextIOWrapper(io.BytesIO(), encoding='cp950', errors='strict')
    sys.stdout = buf
    safe_console()
    print('✅ 成功　❌ 失敗　⚠️ 警告　✓ 通過　✗ 不通過　★ 真根因　█████')
    sys.stdout.flush()
    sys.stdout = sys.__stdout__
    print('cp950 輸出：%s'
          % buf.buffer.getvalue().decode('cp950').strip())
    print('沒有丟 UnicodeEncodeError → 通過')
