# -*- coding: utf-8 -*-
r"""
make_html_report.py  ——  把 pipeline 結果渲染成自足式 HTML 報告

用途
----
run_pipeline.py 會輸出 *_result.json。本腳本把它渲染成單一 HTML 檔，
同仁雙擊即可開啟，不需安裝任何東西、不需連網。

  python make_html_report.py --json results/case1_pipeline_result.json \
      --out results/case1_report.html

可一次合併多個 case：
  python make_html_report.py --json a.json b.json c.json --out all.html

產出內容
--------
  1. 摘要卡片：資料規模、候選數、最終規則的解釋率與誤報率
     候選表直接呈現上游模型提供的欄位，不另加工
  2. 分組結構圖（錨點／上游／下游／間接關聯）
  3. 排查順序
  4. 多變量分析（全組合列舉的規則比較）
  5. 最佳規則與涵蓋率視覺化

所有樣式與圖形皆內嵌，無外部相依。

環境：Python 3.8+（僅用標準庫）
"""

import argparse
import html
import json
import os
import sys

TAG_COLOR = {
    '錨點': '#2a78d6',
    '上游': '#7c4dbd',
    '下游': '#1baf7a',
    '間接關聯': '#9a968c',
}

CSS = """
:root{--bg:#fbfaf7;--fg:#1f1e1c;--mut:#6b6862;--line:#e3e1da;--card:#fff}
@media(prefers-color-scheme:dark){:root{--bg:#1a1917;--fg:#eceae4;--mut:#9a968c;--line:#333029;--card:#232120}}
*{box-sizing:border-box}
body{margin:0;padding:28px 20px 60px;background:var(--bg);color:var(--fg);
 font-family:-apple-system,"Segoe UI","Noto Sans TC","Microsoft JhengHei",sans-serif;
 line-height:1.65;font-size:15px}
.wrap{max-width:1000px;margin:0 auto}
h1{font-size:24px;margin:0 0 4px}
h2{font-size:18px;margin:36px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:15px;margin:20px 0 8px;color:var(--mut);font-weight:600}
.sub{color:var(--mut);font-size:13px;margin-bottom:24px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}
.card{flex:1 1 150px;background:var(--card);border:1px solid var(--line);
 border-radius:10px;padding:14px 16px}
.card .k{font-size:12px;color:var(--mut)}
.card .v{font-size:22px;font-weight:600;margin-top:2px}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:8px 0}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child,th.l,td.l{text-align:left}
th{color:var(--mut);font-weight:600;font-size:12.5px}
tr.hl td{background:rgba(42,120,214,.07)}
.chip{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;
 color:#fff;white-space:nowrap}
.rule{background:var(--card);border:1px solid var(--line);border-left:3px solid #2a78d6;
 border-radius:8px;padding:12px 16px;margin:10px 0;font-size:15px}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13.5px}
.ord{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:10px 0}
.ord .s{color:var(--mut)}
.ord .n{background:var(--card);border:1px solid var(--line);border-radius:6px;
 padding:3px 10px;font-size:13.5px}
.note{color:var(--mut);font-size:12.5px;margin-top:6px}
.grp{background:var(--card);border:1px solid var(--line);border-radius:10px;
 padding:12px 16px;margin:10px 0}
.grp .m{margin:3px 0 3px 8px}
.bar{display:flex;height:22px;border-radius:5px;overflow:hidden;margin:8px 0;
 border:1px solid var(--line)}
.bar div{display:flex;align-items:center;justify-content:center;color:#fff;
 font-size:11.5px;white-space:nowrap;overflow:hidden}
svg{max-width:100%;height:auto}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:var(--mut);margin:6px 0}
.legend span{display:flex;align-items:center;gap:5px}
.dot{width:10px;height:10px;border-radius:2px;display:inline-block}
@media print{body{padding:0}h2{page-break-after:avoid}}
"""


def esc(x):
    return html.escape(str(x))


def chip(text, color):
    return '<span class="chip" style="background:%s">%s</span>' % (
        color, esc(text))


DEC2_COLS = {'granger_score'}          # 固定兩位小數的欄位


def fmt_cell(col, v):
    """依欄位與型別決定顯示方式。"""
    if v is None or (isinstance(v, float) and v != v):
        return ''
    key = str(col).strip().lower()
    if isinstance(v, bool):
        return 'TRUE' if v else 'FALSE'
    sv = str(v).strip()
    if sv.upper() in ('TRUE', 'FALSE'):
        return sv.upper()
    try:
        f = float(v)
    except (TypeError, ValueError):
        return sv
    if key in DEC2_COLS:
        return '%.2f' % f
    if f == int(f) and abs(f) < 1e15:
        return '%d' % int(f)
    return '%.4g' % f


def candidate_table(ct, roots):
    """直接呈現上游模型提供的 csv 欄位，不另加工。"""
    cols = ct.get('columns', [])
    sc = ct.get('sensor_col')
    head = ''.join('<th class="l">%s</th>' % esc(c) if c == sc
                   else '<th>%s</th>' % esc(c) for c in cols)
    body = []
    for r in ct.get('rows', []):
        nm = str(r.get(sc, '')).strip()
        cells = []
        for c in cols:
            txt = fmt_cell(c, r.get(c))
            if c == sc:
                cells.append('<td class="l">%s%s</td>'
                             % (esc(txt), ' \u2605' if nm in roots else ''))
            elif txt in ('TRUE', 'FALSE'):
                cells.append('<td>%s</td>' % chip(
                    txt, '#1baf7a' if txt == 'TRUE' else '#9a968c'))
            else:
                cells.append('<td>%s</td>' % esc(txt))
        body.append('<tr class="%s">%s</tr>'
                    % ('hl' if nm in roots else '', ''.join(cells)))
    return ('<table><tr>%s</tr>%s</table>'
            % (head, '\n'.join(body)))


def group_diagram(groups):
    """分組結構圖（SVG）：每組一欄，錨點在上、其餘依標籤排列。"""
    if not groups:
        return ''
    colw, rowh, pad = 220, 46, 16
    ncol = len(groups)
    maxm = max(len(g['members']) for g in groups)
    W = max(ncol * colw + pad * 2, 420)
    H = maxm * rowh + 60
    out = ['<svg viewBox="0 0 %d %d" role="img" aria-label="分組結構">' % (W, H)]
    for gi, g in enumerate(groups):
        cx = pad + gi * colw + colw / 2
        out.append('<text x="%.1f" y="20" text-anchor="middle" font-size="12.5" '
                   'fill="currentColor" opacity=".65">第 %d 組</text>'
                   % (cx, g['index']))
        prev_y = None
        for mi, m in enumerate(g['members']):
            y = 34 + mi * rowh
            col = TAG_COLOR.get(m['tag'], '#9a968c')
            if prev_y is not None:
                out.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" '
                           'stroke="%s" stroke-width="1.2" opacity=".5"/>'
                           % (cx, prev_y + 26, cx, y, col))
            lab = esc(m['name']) + (' ★' if m.get('is_root') else '')
            out.append('<rect x="%.1f" y="%d" width="180" height="26" rx="6" '
                       'fill="none" stroke="%s" stroke-width="%s"/>'
                       % (cx - 90, y, col, '2' if m['tag'] == '錨點' else '1'))
            out.append('<text x="%.1f" y="%d" text-anchor="middle" '
                       'font-size="12.5" fill="currentColor">%s</text>'
                       % (cx, y + 17, lab))
            out.append('<text x="%.1f" y="%d" font-size="11" fill="%s">%s</text>'
                       % (cx + 94, y + 17, col, esc(m['tag'])))
            prev_y = y
    out.append('</svg>')
    return '\n'.join(out)


def coverage_bar(rule):
    """規則涵蓋率：解釋／未解釋／誤報。"""
    t = rule.get('test', {})
    ex = int(t.get('解釋數', 0))
    un = int(t.get('未解釋數', 0))
    fp = int(t.get('誤報數', 0))
    tot = max(ex + un + fp, 1)
    parts = [('解釋 %d' % ex, ex, '#1baf7a'),
             ('未解釋 %d' % un, un, '#e05252'),
             ('誤報 %d' % fp, fp, '#eda100')]
    segs = ''.join(
        '<div style="width:%.2f%%;background:%s">%s</div>'
        % (100.0 * n / tot, c, esc(lab) if 100.0 * n / tot > 9 else '')
        for lab, n, c in parts if n > 0)
    return '<div class="bar">%s</div>' % segs


def render_case(d):
    o = []
    data = d.get('data', {})
    rule = d.get('rule', {}) or {}
    t = rule.get('test', {})

    o.append('<h2>%s</h2>' % esc(d.get('case', 'case')))

    # 摘要卡片
    cards = [('wafer 數', '%d' % data.get('n_wafer', 0),
              '正常 %d／異常 %d' % (data.get('n_normal', 0),
                                  data.get('n_abnormal', 0))),
             ('參數總數', '%d' % data.get('n_param', 0), ''),
             ('候選數', '%d' % len(d.get('candidates', [])), '')]
    if t:
        cards.append(('解釋比例', '%.0f%%' % (100 * float(t.get('解釋比例', 0))),
                      '驗證集'))
        cards.append(('誤報率', '%.1f%%' % (100 * float(t.get('誤報率', 0))),
                      'F1 %.3f' % float(t.get('F1', 0))))
    o.append('<div class="cards">%s</div>' % ''.join(
        '<div class="card"><div class="k">%s</div><div class="v">%s</div>'
        '<div class="k">%s</div></div>' % (esc(k), esc(v), esc(s))
        for k, v, s in cards))

    # 候選表：直接呈現上游模型的輸出
    roots = set(d.get('true_roots', []))
    ct = d.get('candidate_table')
    o.append('<h3>候選參數</h3>')
    if ct:
        o.append(candidate_table(ct, roots))
    else:
        o.append('<table><tr><th class="l">順序</th>'
                 '<th class="l">參數</th></tr>%s</table>'
                 % ''.join('<tr class="%s"><td class="l">%d</td>'
                           '<td class="l">%s%s</td></tr>'
                           % ('hl' if c.get('is_root') else '', c['order'],
                              esc(c['name']),
                              ' \u2605' if c.get('is_root') else '')
                           for c in d.get('candidates', [])))
    if roots:
        o.append('<div class="note">\u2605 為已知真根因</div>')

    # 分組
    if d.get('groups'):
        o.append('<h3>分組結構</h3>')
        o.append('<div class="legend">%s</div>' % ''.join(
            '<span><i class="dot" style="background:%s"></i>%s</span>'
            % (v, esc(k)) for k, v in TAG_COLOR.items()))
        o.append(group_diagram(d['groups']))

    # 排查順序
    if d.get('order'):
        chips = '<span class="s">→</span>'.join(
            '<span class="n">%s</span>' % esc(n) for n in d['order'])
        o.append('<h3>建議排查順序</h3><div class="ord">%s</div>' % chips)

    # 多變量分析：僅呈現全組合列舉的結果
    if rule.get('enum_top'):
        _mpg = int(rule.get('max_per_group', 1) or 1)
        o.append('<h3>多變量分析</h3>')
        if rule.get('enum_truncate', True):
            o.append('<div class="note">窮舉每組取%s參數的所有組合，'
                     '每個組合再依分組順序加入條件、增益不足即截斷，'
                     '截斷後相同的規則會合併。'
                     '「條件數」為該規則實際採納的條件數 —— 數字小於組數，'
                     '表示後面幾組的參數因增益不足被截斷。%s</div>'
                     % ('一個' if _mpg <= 1 else '1~%d個' % _mpg,
                        '同組配對也在搜尋範圍內。' if _mpg > 1 else
                        '同組配對不在搜尋範圍內。'))
        else:
            o.append('<div class="note">窮舉每組取%s參數的所有組合，'
                     '不套用增益停止，一律評估完整長度的規則。'
                     '條件過多而效能不佳的組合會自然沉到表格下方。%s</div>'
                     % ('一個' if _mpg <= 1 else '1~%d個' % _mpg,
                        '同組配對也在搜尋範圍內。' if _mpg > 1 else
                        '同組配對不在搜尋範圍內。'))
        cols = list(rule['enum_top'][0].keys())
        o.append('<table><tr>%s</tr>%s</table>'
                 % (''.join('<th class="l">%s</th>' % esc(c) if c in
                            ('組合', '條件數') else '<th>%s</th>' % esc(c)
                            for c in cols),
                    ''.join('<tr class="%s">%s</tr>'
                            % ('hl' if i == 0 else '',
                               ''.join('<td class="l">%s</td>' % esc(r[c])
                                       if c in ('組合', '條件數')
                                       else '<td>%s</td>' % esc(r[c])
                                       for c in cols))
                            for i, r in enumerate(rule['enum_top']))))
        o.append('<div class="note">「訓練F1」用於挑選，'
                 '「驗證」各欄為效能估計，兩者取自不同的資料切分。</div>')

    # 最佳規則（表中第一列）與涵蓋率
    if rule.get('enum_rule'):
        o.append('<h3>最佳規則</h3>')
        o.append('<div class="rule">當 <span class="mono">%s</span> 時，'
                 '判定為異常</div>' % esc(rule['enum_rule']))
        en = rule.get('enum', {})
        top0 = rule['enum_top'][0] if rule.get('enum_top') else {}
        cov = {'解釋數': top0.get('驗證解釋數', 0),
               '未解釋數': top0.get('驗證未解釋數', 0),
               '誤報數': top0.get('驗證誤報數', 0),
               '解釋比例': top0.get('驗證解釋比例', 0),
               '誤報率': top0.get('驗證誤報率', 0),
               'F1': en.get('test', 0)}
        o.append(coverage_bar({'test': cov}))
        o.append('<div class="note">驗證集共 %d 片異常、%d 片正常；'
                 '解釋比例（Recall）%.1f%%，誤報率（FPR）%.1f%%，'
                 'F1 %.3f。</div>'
                 % (rule.get('n_test_abnormal', 0),
                    rule.get('n_test_normal', 0),
                    100 * float(cov['解釋比例'] or 0),
                    100 * float(cov['誤報率'] or 0),
                    float(cov['F1'] or 0)))
        rnd = rule.get('random')
        if rnd:
            o.append('<div class="note">隨機對照：隨機挑同樣數量的參數，'
                     '平均 F1 %.3f、95%% 分位 %.3f —— %s</div>'
                     % (rnd['mean'], rnd['p95'],
                        '本規則顯著優於隨機' if rnd['p'] < 0.05
                        else '本規則未顯著優於隨機，說服力有限'))

    return '\n'.join(o)


def main():
    ap = argparse.ArgumentParser(description='把 pipeline 結果渲染成 HTML 報告')
    ap.add_argument('--json', nargs='+', required=True,
                    help='run_pipeline 產出的 *_result.json（可多個）')
    ap.add_argument('--out', required=True, help='輸出的 HTML 路徑')
    ap.add_argument('--title', default='半導體異常根因分析報告')
    args = ap.parse_args()

    bodies = []
    for p in args.json:
        if not os.path.exists(p):
            print('⚠️ 跳過（找不到）：%s' % p)
            continue
        txt = None
        for enc in ('utf-8-sig', 'utf-8', 'cp950', 'latin-1'):
            try:
                with open(p, 'r', encoding=enc) as f:
                    txt = f.read()
                break
            except UnicodeDecodeError:
                continue
        if txt is None:
            print('⚠️ 無法解碼：%s' % p)
            continue
        bodies.append(render_case(json.loads(txt)))
    if not bodies:
        sys.exit('沒有可渲染的結果檔。')

    doc = ('<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width,initial-scale=1">'
           '<title>%s</title><style>%s</style></head><body><div class="wrap">'
           '<h1>%s</h1><div class="sub">共 %d 個案例｜本檔為自足式報告，'
           '不需連網或安裝任何軟體</div>%s</div></body></html>'
           % (esc(args.title), CSS, esc(args.title), len(bodies),
              '\n'.join(bodies)))

    d = os.path.dirname(os.path.abspath(args.out))
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(doc)
    print('已產生：%s（%d 個案例，%.0f KB）'
          % (args.out, len(bodies), len(doc) / 1024.0))


if __name__ == '__main__':
    main()