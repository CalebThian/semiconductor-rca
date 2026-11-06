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
/* 分組卡片：固定寬度 + 換行。字級不隨組數變化，30 組和 3 組一樣清楚。 */
.groups{display:flex;flex-wrap:wrap;align-items:flex-start;gap:12px;
 margin:12px 0}
.grp{margin:0;flex:0 1 234px;min-width:0;background:var(--card);
 border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.grp figcaption{font-size:12px;color:var(--mut);margin-bottom:6px}
.grp ul{list-style:none;margin:0;padding:0}
.grp li{display:flex;align-items:baseline;gap:8px;padding:5px 0 5px 9px;
 border-left:3px solid var(--c)}
.grp li+li{border-top:1px dashed var(--line)}
.grp li.anchor{border-left-width:5px;font-weight:600;padding-left:7px}
/* 感測器名稱常常很長，讓它換行而不是溢出或被截掉 */
.grp .nm{flex:1 1 auto;min-width:0;font-size:13px;overflow-wrap:anywhere}
.grp .tg{flex:0 0 auto;font-size:11px;color:var(--c);white-space:nowrap}
@media print{.grp{break-inside:avoid}}
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


def group_cards(groups):
    """分組結構：每組一張卡片，自動換行。

    原本是單一 SVG、每組一欄，寬度 = 組數 × 220 + 32，再由
    `svg{max-width:100%}` 壓回容器寬（1000px）。於是字級隨組數反比縮小：

        4 組 → 縮放 1.00、字 12.5px
        8 組 → 0.56、7.0px
       16 組 → 0.28、3.5px
       30 組 → 0.15、1.9px

    5 組以後就讀不了，而組數是資料決定的，不是版面能控制的。改成每組
    一張固定寬度的卡片、交給瀏覽器換行，字級就與組數無關 —— 多的是往下
    長，不是全部縮小。

    順帶解掉三件事：原本 H 取全體最大成員數，一個大組會讓每一欄都留白；
    參數名超過 180px 會溢出方框且無法截斷；SVG 文字不易選取複製，而這
    份報告是要拿參數名去 Ctrl+F 和貼到別處用的。
    """
    if not groups:
        return ''
    out = ['<div class="groups">']
    for g in groups:
        out.append('<figure class="grp">')
        out.append('<figcaption>第 %d 組 · %d 個</figcaption>'
                   % (g['index'], len(g['members'])))
        out.append('<ul>')
        for m in g['members']:
            col = TAG_COLOR.get(m['tag'], '#9a968c')
            out.append('<li%s style="--c:%s"><span class="nm">%s%s</span>'
                       '<span class="tg">%s</span></li>'
                       % (' class="anchor"' if m['tag'] == '錨點' else '',
                          col, esc(m['name']),
                          ' ★' if m.get('is_root') else '',
                          esc(m['tag'])))
        out.append('</ul></figure>')
    out.append('</div>')
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
        o.append('<h3>分組結構（%d 組）</h3>' % len(d['groups']))
        o.append('<div class="legend">%s</div>' % ''.join(
            '<span><i class="dot" style="background:%s"></i>%s</span>'
            % (v, esc(k)) for k, v in TAG_COLOR.items()))
        o.append(group_cards(d['groups']))

    # 排查順序
    if d.get('order'):
        chips = '<span class="s">→</span>'.join(
            '<span class="n">%s</span>' % esc(n) for n in d['order'])
        o.append('<h3>建議排查順序</h3><div class="ord">%s</div>' % chips)

    # 逐組序貫：有多變量分析時不呈現 —— 那是中間過程，報告要的是結果。
    # 但列舉跑不出來時（組合數過大而被擋、或走到節點上限）它是唯一有結果
    # 的那條路，拿掉就只剩一條規則、沒有任何支撐的數字。所以保留為後備，
    # 兩種情況下報告都不會是空的。
    if rule.get('path') and not rule.get('enum_top'):
        o.append('<h3>逐組序貫選擇</h3>')
        cols = ['步驟', '組', '選用參數', '條件', '增益', '採納']
        extra = [c for c in rule['path'][0]
                 if c.startswith('訓練') and c not in cols]
        cols = cols[:4] + extra + cols[4:]
        o.append('<table><tr>%s</tr>%s</table>' % (
            ''.join('<th class="l">%s</th>' % esc(c) for c in cols),
            ''.join('<tr class="%s">%s</tr>'
                    % ('hl' if str(r.get('採納', '')).startswith('是') else '',
                       ''.join('<td class="l">%s</td>'
                               % esc(fmt_cell(c, r.get(c, '')))
                               for c in cols))
                    for r in rule['path'])))
        o.append('<div class="note">數字皆為訓練集。增益未達門檻即停止 —— '
                 '合取條件只會使解釋比例下降、精確率上升。</div>')

    # 多變量分析：僅呈現全組合列舉的結果
    if rule.get('enum_top'):
        _mpg = int(rule.get('max_per_group', 1) or 1)
        o.append('<h3>多變量分析</h3>')
        # 這段說明原本寫「依分組順序加入條件」，那是可跳過組之前的行為：
        # 當時每組都必須出一個參數，短規則只能是「第 1..k 組」的前綴。
        # 現在每組多一個「不選」，任意組的子集都在範圍內，所以描述要改。
        _same = ('同組配對也在搜尋範圍內。' if _mpg > 1
                 else '每組最多取一個參數。')
        o.append('<div class="note">窮舉的範圍是「每組至多取%s參數」的'
                 '所有子集 —— 可以只用其中幾組，不必每組都出一個。%s%s</div>'
                 % ('一個' if _mpg <= 1 else '%d個' % _mpg, _same,
                    '增益不足即停止往下加，截斷後相同的規則會合併。'
                    if rule.get('enum_truncate', True)
                    else '不套用增益停止，一律評估到底。'))
        o.append('<div class="note">下表每個條件數各列最佳的一筆，'
                 '用來比較「多一個條件換到什麼」—— 通常是誤報率下降、'
                 '解釋比例也跟著下降。</div>')
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
        if any(str(r.get('條件數', '')).endswith('*')
               for r in rule['enum_top']):
            o.append('<div class="note">條件數標 * 者為保底補上的多參數'
                     '組合：截斷後整張表只剩單一條件時，忽略增益門檻'
                     '往下延伸產生，用來顯示多加一個條件的代價。'
                     '只作對照，不參與最終採用。</div>')
        o.append('<div class="note">「訓練F1」用於挑選，'
                 '「驗證」各欄為效能估計，兩者取自不同的資料切分。</div>')

    # 最終規則與涵蓋率。改用 final_rule／rule['test'] —— 原本綁在
    # enum_rule 上，所以列舉沒跑時這一整段（規則、涵蓋率長條、隨機對照）
    # 都不會出現，即使序貫已經給出一條規則。
    if rule.get('final_rule'):
        src = rule.get('final_source', '')
        o.append('<h3>最終規則%s</h3>'
                 % ('（%s）' % esc(src) if src else ''))
        o.append('<div class="rule">當 <span class="mono">%s</span> 時，'
                 '判定為異常</div>' % esc(rule['final_rule']))
        cov = rule.get('test', {}) or {}
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