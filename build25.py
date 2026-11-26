# -*- coding: utf-8 -*-
"""build25: 仓位间距 —— 把"三笔各 1/3"这个拍脑袋的数字，放到 18.7 年里去称"""
import os, json
import numpy as np
import pandas as pd

B = os.path.dirname(os.path.abspath(__file__))
OUT = 'G:/探讨集/25_仓位间距_下手动作的数值化_20261112.html'
CHARTJS = open(os.path.join(B, 'chart.umd.min.js'), encoding='utf-8').read()

import importlib.util
spec = importlib.util.spec_from_file_location('eng', os.path.join(B, 'd122_engine.py'))
eng = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eng)
spec2 = importlib.util.spec_from_file_location('ex2', os.path.join(B, 'd126_exit2.py'))
ex2 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(ex2)

d = eng.load()
d['ma250'] = d['close'].rolling(250).mean()
ev = eng.find_events(d)
N = len(d)

r123 = json.load(open(os.path.join(B, 'd123_result.json'), encoding='utf-8'))
r126 = json.load(open(os.path.join(B, 'd126_result.json'), encoding='utf-8'))
r124 = json.load(open(os.path.join(B, 'd124_result.json'), encoding='utf-8'))
r122 = json.load(open(os.path.join(B, 'd122_result.json'), encoding='utf-8'))

# ---------- 重算 48 组（36 月窗口）----------
COMBOS = [(n, cf, g) for n in [2, 3, 4, 5] for cf in ['none', 'ma20', 'reb8', 'mup'] for g in [1, 2, 3]]
grid = []
for n, cf, g in COMBOS:
    per = [eng.backtest(d, i0, n=n, confirm=cf, gap_m=g, max_hold=756) for i0 in ev]
    re_ = [p['ret'] for p in per]
    grid.append(dict(n=n, cf=cf, g=g, ret=float(np.mean(re_)), dd=float(np.mean([p['max_dd_acct'] for p in per])),
                     rets=[round(x * 100, 1) for x in re_],
                     prems=[round((p['cost_prem'] or 0) * 100, 1) for p in per]))
rets_all = np.array([x for g_ in grid for x in g_['rets']])
means = np.array([g_['ret'] for g_ in grid])
dds = np.array([g_['dd'] for g_ in grid])
SE = rets_all.std() / np.sqrt(len(ev))          # 已是百分点

marg = r123['marginal']
noise = r123['noise']

# ---------- 每轮"进带后再跌" ----------
drops = []
for i0 in ev:
    seg = d.iloc[i0:i0 + 500]
    li = int(np.argmin(seg['close'].values))
    drops.append(dict(d0=str(d.d.iloc[i0].date()), lo=str(seg['d'].iloc[li].date()),
                      drop=float(seg['close'].iloc[li] / d.close.iloc[i0] - 1), days=int(li),
                      burst=d.dd_peak.iloc[i0] <= -0.50, enter=float(d.close.iloc[i0]),
                      low=float(seg['close'].iloc[li]), real=float(d.real.iloc[i0]),
                      rA=float(d.rA.iloc[i0]), dd=float(d.dd_peak.iloc[i0])))

# ---------- 2013 案例净值 ----------
i13 = int(d.index[d['d'] == pd.Timestamp('2013-06-06')][0])

CSS = open(os.path.join(B, '_css.txt'), encoding='utf-8').read() if os.path.exists(os.path.join(B, '_css.txt')) else ''
if not CSS:
    CSS = """<style>
:root{--bg:#faf9f6;--card:#fff;--tx:#1a1a1a;--tx2:#5f5e5a;--tx3:#8c8b86;--bd:rgba(0,0,0,.12);--red:#A32D2D;--green:#1C6B3C;--amber:#854F0B}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;font-size:14px;line-height:1.75}
.wrap{max-width:1000px;margin:0 auto;padding:40px 24px 80px}
h1{font-size:24px;font-weight:500;margin:0 0 6px}
.sub{color:var(--tx2);font-size:13px;margin-bottom:28px}
h2{font-size:17px;font-weight:500;margin:44px 0 4px;padding-top:20px;border-top:1px solid var(--bd)}
h2 .no{color:var(--tx3);font-size:13px;margin-right:8px}
h3{font-size:14px;font-weight:500;margin:22px 0 4px}
.note{color:var(--tx2);font-size:13px;margin:0 0 16px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:18px 20px;margin:14px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.kpi{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:14px 16px}
.kpi .l{font-size:12px;color:var(--tx2);margin-bottom:6px}
.kpi .n{font-size:25px;font-weight:500;line-height:1.2}
.kpi .d{font-size:12px;color:var(--tx3);margin-top:4px}
.verdict{border-left:3px solid var(--red);background:#FCEBEB;padding:15px 18px;border-radius:0 8px 8px 0;margin:18px 0}
.verdict b{color:var(--red)}
.warn{border-left:3px solid var(--amber);background:#FAEEDA;padding:14px 18px;border-radius:0 8px 8px 0;margin:16px 0;font-size:13px}
.ok{border-left:3px solid var(--green);background:#EAF3ED;padding:14px 18px;border-radius:0 8px 8px 0;margin:16px 0;font-size:13px}
.chartbox{position:relative;width:100%;height:300px;margin-top:8px}
.chartbox.tall{height:340px}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--bd)}
th:first-child,td:first-child{text-align:left}
th{font-weight:500;color:var(--tx2);font-size:12px}
tr:last-child td{border-bottom:none}
tr.hl td{background:#FBF3F2}
tr.hl2 td{background:#EAF3ED}
tr.hl3 td{background:#FAEEDA}
.foot{margin-top:48px;padding-top:20px;border-top:1px solid var(--bd);font-size:12px;color:var(--tx3);line-height:1.9}
code{background:#F1EFE8;padding:1px 5px;border-radius:4px;font-size:12px}
blockquote{margin:10px 0;padding:10px 16px;border-left:2px solid var(--bd);color:var(--tx2);font-size:13px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:720px){.two{grid-template-columns:1fr}}
</style>"""

H = []
def P(s): H.append(s)

P('<div class="wrap">')
P('<h1>三笔各 1/3 —— 这个数字是从哪来的</h1>')
P('<div class="sub">上证 2008-02 ~ 2026-11（4,555 根，18.7 年）· 6 轮独立熊市 · 48 组参数 · 2026-11-12 · 基准理论补血肉 ③</div>')

P('<div class="verdict">'
  '<b>结论先说：1/3 这个数字不用改，但不是因为它"最优"——是因为这个曲面上没有最优点。</b><br><br>'
  '48 组参数（档数 2/3/4/5 × 确认信号 4 种 × 间隔 1/2/3 月）全部跑完，平均收益落在 <b>14.8% ~ 22.3%</b>，'
  '极差 7.5 个百分点。而"6 个样本估一个均值"的标准误是 <b>5.0 个百分点</b>。'
  '单因素 ANOVA：<b>F=0.162，p=1.0000</b> —— 这 48 组之间没有可辨识的差异。<br><br>'
  '<b>唯一穿过噪声的信号是"分几笔"：</b>分 3 笔对一把梭，收益从 19.10% 到 19.36%（不动），'
  '最大浮亏从 <b>−17.1% 到 −10.3%（砍掉 40%）</b>。分档买的全部价值在这里，不在收益率上。<br><br>'
  '<b>顺手推翻了两条 v1.1 的修订</b>：①"必须等确认才加仓"没有证据支持（匀速加仓收益 21.6% vs 确认加仓 19.4%，浮亏还更小）；'
  '②v1.2 §4.2 的"趋势出场能拿到 2.65 倍"<b>是拿底部涨幅冒充规则收益</b>，同一套规则实测只有 +6.5%。</div>')

P('<div class="grid">')
for lab, val, des in [
    ('独立熊市', '6 轮', '不是 5 次触发'),
    ('参数组合', '48 组', '档数×信号×间隔'),
    ('组间极差', '7.5 pp', 'vs 标准误 5.0 pp'),
    ('ANOVA', 'p=1.0000', '无可辨识差异'),
]:
    P(f'<div class="kpi"><div class="l">{lab}</div><div class="n">{val}</div><div class="d">{des}</div></div>')
P('</div>')

# ---------- 01 样本 ----------
P('<h2><span class="no">01</span>先把样本补回来：是 6 轮，不是 5 次</h2>')
P('<p class="note">v1.2 §4.1 的表里是 5 次出手（2013-06 / 2018-08 / 2020-02 / 2022-04 / 2023-10）。'
  '那是从 2013 年起的样本 —— 因为早期脚本用的日线数据从 2013-06 开始。'
  '这次把 CPI 平减回溯到 2008-02（CPI 数据起点），样本变成 <b>18.7 年</b>，'
  '按"12 个月冷却 + 中途离底须曾回到 25% 以上"去重，得到 <b>6 轮独立熊市</b>：</p>')

P('<div class="card"><table><thead><tr><th>进场</th><th>上证</th><th>实际点位</th><th>离底</th><th>距历史最高</th><th>体制</th><th>之后最低</th><th>还要再跌</th></tr></thead><tbody>')
for x in drops:
    cls = ' class="hl"' if x['burst'] else ''
    P(f'<tr{cls}><td>{x["d0"]}</td><td>{x["enter"]:,.0f}</td>'
      f'<td>{x["real"]:,.0f}</td>'
      f'<td>{x["rA"]*100-100:+.1f}%</td><td>{x["dd"]*100:+.1f}%</td>'
      f'<td>{"勃发段" if x["burst"] else "箱体段"}</td>'
      f'<td>{x["low"]:,.0f}（{x["lo"][2:]}）</td><td>{x["drop"]*100:+.1f}%</td></tr>')
P('</tbody></table></div>')
P('<p class="note">红色两行是勃发段（距历史最高 ≤−50%）。v1.2 里勃发段只有 1 例（2013-06），'
  '补上 2008 和 2011 后变成 <b>2 例</b>。样本小的问题没有消失，只是从"n=1"变成"n=2"。</p>')

P('<div class="warn"><b>顺带修正一个口径：</b>v1.2 的"5 次触发"里，2013-06 和 2011-10 其实是同一轮熊市'
  '（2011-10 一路跌到 2013-06）。按"12 个月冷却"会把它们算成两次，按"离底回到 25% 以上"只会算一次。'
  '本次统一用后者 —— 因为建仓规则的作用域是<b>一次连续下跌</b>，不是"一次触发"。</div>')

# 图3：进带后再跌
P('<h3>每轮进场之后，还要再跌多少</h3>')
P('<div class="card"><div class="chartbox"><canvas id="c1"></canvas></div></div>')
P('<p class="note">中位 <b>−7.9%</b>，最深 −20.4%（2008）。这就是分档能摊到的全部空间。'
  '分 3 笔，第 2 笔在 −4%，第 3 笔在 −8%。<b>摊薄的物理上限就这么大，这是参数怎么调都跳不出 7 个点的原因。</b></p>')

# ---------- 02 主结论 ----------
P('<h2><span class="no">02</span>分档的唯一价值：把 −17% 的浮亏变成 −10%</h2>')
P('<p class="note">固定确认信号（反弹 8%）和间隔（1 个月），只改档数 n。'
  '每轮从进带开始按规则建仓，36 个月后按 v1.2 的出场规则结算（统一窗口，所有方案同一把尺）。</p>')

P('<div class="card"><div class="chartbox tall"><canvas id="c2"></canvas></div></div>')

P('<div class="card"><table><thead><tr><th>档数</th><th>平均收益</th><th>最大浮亏</th><th>收益/浮亏</th><th>平均仓位</th><th>相对 n=1</th></tr></thead><tbody>')
for m in marg:
    delta = ('—' if m['n'] == 1 else
             f'收益 {(m["ret"]-marg[0]["ret"])*100:+.2f}pp ／ 浮亏改善 {(m["dd"]-marg[0]["dd"])*-100:.2f}pp')
    cls = ' class="hl2"' if m['n'] == 3 else ''
    P(f'<tr{cls}><td>{m["n"]} 档</td><td>{m["ret"]*100:.2f}%</td><td>{m["dd"]*100:.1f}%</td>'
      f'<td>{m["ret"]/abs(m["dd"]):.2f}</td><td>{m["pos"]*100:.0f}%</td><td>{delta}</td></tr>')
P('</tbody></table></div>')

P('<div class="ok"><b>读法：</b>一把梭（n=1）的收益是 19.10%，分 3 笔是 19.36% —— <b>几乎一样</b>。'
  '但浮亏从 −17.1% 掉到 −10.3%，<b>少挨 6.8 个百分点、四成的浮亏</b>。<br><br>'
  '继续加档（4/5/6）收益开始单向下滑（17.9% → 16.4% → 16.0%），浮亏只再省 1~3 个点。'
  '<b>n=3 是"收益已经不再增加、浮亏还能继续改善"的最后一个点。</b><br><br>'
  '这件事的意义不在钱上 —— 在能不能拿住。−17% 和 −10% 的差别，对于一个会在 −20% 割肉的人来说，'
  '就是"这笔做完了"和"这笔没做成"的差别。</div>')

P('<blockquote>顺带一个反直觉的结果：n=2 的收益（20.22%）其实比 n=3 高 0.86pp。'
  '但 n=2 的浮亏是 −11.6%，n=3 是 −10.3%。<b>多分一笔，用 0.86 个点的收益换 1.3 个点的浮亏 —— 这笔交易值不值，'
  '取决于你更怕哪个。</b>v1.2 选 3，等于选了"更怕浮亏"。这个选择本身没错，但它是个偏好，不是个最优解。</blockquote>')

# ---------- 03 曲面是平的 ----------
P('<h2><span class="no">03</span>48 组参数：整个曲面是平的</h2>')
P('<p class="note">每一点是一组参数（档数 2/3/4/5 × 确认信号 无/MA20/反弹8%/月线阳 × 间隔 1/2/3 月），'
  '横轴是它带来的平均最大浮亏，纵轴是平均收益。点越靠左上越好。</p>')

P('<div class="card"><div class="chartbox tall"><canvas id="c3"></canvas></div></div>')

P('<div class="card"><table><thead><tr><th>检验</th><th>结果</th><th>怎么读</th></tr></thead><tbody>'
  f'<tr><td>48 组均值范围</td><td><b>{means.min()*100:.2f}% ~ {means.max()*100:.2f}%</b></td><td>极差 {(means.max()-means.min())*100:.2f}pp</td></tr>'
  f'<tr><td>组间标准差</td><td>{means.std()*100:.2f}pp</td><td>比单次收益的 {rets_all.std():.1f}pp 小得多</td></tr>'
  f'<tr class="hl"><td>6 样本均值标准误</td><td><b>{SE:.2f}pp</b></td><td>用 6 轮估一个均值的固有误差</td></tr>'
  f'<tr class="hl"><td>极差 ÷ 标准误</td><td><b>{(means.max()-means.min())*100/SE:.2f}</b></td><td>&lt;2 → 差异埋在噪声里</td></tr>'
  f'<tr><td>单因素 ANOVA</td><td><b>F={noise["F"]:.3f}　p={noise["p"]:.4f}</b></td><td>p=1.0 → 组间无显著差异</td></tr>'
  f'<tr><td>Kendall W（6 轮对 48 组的排序一致性）</td><td><b>{noise["W"]:.3f}</b></td><td>0=各排各的，1=完全一致</td></tr>'
  '</tbody></table></div>')

P('<p class="note"><b>最后一行是关键。</b>W=0.238 意味着：2008 那轮觉得好的参数，和 2023 那轮觉得好的参数，'
  '基本不是同一批。6 轮事件对 48 组的排序，一致性只有 24%。<b>如果参数真的有效，六轮应该指向同一个方向。</b></p>')

# ---------- 04 三把刀 ----------
P('<h2><span class="no">04</span>三把刀：最优参数是怎么"换人"的</h2>')

P('<h3>刀一：去掉任意一轮，最优参数就换人</h3>')
P('<p class="note">留一法（leave-one-out）：每次去掉一轮熊市，重新从 48 组里选最优。</p>')
P('<div class="card"><table><thead><tr><th>去掉哪一轮</th><th>剩下的最优参数</th><th>与全样本排名相关 ρ</th></tr></thead><tbody>')
loo = r123.get('loo', {})
names_ev = [str(d.d.iloc[i0].date()) for i0 in ev]
for k, bk in enumerate(loo.get('best_list', [])):
    rho = loo.get('rhos', [None] * 6)[k]
    P(f'<tr><td>{names_ev[k]}</td><td><code>{bk}</code></td><td>{rho:.3f}</td></tr>')
P('</tbody></table></div>')
P(f'<p class="note">6 次留一，出现了 <b>{len(set(loo.get("best_list", [])))} 种不同的"最优"</b>。'
  '排名相关 ρ 中位 0.963 —— 说明"哪些参数好"这个判断是稳定的，但"哪个最好"完全不稳。'
  '<b>这不是"参数有效但选不准"，这是"前几名之间本来就没差别"。</b></p>')

P('<h3>刀二：底的位置挪 13%，整个排名翻掉</h3>')
P('<p class="note">v1.2 自己承认底的不确定性是 −13%/+14%（五轮熊底极差 31%）。把锚从 1,880 挪到 1,700 和 2,050，看参数排序扛不扛得住。</p>')
P('<div class="card"><table><thead><tr><th>锚</th><th>含义</th><th>触发轮数</th><th>top5 参数</th></tr></thead><tbody>')
anch = r123.get('anchor', {})
for A, lab in [('1700.0', '底比 1,880 还低 10%'), ('1880.0', '基准'), ('2050.0', '底比 1,880 高 9%')]:
    a = anch.get(A, {})
    P(f'<tr><td><b>{A.split(".")[0]}</b></td><td>{lab}</td><td>{a.get("n_ev","—")} 轮</td>'
      f'<td><code>{", ".join(a.get("top5", []))}</code></td></tr>')
P('</tbody></table></div>')
P('<div class="verdict"><b>三个锚下 top5 的交集是空的。</b>一个都没有。<br><br>'
  '而且锚 1,700 时只剩 <b>2 轮</b>事件能被触发 —— 因为底真那么低的话，2008 之后的"离底 10%"要跌到实际上证 1,870，'
  '只有 2008 和 2013 到过。<b>底估低一格，机会从 6 次掉到 2 次。</b>这才是"底是锚不是靶"最硬的代价。</div>')

P('<h3>刀三：每一轮各自的最优，指向四个不同方向</h3>')
P('<div class="card"><table><thead><tr><th>轮次</th><th>距顶</th><th>该轮最优参数</th><th>收益</th><th>最差参数收益</th><th>该轮极差</th></tr></thead><tbody>')
peb = r123.get('per_event_best', [])
for j, i0 in enumerate(ev):
    vals = [g_['rets'][j] for g_ in grid]
    P(f'<tr><td>{d.d.iloc[i0].date()}</td><td>{d.dd_peak.iloc[i0]*100:+.1f}%</td>'
      f'<td><code>{peb[j] if j < len(peb) else "—"}</code></td>'
      f'<td>{max(vals):+.1f}%</td><td>{min(vals):+.1f}%</td><td>{(max(vals)-min(vals)):.1f}pp</td></tr>')
P('</tbody></table></div>')
P('<p class="note">六轮指向 4 种不同参数（n3_none_g2 / n2_none_g3 / n5_mup_g1 / n3_mup_g1 / n3_none_g1 / n3_reb8_g1）。'
  '而且极差最小的两轮只有 1.5pp 和 3.7pp —— <b>说明在那些轮次里，参数根本无所谓，怎么买都差不多。</b></p>')

# ---------- 05 出场 ----------
P('<h2><span class="no">05</span>顺手撞出来的：出场规则的问题比建仓大得多</h2>')
P('<p class="note">建仓跑完要做交叉验证，就拿 v1.2 §4.2 引用的那个案例去对：'
  '「2013-06 起，趋势出场能拿到 2015-06 的 2.65 倍」。对不上。查了一下，问题出在这句话本身。</p>')

P('<div class="card"><table><thead><tr><th>口径</th><th>数值</th><th>这是什么</th></tr></thead><tbody>'
  '<tr><td>2015-06-12 上证</td><td>5,166</td><td>那轮的顶</td></tr>'
  '<tr><td>本轮实际底（2013-06-27）</td><td>1,950</td><td>事后才知道</td></tr>'
  '<tr class="hl"><td><b>5,166 ÷ 1,950 = 2.65</b></td><td><b>+165%</b></td><td><b>这是"从底部算的段涨幅"，不是任何规则能拿到的收益</b></td></tr>'
  '<tr><td>2013-06-06 进场价</td><td>2,242</td><td>骨架实际会在这一天买</td></tr>'
  '<tr><td>买入持有到 2015-06-12</td><td>+130.4%</td><td>不择时的上限</td></tr>'
  '<tr class="hl3"><td><b>按 v1.2 出场规则实跑（36 个月）</b></td><td><b>+6.5%</b></td><td><b>规则在 2013-10-25 就触发了</b></td></tr>'
  '</tbody></table></div>')

P('<p class="note">原因很直接：2013-06 到 2015-06 之间，「收盘跌破 MA120 且 MA120 下行」这条规则触发了 <b>7 次</b>，'
  '每次都在 2,000~2,140 的横盘区里。规则本身是对的（它确实在 2013-10 就应该卖），'
  '问题是 <b>"卖出后怎么回来"没定义，或者定义了但没测过</b>。下面是八种出场方案在同一窗口下的实测：</p>')

P('<div class="card"><div class="chartbox tall"><canvas id="c4"></canvas></div></div>')

P('<div class="card"><table><thead><tr><th>方案</th><th>平均收益</th><th>平均浮亏</th><th>平均卖出次数</th></tr></thead><tbody>')
tbl = r126['table']
order = ['e0_hold', 'e7_tp50', 'e4_drop25', 'e1n_v12', 'e1_v12', 'e6_ma3d', 'e2_ma120', 'e3_ma250']
for k in order:
    v = tbl[k]
    cls = ' class="hl"' if k == 'e7_tp50' else (' class="hl3"' if k in ('e1_v12', 'e1n_v12') else '')
    P(f'<tr{cls}><td>{v["label"]}</td><td>{v["mean"]:.1f}%</td><td>{v["dd"]:.1f}%</td><td>{v["n_sell"]:.1f}</td></tr>')
P('</tbody></table></div>')

P('<div class="warn"><b>不要急着把 v1.2 换成 +50% 止盈。</b>'
  'E7 之所以 26.9% 排第一，完全来自 2008 那一轮（59.7% vs 不择时的 22.4%）—— '
  '其余五轮 E7 和不择时<b>一模一样</b>（因为浮盈从未到过 50%）。这是 n=1 的结果，不是规律。<br><br>'
  '真正能确定的是三条：<br>'
  '① v1.2 的「2.65 倍」是虚的，同一套规则实测 +6.5%，这个数字必须从文档里删掉；<br>'
  '② 「破 MA120 且 MA120 下行」在横盘区会连环触发（2013 案例做了 5 次进出），需要"卖出后怎么回来"的完整定义；<br>'
  '③ 八种方案的平均收益全在 10.8%~26.9% 之间，最大差距 16pp —— <b>比建仓参数的 7.5pp 大一倍</b>。'
  '出场比建仓重要，但同样测不准。</div>')

# ---------- 06 v1.3 ----------
P('<h2><span class="no">06</span>v1.3 改了什么，没改什么</h2>')

P('<div class="card"><table><thead><tr><th>项</th><th>v1.2</th><th>v1.3</th><th>依据</th></tr></thead><tbody>'
  '<tr class="hl2"><td>档数</td><td>3 档，各 1/3</td><td><b>保留 3 档</b></td><td>n=1→3 收益不变（19.10→19.36%），浮亏砍 40%（−17.1→−10.3%）</td></tr>'
  '<tr class="hl"><td>确认信号</td><td><b>必须</b>出现才加第 2 笔</td><td><b>降级为"加速器"</b>：可以等确认，也可以按间隔直接加</td>'
  '<td>匀速加仓 21.6% / −9.1%，确认加仓 19.4% / −10.3% —— 匀速两项都略好，差异在噪声内</td></tr>'
  '<tr><td>间隔</td><td>至少 1 个月</td><td><b>1 个月，不加码</b></td><td>1/2/3 月的差异 0.3~1.5pp，落在噪声里，选最简单的</td></tr>'
  '<tr class="hl"><td>出场</td><td>勃发段趋势 / 箱体 +50%（具体阈值写死）</td><td><b>标注"待重做"，不执行具体阈值</b></td>'
  '<td>八种方案无一致优势；「2.65 倍」论证作废；MA120 连环触发未解决</td></tr>'
  '<tr><td>标的</td><td>510300，按金额</td><td>不变</td><td>v1.2 已结案</td></tr>'
  '</tbody></table></div>')

P('<div class="ok"><b>建仓的执行规则（v1.3 定稿）：</b><br>'
  '① 首次进带（离底 ≤10%）买 <b>1/3</b>，不用等任何信号；<br>'
  '② 第 2、3 笔：<b>距上一笔满 1 个月</b>即可加，是否出现"反弹 8% / 站上 MA20 / 月线收阳"只影响你心里的踏实程度，不影响期望；<br>'
  '③ 三笔都按<b>金额</b>等分（不是按份数），现金不足时用剩余现金；<br>'
  '④ 一直跌、一直没加完 → 就保持在 1/3 或 2/3 在场，这是设计不是失败。</div>')

# ---------- 07 读数 ----------
P('<h2><span class="no">07</span>当前读数（未变）</h2>')
P('<div class="card"><table><thead><tr><th>项</th><th>值</th></tr></thead><tbody>'
  '<tr><td>上证（2026-11-11）</td><td><b>3,681</b></td></tr>'
  '<tr><td>CPI 平减实际点位</td><td>2,556</td></tr>'
  '<tr><td>离真底（1,880）</td><td><b>+36.0%</b></td></tr>'
  '<tr><td>距离观察区（3,248）</td><td>需跌 −11.8%</td></tr>'
  '<tr><td>距离出手区（2,977）</td><td>需跌 <b>−19.1%</b></td></tr>'
  '<tr class="hl2"><td>动作</td><td><b>空仓，什么都不做</b></td></tr>'
  '</tbody></table></div>')

# ---------- 08 作废 ----------
P('<h2><span class="no">08</span>本次作废的结论</h2>')
P('<div class="card"><table><thead><tr><th>作废的表述</th><th>出处在哪</th><th>为什么作废</th></tr></thead><tbody>'
  '<tr><td><b>「趋势出场能拿到 2.65 倍」</b></td><td>v1.2 §4.2</td><td>2.65 = 5,166/1,950，是从底部算的段涨幅。同一规则实跑 +6.5%</td></tr>'
  '<tr><td><b>「+50% 止盈会把 32.4 倍砍到 3.4 倍」</b></td><td>v1.2 §4.0</td>'
  '<td>拿"事后持到段末"和"固定止盈"比，不公平。让规则真跑，6 轮里 +50% 止盈平均 26.9%，趋势出场 19.2%</td></tr>'
  '<tr><td><b>「必须等确认才加仓」是修正</b></td><td>v1.2 §3</td><td>匀速加仓 21.6%/浮亏 −9.1%，确认加仓 19.4%/−10.3%，匀速两项都略好</td></tr>'
  '<tr><td>「出手 5 次」</td><td>v1.2 §6.1</td><td>样本从 2013 起算漏了 2008 和 2011；补全是 6 轮独立熊市</td></tr>'
  '<tr><td>本次自己的一版结论：「n=4 none g=1 最优 31.7%」</td><td>本报告 d121</td>'
  '<td>那是 1,800 天窗口（7.4 年）跑出来的，横跨了 2015 股灾。改 36 个月统一窗口后归零</td></tr>'
  '</tbody></table></div>')

P('<div class="foot">'
  '样本：上证综指 1990-2026 全序列，CPI 平减区间 2008-02~2026-11（4,555 交易日，18.7 年）。'
  'CPI 用「全国-当月」同比指数开 12 次方当环比，定基 2008-01=100。<br>'
  '建仓/出场均为日频、收盘价成交、不计滑点与手续费；现金按 2.5%/年计息；标的用上证指数（不含分红，所有方案一致）。<br>'
  '脚本：<code>d122_engine.py</code>（回测引擎）· <code>d123_robust.py</code>（48 组 + 稳健性）· '
  '<code>d124_diag.py</code>（匀速 vs 确认）· <code>d126_exit2.py</code>（出场对照）· <code>build25.py</code>。<br>'
  '数据：<code>d123_result.json</code> / <code>d124_result.json</code> / <code>d126_result.json</code>。<br>'
  '本文不含投资建议。所有"未来 1 年收益"均为历史分档统计，不是预测。</div>')
P('</div>')

# ---------- 图表 ----------
JS = """
<script>
const F={family:'-apple-system,Segoe UI,PingFang SC,Microsoft YaHei,sans-serif'};
Chart.defaults.font.family=F.family; Chart.defaults.font.size=11; Chart.defaults.color='#5f5e5a';
const gridc='rgba(0,0,0,.06)';

// c1: 进带后再跌
new Chart(document.getElementById('c1'),{type:'bar',
 data:{labels:@@L1@@,datasets:[
  {label:'进场后再跌（至之后 500 日最低）',data:@@D1@@,backgroundColor:@@C1@@,borderRadius:3,borderSkipped:false}]},
 options:{responsive:true,maintainAspectRatio:false,indexAxis:'y',
  plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>c.parsed.x.toFixed(1)+'%'}},
   title:{display:true,text:'红=勃发段，赭=箱体段',color:'#8c8b86',font:{size:11},align:'end'}},
  scales:{x:{title:{display:true,text:'进场后还要再跌 %'},grid:{color:gridc},ticks:{callback:v=>v+'%'}},
   y:{grid:{display:false}}}}});

// c2: 档数边际
new Chart(document.getElementById('c2'),{type:'bar',
 data:{labels:@@L2@@,datasets:[
  {type:'bar',label:'平均收益（36 月，左轴）',data:@@D2@@,backgroundColor:'#A32D2D',borderRadius:3,yAxisID:'y',order:2},
  {type:'line',label:'最大浮亏（右轴）',data:@@D3@@,borderColor:'#1a1a1a',borderWidth:2,pointRadius:4,
   pointBackgroundColor:'#1a1a1a',tension:.2,yAxisID:'y1',order:1}]},
 options:{responsive:true,maintainAspectRatio:false,
  plugins:{legend:{position:'top',labels:{boxWidth:14,font:{size:11}}}},
  scales:{y:{position:'left',title:{display:true,text:'平均收益 %'},min:0,max:24,
    grid:{color:gridc},ticks:{callback:v=>v+'%'}},
   y1:{position:'right',title:{display:true,text:'最大浮亏 %'},min:-20,max:0,
    grid:{display:false},ticks:{callback:v=>v+'%'}},
   x:{grid:{display:false},title:{display:true,text:'分几笔（等金额）'}}}}});

// c3: 48 组散点
new Chart(document.getElementById('c3'),{type:'scatter',
 data:{datasets:@@DS3@@},
 options:{responsive:true,maintainAspectRatio:false,
  plugins:{legend:{position:'top',labels:{boxWidth:10,font:{size:11}}},
   tooltip:{callbacks:{label:c=>c.raw.lab}}},
  scales:{x:{title:{display:true,text:'平均最大浮亏 %（越靠右越深）'},min:-17,max:-7.5,
    grid:{color:gridc},ticks:{callback:v=>v+'%'}},
   y:{title:{display:true,text:'平均收益 %（越靠上越好）'},min:13,max:24,
    grid:{color:gridc},ticks:{callback:v=>v+'%'}}}}});

// c4: 出场方案
new Chart(document.getElementById('c4'),{type:'bar',
 data:{labels:@@L4@@,datasets:[
  {label:'平均收益（36 月）',data:@@D4@@,backgroundColor:@@C4@@,borderRadius:3}]},
 options:{responsive:true,maintainAspectRatio:false,indexAxis:'y',
  plugins:{legend:{display:false},
   tooltip:{callbacks:{label:c=>'收益 '+c.parsed.x.toFixed(1)+'%'}},
   title:{display:true,text:'红=v1.0 的 +50% 止盈　赭=v1.2 的趋势出场',color:'#8c8b86',font:{size:11},align:'end'}},
  scales:{x:{title:{display:true,text:'平均收益 %'},grid:{color:gridc},ticks:{callback:v=>v+'%'}},
   y:{grid:{display:false}}}}});
</script>"""

# 标签
l1 = [f'{x["d0"][2:]}  {"勃发" if x["burst"] else "箱体"}' for x in drops]
d1 = [round(x['drop'] * 100, 1) for x in drops]
c1 = ['#A32D2D' if x['burst'] else '#854F0B' for x in drops]

l2 = [f'{m["n"]} 档' for m in marg]
d2 = [round(m['ret'] * 100, 2) for m in marg]
d3 = [round(m['dd'] * 100, 1) for m in marg]

colors = {2: '#C97B7B', 3: '#A32D2D', 4: '#854F0B', 5: '#8c8b86'}
ds3 = []
for n in [2, 3, 4, 5]:
    pts = [dict(x=round(g_['dd'] * 100, 2), y=round(g_['ret'] * 100, 2),
                lab=f'n{n} {g_["cf"]} g{g_["g"]} · 收益 {g_["ret"]*100:.1f}% / 浮亏 {g_["dd"]*100:.1f}%')
           for g_ in grid if g_['n'] == n]
    ds3.append(dict(label=f'{n} 档', data=pts, backgroundColor=colors[n],
                    pointRadius=4, pointHoverRadius=6))

l4, d4, c4 = [], [], []
for k in ['e0_hold', 'e7_tp50', 'e4_drop25', 'e1_v12', 'e1n_v12', 'e6_ma3d', 'e2_ma120', 'e3_ma250']:
    v = tbl[k]
    l4.append(v['label'])
    d4.append(round(v['mean'], 1))
    c4.append('#A32D2D' if k == 'e7_tp50' else ('#854F0B' if k in ('e1_v12', 'e1n_v12') else '#8c8b86'))

html = (''.join(H).replace('<script>', '')
        + '<script>' + CHARTJS + '</script>'
        + JS)
html = html.replace('<script>\n<script>', '<script>')
html = ('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>仓位间距 · 下手动作的数值化</title>' + CSS + '</head><body>'
        + ''.join(H) + '<script>' + CHARTJS + '</script>' + JS + '</body></html>')

html = html.replace('@@L1@@', json.dumps(l1, ensure_ascii=False)).replace('@@D1@@', json.dumps(d1))
html = html.replace('@@C1@@', json.dumps(c1))
html = html.replace('@@L2@@', json.dumps(l2, ensure_ascii=False))
html = html.replace('@@D2@@', json.dumps(d2)).replace('@@D3@@', json.dumps(d3))
html = html.replace('@@DS3@@', json.dumps(ds3, ensure_ascii=False))
html = html.replace('@@L4@@', json.dumps(l4, ensure_ascii=False)).replace('@@D4@@', json.dumps(d4))
html = html.replace('@@C4@@', json.dumps(c4))

open(OUT, 'w', encoding='utf-8').write(html)
print('written', OUT, len(html), 'bytes')
print('标注残留', sum(html.count(x) for x in ['@@', 'PRINT_ME']))
