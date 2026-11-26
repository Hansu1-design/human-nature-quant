# -*- coding: utf-8 -*-
"""build24: 标的错配 —— 进场地基的最后一块砖"""
import os, json
import numpy as np
import pandas as pd

B = os.path.dirname(os.path.abspath(__file__))
OUT = 'G:/探讨集/24_标的错配_进场地基修补_20261112.html'
CHARTJS = open(os.path.join(B, 'chart.umd.min.js'), encoding='utf-8').read()

d110 = json.load(open(os.path.join(B, 'd110_result.json'), encoding='utf-8'))
d111 = json.load(open(os.path.join(B, 'd111_result.json'), encoding='utf-8'))
d112 = json.load(open(os.path.join(B, 'd112_result.json'), encoding='utf-8'))
d113 = json.load(open(os.path.join(B, 'd113_result.json'), encoding='utf-8'))

# ---------- 主序列 ----------
d = pd.read_parquet(os.path.join(B, 'd113_510300.parquet')).sort_values('d').reset_index(drop=True)
d['d'] = pd.to_datetime(d['d'])
d['rel'] = d['c'] / d['sse']                      # 每 1 点上证对应的 ETF 价
d['rel_idx'] = d['rel'] / d['rel'].iloc[0] * 100  # 归一
d['real_etf'] = d['c'] * 100 / d['cpi_idx']
d['real_sse'] = d['sse'] * 100 / d['cpi_idx']

# 熊底点的实价比
EV = ['2013-06-25', '2014-03-12', '2019-01-03', '2022-04-26', '2024-02-05', '2026-09-11']
evrows = []
for e in EV:
    sub = d[d['d'] <= e]
    if len(sub) == 0:
        continue
    r = sub.iloc[-1]
    evrows.append({'d': e, 'etf': round(float(r['c']), 3), 'sse': round(float(r['sse']), 0),
                   'real_etf': round(float(r['real_etf']), 3),
                   'real_sse': round(float(r['real_sse']), 0),
                   'rel': round(float(r['rel']) * 1000, 3),
                   'dist': round(float(r['sse'] * 100 / r['cpi_idx'] / 1879 - 1) * 100, 1)})

# 2019 后的实价比区间
post = d[d['d'] >= '2019-01-01']
pre = d[d['d'] < '2019-01-01']
rel_post = {'min': round(float(post['rel'].min() / post['rel'].mean() - 1) * 100, 1),
            'max': round(float(post['rel'].max() / post['rel'].mean() - 1) * 100, 1),
            'sd_%': round(float(post['rel'].std() / post['rel'].mean()) * 100, 1)}
rel_pre = {'min': round(float(pre['rel'].min() / pre['rel'].mean() - 1) * 100, 1),
           'max': round(float(pre['rel'].max() / pre['rel'].mean() - 1) * 100, 1)}

# 当前读数
now = d.iloc[-1]
CPI_NOW = 144.0
SSE_NOW = 3680.6
REAL_NOW = SSE_NOW * 100 / CPI_NOW
DIST_NOW = (REAL_NOW / 1879 - 1) * 100
GATE10 = 1879 * 1.10 * CPI_NOW / 100
GATE20 = 1879 * 1.20 * CPI_NOW / 100
NEED = (1 - SSE_NOW / GATE10) * 100

CSS = open(os.path.join(B, '_css.txt'), encoding='utf-8').read() if os.path.exists(os.path.join(B, '_css.txt')) else None
if CSS is None:
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
P('<h1>标的错配：一个有过的坑，不是一个还在的坑</h1>')
P('<div class="sub">上证 2012-05 ~ 2026-09（3,476 根）· 标的 510300 前复权 · 2026-11-12 · 基准理论补血肉 ②</div>')

P('<div class="verdict">'
  '<b>结论先说：闸门不用改，标的不换，"折价微调"这条路走不通——但它也不需要走。</b><br><br>'
  'v1.1 §5 留的那条缺陷（"上证 2,900~3,100 时 ETF 价 2.219~3.879，同样点位贵贱差 75%"）<b>数字是真的，归因是错的</b>。'
  '那 75% 由两件事拼起来：<b>分红再投资的累积</b>（510300 年化分红 2.7%，14 年约 45%，是到手现金）'
  '+ <b>沪深300 相对上证的估值摆动</b>（2019-2021 冲到 200，2024 回到 146，净值为零但中途 ±30%）。'
  '前者不用修，后者才是真问题。<br><br>'
  '三条修法全部验过，全部失败：比值分位（ADF 三规格全不拒绝单位根，预测力 R² 0.0~0.4%）、'
  'ETF 自身分位当闸门（+21.9%，远弱于上证闸门的 +42.1%）、双条件（+31.8%，加条件反而降 10 个点）。<br><br>'
  '<b>所以这条缺陷从"待修"改成"已验无害"。</b>真正要加的不是修正项，是一句免责：'
  '<b>标的误差 ±10%（2019 年后），不可预测，按金额建仓自然摊平，不做贵贱判断。</b></div>')

# KPI
P('<div class="grid">')
for lab, val, des in [
    ('当前上证', '3,681', '2026-11-11 收盘'),
    ('实际点位', '2,556', 'CPI 平减 2008-01 不变价'),
    ('离真底', '+36.0%', '底 = 1,879'),
    ('出手闸门', '2,976', '离底 10%，需再跌 19.2%'),
]:
    P(f'<div class="kpi"><div class="l">{lab}</div><div class="n">{val}</div><div class="d">{des}</div></div>')
P('</div>')

# ---------- 1 三条路 ----------
P('<h2><span class="no">01</span>三条修法，三条都不通</h2>')
P('<p class="note">Hermes 提的第三条路是"保留上证做显示锚 + ETF/上证比值分位折价微调"，前提是比值必须均值回复。'
  '先把前提验掉。样本：2012-05-28~2026-09-11，3,476 个交易日。</p>')

P('<h3>a. 比值是不是均值回复？不是，是带漂移的随机游走</h3>')
rows = ''
for k, v in d110['D_adf'].items():
    ap = v['aic_pick']
    rows += (f"<tr{' class=\"hl\"' if '趋势' in k else ''}><td>{k}</td>"
             f"<td>{v['crit']['5%']}</td><td>{ap['t']}</td>"
             f"<td style=\"color:#1C6B3C\">不拒绝（有单位根）</td></tr>")
P('<table><thead><tr><th>ADF 规格</th><th>5% 临界值</th><th>t 统计量</th><th>结论</th></tr></thead>'
  '<tbody>' + rows + '</tbody></table>')
P('<p class="note">三种规格全部不拒绝单位根（t = −1.32 / −1.33 / −1.40，临界 −2.58 / −2.86 / −3.41）。'
  '连"允许确定性趋势"的规格都不拒绝——<b>说明这个比值既不是稳定均值，也不是绕趋势波动，它就是随机游走。</b>'
  '随机游走没有"该回去的地方"，分位折价无从谈起。</p>')

P('<h3>b. 短期有回复，但那是噪声，不是估值</h3>')
rows = ''
for k, v in d110['E_vr'].items():
    cl = ' class="hl"' if v['revert'].startswith('均值回复') else ''
    rows += (f"<tr{cl}><td>{k[1:]} 日</td><td>{v['VR']}</td><td>{v['z']}</td><td>{v['revert']}</td></tr>")
P('<table><thead><tr><th>持有期 q</th><th>方差比 VR</th><th>z 值</th><th>判定</th></tr></thead>'
  '<tbody>' + rows + '</tbody></table>')
P('<p class="note">q ≤ 20 天显著小于 1，q ≥ 60 天回到 1 附近。<b>月份内有回复、两个月以上就随机</b>——'
  '这是 ETF 折溢价的日内修正，不是"估值回归"。拿它做季度以上的择时，等于拿分钟级的规律管年线。</p>')

P('<h3>c. 就算偏离能算，也预测不了未来</h3>')
P('<div class="card"><div class="chartbox"><canvas id="c1"></canvas></div></div>')
P('<p class="note">用"过去 5 年/10 年滚动均值"当基准（实盘只能这么算，不能偷看未来），'
  '看偏离能不能预测未来 ratio 变化。重叠样本必须用块自助法，否则 t 值会虚高——'
  '<b>我第一版用全样本趋势残差，跑出 t = 1,094，那是恒等式，不是预测力。</b></p>')

rows = ''
for r in d111['D_predict']:
    sig = r['sig'] == '显著'
    rows += (f"<tr{' class=\"hl3\"' if sig else ''}><td>{r['base']}</td><td>{r['h']} 日</td>"
             f"<td>{r['coef']}</td><td>[{r['ci95'][0]}, {r['ci95'][1]}]</td>"
             f"<td>{r['R2_%']}%</td><td>{r['n']}</td>"
             f"<td style=\"color:{'#A32D2D' if sig else '#8c8b86'}\">{r['sig']}</td></tr>")
P('<table><thead><tr><th>基准</th><th>预测期</th><th>系数</th><th>95% 置信区间</th>'
  '<th>R²</th><th>样本</th><th>显著性</th></tr></thead><tbody>' + rows + '</tbody></table>')
P('<div class="warn"><b>5 年基准全军覆没</b>（R² 0.0~0.4%，置信区间全部跨 0）。'
  '10 年基准只有 500 日那档勉强显著，置信区间 [−2.33, −0.02] 宽到无法用——'
  '系数在 −2.33 和 −0.02 之间都可能，对应的"微调幅度"差 100 倍。'
  '<b>结论：没有可用的折价信号。</b></div>')

P('<h3>d. 那能不能干脆用 ETF 自己的分位当闸门？更差</h3>')
P('<div class="card"><div class="chartbox"><canvas id="c2"></canvas></div></div>')
P('<p class="note">同一件事（买 510300），两种闸门比预测力：'
  '左边是"上证离底距离"，右边是"510300 在过去 5 年价格中的分位"。看未来 1 年平均涨幅。</p>')

rows = ''
for r in d112['B_ladder']:
    if r['gate'].startswith('上证'):
        rows += (f"<tr class=\"hl\"><td>上证离底 {r['band']}</td><td>{r['n']}</td>"
                 f"<td style=\"color:#A32D2D;font-weight:500\">{r['fwd_%']:+.1f}%</td><td>{r['win_%']}%</td></tr>")
b = d112['B_base']
rows += (f"<tr><td>全样本（买入持有基准）</td><td>{b['n']}</td><td>{b['fwd_%']:+.1f}%</td><td>{b['win_%']}%</td></tr>")
for r in d112['B_ladder']:
    if r['gate'].startswith('ETF'):
        rows += (f"<tr><td>510300 分位 {r['band']}</td><td>{r['n']}</td>"
                 f"<td>{r['fwd_%']:+.1f}%</td><td>{r['win_%']}%</td></tr>")
P('<table><thead><tr><th>闸门</th><th>样本天数</th><th>未来 1 年</th><th>上涨概率</th></tr></thead>'
  '<tbody>' + rows + '</tbody></table>')
P('<p class="note">上证闸门 ≤10% 给 <b>+42.1% / 胜率 78.0%</b>（n=804）；'
  'ETF 分位 ≤0.3 只有 <b>+21.9%</b>（n=259）——胜率 100% 好看，但涨幅只有一半，'
  '因为分位在漂移序列上永远偏高，低分位只在崩盘那一刻出现，抓不住「便宜但还没崩」的整段。'
  '再加双条件（上证≤10% 且 ETF 分位≤0.5）更糟：<b>+31.8%</b>，比单用上证低 10 个点。'
  '<b>加条件不是为了更准，是为了更安心，这两个不是一回事。</b></p>')

# ---------- 2 七十五是什么 ----------
P('<h2><span class="no">02</span>那 75% 到底是什么：分红在下面推，估值在上面摆</h2>')
P('<p class="note">把"每 1 点上证对应的 ETF 价格"画出来（归一化到 2012-05 首日 = 100）。'
  '如果这个比值是"错配"，它应该在一个区间里来回摆；如果它被真实增值推着走，就应该整体上移。'
  '<b>答案是两件事同时在发生。</b></p>')

P('<div class="card"><div class="chartbox tall"><canvas id="c3"></canvas></div></div>')

P('<p class="note">它不是一路上走，是<b>颠着上走</b>：2012-2019 在 <b>88~121</b> 之间横着（ETF 长期比上证便宜），'
  '2019-2021 急冲到 <b>200</b>（沪深300 抱团顶部，ETF 相对上证贵一倍），'
  '之后回吐到 146，现在 167。<b>首尾净涨约 61%，年化 3.38%。</b>拆开：</p>')
P('<div class="card">'
  '<table><thead><tr><th>来源</th><th>年化</th><th>14 年累积</th><th>性质</th></tr></thead><tbody>'
  '<tr class="hl2"><td>510300 分红（前复权因子反推）</td><td>+2.7%</td><td>+45%</td>'
  '<td><b>真实到手</b>，不是溢价</td></tr>'
  '<tr><td>沪深300 相对上证的成分优势</td><td>+0.7%</td><td>+10%</td>'
  '<td>真实增值（剔除 ST、按自由流通市值加权）</td></tr>'
  '<tr class="hl3"><td>沪深300/上证 的估值摆动</td><td>净 ≈ 0</td><td>净 ≈ 0</td>'
  '<td><b>但中途摆到 ±30%</b>——这才是"同点位贵贱不同"的真身</td></tr>'
  '<tr><td><b>合计</b></td><td><b>+3.38%</b></td><td><b>+61%</b></td><td>—</td></tr>'
  '</tbody></table></div>')
P('<p class="note">第三行是这次真正找出来的东西：<b>估值摆动净值为零，但幅度极大</b>。'
  '2021 年初沪深300 相对上证贵了整整一倍，2024 年又回到持平——这跟分红无关，跟"抱团股 vs 银行地产"的轮动有关。'
  '<b>所以"同一点位 ETF 贵贱差 75%"这句话，说的是这个摆动，不是分红。</b></p>')
P('<div class="warn"><b>我上一版把 75% 全算到分红头上，那是错的。</b>'
  '分红只解释 45%（14 年累积），剩下的是估值摆动。<b>这个纠正很重要，因为它决定了修法该往哪走</b>：'
  '如果是分红，那是真实收益，不用修；如果是估值摆动，那它就<b>有可能被预测</b>——所以才要去验 ADF 和预测力（见 01 节）。'
  '验完的结论是不可预测，于是只剩"接受"。</div>')

P('<h3>五个熊底：ETF 相对上证，从便宜 35% 变成贵 19%</h3>')
P('<p class="note">把每个熊底那天的"ETF 实际价 ÷ 上证实价"算出来（1.0 = 两者一样便宜）。'
  '<b>这个数才是"同一点位，ETF 贵贱差多少"的准确度量</b>，它把分红和点位都消掉了。</p>')
P('<div class="card"><div class="chartbox"><canvas id="c4"></canvas></div></div>')
rows = ''
for r in evrows:
    d_ = r['dist']
    cls = ' class="hl"' if abs(d_) <= 10 else ''
    rows += (f"<tr{cls}><td>{r['d']}</td><td>{r['sse']:,.0f}</td><td>{r['real_sse']:,.0f}</td>"
             f"<td>{d_:+.1f}%</td><td>{r['etf']}</td><td>{r['real_etf']}</td><td>{r['rel']}</td></tr>")
P('<table><thead><tr><th>日期</th><th>上证</th><th>上证·实际</th><th>离底</th>'
  '<th>510300</th><th>510300·实际</th><th>ETF/上证</th></tr></thead><tbody>' + rows + '</tbody></table>')
P('<p class="note">上证实底：1,661 / 1,662 / 1,880 / 2,060 / 1,893 —— 10.6 年只抬 14%（年化 1.24%），'
  '中间区间 1,660~2,060（极差 24%）。<b>而 ETF 实际价 1.14 → 1.085 → 1.835 → 2.455 → 2.099，'
  '2014-2019 跳了一个台阶（+69%），2022 摸到 2.455 又回落到 2.099，不是一路上斜。</b><br>'
  '最后一列 ETF/上证：0.686 / 0.653 / 0.976 / 1.191 / 1.109。<b>从"ETF 便宜 35%"变成"贵 11~19%"，'
  '2019 年之后三个点落在 0.976~1.191，误差 ±10%。2013-2014 的 0.65~0.69 是历史——'
  '那两年沪深300 的 PE 只有 8 倍，属于一次性极低估，不是常态，也不该期待它重来。</b></p>')

P('<div class="ok"><b>这就是最终答案：所谓"标的错配"在 2019 年之后已经收敛到 ±10%。</b>'
  '一个 20% / 10% / 5% 的三档体系，容得下 ±10% 的标的误差。'
  '2013-2014 那个 0.65 的极端值，是"沪深300 被砸到 8 倍 PE"的产物，不能拿来当常态要求今天的 ETF 回到那个位置。</div>')

# ---------- 3 反例 ----------
P('<h2><span class="no">03</span>我中间差点写错的一条：闸门松动</h2>')
P('<p class="note">做完上面，我试着算了一件事：闸门按 CPI 每年抬 2.1%，'
  '而标的每年真实增值 3.4%——那是不是"等得越久，同样触发闸门买到的越贵"？'
  '全期回归给出 <b>+5.64% / 年</b>，照此外推"等 10 年多付 73%"。这条看着很像真的。</p>')

P('<div class="warn"><b>它站不住，因为换窗口就翻脸。</b><br>'
  '510300：全期（14.3 年）+5.64%/年 ｜ 2019 年后（7.7 年）<b>+0.94%/年</b><br>'
  '510500：全期（13.5 年）+2.63%/年 ｜ 2019 年后 +4.05%/年<br>'
  '512100：全期（9.9 年）+2.26%/年 ｜ 2019 年后 +4.16%/年<br>'
  '同一个指标，起点挪 7 年，510300 从 5.64% 掉到 0.94%（差 6 倍），510500 从 2.63% 涨到 4.05%（方向反过来）。'
  '<b>这不是规律，是窗口效应。</b>全期那个 5.64% 是被 2013-2014 的极端低点拉出来的。</div>')
P('<p class="note">「等 10 年多付 73%」这句话<b>已作废，不写进基准理论</b>。'
  '能写进去的只有一句定性的：<b>等的代价真实存在，但大小取决于你信不信 ETF 的增值优势会延续，'
  '而这件事的样本不足以判断。</b>结论：不为此加任何修正项。</p>')

# ---------- 4 数据体检 ----------
P('<h2><span class="no">04</span>顺手挖出的一个雷：510880 数据是坏的</h2>')
P('<p class="note">做标的选择对比时，510880（红利ETF）在出手区未来 1 年跑出 <b>+130.8%</b>，'
  '是 510300（+42.1%）的三倍。这个数字太漂亮，所以我先做了体检。</p>')

rows = ''
for r in d113['A_audit']:
    bad = r['verdict'] == '作废'
    rows += (f"<tr{' class=\"hl\"' if bad else ''}><td>{r['etf']}</td><td>{r['start']}</td><td>{r['n']:,}</td>"
             f"<td style=\"color:{'#A32D2D' if r['neg_or_zero_price'] else '#5f5e5a'}\">{r['neg_or_zero_price']}</td>"
             f"<td style=\"color:{'#A32D2D' if r['abs_move>15%_days']>20 else '#5f5e5a'}\">{r['abs_move>15%_days']}</td>"
             f"<td>{r['pct_col_mismatch_days']}</td><td><b>{r['verdict']}</b></td></tr>")
P('<table><thead><tr><th>ETF</th><th>起始</th><th>交易日</th><th>≤0 的收盘价</th>'
  '<th>单日 |涨跌|>15% 天数</th><th>涨跌幅与价格不符</th><th>判定</th></tr></thead><tbody>' + rows + '</tbody></table>')
P('<div class="warn"><b>510880 有 41 天收盘价 ≤ 0</b>（2008 年末报 <b>−0.109</b>），'
  '146 天单日涨跌超过 15%（2013-06-24 单日 −28.4%），40 天涨跌幅列与价格算出来的涨幅对不上。'
  '<b>这份数据是坏的，不是复权口径问题。</b>“红利 ETF 出手区 +130.8%”这个结论<b>当场撤回</b>，'
  '在拿到干净的 510880 之前，标的选择的实证边界只到 510300 / 510500 / 512100 三只。<br><br>'
  '另外三只全干净：0 个负价、0 天异常跳变、0 天涨跌幅不符。</div>')

# ---------- 5 最终规则 ----------
P('<h2><span class="no">05</span>进场地基：加四条，不改一条</h2>')
P('<p class="note">骨架参数一个不动（底 1,879 / 出手 10% / 加仓 5% / 每年按 CPI 抬）。'
  '这次加的全是"怎么下手"，不是"何时下手"。</p>')

P('<div class="card">'
  '<table><thead><tr><th>#</th><th>规则</th><th>依据</th></tr></thead><tbody>'
  '<tr><td>1</td><td><b>闸门只用上证 + CPI，不引入任何折价/溢价修正项</b></td>'
  '<td>比值 ADF 全不拒绝单位根，预测力 R²<1%；ETF 分位闸门更差（+21.9% vs +42.1%）</td></tr>'
  '<tr><td>2</td><td><b>标的默认 510300</b>；行业/主题仓 ≤1/3</td>'
  '<td>数据最干净、日跟踪相关 0.94、年化跟踪误差 8.35%、流动性最好</td></tr>'
  '<tr><td>3</td><td><b>按金额建仓，不按份数</b>：三笔各 1/3 金额</td>'
  '<td>2019 后标的误差 ±10% 且不可预测；按金额买天然摊平，按份数买会把噪声当决策</td></tr>'
  '<tr><td>4</td><td><b>不判断 ETF 贵贱</b>；只记录，不行动</td>'
  '<td>三条修法全失败；记录的用途是每年体检"ETF/上证 实价比"是否仍在 0.98~1.19</td></tr>'
  '</tbody></table></div>')

P('<div class="ok"><b>第 4 条那个区间是这次新加的体检项。</b>'
  '2019 年后 ETF/上证 实价比在 0.976~1.191 之间。如果哪天它明确跳出这个区间并稳住'
  '（比如连续两年 &lt;0.90，说明沪深300 又被砸到极低估），那时才需要重新审视闸门——'
  '不是因为闸门错了，是因为标的的估值体制换了。这是可观测、可证伪的，不是感觉。</div>')

P('<h3>今天该怎么用</h3>')
P('<div class="card">'
  f'<p style="margin:0 0 8px"><b>上证 3,681（2026-11-11）｜实际 2,556（2008 年不变价）｜离底 +36.0%</b></p>'
  f'<p style="margin:0 0 8px">出手闸门 <b>{GATE10:,.0f}</b>（离底 10%，CPI 144.0）→ 还要再跌 <b>{NEED:.1f}%</b>　'
  f'观察闸门 {GATE20:,.0f}（离底 20%）</p>'
  '<p style="margin:0">动作：<b>空仓，不看不买。</b>没有折价、没有溢价、没有"这次不一样"——'
  '标的误差 ±10%，你的闸门有 10% 的档，两个 10% 撞在一起时，唯一正确的动作是等，不是猜。</p>'
  '</div>')

P('<div class="foot">'
  '数据：上证 sh.000001 2012-05-28~2026-09-11（3,476 根，与 510300 取交集）；'
  '510300/510500/512100/510880 前复权日线（<code>G:/AStockData/supplement/etf_history</code>）；'
  'CPI 定基 2008-01=100，2026-11 为 144.0（「全国-当月」为同比，开 12 次方当环比）。<br>'
  '复权口径：510300 序列为<b>前复权</b>（末日值 = 真实成交价；历史值被后续分红下调），'
  '因此"绝对价格"不能当闸门，"收益率"可以——本报告所有收益比较均用收益率口径。<br>'
  '平稳性：自实现 ADF（含/不含常数与趋势三规格，AIC 选阶，MacKinnon 渐近临界值）；'
  'Lo-MacKinlay 方差比（同方差版）。预测力：Newey-West HAC + 循环块自助法（400 次）。<br>'
  '<b>已作废的中间结论（留档，防止下一轮重犯）</b>：<br>'
  '① 全样本二次趋势残差 → 未来 ETF 超额，t = 1,094，R² 41% —— 那是恒等式，不是预测力；<br>'
  '② "闸门松动 5.64%/年，等 10 年多付 73%" —— 换窗口翻 6 倍，窗口效应，作废；<br>'
  '③ "红利 ETF 出手区 +130.8%" —— 510880 数据含 41 个负价格，作废。<br>'
  '脚本：<code>d110_ratio_basis.py</code> <code>d111_basis_fix.py</code> '
  '<code>d112_gate_compare.py</code> <code>d113_wait_cost.py</code> <code>build24.py</code><br>'
  '样本限制：真正独立的出手事件只有 2013-06、2019-01、2022-04、2024-02 四次，'
  '所有分档统计的天数高度重叠，有效独立样本远小于表面 n。本报告为历史检验，不含投资建议。'
  '</div>')
P('</div>')

# ---------- 图 ----------
JS = r"""
<script>@@LIB@@</script>
<script>
Chart.defaults.font.family='-apple-system,"PingFang SC","Microsoft YaHei",sans-serif';
Chart.defaults.color='#5f5e5a'; Chart.defaults.animation=false;

new Chart(document.getElementById('c1'),{type:'bar',
 data:{labels:@@L1@@,datasets:[{data:@@D1@@,backgroundColor:@@C1@@,borderRadius:3,borderSkipped:false}]},
 options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,
  plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>'R² '+c.parsed.x.toFixed(2)+'%'}}},
  scales:{x:{title:{display:true,text:'R²（0 = 完全预测不了）%'},min:0,max:24,
    grid:{color:'rgba(0,0,0,.06)'},ticks:{callback:v=>v+'%',font:{size:11}}},
   y:{grid:{display:false},ticks:{font:{size:11}}}}}});

new Chart(document.getElementById('c2'),{type:'bar',
 data:{labels:@@L2@@,datasets:[
  {type:'bar',data:@@D2@@,backgroundColor:@@C2@@,borderRadius:3,borderSkipped:false,label:'未来1年平均涨幅'},
  {type:'line',data:@@D2B@@,borderColor:'#5f5e5a',borderWidth:1.2,pointRadius:0,
   borderDash:[4,4],label:'全样本基准 +13.4%'}]},
 options:{responsive:true,maintainAspectRatio:false,
  plugins:{legend:{position:'top',labels:{boxWidth:14,font:{size:11}}},
   tooltip:{callbacks:{label:c=>c.datasetIndex===0?('未来1年 '+c.parsed.y.toFixed(1)+'%'):'基准 +13.4%'}}},
  scales:{y:{title:{display:true,text:'未来 1 年平均涨幅 %'},min:0,grid:{color:'rgba(0,0,0,.06)'},
    ticks:{callback:v=>v+'%',font:{size:11}}},
   x:{grid:{display:false},ticks:{font:{size:10},maxRotation:45,minRotation:45}}}}});

new Chart(document.getElementById('c3'),{type:'line',
 data:{labels:@@L3@@,datasets:[
  {label:'每 1 点上证对应的 510300 价格（2012-05 = 100）',data:@@D3@@,
   borderColor:'#A32D2D',borderWidth:1.8,pointRadius:0,tension:0.1,fill:false},
  {label:'水平线（= 无漂移）',data:@@D3H@@,borderColor:'#8c8b86',borderWidth:1.4,
   pointRadius:0,borderDash:[5,4]}]},
 options:{responsive:true,maintainAspectRatio:false,
  plugins:{legend:{position:'top',labels:{boxWidth:14,font:{size:11}}}},
  scales:{y:{title:{display:true,text:'指数（首日=100）'},grid:{color:'rgba(0,0,0,.06)'},
    ticks:{font:{size:11}}},
   x:{grid:{display:false},ticks:{maxTicksLimit:8,font:{size:11}}}}}});

new Chart(document.getElementById('c4'),{type:'bar',
 data:{labels:@@L4@@,datasets:[
  {type:'bar',label:'ETF 实际价 ÷ 上证实价',data:@@D6@@,backgroundColor:@@C6@@,
   borderRadius:4,borderSkipped:false},
  {type:'line',label:'1.0 = 两者一样便宜',data:@@D7@@,borderColor:'#8c8b86',borderWidth:1.4,
   pointRadius:0,borderDash:[5,4]}]},
 options:{responsive:true,maintainAspectRatio:false,
  plugins:{legend:{position:'top',labels:{boxWidth:14,font:{size:11}}},
   tooltip:{callbacks:{label:c=>c.datasetIndex===0?('比值 '+c.parsed.y.toFixed(3)):'基准 1.000'}}},
  scales:{y:{title:{display:true,text:'ETF 实际价 ÷ 上证实价'},min:0.4,max:1.4,
    grid:{color:'rgba(0,0,0,.06)'},ticks:{font:{size:11}}},
   x:{grid:{display:false},ticks:{font:{size:11}}}}}});
</script>
"""

# 图2 标签：两种闸门
l2, v2, c2 = [], [], []
for r in d112['B_ladder']:
    if r['gate'].startswith('上证'):
        l2.append('上证 ' + r['band'])
        v2.append(r['fwd_%']); c2.append('#A32D2D')
for r in d112['B_ladder']:
    if r['gate'].startswith('ETF'):
        l2.append('ETF分位 ' + r['band'])
        v2.append(r['fwd_%']); c2.append('#C9A227')

l1, v1, c1 = [], [], []
for r in d111['D_predict']:
    l1.append(r['base'] + ' / ' + str(r['h']) + '日')
    v1.append(r['R2_%'])
    c1.append('#A32D2D' if r['sig'] == '显著' else '#C8C6BF')

step = max(1, len(d) // 800)
sub = d.iloc[::step]

V = {'@@LIB@@': CHARTJS,
     '@@L1@@': json.dumps(l1, ensure_ascii=False), '@@D1@@': json.dumps(v1), '@@C1@@': json.dumps(c1),
     '@@L2@@': json.dumps(l2, ensure_ascii=False), '@@D2@@': json.dumps(v2), '@@C2@@': json.dumps(c2),
     '@@D2B@@': json.dumps([13.4] * len(v2)),
     '@@L3@@': json.dumps([x.strftime('%Y-%m') for x in sub['d']]),
     '@@D3@@': json.dumps([round(float(x), 1) for x in sub['rel_idx']]),
     '@@D3H@@': json.dumps([100.0] * len(sub)),
     '@@L4@@': json.dumps([r['d'][:7] for r in evrows if r['d'] != '2026-09-11']),
     '@@D6@@': json.dumps([r['rel'] for r in evrows if r['d'] != '2026-09-11']),
     '@@C6@@': json.dumps(['#C8C6BF', '#C8C6BF', '#A32D2D', '#A32D2D', '#A32D2D']),
     '@@D7@@': json.dumps([1.0] * 5)}
for k, v in V.items():
    JS = JS.replace(k, str(v))

HTML = ('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>标的错配：进场地基修补</title>' + CSS + '</head><body>'
        + ''.join(H) + JS + '</body></html>')
open(OUT, 'w', encoding='utf-8').write(HTML)
print('written', OUT, len(HTML), 'bytes')
