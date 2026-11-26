# -*- coding: utf-8 -*-
"""build22: 「接近底」作为出手信号 —— 距离带的检验"""
import os, json
import numpy as np
import pandas as pd

B = os.path.dirname(os.path.abspath(__file__))
OUT = 'G:/探讨集/22_接近底就是出手信号_20261112.html'
CHARTJS = open(os.path.join(B, 'chart.umd.min.js'), encoding='utf-8').read()

d97 = json.load(open(os.path.join(B, 'd97_result.json'), encoding='utf-8'))
d99 = json.load(open(os.path.join(B, 'd99_result.json'), encoding='utf-8'))
d102 = json.load(open(os.path.join(B, 'd102_result.json'), encoding='utf-8'))

# ---------- 重算曲线 ----------
d = pd.read_parquet(os.path.join(B, 'd91_daily.parquet')).sort_values('d').reset_index(drop=True)
d['d'] = pd.to_datetime(d['d'])
c = pd.read_parquet('G:/AStockData/supplement/macro/cpi.parquet')
c['ym'] = c['月份'].str.replace('年', '-').str.replace('月份', '')
c['d'] = pd.to_datetime(c['ym'], format='%Y-%m')
c = c.sort_values('d').reset_index(drop=True)
c['mom'] = (pd.to_numeric(c['全国-当月'], errors='coerce') / 100.0) ** (1 / 12.0)
ix = [100.0]
for i in range(1, len(c)):
    ix.append(ix[-1] * c['mom'].iloc[i])
c['cpi_idx'] = ix
c['avail'] = c['d'] + pd.offsets.MonthBegin(1) + pd.Timedelta(days=14)
d = pd.merge_asof(d, c[['avail', 'cpi_idx']].sort_values('avail'),
                  left_on='d', right_on='avail', direction='backward')
d = d.dropna(subset=['cpi_idx']).reset_index(drop=True)
d['real'] = d['close'] * 100.0 / d['cpi_idx']
d['rA'] = d['real'] / 1880.0
d['th15'] = 1880 * 1.15 * d['cpi_idx'] / 100
d['th10'] = 1880 * 1.10 * d['cpi_idx'] / 100

A = d97['A_绝对锚']
now = d97['now']
need = d97['need']
by_year = d99['by_year']
bands = d102['bands']

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
.big{font-size:15px;line-height:1.9}
blockquote{margin:10px 0;padding:10px 16px;border-left:2px solid var(--bd);color:var(--tx2);font-size:13px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:720px){.two{grid-template-columns:1fr}}
</style>"""

H = []
def P(s): H.append(s)

P('<div class="wrap">')
P('<h1>把「底」从一条线，改成一个距离</h1>')
P('<div class="sub">上证 2013-06 ~ 2026-11（3,264 根）· CPI 平减 · 标的 510300 · 距底 = 实际值 ÷ 1,880 · 2026-11-12</div>')

P('<div class="verdict">'
  '<b>你的改法对，而且比"严格到底"强得多。</b>距离一分开，单调关系立刻出来：'
  '离底 <b>0~5%</b> 买入，未来一年 <b>+46.9%</b>（胜率 97.5%）；'
  '<b>10~15%</b> 是 <b>+11.7%</b>（93.9%）；<b>20~30%</b> 剩 <b>+5.9%</b>；'
  '<b>30~40%</b> 转负 <b>−4.6%</b>（32.1%）；<b>40% 以上 −12.6% ~ −26.8%</b>。'
  '八档单调，没有一档逆序。<br><br>'
  '但"接近"有硬边界：<b>15% 以内有效，20% 就退化成随机</b>（20% 带 1 年 +6.6%，基准 +7.7%）。'
  '你说的"接近"，必须收在 <b>15%，最好 10%</b>。<br><br>'
  '现在的读数是 <b>离底 36%</b>——比"准备区"还远一倍。按这条线，今天什么都不做是对的。</div>')

# ---------- 1 ----------
P('<h2><span class="no">01</span>距离一分开，单调就出来了</h2>')
P('<p class="note">把每一天的"实际点位"除以 1,880（五轮熊底实际值的中位），得到"离底倍数"。'
  '1.00 = 正在真底上，1.15 = 离底 15%。然后看每一档未来一年/两年/三年的涨幅。</p>')

P('<div class="card"><div class="chartbox"><canvas id="c1"></canvas></div></div>')

rows = ''
for r in A:
    if r['n'] < 30:
        continue
    cls = ''
    if 1.05 < r['lo'] <= 1.15:
        cls = ' class="hl3"'
    c1 = '#A32D2D' if r['f1y'] > 0 else '#1C6B3C'
    rows += ('<tr' + cls + '>'
             + ('<td>&le; %.0f%%</td>' % (r['hi'] * 100 - 100) if r['lo'] == 0
                else '<td>%.0f%% ~ %.0f%%</td>' % (r['lo'] * 100 - 100, r['hi'] * 100 - 100))
             + '<td>%s</td>' % format(r['n'], ',')
             + '<td>%s%%</td>' % r.get('days_pct', 0)
             + '<td style="color:%s;font-weight:500">%+.1f%%</td>' % (c1, r['f1y'])
             + '<td>%+.1f%%</td><td>%.1f%%</td><td>%+.1f%%</td><td>%+.1f%%</td></tr>'
             % (r['med1y'], r['win1y'], r['f2y'], r['f3y']))
P('<table><thead><tr><th>离底距离</th><th>天数</th><th>时间占比</th><th>未来1年</th><th>中位</th>'
  '<th>1年上涨概率</th><th>未来2年</th><th>未来3年</th></tr></thead><tbody>' + rows + '</tbody></table>')
P('<p class="note">窗口重叠，天数不等于独立样本。但八档方向全部一致，且 1/2/3 年三个窗口都同序——'
  '这一层不是噪声。</p>')

P('<div class="ok"><b>这一条是整份报告里最硬的东西。</b>它比我前几轮用"点位分档"得到的任何结论都清楚，'
  '因为点位是名义的（会被通胀稀释），距离是实际的。<b>你说的"接近这个底"，本质是把名义点位换成了实际距离。</b></div>')

# ---------- 2 ----------
P('<h2><span class="no">02</span>触带即买：13 年 5 次，不是 3 次</h2>')
P('<p class="note">用"从带外进入带内的第一天"作为买入信号，加 12 个月冷却（避免同一次下跌反复触发），'
  '不做任何出场规则，只算固定期持有。</p>')

rows = ''
for r in bands:
    if r['k'] <= 0:
        continue
    cls = ' class="hl3"' if r['k'] in (5, 10) else ''
    rows += (f"<tr{cls}><td>离底 {r['k']}% 内</td><td>{r['n']}</td>"
             f"<td style=\"color:{'#A32D2D' if r['m1']>0 else '#1C6B3C'};font-weight:500\">{r['m1']:+.1f}%</td>"
             f"<td>{r['w1']:.0f}%</td><td>{r['m2']:+.1f}%</td><td>{r['m20']:+.1f}%</td>"
             f"<td>{r['peak_med']:+.1f}%</td><td>{r['peak_mo']:.1f}</td>"
             f"<td>{r['h50']}/{r['n']}</td><td>{r['h100']}/{r['n']}</td></tr>")
P('<table><thead><tr><th>带宽</th><th>笔数</th><th>持有1年</th><th>胜率</th><th>持有2年</th>'
  '<th>持有20月</th><th>5年内最高<br>（中位）</th><th>到顶<br>月数</th><th>曾≥50%</th><th>曾翻倍</th></tr></thead><tbody>'
  + rows +
  f'<tr style="color:var(--tx3)"><td>基准（任意一天买）</td><td>3,020</td><td>{d102["base"]["m1"]:+.1f}%</td>'
  f'<td>{d102["base"]["w1"]:.0f}%</td><td>{d102["base"]["m2"]:+.1f}%</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>'
  '</tbody></table>')

P('<h3>入场明细（10% 带）</h3>')
b10 = [r for r in bands if r['k'] == 10][0]
P('<p class="note">' + ' · '.join(f'<b>{a}</b>（{b:,}）' for a, b in zip(b10['dates'], b10['closes'])) + '</p>')

P('<div class="warn"><b>和"十年三次"的关系，到这里才说清。</b>'
  '你等的"三次机会"是<b>涨幅 ≥100% 的大段</b>（2005/2008/2013），22 年 3 次。'
  '而"接近底"的<b>出手次数</b>是 13 年 5 次（10% 带）或 4 次（5% 带）。'
  '两者不矛盾：<b>出手 5 次，吃肉 1 次，其余四次是 +7% 到 +42%</b>。'
  '"十年三次"说的是吃肉，不是出手——你不要把出手次数当成失败。</div>')

# ---------- 3 ----------
P('<h2><span class="no">03</span>逐年看：13 年里有 8 年进过出手区</h2>')
P('<p class="note">每一年内，实际值最低走到离底多远。≤10% 记为出手区，≤5% 记为深水区。</p>')

P('<div class="card"><div class="chartbox tall"><canvas id="c2"></canvas></div></div>')

rows = ''
for r in by_year:
    cls = ''
    if r['min_rA'] <= 1.05:
        cls = ' class="hl"'
    elif r['min_rA'] <= 1.10:
        cls = ' class="hl3"'
    c1 = '#A32D2D' if r['min_rA'] > 1 else '#1C6B3C'
    st = '深水区' if r['min_rA'] <= 1.05 else ('出手区' if r['min_rA'] <= 1.10 else '—')
    rows += ('<tr' + cls + '>'
             + '<td>%d</td><td>%s</td><td>%s</td>' % (r['year'], format(r['low_close'], ','), r['low_date'])
             + '<td style="color:%s;font-weight:500">%+.1f%%</td>' % (c1, (r['min_rA'] - 1) * 100)
             + '<td>%s</td></tr>' % st)
P('<table><thead><tr><th>年份</th><th>年内最低上证</th><th>日期</th><th>最低离底</th>'
  '<th>状态</th></tr></thead><tbody>' + rows + '</tbody></table>')
P('<p class="note">13 年 <b>8 年</b>进过 10% 出手区，<b>6 年</b>进过 5% 深水区。'
  '也就是说：<b>平均不到两年就有一次出手机会</b>，但"翻倍"平均七年才一次。'
  '这两个频率差 4 倍，是最容易被混淆的地方。</p>')

# ---------- 4 ----------
P('<h2><span class="no">04</span>吃肥只发生过一次，而且不是每一次</h2>')
P('<p class="note">从出手区买入，5 年内到达的最高点（不是期末收益，是过程中的峰值）。</p>')

P('<div class="grid">'
  '<div class="kpi"><div class="l">5% 带买入 · 5年最高（中位）</div><div class="n" style="color:#A32D2D">+49.3%</div><div class="d">4 笔里 1 笔翻倍</div></div>'
  '<div class="kpi"><div class="l">10% 带买入 · 5年最高（中位）</div><div class="n" style="color:#A32D2D">+44.3%</div><div class="d">5 笔里 1 笔翻倍</div></div>'
  '<div class="kpi"><div class="l">0% 带（真底以下）</div><div class="n" style="color:#854F0B">+129.6%</div><div class="d">2 笔全部翻倍</div></div>'
  '<div class="kpi"><div class="l">从入场到见顶（中位）</div><div class="n">29<span style="font-size:14px"> 个月</span></div><div class="d">不是 3~20 个月</div></div>'
  '</div>')

P('<div class="warn"><b>这里有一个必须说清的事实：翻倍的那一次是 2013-06 入场，2015 年杠杆牛兑现。</b>'
  '之后 2018、2020、2022、2023 四次入场，5 年内最高只到 <b>+33% ~ +47%</b>，一次都没翻倍。'
  '所以"一波吃肥"的<b>入场条件</b>是对的，但<b>能不能吃肥，取决于后面有没有杠杆/改革/大水</b>——那不在你的规则里。<br><br>'
  '另一个反常识：从入场到见顶的<b>中位是 29 个月</b>，最长 46 个月。<b>"持有 3~20 个月"这个窗口偏短</b>，'
  '很多肉是在第 20 个月之后长的。</div>')

# ---------- 5 ----------
P('<h2><span class="no">05</span>现在的位置：离底 36%，还得跌 15% 才进准备区</h2>')

P('<div class="grid">'
  f'<div class="kpi"><div class="l">上证（2026-11-11）</div><div class="n">{now["close"]:,.0f}</div><div class="d">实际值 {now["real"]:,.0f}</div></div>'
  f'<div class="kpi"><div class="l">离底距离</div><div class="n" style="color:#854F0B">+{now["rA"]*100-100:.0f}%</div><div class="d">1.00 = 真底</div></div>'
  f'<div class="kpi"><div class="l">今天"真底"对应的点位</div><div class="n">{now["eq_anchor_today"]:,.0f}</div><div class="d">1,880 × CPI 1.44</div></div>'
  '<div class="kpi"><div class="l">这个位置未来1年（历史同档）</div><div class="n" style="color:#1C6B3C">−4.6%</div><div class="d">上涨概率 32.1%</div></div>'
  '</div>')

rows = ''.join(
    f"<tr{' class=\"hl3\"' if r['k']==10 else ''}><td>离底 {r['k']}% 内</td><td>{r['real']:,.0f}</td>"
    f"<td>{r['nom']:,.0f}</td><td style=\"color:#1C6B3C\">{r['drop']*100:+.1f}%</td>"
    f"<td>{'← 你的 3,000 在这里' if r['k']==10 else ('准备区' if r['k']==15 else '')}</td></tr>"
    for r in need)
P('<table><thead><tr><th>进入该带</th><th>实际值需跌到</th><th>对应上证点位</th><th>距现在</th>'
  '<th>说明</th></tr></thead><tbody>' + rows + '</tbody></table>')

P('<div class="ok"><b>注意 10% 那一行和你的 3,000 的关系。</b>'
  '今天 2,977 点 = 离底 10%，你划的 3,000 正好压在这条线上（差 23 点）。'
  '<b>你不是猜的，你划的线在 2026 年这个时点上，等于"离真底 10%"。</b>'
  '这就是为什么它这些年一直好用——它其实一直是距离线，只是你没意识到。</div>')

# ---------- 6 ----------
P('<h2><span class="no">06</span>最要紧的一条：横盘等不到底</h2>')
P('<p class="note">底每年跟着 CPI 往上爬。如果上证横着不动，距离会自然缩小——但缩得很慢。</p>')

P('<div class="card"><div class="chartbox"><canvas id="c3"></canvas></div></div>')

P('<div class="verdict"><b>上证横盘在 3,681 不动，要多久才进 10% 出手区？</b><br>'
  'CPI 1.5%/年 → <b>14.3 年</b>｜CPI 2.0%/年 → <b>10.7 年</b>｜CPI 3.0%/年 → <b>7.2 年</b>。<br><br>'
  '也就是说：<b>你不等跌，只靠通胀慢慢磨，这辈子大概只能再赶上一次。</b>'
  '这条线的前提是"真的会跌下来"，不是"时间久了自然会便宜"。<b>这是整个骨架里最容易被误解的地方。</b></div>')

# ---------- 7 ----------
P('<h2><span class="no">07</span>空仓这件事，你说得对</h2>')
P('<p class="note">上一轮我算"空仓成本"，你说无所谓——数据站在你这边，而且理由比"心态"硬。</p>')

P('<div class="ok"><b>区外买入的期望是负的。</b>离底 30~40% 买，一年 −4.6%（胜率 32%）；40~60% 买，−12.6%（10%）；'
  '60% 以上，−26.8%（0%）。<br><br>所以"平时正常生活、机会来了投一波"不是心理安慰，'
  '<b>它是策略的必要组成部分</b>：在区外动手，你不是"少赚"，是<b>负期望</b>。'
  '空仓那 7 年不是成本，是<b>避免负期望的成本</b>。<br><br>'
  '唯一要守的是：空仓期的钱买货币基金或国债，别躺在活期——那是唯一能改进的地方。</div>')

# ---------- 8 ----------
P('<h2><span class="no">08</span>怎么卖：+50% 有点用，回撤 20% 基本没用</h2>')
P('<p class="note">同一个入场点，三种出场方式。</p>')

P('<table><thead><tr><th>出场方式</th><th>5% 带（4笔）</th><th>10% 带（5笔）</th><th>说明</th></tr></thead><tbody>'
  '<tr><td>纯持有 20 个月</td><td>+32.2%</td><td>+18.4%</td><td>不设任何规则</td></tr>'
  '<tr class="hl3"><td>+50% 止盈（否则 20 月到期）</td><td><b>+40.3%</b></td><td><b>+21.7%</b></td>'
  '<td>只 2013 那笔触发，其余到期</td></tr>'
  '<tr><td>从峰值回撤 20% 离场</td><td>+39.0%</td><td>+20.7%</td><td>只在 2013 那笔触发</td></tr>'
  '<tr><td>两条都要</td><td>+40.3%</td><td>+21.7%</td><td>与"仅 +50%"完全相同</td></tr>'
  '</tbody></table>')

P('<div class="note"><b>读法：</b>5 笔里有 4 笔（2018/2020/2022/2023）20 个月内<b>从来没摸到 +50%</b>，'
  '所以出场规则根本没机会起作用。<b>决定收益的始终是买入位置，不是卖出技巧。</b>'
  '+50% 保留着，因为它不花钱；但别指望它。</div>')

# ---------- 判决 ----------
P('<h2><span class="no">09</span>合起来的判决</h2>')

P('<div class="verdict">'
  '<b>一、你的改动是对的，而且是这一轮最重要的修正。</b>'
  '"严格等到 1,880" 11 年只给 2 次机会、无法评估；'
  '"接近到 15% 以内"给 6 次、胜率 67%；"接近到 10% 以内"给 5 次、胜率 80%。'
  '<b>带宽换样本量，这一步换得值。</b><br><br>'

  '<b>二、"接近"的边界是 15%，不是 25%。</b>'
  '20% 带一年 +6.6%，和"随便哪天买"（+7.7%）没区别；15% 带 +9.9%、胜负比 4:1；'
  '10% 带 +11.0%、胜负比 4:1；5% 带 +13.1%、一年胜率 75%。<b>越紧越好，但样本越少。</b>'
  '折中：<b>20% 开始准备，10% 出手，5% 加仓。</b><br><br>'

  '<b>三、出手频率是 13 年 5 次，不是十年三次。</b>'
  '十年三次说的是"翻倍大段"，出手说的是"进入 10% 带"。'
  '<b>5 次出手里只有 1 次翻倍</b>（2013-06 → 2015），其余 4 次是 +7% ~ +42%。'
  '把这两个数分开记，你才不会因为"只赚 20%"而怀疑自己。<br><br>'

  '<b>四、你的规则里唯一该改的是持有时长。</b>'
  '从入场到见顶的中位是 <b>29 个月</b>，最长 46 个月。'
  '你原来设的"3~20 个月"会砍掉一半以上的肉。<b>改成"20 个月起评，不设上限，直到接近顶"。</b><br><br>'

  '<b>五、现在什么都不做，是对的。</b>'
  f'离底 {now["rA"]*100-100:.0f}%，同档历史 1 年 −4.6%、胜率 32%。'
  '要跌到 <b>3,112</b>（−15.4%）才进准备区，<b>2,977</b>（−19.1%）才进出手区。'
  '而你划的 3,000，正好就是出手区。<b>你的线没错，只是还没到。</b>'
  '</div>')

P('<h2><span class="no">10</span>什么情况下这份报告作废</h2>')
P('<div class="card big">'
  '① 出现一次"跌破 1,880 实际值后 5 年未收回"——下有底被证伪，"距离"这个坐标系就失效，得换锚。<br>'
  '② 上证站上 4,500 并维持两年以上——"顶递降"被打破，箱体上沿移动，所有距离要重算。<br>'
  '③ 全 A PE 中位上穿 60 且维持一年——估值中枢上移，"实际点位"不再是有效的便宜度度量。<br>'
  '④ 本轮 5 笔里，若把 2013-06 单独剔除后，10% 带的 1 年胜率从 80% 掉到 50%（n=4），'
  '<b>那"出手区有效"就只剩 1 个样本支撑</b>，结论必须降级为"无法评估"。'
  '</div>')

P('<div class="foot">'
  '数据：上证 sh.000001 2013-06-06~2026-11-11（3,264 根，受 M2/CPI 月度数据起点限制）；'
  'CPI 定基 2008-01=100，2026-11 为 144.0；执行标的 510300 沪深300ETF（前复权）。<br>'
  '方法：离底倍数 = 上证收盘 × 100 ÷ CPI定基 ÷ 1880。1,880 取自五轮熊底（2008/2013/2016/2019/2024）'
  'CPI 实际值 1,631/1,661/2,140/1,880/1,893 的中位，属<b>样本内锚</b>，存在前视偏差；'
  '"过去 8 年最低值"的事前锚给出同向结果，可作交叉验证。<br>'
  '入场定义：从带外进入带内的第一天，12 个月冷却期；不含交易成本；不含分红；买入持有基准为同区间上证。<br>'
  '<b>颜色约定：红色表示正收益/警示，绿色表示负收益，与行情软件的涨跌配色相反。</b><br>'
  '本报告为历史数据检验，样本量 2~6 笔，不含投资建议。<br>'
  '脚本：<code>d97_near.py</code> <code>d98_band_buy.py</code> <code>d99_dedupe.py</code> '
  '<code>d100_exit.py</code> <code>d102_fix.py</code> <code>build22.py</code>'
  '</div>')
P('</div>')

# ---------- 图表 ----------
JS = r"""
<script>@@LIB@@</script>
<script>
Chart.defaults.font.family='-apple-system,"PingFang SC","Microsoft YaHei",sans-serif';
Chart.defaults.color='#5f5e5a'; Chart.defaults.animation=false;

new Chart(document.getElementById('c1'),{type:'bar',
 data:{labels:@@L1@@,datasets:[{label:'未来1年平均涨幅',data:@@D1@@,
  backgroundColor:@@C1@@,borderRadius:3,borderSkipped:false}]},
 options:{responsive:true,maintainAspectRatio:false,
  plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>'未来1年 '+c.parsed.y.toFixed(1)+'%'}}},
  scales:{y:{title:{display:true,text:'未来 1 年平均涨幅 %'},grid:{color:'rgba(0,0,0,.06)'},
    ticks:{callback:v=>v+'%',font:{size:11}}},
   x:{title:{display:true,text:'买入时离真底的距离'},grid:{display:false},ticks:{font:{size:11}}}}}});

new Chart(document.getElementById('c2'),{type:'bar',
 data:{labels:@@L2@@,datasets:[{label:'年内最低离底距离',data:@@D2@@,
  backgroundColor:@@C2@@,borderRadius:3,borderSkipped:false}]},
 options:{responsive:true,maintainAspectRatio:false,
  plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>'最低离底 '+(c.parsed.y*100-100).toFixed(1)+'%'}}},
  scales:{y:{title:{display:true,text:'离底倍数（1.00=真底）'},min:0.85,max:1.40,
    grid:{color:'rgba(0,0,0,.06)'},ticks:{callback:v=>v.toFixed(2),font:{size:11}}},
   x:{grid:{display:false},ticks:{font:{size:11}}}}}});

new Chart(document.getElementById('c3'),{type:'line',
 data:{labels:@@L3@@,datasets:[
  {label:'上证',data:@@D3@@,borderColor:'#1a1a1a',borderWidth:1.6,pointRadius:0,tension:0.1},
  {label:'出手线（离底10%）',data:@@D4@@,borderColor:'#A32D2D',borderWidth:2,pointRadius:0,borderDash:[6,4],tension:0.1},
  {label:'准备线（离底15%）',data:@@D4b@@,borderColor:'#854F0B',borderWidth:1.4,pointRadius:0,borderDash:[3,3],tension:0.1}]},
 options:{responsive:true,maintainAspectRatio:false,
  plugins:{legend:{position:'top',labels:{boxWidth:14,font:{size:12}}}},
  scales:{y:{title:{display:true,text:'上证点位'},grid:{color:'rgba(0,0,0,.06)'},ticks:{font:{size:11}}},
   x:{grid:{display:false},ticks:{maxTicksLimit:8,font:{size:11}}}}}});
</script>
"""

lab = [('≤%d%%' % (r['hi']*100-100)) if r['lo'] == 0 else (f"{r['lo']*100-100:.0f}~{r['hi']*100-100:.0f}%" if r['hi'] < 99 else '60%+')
       for r in A if r['n'] >= 30]
val = [r['f1y'] for r in A if r['n'] >= 30]
col = ['#A32D2D' if v > 0 else '#1C6B3C' for v in val]
yl = [str(r['year']) for r in by_year]
yv = [r['min_rA'] for r in by_year]
yc = ['#A32D2D' if v <= 1.05 else ('#C08A3E' if v <= 1.10 else '#C8C6BF') for v in yv]

step = max(1, len(d) // 700)
sub = d.iloc[::step]

V = {'@@LIB@@': CHARTJS,
     '@@L1@@': json.dumps(lab, ensure_ascii=False), '@@D1@@': json.dumps(val), '@@C1@@': json.dumps(col),
     '@@L2@@': json.dumps(yl), '@@D2@@': json.dumps(yv), '@@C2@@': json.dumps(yc),
     '@@L3@@': json.dumps([x.strftime('%Y-%m') for x in sub['d']]),
     '@@D3@@': json.dumps([round(float(x)) for x in sub['close']]),
     '@@D4@@': json.dumps([round(float(x)) for x in sub['th10']]),
     '@@D4b@@': json.dumps([round(float(x)) for x in sub['th15']])}
for k, v in V.items():
    JS = JS.replace(k, str(v))

HTML = ('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>接近底就是出手信号</title>' + CSS + '</head><body>'
        + ''.join(H) + JS + '</body></html>')
open(OUT, 'w', encoding='utf-8').write(HTML)
print('written', OUT, len(HTML), 'bytes')
