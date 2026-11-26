# -*- coding: utf-8 -*-
"""生成 21_货币口径的等值点位 报告"""
import os, json

B = os.path.dirname(os.path.abspath(__file__))
OUT = 'G:/探讨集/21_货币口径等值点位_算回来的三千点_20261112.html'
CHARTJS = open(os.path.join(B, 'chart.umd.min.js'), encoding='utf-8').read()

d90 = json.load(open(os.path.join(B, 'd90_result.json'), encoding='utf-8'))
d91 = json.load(open(os.path.join(B, 'd91_result.json'), encoding='utf-8'))
d92 = json.load(open(os.path.join(B, 'd92_result.json'), encoding='utf-8'))
d93 = json.load(open(os.path.join(B, 'd93_result.json'), encoding='utf-8'))
d94 = json.load(open(os.path.join(B, 'd94_result.json'), encoding='utf-8'))
d95 = json.load(open(os.path.join(B, 'd95_result.json'), encoding='utf-8'))
d96 = json.load(open(os.path.join(B, 'd96_result.json'), encoding='utf-8'))

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
.foot{margin-top:48px;padding-top:20px;border-top:1px solid var(--bd);font-size:12px;color:var(--tx3);line-height:1.9}
code{background:#F1EFE8;padding:1px 5px;border-radius:4px;font-size:12px}
.big{font-size:15px;line-height:1.9}
blockquote{margin:10px 0;padding:10px 16px;border-left:2px solid var(--bd);color:var(--tx2);font-size:13px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:720px){.two{grid-template-columns:1fr}}
</style>"""

# ---------- 头部 ----------
H = []
H.append('<div class="wrap">')
H.append('<h1>货币口径的等值点位：把你那条 3,000 点算回来</h1>')
H.append('<div class="sub">上证 2008-02 ~ 2026-11（4,555 根）× M2 / CPI 双平减 · 执行标的 510300 沪深300ETF · '
         '回测窗口 2015-01 ~ 2026-11 · 2026-11-12</div>')

H.append('<div class="verdict"><b>结论先说：你的方法对，但你选错了平减物。</b>'
         '用 M2 算，五轮熊底从 13,397 掉到 3,287（<b>−75%</b>），"下有底"被直接摧毁，'
         '拿它做闸门 11 年交易 19 笔、只有 1 次跑赢买入持有；<b>换成 CPI 平减，五轮熊底是 '
         '1,631 / 1,661 / 2,140 / 1,880 / 1,893，变异系数只有 10.0%——"下有底"完美成立。</b><br>'
         '算回来的结果是：<b>今天的真底在 2,707 点，而你的 3,000 点比它高 10.9%。这个"高"不是错，是对的</b>——'
         '因为严格等 2,707，11 年只等到 2 次、只有 2 次跑赢。<br>'
         '但有一条你必须改：<b>这条线每年要往上爬约 2%（CPI）。不爬，2030 年你的 3,000 点只值今天的 2,828（实际 1,964），'
         '2040 年只值今天的 2,436（实际 1,692）。你会把"比今天贵 40%"当成便宜。</b></div>')

# ---------- 1. 三个口径 ----------
H.append('<h2><span class="no">01</span>先看三个口径，它们给出三个完全不同的答案</h2>')
H.append('<p class="note">同一段数据（上证 2008-02 ~ 2026-11），只是换了把尺子。'
         '<b>注意：M2 月度数据只有 2008-01 起，所以三个口径的均值不是同一样本，不能互相换算。</b></p>')

c3 = d91['center3']
rows = [('名义点位（就是你在看的那个数）', '名义点位', '你每天看到的 3,681', '#1a1a1a'),
        ('CPI 平减（2008-01 不变价）', 'CPI平减(2008年不变价)', '扣掉通货膨胀', '#854F0B'),
        ('M2 平减（折算到 2026-11 货币水平）', 'M2平减(2026货币水平)', '扣掉货币扩张', '#A32D2D')]
H.append('<table><tr><th>口径</th><th>均值</th><th>中位</th><th>P20</th><th>P80</th><th>起点→终点</th></tr>')
for lab, k, _, _c in rows:
    v = c3[k]
    H.append(f'<tr><td>{lab}</td><td><b>{v["mean"]:,}</b></td><td>{v["med"]:,}</td>'
             f'<td>{v["p20"]:,}</td><td>{v["p80"]:,}</td><td>{v["first"]:,} → {v["last"]:,}</td></tr>')
H.append('</table>')
H.append('<p class="note" style="margin-top:10px">名义均值 <b>3,006</b>、中位 <b>3,051</b>——'
         '<b>你說的"长期震荡均值是 3000"，实测完全正确，一个点都不差。</b>但同一个 3,006，'
         '换成 CPI 口径只有 2,380，换成 M2 口径是 8,305。三个都对，因为它们回答三个不同的问题。</p>')

# ---------- 2. 五轮熊底 ----------
H.append('<h2><span class="no">02</span>五轮熊底：一把尺子说"底在崩塌"，另一把说"底是条水平线"</h2>')
H.append('<p class="note">同一个日期、同一个点位，三种读法。这是整份报告最关键的一张表。</p>')

bt = d92['bottoms']
H.append('<table><tr><th>熊底</th><th>名义</th><th>CPI 不变价</th><th>M2 等价</th><th>ratio（点/万亿）</th></tr>')
for r in bt:
    H.append(f'<tr><td>{r["date"]}</td><td><b>{r["nom"]:,}</b></td><td>{r["real"]:,}</td>'
             f'<td style="color:#A32D2D">{r["m2"]:,}</td><td>{r["ratio"]}</td></tr>')
bc = d91['bottom_change']
H.append(f'<tr class="hl"><td>2008 底 → 2024 底</td><td style="color:#A32D2D"><b>+{bc["nom"]}%</b></td>'
         f'<td style="color:#1C6B3C"><b>+{bc["cpi"]}%</b></td>'
         f'<td style="color:#A32D2D"><b>{bc["m2"]}%</b></td><td>37.68 → 9.25</td></tr>')
H.append('</table>')

H.append('<div class="chartbox"><canvas id="c1"></canvas></div>')
H.append('<p class="note">以 2008-11-04 为 100 归一化。名义底在往上抬（+58%），CPI 实际底几乎是水平线（+16%），'
         '<b>M2 等价底一路塌到 24.5（−75%）</b>。三条线说的是同一件事，结论却相反——这就是"选错平减物"的代价。</p>')

bs = d92['bottom_stat']
H.append('<div class="grid" style="margin-top:14px">')
H.append(f'<div class="kpi"><div class="l">CPI 实际底的中位</div><div class="n">{bs["real_med"]:,}</div>'
         f'<div class="d">2008-01 不变价 · 五轮范围 {bs["real_min"]:,}~{bs["real_max"]:,}</div></div>')
H.append(f'<div class="kpi"><div class="l">变异系数</div><div class="n" style="color:#1C6B3C">{bs["cv"]}%</div>'
         f'<div class="d">五轮熊底的实际值离散度</div></div>')
H.append(f'<div class="kpi"><div class="l">折算到今天（CPI=144.0）</div><div class="n" style="color:#A32D2D">{bs["nom_today_med"]:,}</div>'
         f'<div class="d">范围 {bs["nom_today_min"]:,}~{bs["nom_today_max"]:,}</div></div>')
H.append(f'<div class="kpi"><div class="l">你的 3,000 点偏高</div><div class="n">+10.9%</div>'
         f'<div class="d">3,000 ÷ {bs["nom_today_med"]:,} − 1</div></div>')
H.append('</div>')

H.append('<div class="warn"><b>必须标注的三个边界：</b>'
         '① 变异系数 10.0% 是 <b>5 个点</b>算出来的，最小值 1,631（2008）与最新值 1,893（2024）实际差 16%——'
         '<b>n=5 的离散度不能当精度用</b>；'
         '② CPI 官方口径系统性低估居住与服务成本，所以"底稳"这个结论有<b>向上的偏</b>（真实的底可能更低、更不稳）；'
         '③ 2024 年是双底——2024-02-05 收盘 2,702，2024-09-13 收盘 2,704，几乎相同，取哪天都行，'
         '但说明"一轮熊底"可以有多个低点，取点本身有主观性。</div>')

# ---------- 3. 3000 点在贬值 ----------
H.append('<h2><span class="no">03</span>同样一个 3,000 点，2008 年和 2026 年不是同一个东西</h2>')
H.append('<p class="note">这就是你说的"算回来"。把每年的 3,000 点换算成 2008-01 不变价：</p>')
tk = d92['three_k']
H.append('<table><tr><th>年份</th><th>CPI 定基（2008-01=100）</th><th>3,000 点 = 实际</th><th>相对 2008 年</th></tr>')
base = tk[0]['real']
for r in tk:
    chg = r['real'] / base - 1
    H.append(f'<tr><td>{r["year"]}</td><td>{r["cpi"]}</td><td><b>{r["real"]:,}</b></td>'
             f'<td style="color:{"#1C6B3C" if chg<0 else "#5f5e5a"}">{chg*100:+.1f}%</td></tr>')
H.append('</table>')
H.append('<div class="chartbox"><canvas id="c2"></canvas></div>')
H.append('<p class="note"><b>同一个 3,000 点，实际购买力从 2,915 掉到 2,089，缩水 28%。</b>'
         '换句话说：这条线每年自己往下掉约 1.7%——你以为门槛没变，其实它每年在偷偷变松。</p>')

# ---------- 4. 两条线交叉 ----------
H.append('<h2><span class="no">04</span>两条线：固定的 3,000 vs "实际 2,100" 等值线</h2>')
H.append('<p class="note">把你说的"等值"做成一条会动的线：锚定"CPI 实际 2,100"，每年按 CPI 上调。'
         '它和你的固定 3,000 在 2024 年交叉。</p>')
tl = d94['two_lines']
H.append('<table><tr><th>年份</th><th>固定 3,000 点的实际值</th><th>"实际 2,100" 对应的名义点位</th><th>差</th></tr>')
for r in tl:
    H.append(f'<tr{" class=hl" if r["year"]==2024 else ""}><td>{r["year"]}</td><td>{r["real3000"]:,}</td>'
             f'<td><b>{r["nom2100"]:,}</b></td><td style="color:{"#A32D2D" if r["diff"]>0 else "#1C6B3C"}">{r["diff"]:+,}</td></tr>')
H.append('</table>')
H.append('<div class="chartbox"><canvas id="c3"></canvas></div>')
H.append(f'<p class="note"><b>2015 年：你的 3,000 比等值线贵 394 点（13%）。2024 年：等值线 3,006，首次超过 3,000'
         f'（{d94["cross"]["nom2100"]:,} vs 3,000）。2026 年：3,023，只高 23 点。</b>'
         '所以今天这两条线几乎重合——<b>你的 3,000 现在是对的，但它对得刚好，没有任何余量。</b></p>')

H.append('<div class="warn"><b>但 Hermes 提醒我，这里有个反向的事实，我必须说：</b>'
         '2015 年你用 3,000 进场，比等值线贵 13%，<b>结果 8 笔赚了 +155.8%；换成"实际 2,100"（当年 2,606），'
         '只有 5 笔、+69.5%</b>。贵 13%，收益高一倍多。'
         '原因不是"贵更好"，是<b>贵让我多做了 3 笔，其中包括 2016-02-16 那笔 +63.7%</b>。'
         '所以"3,000 当年太贵"这个判断，不能用来解释成败——<b>精度从来不是瓶颈，能不能触发才是。</b></div>')

# ---------- 5. ratio 为什么不能当锚 ----------
H.append('<h2><span class="no">05</span>你最初提的那个 ratio（10.8），为什么不能当锚</h2>')
H.append('<p class="note">你说的 2024-09 的 3,336 与 2026-07 的 3,832 都是 ratio 10.8——这个观察是真的。'
         '但把 ratio 做成信号，它失效了。</p>')
rt = d93['ratio_trend']
H.append('<div class="grid">')
H.append(f'<div class="kpi"><div class="l">ratio 的年化趋势</div><div class="n" style="color:#A32D2D">{rt["ann"]}%</div>'
         f'<div class="d">log 线性拟合，每年下降</div></div>')
H.append(f'<div class="kpi"><div class="l">滚动 5 年分位 ≤0.1 的时间占比</div><div class="n" style="color:#A32D2D">{rt["p10"]}%</div>'
         f'<div class="d">对照：名义分位只有 {rt["nom_p10"]}%</div></div>')
H.append(f'<div class="kpi"><div class="l">ratio 分位做闸门（11 起点）</div><div class="n" style="color:#A32D2D">1/11</div>'
         f'<div class="d">19 笔 · 中位 23.3% · 最差 −19.7%</div></div>')
H.append('</div>')
H.append('<p class="note" style="margin-top:12px"><b>机制很简单：ratio 每年降 6.8%，滚动 5 年分位就常年贴在 0 附近。'
         '闸门一直开着，等于没有闸门。</b>19 笔交易被止损反复收割，只有 1 次跑赢买入持有。'
         '这不是 ratio 没用，是<b>"分位"这个工具用在单调下降的序列上必然失效</b>——'
         '它会把"相对自己在下降"误读成"绝对便宜"。</p>')

# ---------- 6. 闸门细扫 ----------
H.append('<h2><span class="no">06</span>闸门细扫：不是悬崖，是"断崖 + 150 点平台"</h2>')
H.append('<p class="note">把买点从 2,700 扫到 3,450（步长 50），其他一切不变。'
         '我上一轮说"2,800 是悬崖"，<b>这一轮细扫推翻了那个说法</b>。</p>')
sc = d96['scan']
H.append('<div class="chartbox tall"><canvas id="c4"></canvas></div>')
H.append('<table><tr><th>买点</th><th>笔数</th><th>在场</th><th>总收益</th><th>去掉最大一笔</th><th>在场年化</th></tr>')
for r in sc:
    hl = ' class="hl2"' if 2900 <= r['th'] <= 3050 else ''
    H.append(f'<tr{hl}><td>{r["th"]:,}</td><td>{r["n"]}</td><td>{r["expo"]}%</td>'
             f'<td><b>{r["total"]}%</b></td><td>{r["total_db"]}%</td><td>{r["cagr"]}%</td></tr>')
H.append('</table>')
H.append('<div class="ok"><b>三个发现：</b><br>'
         '① <b>2,800 → 2,850 是断崖</b>：2.8% 直接跳到 124.8%。差别只有 50 点，结果差 44 倍。'
         '原因单一——2,850 才够得着 2016-02-16 那天（上证 2,837），差 13 点就整段错过。<br>'
         '② <b>2,900 ~ 3,050 是平台，跨 150 点</b>，收益 155.8~166.9%。'
         '<b>你的 3,000 落在平台中段，不是刀尖。</b><br>'
         '③ <b>3,100 以上开始衰减</b>：笔数从 8 涨到 11、14、18，收益反而从 155.8% 掉到 141.4%、81.7%、76.0%。'
         '这是"太松"被止损磨损的证据。<br>'
         '<b>去掉最大一笔后，平台仍在（2,900 是 63.0%、3,000 是 56.3%），说明不是单笔驱动。</b></div>')
H.append('<p class="note">反过来读：<b>你这条线的价值不在"精确"，在"留了 150 点余量"。</b>'
         '如果它是 2,800，会被那 13 点卡死，11 年只做 2 笔、赚 2.8%。'
         '3,000 之所以能干活，是因为它在断崖之上 150 点。</p>')

# ---------- 7. 11 起点 ----------
H.append('<h2><span class="no">07</span>11 个起点：为什么我不该给你"胜率 8/11"这个数</h2>')
H.append('<div class="warn"><b>先自我否定：11 个起点不是 11 个样本。</b>它们是同一条价格路径的 11 个切片，'
         '差异 100% 来自"第一个信号什么时候到来"，那是运气。'
         '<b>报"胜率 8/11"和报"中位 69.5%"一样，都是伪统计。</b>该看的是逐笔分布（下一节）。'
         '下表留着，只用来展示"不同起点会差多少"。</div>')

starts = ['2015-01-05', '2015-06-12', '2016-01-04', '2017-01-03', '2018-01-02', '2019-01-02',
          '2020-01-02', '2021-01-04', '2022-01-04', '2023-01-03', '2024-01-02']
g = d93['gates']
keys = [('real<=2100', '实际≤2,100'), ('名义<=3000', '名义≤3,000'), ('名义分位<=0.25', '名义分位≤0.25')]
H.append('<table><tr><th>起点</th><th>买入持有</th>' + ''.join(f'<th>{lab}</th>' for k, lab in keys) + '</tr>')
for i, st in enumerate(starts):
    row = f'<tr><td>{st}</td><td>{g[keys[0][0]]["rows"][i]["bh"]}%</td>'
    for k, _ in keys:
        r = g[k]['rows'][i]
        row += f'<td><b>{r["total"]}%</b> <span style="color:#8c8b86;font-size:11px">({r["n"]}笔)</span></td>'
    H.append(row + '</tr>')
H.append(f'<tr class="hl"><td>中位</td><td>31.7%</td><td>{g["real<=2100"]["median"]}%</td>'
         f'<td>{g["名义<=3000"]["median"]}%</td><td>{g["名义分位<=0.25"]["median"]}%</td></tr>')
H.append('</table>')
H.append('<p class="note">看出问题了吗：<b>2021 年以后所有起点都收敛到 12.2%</b>，因为那时才出现第一个信号，'
         '前面几年全程空仓。三个闸门在这一段做的<b>是同一批交易</b>——所以那不是"三套风控都很稳"，'
         '是<b>三把尺子退化成了一把</b>。这也解释了为什么"最差 +12.2%"在三行里一模一样。</p>')

# ---------- 8. 逐笔 ----------
H.append('<h2><span class="no">08</span>逐笔：8 笔 vs 5 笔，差在哪三笔</h2>')
H.append('<p class="note">2015-01-05 起点。名义 3,000 做 8 笔，CPI 实际 2,100 做 5 笔，差的正好是这三笔：</p>')
t3 = d94['trades_3000']
t5 = d94['trades_2100']
set5 = set((x[0], x[1]) for x in t5)
H.append('<table><tr><th>买入</th><th>卖出</th><th>收益</th><th>出场</th><th>在 5 笔里？</th></tr>')
for x in t3:
    extra = (x[0], x[1]) not in set5
    H.append(f'<tr{" class=hl" if extra else ""}><td>{x[0]}</td><td>{x[1]}</td>'
             f'<td style="color:{"#A32D2D" if x[2]>0 else "#1C6B3C"}"><b>{x[2]:+.1f}%</b></td>'
             f'<td>{x[3]}</td><td style="color:{"#854F0B" if extra else "#8c8b86"}">{"多出来" if extra else "相同"}</td></tr>')
H.append('</table>')
H.append('<p class="note">8 笔合计 <b>+155.8%</b>，5 笔合计 <b>+69.5%</b>。多出来的三笔是 '
         '<b>+63.7% / −8.6% / +0.9%</b>，合起来贡献 +51%。'
         '<b>其中 2016-02-16 那一笔（+63.7%）是全部收益的支柱</b>：去掉它，8 笔只剩 56.3%。</p>')
H.append('<div class="warn"><b>为什么"实际 2,100"抓不到 2016 那笔？</b>2016 年 2 月 CPI 定基 126，'
         '"实际 2,100" 换算成名义是 2,646；而上证当天 2,837，高出 7%。'
         '<b>通胀低的年份，CPI 等值线会把门槛压得更低，于是漏掉机会。</b>'
         '这是 CPI 平减的固有代价，不是我的口误。</div>')

# ---------- 9. 等真底 ----------
H.append('<h2><span class="no">09</span>最反直觉的一条：严格等"真底"，反而输</h2>')
H.append('<div class="two">')
H.append('<div class="card"><h3 style="margin-top:0">等真底（2,707 / 实际 2,000）</h3>'
         '<div style="font-size:30px;font-weight:500;color:#854F0B">2 笔</div>'
         '<p class="note" style="margin:8px 0 0">11 年只等到两次：2019-01-09（+20.4%）、2024-02-07（−0.5%）。'
         '合计 +19.7%，跑赢买入持有 2/11。</p></div>')
H.append('<div class="card"><h3 style="margin-top:0">等"合适"（3,000 / 实际 2,100）</h3>'
         '<div style="font-size:30px;font-weight:500;color:#1a1a1a">5~8 笔</div>'
         '<p class="note" style="margin:8px 0 0">5 笔 +69.5% / 8 笔 +155.8%。'
         '同样 11 年，同样出场规则，唯一差别是肯不肯在"没那么便宜"的地方动手。</p></div>')
H.append('</div>')
H.append('<div class="warn"><b>但我不能就此说"等真底是错的"——样本不够，说不出口。</b>'
         '2 笔交易、每笔标准差约 25%，标准误 17.7pt，<b>连 20pt 的差异都测不出来（需要约 7 笔），'
         '10pt 更要约 25 笔</b>。所以正确的表述是：<b>"等真底"在 2015-2026 这段里只触发了 2 次（0.18 次/年），'
         '频率太低，任何收益结论都没有统计意义。</b>'
         '不能写"太严所以输"，只能写"<b>无法评估</b>"。<br>'
         '另外，我之前把"实际≤2,000"和"名义≤2,707"当成两组独立证据，<b>其实它们有一笔是同一笔交易'
         '（2019-01-09 → 2019-05-22）</b>，不是两次独立确认。</div>')

# ---------- 10. 空仓成本 ----------
H.append('<h2><span class="no">10</span>空仓的代价：这才是 M2 该用的地方</h2>')
H.append('<p class="note">你说"其余任何时候都收手不做"。收手是有成本的，但成本要用对尺子量。</p>')
H.append('<table><tr><th>空仓期拿到的收益</th><th>相对 M2（12.27%/年）</th><th>10 年后剩</th>'
         '<th>相对 CPI（1.99%/年）</th><th>10 年后剩</th></tr>')
cash_rows = [('0%（活期）', -12.27, 31, -1.99, 82), ('1.5%', -10.77, 36, -0.49, 95),
             ('2.5%', -9.77, 40, 0.51, 105), ('3.5%', -8.77, 44, 1.51, 116)]
for a, b, c, d, e in cash_rows:
    hl = ' class="hl2"' if a.startswith('2.5') else ''
    H.append(f'<tr{hl}><td>{a}</td><td style="color:#A32D2D">{b:+.2f}%/年</td><td>{c}%</td>'
             f'<td style="color:{"#1C6B3C" if d>0 else "#A32D2D"}">{d:+.2f}%/年</td><td>{e}%</td></tr>')
H.append('</table>')
H.append('<div class="ok"><b>一句话：空仓期只要把钱放在 2.5% 以上，购买力十年不亏（105%）；'
         '但相对货币池，十年只剩 40%。</b>所以"等待"的真实代价不是通胀，是<b>相对社会总财富掉 60%</b>。'
         '这条正好支持你"只做勃发段"——<b>但前提是空仓期的钱必须生息，否则等待本身就在亏。</b></div>')

# ---------- 11. 判决 ----------
H.append('<h2><span class="no">11</span>判决</h2>')
H.append('<div class="verdict" style="border-left-color:#1C6B3C;background:#EAF3ED">'
         '<b style="color:#1C6B3C">你的方法成立，但要换一把尺子，并且记住这条线每年会动。</b></div>')
H.append('<div class="big">'
         '<p><b>1. 你的观察全部正确。</b>上证 18.7 年名义均值 3,006、中位 3,051——'
         '"长期震荡均值 3,000"不是感觉，是事实。2024-09 的 3,336 与 2026-07 的 3,832 在 M2 口径下同为 10.8，'
         '这个读数也对。</p>'
         '<p><b>2. 但"等值"必须用 CPI，不能用 M2。</b>M2 是信用扩张（年均 12.27%），'
         '它衡量的是"你的钱相对社会总财富缩水多少"，不是"股票贵不贵"。'
         '用它定买点，会把底算成一条向下 75% 的斜线，闸门永远开着（19 笔、1/11）。'
         '换成 CPI，五轮熊底立刻变成一条水平线（1,631~2,140），<b>"下有底"才真正成立</b>。</p>'
         '<p><b>3. 算回来的答案是 2,707，不是 3,000。</b>五轮熊底的实际值中位 1,880，'
         '折算到今天的 CPI 是 2,707（范围 2,348~3,081）。<b>你的 3,000 比真底高 10.9%——'
         '而这个"高"是对的</b>，因为严格等 2,707 在 11 年里只触发 2 次，无法评估。</p>'
         '<p><b>4. 3,000 不是刀尖，是平台中段。</b>2,900~3,050 连成一片（155.8~166.9%），'
         '下沿到 2,850 才断崖（124.8%），再下到 2,800 归零（2.8%）。'
         '<b>你这条线的价值不是"精确"，是"刚好在断崖之上 150 点"。</b></p>'
         '<p style="margin-bottom:0"><b>5. 唯一必须改的一条：这条线每年要往上爬约 2%。</b>'
         '2026 年 3,023、2030 年 3,209、2040 年 3,724。'
         '<b>死守 3,000 的后果不是"买贵了"，是"你以为便宜，其实比今天贵 40%"</b>——'
         '2030 年你的 3,000 只等于今天的 2,828，2040 年只等于 2,436。</p>'
         '</div>')

H.append('<h3>可执行版本</h3>')
H.append('<div class="card"><p class="big" style="margin:0">'
         '<b>买入线 = 2,100 × (CPI定基 ÷ 100)</b>，2026 年 = <b>3,023 点</b>，每年 1 月按上年 CPI 调一次。<br>'
         '实际执行时用 <b>2,900~3,050</b> 这个区间，不要用单点——平台有 150 点宽，'
         '差 13 点就可能整段错过（2016-02-16 就是差 13 点）。<br>'
         '<b>不要设"跌破 2,800 才买"</b>：那是断崖以下是 2 笔、+2.8%，11 年白等。<br>'
         '<b>空仓期钱放 2.5% 以上</b>，否则等待本身在亏。</p></div>')

H.append('<h3>什么情况下这份判决作废</h3>')
H.append('<div class="card"><p style="margin:0">'
         '① CPI 累计涨幅连续 3 年超 4%，导致买入线一年上抬 120 点以上，2,900~3,050 平台失效；<br>'
         '② 出现一次"跌破 2,707（实际 1,880）后 5 年内未收回"——"下有底"被证伪；<br>'
         '③ 上证在 3,000 以上连续停留超过 36 个月且期间无 20% 回调——箱体结构改变，平台不再成立；<br>'
         '④ 换成全 A 中位 PE 十年分位做锚，在同一 11 起点上跑赢本报告 8 笔版本——说明 CPI 点位不是最好的尺子。</p></div>')

# 证伪声明
H.append('<div class="foot">'
         '数据：上证 sh.000001 2008-02-15 ~ 2026-11-11（baostock，4,555 根）；M2 月度 2008-01 ~ 2026-07（223 期，'
         '滞后 1 月 15 日生效避免前视）；CPI 月度同比 → 12 次方根还原环比 → 定基（2008-01=100，末值 144.0）；'
         '执行标的 510300 沪深300ETF（前复权收盘），双边万五成本。<br>'
         '脚本：<code>d90_m2_equiv.py</code>（三口径换算）、<code>d91_deflate.py</code>（中枢/分位/单点诊断）、'
         '<code>d92_backcalc.py</code>（熊底牛顶折算）、<code>d93_absolute.py</code>（绝对值闸门）、'
         '<code>d94_cross.py</code>（两线交叉/空仓成本/逐笔）、<code>d95_verify.py</code>（在场比例/收益驱动）、'
         '<code>d96_scan.py</code>（细扫/去最大一笔/ETF 错配）。<br>'
         '已知缺陷：① 标的用 510300、闸门用上证点位，上证 2,900~3,100 时 510300 价格跨度 2.219~3.879'
         '（变异系数 12.6%，相关 0.848）——<b>同样的 3,000 点，ETF 贵贱差 75%</b>，本报告所有收益都继承这个错配；'
         '② M2 只有 2008 起，三口径均值非同样本；③ 11 起点非独立样本；④ n=5、n=2 的统计量不可当精度。<br>'
         '<b>颜色约定：本报告的红色/绿色只表示数值正负与警示级别，不表示涨跌</b>（与行情软件的习惯不同）。<br>'
         '本报告为历史数据检验，不含投资建议。</div>')

H.append('</div>')

# ---------- 图表脚本 ----------
bt_n = [r['nom'] for r in bt]
bt_c = [r['real'] for r in bt]
bt_m = [r['m2'] for r in bt]
lab_bt = [r['date'][:7] for r in bt]
nrm_n = [round(x / bt_n[0] * 100, 1) for x in bt_n]
nrm_c = [round(x / bt_c[0] * 100, 1) for x in bt_c]
nrm_m = [round(x / bt_m[0] * 100, 1) for x in bt_m]
SC = d96['scan']

JS_T = r"""
<script>@@LIB@@</script>
<script>
Chart.defaults.font.family='-apple-system,"PingFang SC","Microsoft YaHei",sans-serif';
Chart.defaults.color='#5f5e5a'; Chart.defaults.animation=false;

new Chart(document.getElementById('c1'),{type:'line',
 data:{labels:@@L1@@,datasets:[
  {label:'名义',data:@@D1@@,borderColor:'#1a1a1a',borderWidth:2,pointRadius:4,tension:0.1},
  {label:'CPI 实际',data:@@D2@@,borderColor:'#854F0B',borderWidth:2,pointRadius:4,tension:0.1},
  {label:'M2 等价',data:@@D3@@,borderColor:'#A32D2D',borderWidth:2,pointRadius:4,tension:0.1}]},
 options:{responsive:true,maintainAspectRatio:false,
  plugins:{legend:{position:'top',labels:{boxWidth:12,font:{size:12}}}},
  scales:{y:{title:{display:true,text:'以 2008-11 底 = 100'},grid:{color:'rgba(0,0,0,.06)'},ticks:{font:{size:11}}},
   x:{grid:{display:false},ticks:{font:{size:11}}}}}});

new Chart(document.getElementById('c2'),{type:'line',
 data:{labels:@@L2@@,datasets:[{label:'3,000 点的实际价值',data:@@D4@@,borderColor:'#A32D2D',backgroundColor:'rgba(163,45,45,.08)',borderWidth:2,fill:true,pointRadius:4,tension:0.1}]},
 options:{responsive:true,maintainAspectRatio:false,
  plugins:{legend:{display:false}},
  scales:{y:{title:{display:true,text:'2008-01 不变价'},grid:{color:'rgba(0,0,0,.06)'},ticks:{font:{size:11}}},
   x:{grid:{display:false},ticks:{font:{size:11}}}}}});

new Chart(document.getElementById('c3'),{type:'line',
 data:{labels:@@L3@@,datasets:[
  {label:'固定 3,000',data:@@D5@@,borderColor:'#1a1a1a',borderWidth:2,pointRadius:3,borderDash:[5,3],tension:0},
  {label:'实际 2,100 等值线',data:@@D6@@,borderColor:'#A32D2D',borderWidth:2.5,pointRadius:4,tension:0.1}]},
 options:{responsive:true,maintainAspectRatio:false,
  plugins:{legend:{position:'top',labels:{boxWidth:12,font:{size:12}}}},
  scales:{y:{title:{display:true,text:'上证点位'},grid:{color:'rgba(0,0,0,.06)'},ticks:{font:{size:11}}},
   x:{grid:{display:false},ticks:{font:{size:11}}}}}});

new Chart(document.getElementById('c4'),{type:'bar',
 data:{labels:@@L4@@,datasets:[{label:'总收益',data:@@D7@@,backgroundColor:@@D8@@,borderRadius:3,borderSkipped:false}]},
 options:{responsive:true,maintainAspectRatio:false,
  plugins:{legend:{display:false}},
  scales:{y:{title:{display:true,text:'总收益 %'},grid:{color:'rgba(0,0,0,.06)'},ticks:{font:{size:11}}},
   x:{title:{display:true,text:'买入线（上证点位）'},grid:{display:false},ticks:{font:{size:10}}}}}});
</script>
"""

VALS = {
    '@@LIB@@': CHARTJS,
    '@@L1@@': json.dumps(lab_bt, ensure_ascii=False),
    '@@D1@@': json.dumps(nrm_n), '@@D2@@': json.dumps(nrm_c), '@@D3@@': json.dumps(nrm_m),
    '@@L2@@': json.dumps([r['year'] for r in tk], ensure_ascii=False),
    '@@D4@@': json.dumps([r['real'] for r in tk]),
    '@@L3@@': json.dumps([r['year'] for r in tl], ensure_ascii=False),
    '@@D5@@': json.dumps([3000] * len(tl)),
    '@@D6@@': json.dumps([r['nom2100'] for r in tl]),
    '@@L4@@': json.dumps([r['th'] for r in SC], ensure_ascii=False),
    '@@D7@@': json.dumps([r['total'] for r in SC]),
    '@@D8@@': json.dumps(['#854F0B' if 2900 <= r['th'] <= 3050 else ('#9C9A93' if r['th'] < 2900 else '#6B6A65') for r in SC]),
}
JS = JS_T
for k, v in VALS.items():
    JS = JS.replace(k, str(v))

HTML = ('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>货币口径等值点位 · 算回来的三千点</title>' + CSS + '</head><body>'
        + ''.join(H) + JS + '</body></html>')

open(OUT, 'w', encoding='utf-8').write(HTML)
print('written', OUT, len(HTML), 'bytes')
