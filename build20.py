# -*- coding: utf-8 -*-
"""生成 20_三次机会与真实总价值 报告"""
import os, json

B = os.path.dirname(os.path.abspath(__file__))
OUT = 'G:/探讨集/20_三次机会与真实总价值_20261112.html'
D = json.load(open(os.path.join(B, 'd86_pack.json'), encoding='utf-8'))

CHARTJS = open(os.path.join(B, 'chart.umd.min.js'), encoding='utf-8').read()
SEG = D['segs']
LAY = D['layers']
ZZ = D['zz_sens']
TOP = D['tops']
ST = D['starts']
M2 = D['m2']
TR = D['trades']
DDS = D['dd_sig']
PF = D['perf']
W = D['sse_week']


def col(v):
    if v is None:
        return '#5f5e5a'
    return '#A32D2D' if v > 0 else ('#1C6B3C' if v < 0 else '#5f5e5a')


# ---------- 表：三段 ----------
segrows = ''
for r in SEG:
    segrows += (f"<tr><td>{r['start']} → {r['end']}</td>"
                f"<td>{r['p0']:,} → {r['p1']:,}</td>"
                f"<td style=\"color:#A32D2D;font-weight:500\">+{r['ret']}%</td>"
                f"<td>{r['months']}</td><td>{r['dd']}%</td><td>{r['mdd']}%</td>"
                f"<td>{r['pe']}</td></tr>")

# ---------- 表：阈值 ----------
layrows = ''
for r in LAY:
    hl = ' class="hl"' if r['th'] == 100 else ''
    layrows += (f"<tr{hl}><td>涨幅 ≥ {r['th']}%</td><td><b>{r['n']}</b> 次</td>"
                f"<td>{r['share']}%</td><td>+{r['comp']:,}%</td></tr>")

# ---------- 表：顶 ----------
toprows = ''
for i, r in enumerate(TOP):
    chg = ''
    if i > 0:
        c = r['lv'] / TOP[i - 1]['lv'] - 1
        chg = f"<td style=\"color:{col(c)}\">{c*100:+.1f}%</td>"
    else:
        chg = '<td>—</td>'
    toprows += f"<tr><td>{r['date']}</td><td><b>{r['lv']:,}</b></td>{chg}<td>{r['pe']}</td></tr>"

# ---------- 表：多起点 ----------
strows = ''
for r in ST:
    real = f"{r['real']:+.2f}%" if r['real'] is not None else '—'
    infl = f"{r['infl']:.2f}%" if r['infl'] is not None else '—'
    strows += (f"<tr><td>{r['s']}</td><td>{r['v0']:,}</td><td>{r['yrs']}</td>"
               f"<td style=\"color:{col(r['nom'])}\">{r['nom']:+.2f}%</td>"
               f"<td>{infl}</td><td style=\"color:{col(r['real'] if r['real'] is not None else 0)}\">{real}</td></tr>")

# ---------- 表：M2 ----------
m2rows = ''
for r in M2:
    hl = ' class="hl"' if r['t'] in ('2024-09', '2026-07') else ''
    m2rows += (f"<tr{hl}><td>{r['t']}</td><td>{r['sse']:,}</td><td>{r['m2']:.1f} 万亿</td>"
               f"<td><b>{r['ratio']:.1f}</b></td></tr>")

# ---------- 表：8 笔 ----------
trrows = ''
for r in TR:
    bad = r['dd'] > -50
    trrows += (f"<tr><td>{r['t']}</td><td>{r['sse']:,}</td><td>{r['cycle_hi']:,}</td>"
               f"<td style=\"color:{'#854F0B' if bad else '#1C6B3C'}\">{r['dd']}%</td>"
               f"<td style=\"color:#854F0B\">差 {abs(r['dd']+55):.1f}pt</td></tr>")

# ---------- 表：收益 ----------
pfrows = ''
for r in PF:
    c = col(r['cagr'] - 5)
    pfrows += (f"<tr><td style=\"text-align:left\">{r['case']}</td><td>{r['mult']}x</td>"
               f"<td>{r['yrs']}</td><td style=\"color:{c};font-weight:500\">{r['cagr']:.2f}%</td>"
               f"<td style=\"text-align:left;color:#8c8b86;font-size:12px\">{r['note']}</td></tr>")

# 三段高亮序列
lbl = W['labels']
val = W['values']
hi = [None] * len(val)
for r in SEG:
    s, e = r['start'][:7], r['end'][:7]
    for i, x in enumerate(lbl):
        if s <= x <= e:
            hi[i] = val[i]

html = f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>三次机会与真实总价值</title>
<script>{CHARTJS}</script>
<style>
:root{{--bg:#faf9f6;--card:#fff;--tx:#1a1a1a;--tx2:#5f5e5a;--tx3:#8c8b86;--bd:rgba(0,0,0,.12);--red:#A32D2D;--green:#1C6B3C;--amber:#854F0B}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--tx);font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;font-size:14px;line-height:1.75}}
.wrap{{max-width:1000px;margin:0 auto;padding:40px 24px 80px}}
h1{{font-size:24px;font-weight:500;margin:0 0 6px}}
.sub{{color:var(--tx2);font-size:13px;margin-bottom:28px}}
h2{{font-size:17px;font-weight:500;margin:44px 0 4px;padding-top:20px;border-top:1px solid var(--bd)}}
h2 .no{{color:var(--tx3);font-size:13px;margin-right:8px}}
h3{{font-size:14px;font-weight:500;margin:22px 0 4px}}
.note{{color:var(--tx2);font-size:13px;margin:0 0 16px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:18px 20px;margin:14px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}}
.kpi{{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:14px 16px}}
.kpi .l{{font-size:12px;color:var(--tx2);margin-bottom:6px}}
.kpi .n{{font-size:25px;font-weight:500;line-height:1.2}}
.kpi .d{{font-size:12px;color:var(--tx3);margin-top:4px}}
.verdict{{border-left:3px solid var(--red);background:#FCEBEB;padding:15px 18px;border-radius:0 8px 8px 0;margin:18px 0}}
.verdict b{{color:var(--red)}}
.warn{{border-left:3px solid var(--amber);background:#FAEEDA;padding:14px 18px;border-radius:0 8px 8px 0;margin:16px 0;font-size:13px}}
.ok{{border-left:3px solid var(--green);background:#EAF3ED;padding:14px 18px;border-radius:0 8px 8px 0;margin:16px 0;font-size:13px}}
.chartbox{{position:relative;width:100%;height:300px;margin-top:8px}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}}
th,td{{padding:7px 10px;text-align:right;border-bottom:1px solid var(--bd)}}
th:first-child,td:first-child{{text-align:left}}
th{{font-weight:500;color:var(--tx2);font-size:12px}}
tr:last-child td{{border-bottom:none}}
tr.hl td{{background:#FBF3F2}}
.foot{{margin-top:48px;padding-top:20px;border-top:1px solid var(--bd);font-size:12px;color:var(--tx3);line-height:1.9}}
code{{background:#F1EFE8;padding:1px 5px;border-radius:4px;font-size:12px}}
.big{{font-size:15px;line-height:1.9}}
blockquote{{margin:10px 0;padding:10px 16px;border-left:2px solid var(--bd);color:var(--tx2);font-size:13px}}
</style></head><body><div class="wrap">

<h1>三次机会、顶在递降，与「A 股真实总价值」</h1>
<div class="sub">上证 sh.000001 · 1990-2026（7,482 根）；勃发段用 2000 起 6,507 根 · 执行标的 510300 前复权 · 2026-11-12</div>

<div class="verdict">
<b>你说的「十年磨一剑，三次机会」，在数字上成立——但成立的方式和你以为的不一样。</b><br><br>
<b>对的部分：</b>只做勃发段、其余收手，22 年确实只有 <b>3 段 ≥100%</b>；只做这三段是 <b>32.4 倍 / CAGR 17.7%</b>，买入持有只有 5.09%。<b>「只吃能看懂的那一段」在数学上是成立的，一段就够。</b><br><br>
<b>三处偏差：</b><br>
<b>① 频率。</b>22 年 3 次 = <b>7.3 年一次</b>，不是 10 年 3 次。你的期待值快了 2.2 倍。<br>
<b>② 时间。</b>不是「可能等 11 年」，是 <b>已经等了 11 年 4 个月</b>（2015-06 → 2026-11 零次 ≥100%，最大只有 +54%）。<br>
<b>③ 执行。</b>严格执行骨架，这 11 年应该是 <b>0 笔交易</b>；实际做了 <b>8 笔</b>，而且没有一笔达到勃发段起点的深度。你那 +211.6% 不是勃发段的收益，是箱体的收益。<br><br>
<b>「真实总价值」这一问：A 股不会消失，但它不增值。</b>多起点看，名义 4~5%/年、扣平减后 3% 左右——跑赢 CPI 一点，跑输 M2 一大截。<b>它是保值品，不是增值品。这恰好是「只能靠勃发段赚钱」的另一种说法。</b>
</div>

<h2><span class="no">一</span>三次机会：是真的</h2>
<p class="note">用 ZigZag（25% 反转）切上证的摆动，取涨幅 ≥100% 的上涨段。22 年只有三段：</p>
<table>
<tr><th>区间</th><th>点位</th><th>涨幅</th><th>月数</th><th>起点距周期顶</th><th>段内日频最大回撤</th><th>起点 PE 中位</th></tr>
{segrows}
</table>
<div class="chartbox" style="height:220px"><canvas id="c1"></canvas></div>
<p class="note">三段起点的共同特征只有一个：<b>距上一轮牛市顶 −55% / −72% / −68%</b>。PE 中位 25.3 / 18.4 / 36.4 差了一倍，<b>不是估值信号</b>。段内日频回撤 −16.6 / −13.3 / −11.7%，持仓体验比想象的好。<b>你要的门槛，历史上就是「从顶上砍掉一半以上」。</b></p>

<h3>只做这三段 vs 买入持有</h3>
<table>
<tr><th>做法</th><th>倍数</th><th>年数</th><th>CAGR</th><th>口径</th></tr>
{pfrows}
</table>
<div class="warn">
<b>先说清楚这三行里哪一行不能信。</b>第一行「32.4 倍」是<b>上帝视角上界</b>——你必须事前知道哪一段会到 100%，而这正是本节要否掉的东西，它只能标为「不可实现」。真正可比的是第二行：<b>只做第三段（+165%）、之后 11.4 年空仓拿 2% 现金，CAGR 9.39%</b>，同期买入持有 4.86%。<b>一段就够跑赢一倍。</b>
</div>

<h2><span class="no">二</span>但「三次」不是市场常数，是你选的阈值</h2>
<p class="note">同样一段历史，只换门槛：</p>
<table>
<tr><th>门槛</th><th>次数</th><th>占全部时间</th><th>复合收益</th></tr>
{layrows}
</table>
<p class="note">ZigZag 反转阈值调到 15% / 20% / 25% / 30% / 40%，<b>≥100% 的段始终是 3 次</b>（时间占比 17~21%）——这个稳健。但<b>「三次」本身完全由 100% 这条线定义</b>：降到 50% 是 6 次，升到 150% 是 1 次。<b>「三次机会」是真实的历史形态，但它不是自然常数，是你把靶子钉在「翻倍」上得到的读数。</b></p>

<h2><span class="no">三</span>2015 年之后，11 年 4 个月，零次</h2>
<div class="grid">
<div class="kpi"><div class="l">2015-06 → 2026-11</div><div class="n" style="color:#854F0B">0 次</div><div class="d">≥100% 的段</div></div>
<div class="kpi"><div class="l">这段时间最大一段</div><div class="n">+54%</div><div class="d">2024-01→2026-05</div></div>
<div class="kpi"><div class="l">上一轮最长空窗</div><div class="n">46.7 月</div><div class="d">2009-08 → 2013-06</div></div>
<div class="kpi"><div class="l">按 22 年 3 次推算</div><div class="n">≈21%</div><div class="d">11.4 年空手的概率</div></div>
</div>
<div class="chartbox"><canvas id="c2"></canvas></div>
<p class="note">灰线是上证周线，<b>红色区段就是那三次</b>。注意右边那一大段空白：<b>2015 年 6 月之后，整整 11 年没有出现过一次翻倍行情。</b>你现在的处境不是「准备开始等」，是「已经等了 11 年」。</p>
<p class="note">按 22 年 3 次（λ≈0.136/年）算，连续 11.4 年不出事的概率约 <b>21%</b>——<b>五分之一，属于常态，不是尾部</b>。所以「等到第十一年还没有」这件事不会给你任何「快来了」的信息。n=3 的到达率区间极宽，谁也算不准。</p>

<h2><span class="no">四</span>顶在往下走，这才是箱体收缩的机制</h2>
<table>
<tr><th>牛市顶</th><th>点位</th><th>较上轮</th><th>全A PE 中位</th></tr>
{toprows}
</table>
<p class="note">三轮顶：<b>6,092 → 5,166 → 4,243</b>，依次 −15.2%、−17.9%。<b>但 PE 中位是 53.6 → 70.0 → 42.1</b>——所以不是「市场变差」，是<b>盈利在涨、给的倍数在降</b>。</p>
<div class="warn">
<b>这条比「三次机会」更值得警惕。</b>如果估值中枢继续下移，下一个顶可能只有 3,600。那么「距顶 −55%」的买入门槛就是 <b>1,620 点</b>——而你按 3,000 点等，会发现自己<b>永远不够便宜</b>。<br>
反过来说：<b>「跌 50%」这个门槛是跟着顶走的，顶降，门槛也降。</b>用固定点位（3,000）当门槛，就是在赌顶不降。这 11 年顶已经降了两次。
</div>

<h2><span class="no">五</span>真实总价值：不消失，也不增值</h2>
<p class="note">先否掉一个我自己算错的数：本地库 <code>daily_tencent_full</code> 只有 3,179 只（2025 年末实际约 5,400 只），算出的总市值 48.8 万亿只有真实的 <b>44%</b>。<b>绝对市值不可用，下面全部改用指数口径。</b></p>

<h3>五个起点的名义与实际</h3>
<table>
<tr><th>起点</th><th>点位</th><th>年数</th><th>名义 CAGR</th><th>平减(GDP)</th><th>实际 CAGR</th></tr>
{strows}
</table>
<p class="note">中位约 <b>4.4% 名义 / 3.3% 实际</b>。极值来自起点：2008-01（大牛后下跌初段）起算是 <b>−1.05%</b>，2005-06（底部）起算是 <b>+5.80%</b>。<b>这两个数都对，差别全在你在哪一年开始数。</b>任何单一起点的长期收益都是选择性叙事，包括我上一轮那个「11 年 +18.5%」。</p>

<h3>「算上通货膨胀」到底算哪个</h3>
<div class="ok">
<b>他自己问的是 CPI 口径，那个答案是：约 −0.5%/年（实际购买力）。</b>上证名义 4~5%，CPI 约 2%，扣完剩 2~3%；把股息（2~2.5%）算回去，<b>实际购买力大致打平到微亏</b>。<b>A 股不是「不保值」，是「保值但不增值」。</b><br><br>
<b>M2 口径不能用做判决，只能做位置读数。</b>上证/M2 从 2008 的 104.9 掉到 2026 的 10.8（−90%），但这不 indict A 股——同期存款（1.5% vs M2 12%）同样白干，房子同样白干。<b>用 M2 做基准，是 indict 所有人民币资产，不是 indict 股票。</b>
</div>

<table>
<tr><th>时点</th><th>上证</th><th>M2</th><th>上证 / M2</th></tr>
{m2rows}
</table>
<p class="note">这张表唯一的正确用法：<b>2024-09 的 3,336 点和 2026-07 的 3,832 点，比值都是 10.8——两年涨的 15%，正好被货币增量吃掉。</b>这是「位置读数」：<b>从货币池的角度看，2026 年 7 月和 2024 年 9 月是同一个地方。</b>不是质量判决。</p>

<h2><span class="no">六</span>最要紧的一条：8 笔 vs 0 笔</h2>
<p class="note">上一轮那套可执行规则（跌破 3,000 磨底 + 上穿 MA20 入场、回撤 10% 或热度出场），2015-2026 做了 8 笔、+211.6%。把这 8 笔的进场日，放到「距上一轮牛市顶」这把尺子上：</p>
<table>
<tr><th>进场</th><th>上证</th><th>上一轮牛市顶</th><th>距顶</th><th>够不够 −55%</th></tr>
{trrows}
</table>
<div class="verdict">
<b>8 笔最深 −48.4%（2018-11-05），没有一笔到 −55%。</b>而三次大段起点是 −54.9% / −72.0% / −68.0%。<br><br>
<b>也就是说：严格执行你的骨架，2015-2026 应该是 0 笔交易、100% 空仓。</b>你实际做的 8 笔，全部落在骨架本该收手的时段里——<b>它们不是骨架的血肉，是另一套策略（箱体来回）</b>。<br><br>
<b>所以 +211.6% 不能用来证明「十年三次勃发」这个理论。</b>它是「3,000 点箱体」的成绩单。<b>你一直在用箱体的收益，验证勃发的信念。</b>这两件事各自可能都对，但它们不能互相证明。
</div>
<p class="note">顺带一个容易骗人的陷阱：如果拿 <b>2007 年的 6,092</b> 当分母，2016 年之后每一笔都显示「跌了 53~56%」，看起来全都够深。但那样算，<b>6,092×0.5 = 3,046</b>，「距历史顶跌一半」在 2016-2025 就等价于「上证低于 3,046」——<b>又退化成你那个 3,000 点</b>。上一轮回吐锚翻车就是这个原因。分母必须用「上一轮牛市顶」，不能用「历史最高」。</p>

<h2><span class="no">七</span>「跌一半就买」这个信号本身有多硬</h2>
<table>
<tr><th>门槛</th><th>触发次数</th><th>独立熊市</th><th>买后平均再跌</th><th>3 年平均</th><th>3 年最差</th></tr>
<tr><td>回撤 ≤ −50%</td><td>9</td><td>约 5</td><td style="color:#1C6B3C">−15.1%</td><td style="color:#A32D2D">+23%</td><td style="color:#1C6B3C">−25%</td></tr>
<tr><td>回撤 ≤ −60%</td><td>3</td><td>3</td><td style="color:#1C6B3C">−16.3%</td><td style="color:#1C6B3C">−2%</td><td style="color:#1C6B3C">−13%</td></tr>
</table>
<div class="warn">
<b>这张表杀掉了「重仓」的唯一理由。</b>−60% 档（更深的坑）3 年均值 <b>−2%</b>，比 −50% 档的 +23% 差得多——<b>「跌得越深越安全」是错的</b>。而重仓的全部依据就是「深 = 安全」。<br><br>
另外 9 次不是 9 个独立样本：1996-2026 只有约 5 个熊市，同一轮熊市反复触发。<b>有效样本 n≈5。</b><br><br>
<b>可执行的结论：「重仓」只能是最终仓位，不能是入场仓位。</b>按「−50% 触发后平均还要再跌 15%、最深再跌 43%」，必须分 4 档建仓（−50 / −60 / −70 / 跌破前低再补），<b>第一笔只上 1/4</b>。这不是保守，是「你不可能买在最低点」这件事的价格。
</div>

<h2><span class="no">八</span>关于日本：你说错了，但结论站在你这边</h2>
<blockquote>「如果真到了那种地步，不是 A 股能不能赚钱了，那是一代人的灾难，股市只是一部分体现，好比天灾人祸，没用。」</blockquote>
<p class="note">这个反驳我接受一半，但事实要摆清楚：<b>1990-2012 的日本不是「一代人的灾难」。</b>日经从 38,957 跌到 8,000（−79%，23 年），同期日本名义 GDP 基本持平、失业率 2~5%、企业照常经营、东京照常是全世界最贵的城市之一。<b>社会运转正常，只有股市惨。</b></p>
<div class="verdict">
<b>这个事实比你的版本更难反驳，因为它正好否掉了你的挡箭牌。</b>如果日本真是「一代人灾难」，你还能说「炒股不炒股都一样」；<b>正因为社会正常、只有股市独自惨了 23 年，才说明「股市可以独立于经济长期不回报」——这才是持股人的真实风险。</b><br><br>
而且<b>你不需要举日本</b>：你自己的数据已经证明了同一件事——<b>2006-2025 这 19 年，中国 GDP 从 22.3 万亿涨到 140.2 万亿（6.3 倍），而上证从 2,675 只涨到 3,681（1.38 倍）。「股市回报与经济增长脱钩」在中国已经发生了 19 年。</b>这是你自己数据里的、小一号的日本，比举日本更难反驳。
</div>

<h2><span class="no">九</span>总判决</h2>
<div class="card big">
<b>1. 骨架成立。</b>「只做熊市后勃发的那一段、其余收手」在 22 年数据上是最优形态——不是「之一」，是明确的：一段足够跑赢买入持有近一倍。<br><br>
<b>2. 三处要改口。</b>7.3 年一次（不是 10 年 3 次）；已经等了 11.4 年（不是「可能等」）；执行要 0 笔（不是 8 笔）。<b>第三条最贵——它意味着过去 11 年的收益不该记在这个理论账上。</b><br><br>
<b>3. 门槛要跟着顶走。</b>你的 −55% 是真的历史门槛，但顶在递降（6,092→5,166→4,243），所以这个门槛也在降。用固定的 3,000 点，就是赌顶不降——11 年里顶已经降了两次。<br><br>
<b>4. 「真实总价值」：不消失，也不增值。</b>名义 4~5%、实际 3% 左右，跑赢 CPI 一点点，跑输 M2 一大截。它的价值不在「持有」，在「那三段」——<b>这恰恰是你说「别的任何时候都收手不做」的经济学理由，比「人性」两个字硬。</b><br><br>
<b>5. 最大的风险不是崩盘，是空窗。</b>如果 2015 后的形态（顶递降 + 零翻倍）延续到下一个十年，你不是亏钱，是<b>十年一动不动，而且规则不会报警</b>。这一条无法用「国家不会允许」对冲，因为日本的 23 年就是在这种信念下发生的。
</div>
<p class="note"><b>若观察到以下任一事实，本报告第 3、5 条作废：</b>① 上证出现一段 ≥100% 且起点距 <b>2026 年那个顶</b> ≥50% 的上涨；② 全 A PE 中位上穿 60（估值中枢重新上移，顶不再递降）；③ 出现「跌破 3,000 后 5 年内未收回」的一次（下有底被证伪）。</p>

<div class="foot">
数据：<code>idx_sse_all.parquet</code>（baostock sh.000001，1990-12-19 ~ 2026-11-11，8,761 根，本节用 1996 起 7,482 根 / 2000 起 6,507 根）；<code>all_a_pe_ttm.parquet</code>（262 期月度）；<code>macro/gdp.parquet</code>（2006-2025 年度）；<code>macro/money_supply.parquet</code>（2008-01 ~ 2026-07）；<code>510300_沪深300ETF.parquet</code>（前复权）。<br>
脚本：<code>d80_value.py</code>（总市值，已作废）· <code>d81_real.py</code>（多起点名义/实际）· <code>d82_three.py</code>（ZigZag 勃发段）· <code>d83_start.py</code>（起点特征/空窗/敏感性）· <code>d84_dd.py</code>（跌一半信号）· <code>d85_check.py</code>（核 8 笔与段内回撤）· <code>d86_pack.py</code> · <code>build20.py</code>。<br>
口径声明：上证为价格指数（不含分红），文中「实际 CAGR」用 GDP 平减；「8 笔」出自 d63 机械回测，标的 510300 前复权，双边万五。<b>本报告为历史检验，不含投资建议。</b>
</div>
</div>

<script>
const LBL = {json.dumps(lbl)};
const VAL = {json.dumps(val)};
const HI  = {json.dumps(hi)};
const opt = (y) => ({{animation:false, responsive:true, maintainAspectRatio:false,
  plugins:{{legend:{{display:false}}}},
  scales:{{x:{{grid:{{display:false}},ticks:{{maxTicksLimit:8,font:{{size:10}},color:'#8c8b86'}}}},
           y:{{grid:{{color:'rgba(0,0,0,.06)'}},ticks:{{font:{{size:10}},color:'#8c8b86'}}}}}}}});

new Chart(document.getElementById('c1'), {{
  type:'bar',
  data:{{labels:['2005-07 → 2007-10   +502%','2008-11 → 2009-08   +103%','2013-06 → 2015-06   +165%'],
    datasets:[{{data:[502,103,165],
      backgroundColor:['#A32D2D','#C08080','#A32D2D'],
      borderRadius:3, barThickness:60}}]}},
  options:{{animation:false, responsive:true, maintainAspectRatio:false, indexAxis:'y',
    plugins:{{legend:{{display:false}},
      tooltip:{{callbacks:{{label:(c)=>'+'+c.raw+'%'}}}}}},
    scales:{{x:{{grid:{{color:'rgba(0,0,0,.06)'}},ticks:{{callback:v=>'+'+v+'%',font:{{size:11}},color:'#8c8b86'}}}},
             y:{{grid:{{display:false}},ticks:{{font:{{size:11}},color:'#5f5e5a'}}}}}}}}
}});

new Chart(document.getElementById('c2'), {{
  type:'line',
  data:{{labels:LBL, datasets:[
    {{data:VAL, borderColor:'#C9C7C0', borderWidth:1.2, pointRadius:0, tension:0, fill:false}},
    {{data:HI, borderColor:'#A32D2D', borderWidth:2.4, pointRadius:0, tension:0, fill:false, spanGaps:false}}
  ]}},
  options:opt()
}});
</script>
</body></html>'''

open(OUT, 'w', encoding='utf-8').write(html)
print('written', OUT, len(html))
