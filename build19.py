# -*- coding: utf-8 -*-
"""生成 19_四条反驳的检验 报告"""
import os, json

B = os.path.dirname(os.path.abspath(__file__))
CHARTJS = open(os.path.join(B, 'chart.umd.min.js'), encoding='utf-8').read()
OUT = 'G:/探讨集/19_四条反驳的检验_下有底与回吐_20261112.html'

d70 = json.load(open(os.path.join(B, 'd70_result.json'), encoding='utf-8'))
d72 = json.load(open(os.path.join(B, 'd72_result.json'), encoding='utf-8'))
d73 = json.load(open(os.path.join(B, 'd73_result.json'), encoding='utf-8'))
d74 = json.load(open(os.path.join(B, 'd74_result.json'), encoding='utf-8'))

A = {r['L']: r for r in d70['A_bottom']}
BULL = d70['B_bull_giveback']
HOLD = d72['hold']
SENS = d74['sens_corrected']

def col(v, inv=False):
    if v > 0:
        return '#A32D2D'
    if v < 0:
        return '#1C6B3C'
    return '#5f5e5a'

# ---- 下有底表 ----
arows = ''
for L in [2000, 2400, 2500, 2800, 3000]:
    r = A[L]
    recs = r['recs']
    # 排除样本起点被左截断的那一段（起始日 = 1996-01-02）
    real = [x for x in recs if x['start'] != '1996-01-02']
    mx = max(real, key=lambda x: x['dur']) if real else None
    arows += (f"<tr><td>{L}</td><td>{len(real)}</td><td>{r['never']}</td>"
              f"<td>{mx['dur']} 日</td><td>{mx['dur']/244:.1f} 年</td>"
              f"<td style=\"text-align:left\">{mx['start'][:7]} → {mx['end'][:7]}</td></tr>")

# ---- 牛市回吐表 / 图 ----
brows = ''
blabels, bvals = [], []
for r in BULL:
    g = r['give_pct']
    blabels.append(r['peak'][:7])
    bvals.append(g)
    br = '#A32D2D' if g >= 75 else '#854F0B'
    brows += (f"<tr><td>{r['start'][:7]} {r['s_lvl']:.0f}</td>"
              f"<td>{r['peak'][:7]} <b>{r['p_lvl']:.0f}</b></td>"
              f"<td>{r['trough'][:7]} {r['e_lvl']:.0f}</td>"
              f"<td>+{r['up']:.0f}%</td>"
              f"<td style=\"color:{br};font-weight:500\">{g:.0f}%</td>"
              f"<td>{r['trough_vs_start']:.2f}</td></tr>")

# ---- 个股持有表 ----
hrows = ''
for r in HOLD:
    hrows += (f"<tr><td>{r['start']}</td><td>{r['n']}</td>"
              f"<td style=\"color:{col(r['med'])}\">{r['med']:+.1f}%</td>"
              f"<td style=\"color:{col(r['mean'])}\">{r['mean']:+.1f}%</td>"
              f"<td>{r['pos']:.1f}%</td><td>{r['dbl']:.1f}%</td>"
              f"<td>{r['half']:.1f}%</td>"
              f"<td style=\"color:{col(r['etf'])}\">{r['etf']:+.1f}%</td>"
              f"<td>{r['med'] - r['etf']:+.0f}pt</td></tr>")

# ---- 四锚 ----
GRID = d74['grid']
STARTS = list(GRID.keys())
grows = ''
for sd in STARTS:
    r = GRID[sd]
    bh = r['A']['bh']['total']
    grows += (f"<tr><td>{sd}</td><td>{bh:+.1f}%</td>")
    for m in ['A', 'B', 'C1', 'C2']:
        v = r[m]['stats']['total']
        grows += f"<td style=\"color:{col(v - bh)};font-weight:{'500' if v > bh else '400'}\">{v:+.1f}%</td>"
    grows += f"<td>{r['C2']['expo']:.0f}%</td></tr>"
grows += ('<tr class="hl"><td><b>跑赢买入持有</b></td><td>—</td>' +
          ''.join(f"<td><b>{sum(1 for sd in STARTS if GRID[sd][m]['stats']['total'] > GRID[sd]['A']['bh']['total'])}/{len(STARTS)}</b></td>"
                  for m in ['A', 'B', 'C1', 'C2']) + '<td>—</td></tr>')

# ---- 阈值敏感性 ----
slabels = [f"≥{int(r['th']*100)}%" for r in SENS]
svals = [r['total'] for r in SENS]

# ---- 逐年 ----
yrows = ''
for r in d73['yearly']:
    yrows += (f"<tr><td>{r['y']}</td><td>{r['n']}</td>"
              f"<td style=\"color:{col(r['med'])}\">{r['med']:+.1f}%</td>"
              f"<td>{r['pos']:.0f}%</td>"
              f"<td style=\"color:{col(r['idx'] or 0)}\">{(r['idx'] or 0):+.1f}%</td>"
              f"<td>{r['beat']:.0f}%</td></tr>")

S = {r['th']: r for r in SENS}

html = f'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>四条反驳的检验 · 下有底与回吐</title>
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
.chartbox{{position:relative;width:100%;height:290px;margin-top:8px}}
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
@media(max-width:700px){{}}
</style></head><body><div class="wrap">

<h1>四条反驳的检验</h1>
<div class="sub">针对「下有底 / 只涨 / 不退市就能拿 / 牛市后回吐 3/5~3/4」逐条实测 ·
上证 sh.000001 1996-01~2026-11（7,482 根）· 个股 3,169 只到 2025-12-31 · 执行 510300 · 双边万五</div>

<div class="verdict big">
<b>四条里三条成立，但成立的方式各不一样。</b><br>
① <b>下有底 —— 方向成立，证据很薄。</b>三十年里确实没有一次「跌破后永不收回」，但 2500 以下只有 <b>2 段</b>，最长在下方 777 日。这是 n=2 的归纳，不是规律。<br>
② <b>只涨 —— 涨是真，慢也是真。</b>2015-2025 上证 +18.5%（年化 1.6%），300ETF +60.3%（年化 4.3%）。「没赶上车不考虑」等于把目标设成不亏，和「年入 10 万」直接打架。<br>
③ <b>不退市就能长期持有 —— 对，但对的是拿着 ETF，不是拿着你选的股。</b>个股中位 +26.9%，ETF +60.3%，只有 34.7% 的股票跑赢。<br>
④ <b>回吐 3/5~3/4 —— 描述对（历史中位 81%），当买点就崩。</b>我第一版做出来 +215%，第二版修正顶的定义后掉到 +65.8%。<b>翻车的是我，不是他。</b>
</div>

<h2><span class="no">一</span>「跌到 2500 以下数十年不回弹，国家不会允许」</h2>
<p class="note">把「低于某点位的连续区段」按 gap 20 日合并（20 日内的反复算同一段），排除 1996 年样本起点被左截断的那一段（那时指数本来就在 538 点）。</p>
<table>
<tr><th>点位</th><th>真实段数</th><th>永不收回</th><th>最长在下方</th><th>折合年数</th><th>那一段</th></tr>
{arows}
</table>
<p class="note">另：各点位最后一次低于 —— 2500 是 <b>2019-01-03（2,440 点）</b>，距今 <b>2,869 天（7.9 年）</b>；3000 是 <b>2024-09-25</b>，距今 777 天（2.1 年）。</p>

<div class="ok">
<b>成立的部分：</b>三十年内，没有任何一次「跌破后永不收回」。最长的一次是 2010-04→2014-12 的 4.7 年（3000 点以下）。
</div>
<div class="warn">
<b>但不能当公理的三条：</b><br>
<b>① n=2。</b>2500 以下只有两段（2008 年的 175 日、2011-14 年的 777 日）。用两个观察说「不可能」，比用三十年说「不会」更弱。<br>
<b>② 机制不可复制。</b>撑起这三次回弹的是三件互不重复的事：股权分置改革（2006）、四万亿（2009）、杠杆牛（2015）。「从未发生」是三次不同原因的合并，不是一条规律。<br>
<b>③ 同一条信念在别处被证伪过。</b>1989-2012 的日本同样有「政府不会允许」的共识（PKO 价格维持、托市基金），日经从 38,957 跌到 8,000 用了 22 年，其间 21 年低于起点。这不是说中国会重演，是说「国家不会允许」<b>可以是对的，但它不是证据</b>。
</div>
<p class="note">可检验形式：「跌破 X 后 N 年内收回的概率」。按上表，N=5 年时是 5/5，但分母只有 5。</p>

<h2><span class="no">二</span>「A 股指数只可能涨」</h2>
<div class="grid">
<div class="kpi"><div class="l">上证 2015-01 → 2025-12</div><div class="n" style="color:#A32D2D">+18.5%</div><div class="d">11 年 · 年化 1.6%</div></div>
<div class="kpi"><div class="l">300ETF 同期</div><div class="n" style="color:#A32D2D">+60.3%</div><div class="d">年化 4.3%</div></div>
<div class="kpi"><div class="l">2015-06 高点 5,166</div><div class="n">10.4 年</div><div class="d">至今未收回</div></div>
<div class="kpi"><div class="l">现在（2026-11）</div><div class="n">3,681</div><div class="d">距顶 −28.7%</div></div>
</div>
<p class="note">「涨到 6000-7000 没赶上车，不在考虑范围」——这句话把目标函数设成了「不亏」而不是「赚」。空仓 11 年，钱还在，但那 10 万年化永远来不了。<b>这不是数据问题，是两个目标在打架，我不替你选，但必须指出来。</b></p>

<h2><span class="no">三</span>「只要不退市，长期持有就行」</h2>
<p class="note">本地库 <code>daily_tencent_full</code> 3,169 只，前复权收盘（含分红调整），持有到 2025-12-31。</p>
<table>
<tr><th>起点</th><th>n</th><th>中位</th><th>均值</th><th>正收益</th><th>翻倍</th><th>腰斩</th><th>同期 300ETF</th><th>中位−ETF</th></tr>
{hrows}
</table>

<div class="grid" style="margin-top:16px">
<div class="kpi"><div class="l">持有过程最大回撤（中位）</div><div class="n" style="color:#1C6B3C">−60.3%</div><div class="d">2015 起点，path 内最低 / 起点价</div></div>
<div class="kpi"><div class="l">曾经腰斩过的</div><div class="n">77.7%</div><div class="d">−70% 以上的占 24.6%</div></div>
<div class="kpi"><div class="l">11 年里中位为正的年数</div><div class="n">5 / 11</div><div class="d">跑赢上证也是 5 / 11</div></div>
<div class="kpi"><div class="l">跑赢 300ETF 的个股</div><div class="n" style="color:#1C6B3C">34.7%</div><div class="d">2015 起点</div></div>
</div>

<h3>逐年的真实体验</h3>
<table>
<tr><th>年份</th><th>个股数</th><th>个股中位</th><th>正收益占比</th><th>上证</th><th>跑赢上证</th></tr>
{yrows}
</table>
<p class="note">2017 年个股中位 <b>−18.5%</b> 而上证 +5.5%（漂亮 50 行情）；2018 年 −34.4%；2024 年 −4.8% vs +13.1%。<b>「长期持有」是一半年份在亏、且多数年份跑输指数的活。</b></p>

<div class="warn">
<b>存活偏差放在前面说：</b>本地库退市股 <b>0 只</b>（存活率 100%）——这是「当前仍在交易」的快照。按 3 / 6 / 9 / 12% 退市率插入，中位降到 <b>24.0 / 19.6 / 16.4 / 12.7%</b>，正收益比例 62.0 / 60.1 / 58.2 / 56.3%。中位数对「亏多少」不敏感（−60/−80/−95 同结果），对「退市几家」敏感——别拿前者替后者挡。
</div>

<div class="verdict">
<b>定性：动作对，容器错。</b>「不退市就能拿着」是真的（中位为正），但<b>对的是拿着 ETF，不是拿着自己选的股</b>。均值 +82.4% 与中位 +26.9% 的缺口说明这个「不亏」全靠少数暴涨股——「找票一波吃肥」正是赌这个右尾，逻辑自洽，但那需要选股能力，而这 33 笔交易（净 −6,247 元）没提供这种能力的证据。<br><br>
真正的门槛不在收益，在<b>过程</b>：回撤中位 −60.3%、77.7% 曾腰斩，而实测平均持仓 1-3 天、单笔最大 44% 仓位。<b>「拿不住」不是意志问题，是这份回撤数据本身就在劝退。</b>
</div>

<h2><span class="no">四</span>「大牛之后必有大熊，起码回吐五分之二到四分之一」</h2>
<p class="note">先拆语义：<b>1−2/5 = 60%，1−3/4 = 75%</b>——「跌到峰值的 2/5、1/4」与「回吐涨幅的 3/5、3/4」是同一句话。只有把「高度」读成「牛市启动低点」时才是 0.4 / 0.25 倍启动点，那种情形历史从未出现（底/启动 = 1.97 / 1.71 / 1.11 / 1.36 / 1.07 / 1.14，全部 ≥0.90）。所以按 60~75% 回吐推。</p>

<table>
<tr><th>启动 → 顶</th><th>顶部</th><th>熊底</th><th>涨幅</th><th>回吐</th><th>底 / 启动</th></tr>
{brows}
</table>
<p class="note">中位 <b>81%</b>，范围 46~89%。<b>你的「3/5~3/4」方向对，但偏浅——中位比 3/4 还深。</b>而「跌回牛市启动点」从未发生，底/启动最低 1.07。</p>

<div class="card">
<h3 style="margin-top:0">2016-03-02 那次，你说对了</h3>
<p class="note" style="margin-bottom:8px">当日上证 2,850，我上一版判它「贵」（5 年价格分位 0.717）。三种锚的读数：</p>
<div class="grid">
<div class="kpi"><div class="l">5 年价格分位</div><div class="n" style="color:#A32D2D">0.717</div><div class="d">判「贵」· 错了</div></div>
<div class="kpi"><div class="l">10 年价格分位</div><div class="n">0.616</div><div class="d">拉长窗口仍偏高</div></div>
<div class="kpi"><div class="l">距历史顶</div><div class="n">−53.2%</div><div class="d">2016-03-02</div></div>
<div class="kpi"><div class="l">回吐 2015 牛市涨幅</div><div class="n" style="color:#1C6B3C">72.2%</div><div class="d">判「刚好」· 对了</div></div>
</div>
<p class="note" style="margin-top:12px">后续：2,850 → 2018-01 的 3,547，<b>+61.6%</b>。分位说贵、回吐说刚好，<b>回吐对</b>。原因就是你说的：5 年窗口里塞了 5,166 这个顶，把参照系整体抬高了。<b>「分位被大牛市污染」这个诊断是准确的。</b></p>
</div>

<h3>但把它当买点，我翻车了</h3>
<p class="note">第一版把「顶」设成历史最高。2015 之后 peak 永远卡在 5,166、start 卡在 1,950，于是「回吐 ≥70%」在这 11 年退化成「上证 ≲2,900」——和 3,000 点是同一个东西。</p>
<div class="chartbox"><canvas id="c1"></canvas></div>
<p class="note">同一段历史，改成「滚动 1,000 日最高」作顶、「顶之前 1,000 日最低」作启动之后：回吐锚从 <b>+215.3% 掉到 +65.8%</b>，跑赢买入持有从 9/10 掉到 <b>4/10</b>。原因：2020-03 时 peak=3,559 但 start 被拉回 2014 年的 1,974（不是牛市真正启动的 1,849 / 2,638），回吐只算 56.7%，判不进——整段 2020-04→2021-02 的 <b>+57% 全错过</b>。</p>

<table>
<tr><th>起点</th><th>买入持有</th><th>A：&lt;3000</th><th>B：分位≤25</th><th>C1：回吐≥70</th><th>C2：分档</th><th>C2 空仓</th></tr>
{grows}
</table>

<div class="chartbox" style="height:260px"><canvas id="c2"></canvas></div>
<p class="note">阈值敏感性（2015 起）：50%→{S[0.5]['total']}%｜55%→{S[0.55]['total']}%｜60%→{S[0.6]['total']}%｜65%→{S[0.65]['total']}%｜70%→{S[0.7]['total']}%｜75%→{S[0.75]['total']}%｜80%→{S[0.8]['total']}%｜85%→0 笔。<b>锯齿，没有平台。</b>一个参数若唯一依据是「它让回测最好」，它不是参数，是拟合点。</p>

<div class="verdict">
<b>回吐深度是对的常识，不是可执行的信号。</b>三条硬理由：<br>
<b>① 触发点会飘。</b>顶的定义一换（历史最高 / 滚动 4 年 / 上一轮牛市起点），同一时点的读数能从 56.7% 变到 80.7%。「顶」是事后量。<br>
<b>② 60% 只是进区间。</b>历史上回吐 60% 触发后还会再跌：2007 再跌 42%、2009 再跌 20%、2015 再跌 18%、2021 再跌 9%。单点入场必然买在半山腰。<br>
<b>③ 抓不住浅回撤转牛。</b>2020-04 那轮只回撤 34% 就转牛，任何 ≥60% 的闸门都会错过。<br><br>
→ <b>正确用法：当「这轮跌够了没有」的常识尺（中位 81%），不当买入触发器。</b>这正是你自己说的「骨架不需要针灸」——它是骨架，不是扳机。
</div>

<h2><span class="no">五</span>分位：你说对了一半</h2>
<p class="note">B（5 年分位 ≤25）是这轮所有对照里<b>胜率最高（8/10）、回撤最小（−17.1%）</b>的那条，代价是收益只有 A 的一半（+79.1% vs +166.3%）。所以「分位并非不可取」成立——它是<b>更稳但更穷</b>的路，适合「宁可少赚也别深亏」的人。</p>
<p class="note">但诊断只对一半：2016-03 的 10 年分位仍 0.616，说明那个点即使拉长窗口也不算便宜（2,850 相对 2013 年的 1,960 已涨 45%）。<b>分位真正的病不是窗口长度，是它拿「过去的价格」当价值——它永远不知道指数成分股已经换过一轮。</b></p>

<h2><span class="no">六</span>四条合起来</h2>
<div class="verdict big">
<b>三条成立，等级各不同：</b>下有底（方向对，n=2，不能用「不可能」）；长期持有（动作对，但该拿 ETF 不是拿票）；牛市后回吐 3/5~3/4（描述对，中位 81%，当信号就崩）。<br>
<b>第四条被数据推翻：</b>「分位不可取」——分位是三条锚里最稳的一条（8/10 胜，回撤 −17.1%）。<br><br>
唯一没被推翻的是<b>你的直觉精度</b>：2016-03 那次，你比我用的 5 年分位准。但同一个直觉做成规则就失效——因为「回吐够了」是一个判断，「回吐 ≥70%」是一个参数，前者能活，后者会被下一轮牛市的形状打死。
</div>

<div class="foot">
数据：上证 sh.000001 1996-01-02~2026-11-11（baostock，7,482 根）；执行标的 510300 收盘（2012-05-28~2026-09-11）；个股 daily_tencent_full 3,169 只（2010-2025，前复权收盘）。<br>
口径：下线无风险利率，双边万五成本；出场规则统一为「&gt;3500 减半 / 回撤 10% 且距顶 ≥20 日 / 成交额 250 日分位 ≥0.95 且 &gt;3500 清仓」；四锚对照<b>不含</b>「站回年线回补」，故 A 为 +166.3%（含回补是 +211.6%，见 18 号报告）。<br>
脚本：<code>d70_bottom_bull.py</code>（下有底 / 牛市回吐 / 三锚读数）、<code>d71_giveback.py</code>（回吐锚 v1，已作废）、<code>d72_stock_hold.py</code>（个股持有）、<code>d73_surv.py</code>（退市偏差 / 逐年）、<code>d74_gb_fix.py</code>（修正版）。分析写回 <code>_人性理论/README.md</code> §5.8。<br>
局限：个股样本<b>不含退市股</b>（存活偏差已在正文做敏感性）；ETF 数据止于 2026-09-11，最后两个月按上证口径不外推；「下有底」的样本实质是 n=2。
</div>

</div>
<script>
const LBL = {json.dumps(blabels, ensure_ascii=False)};
new Chart(document.getElementById('c1'), {{
  type: 'bar',
  data: {{labels: LBL, datasets: [{{
    data: {json.dumps(bvals)},
    backgroundColor: '#6B6A65',
    borderRadius: 3, barPercentage: .55
  }}]}},
  options: {{animation:false, responsive:true, maintainAspectRatio:false,
    plugins:{{legend:{{display:false}}, tooltip:{{callbacks:{{label:c=>'回吐 '+c.raw+'%'}}}}}},
    scales:{{y:{{beginAtZero:true, max:100, grid:{{color:'rgba(0,0,0,.06)'}},
      ticks:{{color:'#5f5e5a', callback:v=>v+'%'}}}},
      x:{{grid:{{display:false}}, ticks:{{color:'#5f5e5a', font:{{size:11}}}}}}}}}}
}});
new Chart(document.getElementById('c2'), {{
  type: 'line',
  data: {{labels: {json.dumps(slabels)},
    datasets: [{{data: {json.dumps(svals)}, borderColor:'#A32D2D',
      backgroundColor:'rgba(163,45,45,.08)', fill:true, tension:.25,
      pointRadius:4, pointBackgroundColor:'#A32D2D'}}]}},
  options: {{animation:false, responsive:true, maintainAspectRatio:false,
    plugins:{{legend:{{display:false}}, tooltip:{{callbacks:{{label:c=>'总收益 '+c.raw+'%'}}}}}},
    scales:{{y:{{grid:{{color:'rgba(0,0,0,.06)'}}, ticks:{{color:'#5f5e5a', callback:v=>v+'%'}}}},
      x:{{grid:{{display:false}}, ticks:{{color:'#5f5e5a'}},
         title:{{display:true, text:'回吐阈值（2015-01 起单点）', color:'#8c8b86', font:{{size:11}}}}}}}}}}
}});
</script>
</body></html>'''

open(OUT, 'w', encoding='utf-8').write(html)
print('written', OUT, len(html))
