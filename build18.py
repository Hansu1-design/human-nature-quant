# -*- coding: utf-8 -*-
"""生成 18_指数骨架验证 报告"""
import os, json
import pandas as pd, numpy as np

B = os.path.dirname(os.path.abspath(__file__))
CHARTJS = open(os.path.join(B, 'chart.umd.min.js'), encoding='utf-8').read()
OUT = 'G:/探讨集/18_指数骨架验证_3000与3500_20261112.html'

d61 = json.load(open(os.path.join(B, 'd61_result.json'), encoding='utf-8'))
d64 = json.load(open(os.path.join(B, 'd64_result.json'), encoding='utf-8'))
d65 = json.load(open(os.path.join(B, 'd65_result.json'), encoding='utf-8'))
d66 = json.load(open(os.path.join(B, 'd66_result.json'), encoding='utf-8'))
cur = pd.read_parquet(os.path.join(B, 'd64_curves.parquet'))
cur = cur.iloc[::5].reset_index(drop=True)          # 周采样
dates = [str(x)[:10] for x in cur['d']]

band = [r for r in d61['A_band'] if r['h'] == '1年']
CSS = open(os.path.join(B, '_report.css'), encoding='utf-8').read() if os.path.exists(os.path.join(B, '_report.css')) else ''

def kpi(l, n, d, color=''):
    c = f' style="color:{color}"' if color else ''
    return f'<div class="kpi"><div class="l">{l}</div><div class="n"{c}>{n}</div><div class="d">{d}</div></div>'

# 分档表
brows = ''
for r in band:
    cls = ' class="hl"' if r['band'] in ('2800~3000', '3500~3800') else ''
    sign = '+' if r['mean_%'] > 0 else ''
    col = '#A32D2D' if r['mean_%'] > 0 else '#1C6B3C'
    brows += (f"<tr{cls}><td>{r['band']}</td><td>{r['n']}</td>"
              f"<td style=\"color:{col}\">{sign}{r['mean_%']}%</td><td>{r['win_%']}%</td>"
              f"<td>{r['worst_%']}%</td></tr>")

# 波段表
wrows = ''
for w in d65['waves']:
    wrows += (f"<tr><td>{w['buy']} @ {w['buy_lv']:.0f}</td><td>{w['sell']} @ {w['sell_lv']:.0f}</td>"
              f"<td style=\"color:#A32D2D\">+{w['gain_%']}%</td><td>{w['months']} 个月</td></tr>")

# 七笔
erows = ''
for e in d64['eps_A']:
    col = '#A32D2D' if e['ret'] > 0 else '#1C6B3C'
    erows += (f"<tr><td>{e['entry']}</td><td>{e['entry_lv']:.0f}</td><td>{e['exit']}</td><td>{e['exit_lv']:.0f}</td>"
              f"<td style=\"color:{col}\">{'+' if e['ret']>0 else ''}{e['ret']}%</td>"
              f"<td>{e['hold']} 天</td><td>{e['reason']}</td></tr>")

# PE 对照
perows = ''
for r in d66['pe_at_3000']:
    pe = f"{r['pe_at_3000']} 倍" if r['pe_at_3000'] else '未触及'
    perows += f"<tr><td>{r['years']}</td><td>{pe}</td><td>{r['pe_avg']} 倍</td></tr>"

hs = d66['hs300_equiv_of_3000']
_yr = list(hs.keys()); _vv = list(hs.values())
_g = [(f"{_yr[i]}-{_yr[i+1][2:]}", (_vv[i] + _vv[i+1]) / 2) for i in range(0, len(_yr) - 1, 2)]
hsrows = ''.join(f"<tr><td>{k}</td><td>{v:.0f}</td></tr>" for k, v in _g)

m = d64['main']
kA, kA2, kB = m['A_绝对点位3000/3500'], m['A2_绝对点位+回补'], m['B_自适应分位25/75']

html = f'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>指数骨架验证 · 3000 与 3500</title>
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
.legend{{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--tx2);margin:6px 0 0}}
.legend i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:middle}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}}
th,td{{padding:7px 10px;text-align:right;border-bottom:1px solid var(--bd)}}
th:first-child,td:first-child{{text-align:left}}
th{{font-weight:500;color:var(--tx2);font-size:12px}}
tr:last-child td{{border-bottom:none}}
tr.hl td{{background:#FBF3F2}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.foot{{margin-top:48px;padding-top:20px;border-top:1px solid var(--bd);font-size:12px;color:var(--tx3);line-height:1.9}}
code{{background:#F1EFE8;padding:1px 5px;border-radius:4px;font-size:12px}}
.big{{font-size:15px;line-height:1.9}}
@media(max-width:700px){{.two{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">

<h1>指数骨架验证：3000 买、3500 卖</h1>
<div class="sub">上证指数 sh.000001 · 2009-01 ~ 2026-11-11（4,336 根）· 执行标的 510300 · 回测窗口 2015-01-05 ~ 2026-11-11（2,880 个交易日）· 双边万五成本</div>

<div class="verdict big">
<b>三句话：</b><br>
① <b>存在。</b>“位置越低、后续一年越涨”在这 11 年是单调的，跨 6 个档位、几万个观测，不是噪声（低档 − 高档 = 23.1pt，块自助法 95%CI 16.8~28.9，P&gt;0 = 1.000）。<br>
② <b>对，但一半对错了理由。</b>你以为在买“便宜”，其实在买“箱体下沿”。同一个 3000 点，2015 年对应全 A 市盈率 57.7 倍，2024 年 30.5 倍；换成沪深 300 是 3159 点 vs 3515 点。点位与估值的秩相关只有 0.43。<br>
③ <b>有价值，但价值不在“人性”，在“出场”。</b>同一套出场规则下把买点完全去掉，11 年还有 +124.9%（买入持有是 +65.0%）。而且 +166% 里有 154 个点是两笔交易贡献的——<b>有效样本 n≈2</b>。
</div>

<h2><span class="no">01</span>位置 → 未来一年收益：单调，且是这 11 年最硬的一条</h2>
<p class="note">每一天按当日上证收盘归入一档，算“往后 250 个交易日的涨幅”。窗口重叠，所以用块自助法（block=60 日，B=2000）校正自相关，置信区间仍然远离 0。</p>
<div class="card">
  <div class="chartbox"><canvas id="c1" role="img" aria-label="不同点位区间的未来一年平均收益"></canvas></div>
  <div class="legend"><span><i style="background:#A32D2D"></i>正收益</span><span><i style="background:#1C6B3C"></i>负收益</span></div>
</div>
<table>
<tr><th>点位区间</th><th>天数</th><th>未来 1 年平均</th><th>上涨概率</th><th>最差一次</th></tr>
{brows}
</table>
<div class="ok">
<b>这条经得起看。</b>不是“某一档特别好”，是六档连成一条斜线；2 年、3 年窗口同向。低档与高档的差距 23.1pt，即使把重叠窗口按 60 日一块重新抽样 2000 次，95% 区间仍是 [16.8, 28.9]。<b>“3000 以下买、3500 以上卖”在方向上是对的。</b>
</div>

<h2><span class="no">02</span>11 年只有 4 次完整波段，每次都赚 19%——这句话是循环的</h2>
<p class="note">“跌破 3000 之后第一次涨破 3500”，2015 年以来一共 4 次。</p>
<table>
<tr><th>买入</th><th>卖出</th><th>区间涨幅</th><th>耗时</th></tr>
{wrows}
</table>
<div class="warn">
<b>别被“4 战 4 胜”骗了。</b>①样本是规则自己挑出来的——跌破 3000 之后再也没回去的情况，在这 11 年根本不存在，因为 70% 的时间指数就待在这个箱子里。所以 4/4 说的是“<b>体制没变</b>”，不是“规则有效”。<br>
②每次都赚 18.7~20.5%，不是市场有节律，是因为<b>你把靶子钉死在 3500</b>，收益必然聚在 (3500/3000−1)=16.7% 附近。真正有信息量的是<b>耗时</b>：2.2 / 24.4 / 30.5 / 38.3 个月——四次里有三次要拿 2 年以上，最长一次 3.2 年。
</div>

<h2><span class="no">03</span>机械执行一遍：+166% vs 买入持有 +65%，回撤减半</h2>
<p class="note">买点：连续 20 日跌破 3000 后，首次站上 20 日线。卖点：站上 3500 先减半；之后“从持仓最高点回撤 10% 且已距顶 20 个交易日”或“成交额 250 日分位 ≥0.95 且指数 &gt;3500”清仓。</p>
<div class="grid">
{kpi('总收益（11.9 年）', '+166.3%', '买入持有 +65.0%')}
{kpi('年化（在场时间口径）', '19.6%', '买入持有 4.32%')}
{kpi('最大回撤', '−26.1%', '买入持有 −53.0%', '#1C6B3C')}
{kpi('在场时间', '46.2%', '空仓 53.8%，最长连续 714 个交易日')}
</div>
<div class="card">
  <div class="chartbox"><canvas id="c2" role="img" aria-label="策略与买入持有的净值曲线"></canvas></div>
  <div class="legend"><span><i style="background:#A32D2D"></i>骨架 + 回撤回补</span><span><i style="background:#C98A5B"></i>原骨架 3000/3500</span><span><i style="background:#7A8C99"></i>买入持有 510300</span></div>
</div>
<table>
<tr><th>买入</th><th>点位</th><th>卖出</th><th>点位</th><th>收益</th><th>持有</th><th>出场原因</th></tr>
{erows}
</table>
<div class="warn">
<b>看这七行：</b>两笔赚大钱（+61.6%、+57.2%），两笔小赚，两笔打平，一笔小亏。去掉任一笔重算：去掉第一笔剩 +72.2%，去掉第五笔剩 +77.0%。<b>93% 的收益来自两笔，而且这两笔本质是同一笔交易做了两次</b>——2850 附近买、3550/3650 靠热度卖。<b>所以“这套方法有效”这句话的有效样本量是 2。</b>
</div>

<h2><span class="no">04</span>拆开看：钱是哪一条规则挣的</h2>
<p class="note">买点固定为“跌破 3000 即买”，逐条加出场规则；最后一行是把买点完全去掉。</p>
<div class="card">
  <div class="chartbox"><canvas id="c3" role="img" aria-label="不同规则组合的期末总收益"></canvas></div>
</div>
<p class="note" style="margin-top:10px">只有“3500 减半、不止损不清仓” = +77.9%（几乎等于买入持有）；加<b>热度清仓</b> → +194.5%；改加<b>回撤止损</b> → +162.1%；两条都要 → +221.0%。<b>抗跌那一条（回撤止损）贡献最小，卖在热闹处那一条（热度）贡献最大。</b></p>
<div class="warn">
<b>但“热度出场”是全场最脆的一环：</b>它扛着七成利润，样本却只有 2018-01 和 2021-02 两个顶。我换了 3 种窗口（120/250/500 日）× 4 种阈值（0.85/0.90/0.95/0.99）共 12 组：10 组都能同时抓到那两个顶，总收益落在 +135%~+168%，只有最极端的 0.99 会漏掉 2018 年那个顶。结论是<b>它不是针尖，但它只有两个样本</b>——这不是统计证据，是两次观察。
</div>

<h2><span class="no">05</span>三个洞，按危险程度排</h2>

<h3>洞一：“3000 点”不是尺子，是这 11 年箱体的中轴</h3>
<div class="two">
<div>
<p class="note">同一个 3000 点，在不同年份对应的估值：</p>
<table><tr><th>年份</th><th>3000 点时的全 A PE(TTM)</th><th>该期平均</th></tr>{perows}</table>
</div>
<div>
<p class="note">同一个 3000 点，折算成沪深 300 的点位：</p>
<table><tr><th>年份</th><th>沪深 300 等价点位（期内均值）</th></tr>{hsrows}</table>
</div>
</div>
<div class="warn">
上证与沪深 300 的比价在 0.666~0.966 之间摆动（幅度 45%）。所以“3000 点”在 2015 年 ≈ 沪深 300 的 3159 点，2020 年 ≈ 4244 点，2026 年 ≈ 3508 点。<b>而你的买卖是用 510300 执行的——信号和标的不是同一把尺子。</b>再叠加一层：上证是价格指数，11 年漏掉约 25~30% 的分红，且你有一半时间空仓，等于再放弃约 1.1pt/年。<br>
<b>结论：这套规则成立的原因是“指数 70.1% 的时间待在 2800~3500 这个箱子里”，不是“3000 点便宜”。它本质是箱体交易，不是价值投资。</b>
</div>

<h3>洞二：出场用趋势，进场用估值——两套坐标系，牛市里会被永久甩下车</h3>
<p class="note">2025-04-07，关税暴跌当天，规则在 3097 点“回撤 10% + 距顶 20 日”清仓——正好卖在坑里。之后指数一路到 4243，而回补条件是“跌回 3000 以下”，<b>19 个月没成立过</b>。这一段踏空 +37%。</p>
<div class="ok">
<b>只改一处：出场用趋势，回补也用趋势。</b>把“清仓后必须跌回 3000 才买”改成“站回年线且高于上次卖价就买回”。同一段历史：总收益 +166.3% → <b>+211.6%</b>，回撤不变（−26.1%）。2025-04-09 在 3187 回补，2025-08-13 在 3683 靠热度出场，这一笔 +16.5%。<br>
注意：这是<b>第六条规则</b>，加它就离“骨架”远一步、离“调参”近一步。要不要加，是你的选择，不是数据的选择。
</div>

<h3>洞三：现在的状态是“既不能买、也没让你卖”</h3>
<div class="grid">
{kpi('上证（2026-11-11）', '3,681', '5 年相对分位 0.76（偏高）')}
{kpi('距上次清仓', '19 个月', '2025-04-07 @ 3097')}
{kpi('2026 年跌破 3000 的天数', '0 天', '2025 年以来最低 3097')}
{kpi('成交额热度分位', '0.12', '冷，但不是清仓信号')}
</div>
<p class="note" style="margin-top:12px">换句话说：按你的骨架，现在该做的动作是<b>什么都不做</b>——已经在 3500 以上，只该逐步减；但热度只有 0.12，没有任何清仓信号。真正的问题不是现在，是<b>下一次跌破 3000 如果永远不来</b>。</p>

<h2><span class="no">06</span>如果非要换一个锚，会怎样</h2>
<p class="note">把 3000/3500 换成“过去 5 年价格分位 25%/75%”（自适应，不依赖固定数字），同样规则、同样出场：</p>
<div class="grid">
{kpi('总收益', f"+{kB['total_%']}%", '原骨架 +166.3%')}
{kpi('最大回撤', f"{kB['mdd_%']}%", '原骨架 −26.1%', '#1C6B3C')}
{kpi('空仓时间', f"{kB['flat_%']}%", '原骨架 53.8%')}
{kpi('11 个起点里跑赢买入持有', '10 / 11', '原骨架 8 / 11')}
</div>
<div class="warn">
换了锚之后<b>更稳但更穷</b>：收益腰斩，回撤减半，几乎全程空仓。它输在一个具体的地方——2016 年 3 月上证 2850 点时，在过去 5 年里排 72 分位，被判为“贵”，于是整段 +61.6% 没吃到。<b>“相对位置”也不是好尺子，它只是把固定数字换成了浮动数字，仍然是价格、不是估值。</b><br>
如果要真正换尺子，该用的是全 A 中位 PE 的十年分位（本机 <code>all_a_pe_ttm</code> 有 262 期月度数据）＋ 证券化率（总市值/GDP）。这件事不是针灸，是换尺子；尺子错了，调参数只会更错。
</div>

<h2><span class="no">07</span>所以，你的骨架值多少钱</h2>
<p class="note">11 个起点分别开跑（含 2015-06 的 5166 点高位），对比同期买入持有。前三个起点（2015-01、2015-06、2016-01）结果完全相同，因为规则在 2016-03-02 第一次信号之前一直空仓——这正是“3000 以上不动手”的价值：<b>2015 年那场 −53% 的崩盘，它一分钟都没在场</b>。</p>
<div class="card">
  <div class="chartbox"><canvas id="c4" role="img" aria-label="不同起点下策略与买入持有的总收益对比"></canvas></div>
  <div class="legend"><span><i style="background:#A32D2D"></i>骨架</span><span><i style="background:#9AA5AD"></i>买入持有</span></div>
</div>
<table>
<tr><th>起点</th><th>上证</th><th>买入持有</th><th>骨架</th><th>骨架回撤</th><th>跑赢？</th></tr>
{''.join(f"<tr><td>{r['start']}</td><td>{r['lvl']:.0f}</td><td>{r['bh_%']}%</td><td>{r['A_%']}%</td><td>{('−' if r['A_mdd_%']<0 else '')}{abs(r['A_mdd_%'])}%</td><td>{'是' if r['A_beat']=='Y' else '否'}</td></tr>" for r in d64['starts'])}
</table>
<div class="verdict">
<b>最终判决：</b><br>
<b>存不存在——存在。</b>位置与后续收益的关系、以及“低买高卖能大幅降低回撤”，在这 11 年都成立，且不是统计噪声。<br>
<b>对不对——方向对，理由错一半。</b>你以为是估值，实际是箱体；你以为是“人性”，实际是“这个指数 11 年没涨”。这两件事一旦分开，就知道它会在哪里失效。<br>
<b>有没有价值——有，但价值集中在出场，不在选股、也不在人格。</b>同一套出场去掉买点还剩 +124.9%；而全套的 93% 来自两笔。<br><br>
<b>一句话：3000 点不是你的线，是这 11 年箱体的中轴。箱体在，这是你能拿到的最好的东西；箱体破，你不是亏钱，是永久空仓——而且它不会报警，只会让你在牛市里一动不动。</b>
</div>

<div class="foot">
口径说明：信号用上证指数收盘，执行用 510300 沪深 300ETF 收盘，未计分红与冲击成本（仅双边万五）；“年化（在场时间口径）”＝总收益按实际在场交易日数年化，用于和 100% 满仓的买入持有做同口径比较。<br>
“热度”＝上证成交额在过去 250 个交易日中的分位（不含当日）。<br>
数据：baostock sh.000001 日线；G:/AStockData/supplement/etf_history；valuation/all_a_pe_ttm、index_pe_沪深300。<br>
脚本：<code>_人性理论/d61_index_band.py</code>（分档/主策略）、<code>d64_decisive.py</code>（自适应分位对照）、<code>d65_robust.py</code>（块自助法/贡献拆解）、<code>d66_hermes_holes.py</code>（锚漂移/热度敏感性）。<br>
2026-11-12 · 本报告只做骨架级验证，不含参数优化建议。
</div>
</div>
<script>
const D={json.dumps(dates)};
const opt=(y)=>({{animation:false,responsive:true,maintainAspectRatio:false,interaction:{{mode:'index',intersect:false}},
 plugins:{{legend:{{display:false}},tooltip:{{backgroundColor:'#1a1a1a',padding:10,cornerRadius:6}}}},
 scales:{{x:{{grid:{{display:false}},ticks:{{maxTicksLimit:7,color:'#8c8b86',font:{{size:11}}}}}},
 y:{{grid:{{color:'rgba(0,0,0,.06)'}},ticks:{{color:'#8c8b86',font:{{size:11}},callback:y.tick}}}}}}}});

new Chart(document.getElementById('c1'),{{type:'bar',data:{{
 labels:{json.dumps([r['band'] for r in band])},
 datasets:[{{data:{json.dumps([r['mean_%'] for r in band])},
 backgroundColor:{json.dumps(['#A32D2D' if r['mean_%']>0 else '#1C6B3C' for r in band])},borderRadius:3}}]}},
 options:opt((v)=>v+'%')}});

new Chart(document.getElementById('c2'),{{type:'line',data:{{labels:D,datasets:[
 {{data:{json.dumps([round(x,3) for x in cur['A2_绝对点位+回补']])},borderColor:'#A32D2D',borderWidth:2,pointRadius:0,tension:.1}},
 {{data:{json.dumps([round(x,3) for x in cur['A_绝对点位3000/3500']])},borderColor:'#C98A5B',borderWidth:2,pointRadius:0,tension:.1}},
 {{data:{json.dumps([round(x,3) for x in cur['bh']])},borderColor:'#7A8C99',borderWidth:1.6,pointRadius:0,borderDash:[5,4],tension:.1}}]}},
 options:opt((v)=>v.toFixed(1)+'×')}});

new Chart(document.getElementById('c3'),{{type:'bar',data:{{
 labels:['只有3500减半','+热度清仓','+回撤止损','两者都要','去掉买点'],
 datasets:[{{data:[77.9,194.5,162.1,221.0,124.9],backgroundColor:['#9AA5AD','#A32D2D','#C98A5B','#A32D2D','#7A8C99'],borderRadius:3}}]}},
 options:Object.assign(opt((v)=>'+'+v+'%'),{{indexAxis:'y',scales:{{x:{{grid:{{color:'rgba(0,0,0,.06)'}},ticks:{{color:'#8c8b86',callback:v=>'+'+v+'%'}}}},y:{{grid:{{display:false}},ticks:{{color:'#5f5e5a',font:{{size:12}}}}}}}}}})}});

new Chart(document.getElementById('c4'),{{type:'bar',data:{{
 labels:{json.dumps([r['start'][:7] for r in d64['starts']])},
 datasets:[
 {{label:'骨架',data:{json.dumps([r['A_%'] for r in d64['starts']])},backgroundColor:'#A32D2D',borderRadius:3}},
 {{label:'买入持有',data:{json.dumps([r['bh_%'] for r in d64['starts']])},backgroundColor:'#9AA5AD',borderRadius:3}}]}},
 options:Object.assign(opt((v)=>v+'%'),{{scales:{{x:{{grid:{{display:false}},ticks:{{color:'#8c8b86',font:{{size:11}}}}}},y:{{grid:{{color:'rgba(0,0,0,.06)'}},ticks:{{color:'#8c8b86',callback:v=>v+'%'}}}}}}}})}});
</script>
</body></html>'''

open(OUT, 'w', encoding='utf-8').write(html)
print('written', OUT, len(html))
