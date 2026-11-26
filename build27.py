# -*- coding: utf-8 -*-
"""build27: 月线级别的回调，能不能吃一口"""
import os, json
import numpy as np
import pandas as pd

B = os.path.dirname(os.path.abspath(__file__))
OUT = 'G:/探讨集/27_月线回调_能不能吃一口_20261112.html'
CHARTJS = open(os.path.join(B, 'chart.umd.min.js'), encoding='utf-8').read()

import importlib.util
spec = importlib.util.spec_from_file_location('m', os.path.join(B, 'd141_eat_bite.py'))
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)
spec2 = importlib.util.spec_from_file_location('m2', os.path.join(B, 'd142_dedup.py'))
m2 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(m2)

d = mm.load()
c = d['close'].values
n = len(c)
yrs = (d.d.iloc[-1] - d.d.iloc[0]).days / 365.25
ra = d['rA'].values
pre = np.array([c[i] / c[max(0, i - 250)] - 1 for i in range(n)])
dd250 = np.array([c[i] / c[max(0, i - 250):i + 1].max() - 1 for i in range(n)])
ma250 = pd.Series(c).rolling(250).mean().values
RF = 0.025
DAILY_RF = (1 + RF) ** (1 / 243) - 1


def fwd(i, h):
    return c[min(n - 1, i + h)] / c[i] - 1


def trig(cond, cool=180):
    ev, last = [], -999
    for i in range(300, n):
        if i - last <= cool:
            continue
        if cond(i):
            ev.append(i)
            last = i
    return ev


# ============ 1. 阈值扫描 ============
scan = []
for thr in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
    ev, last = [], -999
    for i in range(250, n):
        if dd250[i] <= -thr and i - last > 60:
            ev.append(i)
            last = i
    if not ev:
        continue
    r250 = np.array([fwd(i, 250) for i in ev])
    r60 = np.array([fwd(i, 60) for i in ev])
    low = np.array([min(c[i:min(n, i + 251)]) / c[i] - 1 for i in ev])
    prior = np.array([1.0 if max(c[i:min(n, i + 251)]) >= c[max(0, i - 250):i].max() else 0.0
                      for i in ev])
    scan.append(dict(thr=thr * 100, nev=len(ev), per=len(ev) / yrs,
                     med250=float(np.median(r250) * 100), mean250=float(r250.mean() * 100),
                     med60=float(np.median(r60) * 100), low=float(np.median(low) * 100),
                     rec=float(prior.mean() * 100)))

# ============ 2. 离底分层 ============
EV15 = m2.events_dedup(d, 0.15, cool=180)   # 与 d142 同口径
layers = []
for lo_, hi_, lab in [(0, 1.10, '离底 ≤10%'), (1.10, 1.30, '10~30%'),
                      (1.30, 1.60, '30~60%'), (1.60, 99, '>60%')]:
    idx = [i for i in EV15 if not np.isnan(ra[i]) and lo_ < ra[i] <= hi_]
    if not idx:
        continue
    r = np.array([fwd(i, 250) for i in idx])
    layers.append(dict(lab=lab, n=len(idx), mean=float(r.mean() * 100),
                       med=float(np.median(r) * 100), win=float((r > 0).mean() * 100),
                       lo=float(np.percentile(r, 5) * 100) if len(r) > 2 else None,
                       hi=float(np.percentile(r, 95) * 100) if len(r) > 2 else None))

# ============ 3. 增量测试 ============
inc = []
for h in [60, 120, 250]:
    idx = [i for i in EV15 if not np.isnan(ra[i]) and 1.10 < ra[i] <= 1.30]
    r = np.array([fwd(i, h) for i in idx])
    inc.append(dict(h=h, mean=float(r.mean() * 100), med=float(np.median(r) * 100),
                    win=float((r > 0).mean() * 100),
                    low=float(np.mean([min(c[i:min(n, i + h)]) / c[i] - 1 for i in idx]) * 100),
                    rf=float(RF * h / 243 * 100), n=len(idx)))

# ============ 4. 出场方式 ============
EV_A = trig(lambda i: (not np.isnan(ra[i])) and ra[i] <= 1.10, 180)


def sim(ev, hold=250, tp=None, half=None):
    rs = []
    for i in ev:
        w, cash, done = 1.0, 0.0, False
        for j in range(i + 1, min(n, i + hold + 1)):
            ret = c[j] / c[i] - 1
            if half is not None and not done and ret >= half:
                cash += 0.5 * (1 + ret); w = 0.5; done = True
            if tp is not None and ret >= tp:
                cash += w * (1 + ret); w = 0.0; break
        j = min(n - 1, i + hold)
        cash += w * (c[j] / c[i])
        rs.append(cash - 1)
    return np.array(rs)


exits = []
for nm, kw in [('全跑 +10%', dict(tp=0.10)), ('全跑 +15%', dict(tp=0.15)),
               ('全跑 +20%', dict(tp=0.20)),
               ('涨10% 卖一半', dict(half=0.10)), ('涨20% 卖一半', dict(half=0.20)),
               ('涨30% 卖一半', dict(half=0.30)), ('一直不动', dict())]:
    r = sim(EV_A, 250, **kw)
    exits.append(dict(lab=nm, mean=float(r.mean() * 100), med=float(np.median(r) * 100),
                      win=float((r > 0).mean() * 100)))

holds = []
for h in [20, 60, 120, 250, 500, 750]:
    r = sim(EV_A, h)
    holds.append(dict(h=h, mean=float(r.mean() * 100), med=float(np.median(r) * 100),
                      win=float((r > 0).mean() * 100)))

# 右尾依赖
r250_A = sim(EV_A, 250)
srt = np.sort(r250_A)
tail = dict(all=float(r250_A.mean() * 100),
            drop3=float(srt[:-3].mean() * 100),
            drop5=float(srt[:-5].mean() * 100),
            gt30=float((r250_A > 0.30).mean() * 100),
            gt50=float((r250_A > 0.50).mean() * 100),
            pos=float((r250_A > 0).mean() * 100))

# ============ 5. 月K 连阴 ============
per = d.d.dt.to_period('M')
gb = d.groupby(per)
last_i = gb.apply(lambda x: int(x.index[-1]), include_groups=False).tolist()
first_i = gb.apply(lambda x: int(x.index[0]), include_groups=False).tolist()
mv = c[last_i]
nm_ = len(mv)
monthK = []
for k in [1, 2, 3, 4]:
    ev, last = [], -99
    for j in range(k, nm_ - 1):
        if all(mv[j - t] < mv[j - t - 1] for t in range(k)) and j - last >= k:
            ev.append((j, first_i[j + 1]))
            last = j
    out = []
    for h in [3, 6, 12, 24]:
        out.append(float(np.nanmedian([mv[min(nm_ - 1, j + 1 + h)] / mv[j + 1] - 1
                                       for j, fi in ev]) * 100))
    r12 = np.array([mv[min(nm_ - 1, j + 1 + 12)] / mv[j + 1] - 1 for j, fi in ev])
    monthK.append(dict(k=k, nev=len(ev), m3=out[0], m6=out[1], m12=out[2], m24=out[3],
                       win12=float((r12 > 0).mean() * 100)))
mk_with = []
for k in [1, 2, 3]:
    ev, last = [], -99
    for j in range(k, nm_ - 1):
        if all(mv[j - t] < mv[j - t - 1] for t in range(k)) and j - last >= k:
            i = first_i[j + 1]
            if (not np.isnan(ra[i])) and ra[i] <= 1.10:
                ev.append(i)
            last = j
    if len(ev) < 2:
        continue
    r = np.array([fwd(i, 250) for i in ev])
    low = np.array([min(c[i:min(n, i + 251)]) / c[i] - 1 for i in ev])
    mk_with.append(dict(k=k, n=len(ev), mean=float(r.mean() * 100),
                        med=float(np.median(r) * 100), win=float((r > 0).mean() * 100),
                        low=float(low.mean() * 100)))

# ============ 6. 牛市回调 ============
bull = []
base_cond = lambda i: (not np.isnan(ma250[i])) and c[i] > ma250[i] and 0.0 < pre[i] <= 0.40
for nm2, cond, dedup in [('牛市里随便买（每个满足条件的交易日）', base_cond, False),
                         ('牛市 + 跌 10%', lambda i: base_cond(i) and dd250[i] <= -0.10, True)]:
    ev = [i for i in range(300, n) if cond(i)]
    if dedup:
        tmp, last = [], -999
        for i in ev:
            if i - last > 120:
                tmp.append(i)
                last = i
        ev = tmp
    r = np.array([fwd(i, 250) for i in ev])
    low = np.array([min(c[i:min(n, i + 251)]) / c[i] - 1 for i in ev])
    bull.append(dict(lab=nm2, n=len(ev), mean=float(r.mean() * 100),
                     win=float((r > 0).mean() * 100), low=float(low.mean() * 100)))

# ============ 7. 否决权 ============
EV60 = EV15
keep = [i for i in EV60 if pre[i] <= 0.40]
drop = [i for i in EV60 if pre[i] > 0.40]
veto = []
for nm3, ev in [('全部跌 15% 事件', EV60), ('否决后保留', keep), ('被否决掉的', drop)]:
    if not ev:
        continue
    r = np.array([fwd(i, 250) for i in ev])
    low = np.array([min(c[i:min(n, i + 251)]) / c[i] - 1 for i in ev])
    veto.append(dict(lab=nm3, n=len(ev), mean=float(r.mean() * 100),
                     win=float((r > 0).mean() * 100), low=float(low.mean() * 100)))
veto_dates = [str(d.d.iloc[i].date()) for i in drop]

# ============ 8. 资金曲线 ============
def equity(sig, hold=250):
    nav = np.ones(n); cash = 1.0; sh = 0.0; entry = -1
    s = set(sig)
    for i in range(n):
        if sh > 0:
            nav[i] = sh * c[i]
            if i - entry >= hold:
                cash = sh * c[i]; sh = 0.0
        else:
            nav[i] = cash
            if i in s and i > 0:
                sh = cash / c[i]; cash = 0.0; entry = i
        if sh == 0 and i < n - 1:
            cash *= (1 + DAILY_RF)
    return nav


nav0 = np.ones(n) * (1 + DAILY_RF) ** np.arange(n)
nav1 = c / c[0]
sigH2 = EV_A
sigH5 = trig(lambda i: (not np.isnan(ra[i])) and ra[i] <= 1.10 and dd250[i] <= -0.15, 180)
navH2 = equity(sigH2)
navH5 = equity(sigH5)


def st(nv):
    return dict(cagr=float((nv[-1] ** (1 / yrs) - 1) * 100),
                dd=float((nv / np.maximum.accumulate(nv) - 1).min() * 100),
                nav=float(nv[-1]))


eqs = [dict(lab='一直空仓生息', **st(nav0)),
       dict(lab='买入持有', **st(nav1)),
       dict(lab='骨架：进带即买', **st(navH2)),
       dict(lab='骨架 + 跌15%', **st(navH5))]

# 2014-10-20 那笔
i14 = int(d.index[d.d == pd.Timestamp('2014-10-20')][0])
seg14 = c[i14:i14 + 251]
case14 = dict(buy=float(c[i14]), peak=float(seg14.max()),
              peak_r=float(seg14.max() / c[i14] - 1) * 100,
              low=float(seg14.min()), low_r=float(seg14.min() / c[i14] - 1) * 100,
              end=float(seg14[-1]), end_r=float(seg14[-1] / c[i14] - 1) * 100,
              giveback=float(seg14.min() / seg14.max() - 1) * 100)

# 采样资金曲线（每月一个点）
samp = list(range(0, n, 21))
if samp[-1] != n - 1:
    samp.append(n - 1)
L6 = [str(d.d.iloc[i])[:7] for i in samp]
D6 = [[round(float(nv[i]), 3) for i in samp] for nv in (nav0, nav1, navH2, navH5)]

# ============ 组装 HTML ============
CSS = open(os.path.join(B, '_css.txt'), encoding='utf-8').read() if os.path.exists(
    os.path.join(B, '_css.txt')) else None
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
.tleft td,.tleft th{text-align:left}
@media(max-width:720px){.chartbox{height:260px}}
</style>"""

H = []
def P(s):
    H.append(s)

P('<div class="wrap">')
P('<h1>月线级别的回调，能不能吃一口</h1>')
P(f'<div class="sub">上证 {d.d.iloc[0].date()} ~ {d.d.iloc[-1].date()}（{n:,} 根，{yrs:.1f} 年）· '
  f'回撤 15% 触发 65 次 · 去重后 33 次 · 2026-11-12</div>')

P('<div class="verdict">'
  '<b>结论先说：能吃的不是"回调"，是"位置"。</b><br><br>'
  '同样跌 15%，在离底 10% 以内买，6 次赚 <b>+20.2%</b>、胜率 83%；'
  '在离底 30% 以外买，7 次 <b>−2.6%</b>、胜率 43%；离底 60% 以外买，那次是 2015-06-26'
  '（4,193 点），之后 <b>−28.3%</b>。<br><br>'
  '而在骨架<b>本来不买</b>的区间（离底 10~30%），"跌 15% 就买"持有 250 日只有 <b>+3.16%</b>，'
  '同期空仓生息 <b>2.57%</b> —— <b>p=0.926，等于白折腾，还多扛 −10.4% 浮亏。</b><br><br>'
  '更狠的一条：<b>"吃一口就跑"是这个想法里唯一确定错的部分。</b>'
  '在出手区里，+10% 止盈得 +7.5%，卖一半得 +15.4%，一直不动得 <b>+19.2%</b>；'
  '持有 20 日 +2.5% → 250 日 +19.2% → 750 日 +32.1%，<b>单调到 750 天，胜率 100%</b>。'
  '这一层不是"吃一口"的层，是"吃一段"的层。</div>')

P('<div class="grid">')
P('<div class="kpi"><div class="l">回撤 15% 的频率</div><div class="n">2.0 次/年</div>'
  '<div class="d">去重后 1.1 次/年（33 次/29.9 年）</div></div>')
P('<div class="kpi"><div class="l">不加筛选的中位收益</div><div class="n">−0.6%</div>'
  '<div class="d">持有 250 日 · 胜率 48%</div></div>')
P('<div class="kpi"><div class="l">离底 ≤10% 时做</div><div class="n">+20.2%</div>'
  '<div class="d">n=6 · 胜率 83%</div></div>')
P('<div class="kpi"><div class="l">吃一口（+10% 止盈）</div><div class="n">+7.5%</div>'
  '<div class="d">比不动少赚 11.7pp</div></div>')
P('</div>')

# ---------- 01 ----------
P('<h2><span class="no">01</span>先把这个东西量出来</h2>')
P('<p class="note">"月线级别的回调"没有标准定义。我用两条口径互相校验：'
  '<b>日线口径</b>＝从滚动 250 日高点回撤超过 X%；<b>月线口径</b>＝月收盘从月收盘高点回撤。'
  '结果如下（日线口径，冷却 60 日去重）：</p>')
P('<div class="card"><table><thead><tr><th>回撤阈值</th><th>触发次数</th><th>次/年</th>'
  '<th>之后 60 日中位</th><th>之后 250 日中位</th><th>之后 250 日均值</th>'
  '<th>期间最大浮亏</th><th>一年内回前高</th></tr></thead><tbody>')
for x in scan:
    cls = ' class="hl"' if abs(x['thr'] - 15) < 0.1 else ''
    P(f'<tr{cls}><td>{x["thr"]:.0f}%</td><td>{x["nev"]}</td><td>{x["per"]:.2f}</td>'
      f'<td>{x["med60"]:+.1f}%</td><td>{x["med250"]:+.1f}%</td>'
      f'<td>{x["mean250"]:+.1f}%</td><td>{x["low"]:+.1f}%</td><td>{x["rec"]:.0f}%</td></tr>')
P('</tbody></table></div>')
P('<p class="note">看最后两列就够了：<b>不管多深的回撤，之后 250 日的中位收益都在 '
  '−0.6%~+2.7% 之间（30% 那档 +6.2% 只有 17 个样本，且被 2005 年那一笔拉动），'
  '而一年内能回前高的比例从 36% 一路掉到 6%</b>。跌得越深，回不去的概率越大 —— '
  '这不是"跌多了就便宜"，是"跌多了可能就是转熊"。</p>')
P('<div class="card"><div class="chartbox"><canvas id="c1"></canvas></div></div>')

P('<h3>月线口径更直白：全是负的</h3>')
P('<p class="note">月收盘回撤 8%~20% 各档，之后 3/6/12 个月的中位收益：'
  '<b>−0.3% / −0.7% / −1.1%</b>，回前高比例只有 <b>5%~13%</b>。'
  '月线口径比日线口径更悲观，因为月收盘已经是"月末结果"，没有日内的假摔。</p>')

P('<div class="warn"><b>这里有个必须说清的口径：</b>本轮<b>不折 CPI</b>。'
  '持有几个月到两年，通胀累计 1%~4%，会略微高估收益。但所有方案用的是同一把尺，'
  '横向比较不受影响；绝对的"能不能赚钱"要再打个 2%~3% 的折扣。<br>'
  '另一个限制：<b>离底距离那一列只有 2008 年以后有</b>（CPI 数据从 2008-01 起），'
  '所以分层那几张表是 18.7 年的子样本，不是 29.9 年。</div>')

# ---------- 02 ----------
P('<h2><span class="no">02</span>核心答案：不是回调能不能吃，是你在哪儿吃</h2>')
P('<p class="note">把 33 次去重后的 15% 回撤，按<b>触发当时的离底距离</b>（实际点位 ÷ 1,880）分成四档，'
  '每档看之后 250 日：</p>')
P('<div class="card"><table><thead><tr><th>触发时离底</th><th>次数</th><th>均值</th>'
  '<th>中位</th><th>胜率</th><th>最差</th><th>最好</th></tr></thead><tbody>')
for x in layers:
    cls = ' class="hl2"' if '≤10' in x['lab'] else (' class="hl"' if '>60' in x['lab'] else '')
    lo_s = f'{x["lo"]:+.1f}%' if x['lo'] is not None else '—'
    hi_s = f'{x["hi"]:+.1f}%' if x['hi'] is not None else '—'
    P(f'<tr{cls}><td>{x["lab"]}</td><td>{x["n"]}</td><td><b>{x["mean"]:+.1f}%</b></td>'
      f'<td>{x["med"]:+.1f}%</td><td>{x["win"]:.0f}%</td><td>{lo_s}</td><td>{hi_s}</td></tr>')
P('</tbody></table></div>')
P('<p class="note">四组 ANOVA <b>F=4.10，p=0.0246</b>；离底倍数与之后 250 日收益的 Spearman '
  '<b>ρ=−0.589，p=0.0001</b>（n=40）。<b>这是本轮唯一一个真正穿过统计门槛的信号。</b></p>')
P('<div class="card"><div class="chartbox"><canvas id="c2"></canvas></div></div>')
P('<p class="note">顺带一个当下的坐标：<b>最近一次触发是 2026-10-30（3,599 点，离底 +33%）</b>，'
  '落在"30~60%"那一档 —— 那一档 7 次里只有 2 次为正、均值 −2.6%。'
  '它现在刚好走到 250 日窗口的开头，之后怎么走还不知道，但<b>这一档的历史记录不允许你现在进去抄</b>。</p>')
P('<div class="ok"><b>但这个发现没有你想的那么有用。</b>因为"离底 ≤10%"本来就是骨架的出手区 —— '
  '换成一句人话：<b>月线回调能吃的那一半，恰好就是骨架已经在吃的那一半。</b>'
  '它不是新战场，是旧战场的一个入场时点。</div>')

# ---------- 03 ----------
P('<h2><span class="no">03</span>那么，在骨架不买的区间吃回调，值不值</h2>')
P('<p class="note">这是决定性的一问。骨架规定离底 ≤10% 才出手，'
  '所以"离底 10~30%"这一段是骨架<b>主动放弃</b>的。如果月线回调在这里能吃，'
  '它就是真的第二个战场；如果吃不到，这一层就白研究了。<br>'
  '样本：2011-08、2016-03、2018-06、2019-08、2022-03、2022-09，共 6 次。</p>')
P('<div class="card"><table><thead><tr><th>持有期</th><th>均值</th><th>中位</th><th>胜率</th>'
  '<th>平均最深浮亏</th><th>同期空仓生息</th><th>超额</th></tr></thead><tbody>')
for x in inc:
    exc = x['mean'] - x['rf']
    cls = ' class="hl"' if x['h'] == 250 else ''
    P(f'<tr{cls}><td>{x["h"]} 日</td><td><b>{x["mean"]:+.2f}%</b></td>'
      f'<td>{x["med"]:+.2f}%</td><td>{x["win"]:.0f}%</td>'
      f'<td>{x["low"]:+.1f}%</td><td>{x["rf"]:+.2f}%</td><td>{exc:+.2f}pp</td></tr>')
P('</tbody></table></div>')
P('<div class="verdict"><b>250 日 +3.16%，空仓生息 2.57%，t=0.10，p=0.926。</b><br>'
  '这不是"略好"，是<b>统计上和零没有区别</b>，而你为此多扛了 <b>−10.3% 的平均浮亏</b>、'
  '锁死资金 8 个月。<br><br>'
  '60 日和 120 日更糟：<b>−0.40% / −1.49%</b>，连生息都跑不过。'
  '<b>骨架不买的地方，确实不该买 —— 这条纪律经受住了检验。</b></div>')

# ---------- 04 ----------
P('<h2><span class="no">04</span>"吃一口就跑"的死刑</h2>')
P('<p class="note">假设你判断对了位置（离底 ≤10%，11 次触发），剩下的问题是<b>怎么卖</b>。'
  '这正是"吃一口"能救命的地方吗？</p>')
P('<div class="card"><table><thead><tr><th>出场方式</th><th>均值</th><th>中位</th><th>胜率</th></tr></thead><tbody>')
for x in exits:
    cls = ' class="hl2"' if x['lab'] == '一直不动' else (' class="hl"' if '10%' in x['lab'] else '')
    P(f'<tr{cls}><td>{x["lab"]}</td><td><b>{x["mean"]:+.1f}%</b></td>'
      f'<td>{x["med"]:+.1f}%</td><td>{x["win"]:.0f}%</td></tr>')
P('</tbody></table></div>')
P('<p class="note"><b>单调得可怕：卖得越早，赚得越少。</b>'
  '+10% 全跑 7.5% → +20% 全跑 11.7% → 卖一半 13~17% → 不动 19.2%。'
  '注意"卖一半"这一列：涨 20% 卖一半（+15.4%）比全跑 20%（+11.7%）好，'
  '因为它把右尾留在了场里。</p>')
P('<div class="card"><div class="chartbox"><canvas id="c3"></canvas></div></div>')

P('<h3>把持有期拉长，一路往上涨</h3>')
P('<div class="card"><table><thead><tr><th>持有期</th><th>均值</th><th>中位</th><th>胜率</th></tr></thead><tbody>')
for x in holds:
    cls = ' class="hl2"' if x['h'] >= 500 else ''
    P(f'<tr{cls}><td>{x["h"]} 日（约 {x["h"]/243:.1f} 年）</td>'
      f'<td><b>{x["mean"]:+.1f}%</b></td><td>{x["med"]:+.1f}%</td><td>{x["win"]:.0f}%</td></tr>')
P('</tbody></table></div>')
P('<div class="card"><div class="chartbox"><canvas id="c4"></canvas></div></div>')
P('<p class="note">750 个交易日（约 3 年）平均 <b>+32.1%、胜率 100%</b>。'
  '当然这里面有幸存偏差 —— 能持到 750 日的只剩 2013、2014、2020 那几笔。'
  '但方向很清楚：<b>这一层的收益是"等"出来的，不是"跑"出来的。</b></p>')

P('<h3>根子：收益全在右尾，而"吃一口"砍的正好是右尾</h3>')
P('<div class="card"><table><tbody>'
  f'<tr><td>11 次全部算进去</td><td><b>{tail["all"]:+.1f}%</b></td></tr>'
  f'<tr class="hl"><td>去掉涨幅最大的 3 笔</td><td><b>{tail["drop3"]:+.1f}%</b></td></tr>'
  f'<tr class="hl"><td>去掉涨幅最大的 5 笔</td><td><b>{tail["drop5"]:+.1f}%</b></td></tr>'
  f'<tr><td>涨幅 &gt;30% 的比例</td><td>{tail["gt30"]:.0f}%</td></tr>'
  f'<tr><td>涨幅 &gt;50% 的比例</td><td>{tail["gt50"]:.0f}%</td></tr>'
  f'<tr><td>为正的比例</td><td>{tail["pos"]:.0f}%</td></tr>'
  '</tbody></table></div>')
P(f'<div class="verdict">去掉 3 笔，均值就从 +19.2% 掉到 <b>{tail["drop3"]:+.1f}%</b>；'
  f'去掉 5 笔只剩 <b>{tail["drop5"]:+.1f}%</b>，几乎归零。'
  '<b>这意味着：任何"赚 10% 就跑"的规则，'
  '都会把唯一能赚钱的那几笔砍在半路上。</b><br><br>'
  '顺带一个反直觉的事实：胜率 45%~55% 看着像抛硬币，'
  '但这不是坏事 —— <b>低胜率 + 高盈亏比</b>才是这个游戏的形状。'
  '追求"吃一口"的高胜率，本质是用盈亏比换胜率，在这条收益分布上是亏本买卖。</div>')

# ---------- 05 ----------
P('<h2><span class="no">05</span>换成月 K 看会不会不一样</h2>')
P('<p class="note">"月线级别"最字面的读法是<b>月 K 连阴</b>。单独用它：</p>')
P('<div class="card"><table><thead><tr><th>规则</th><th>次数</th><th>之后 3 月</th><th>6 月</th>'
  '<th>12 月</th><th>24 月</th><th>12 月胜率</th></tr></thead><tbody>')
for x in monthK:
    P(f'<tr><td>连阴 {x["k"]} 个月后买入</td><td>{x["nev"]}</td><td>{x["m3"]:+.1f}%</td>'
      f'<td>{x["m6"]:+.1f}%</td><td>{x["m12"]:+.1f}%</td><td>{x["m24"]:+.1f}%</td>'
      f'<td>{x["win12"]:.0f}%</td></tr>')
P('</tbody></table></div>')
P('<p class="note">连阴 1 个月 166 次，之后 12 个月只有 <b>+1.3%</b>、胜率 52% —— 等于抛硬币。'
  '连阴 3 个月略好（24 次，+2.5%，胜率 58%），但 24 个月 +11.5% 也就跑平买入持有。'
  '<b>单独用月 K 连阴，没有信息量。</b></p>')
P('<p class="note">加上"离底 ≤10%"之后：</p>')
P('<div class="card"><table><thead><tr><th>规则</th><th>次数</th><th>250 日均值</th><th>中位</th>'
  '<th>胜率</th><th>平均浮亏</th></tr></thead><tbody>')
for x in mk_with:
    P(f'<tr><td>连阴 {x["k"]} 月 + 离底≤10%</td><td>{x["n"]}</td><td><b>{x["mean"]:+.1f}%</b></td>'
      f'<td>{x["med"]:+.1f}%</td><td>{x["win"]:.0f}%</td><td>{x["low"]:+.1f}%</td></tr>')
P('</tbody></table></div>')
P('<p class="note"><b>看起来比骨架的 +19.2% 还好（+24%~+25%），浮亏也更小（−4.5%~−6.2%）。</b>'
  '但这是三层条件叠出来的：离底 ≤10% ＋ 上月收阴 ＋ 更晚的入场。<b>换算到资金曲线就不成立了</b>，'
  '见第 07 节。这就是"单笔收益"和"全期年化"的区别 —— 买得晚，单笔更漂亮，但机会也少了。</p>')

# ---------- 06 ----------
P('<h2><span class="no">06</span>牛市里的回调呢：一个干净的反证</h2>')
P('<p class="note">你说的是"牛市熊市过程中的波动"。那就专门看牛市：'
  '当时在年线上方、前一年涨幅 0~40%（排除泡沫顶）。</p>')
P('<div class="card"><table><thead><tr><th>条件</th><th>样本数</th><th>之后 250 日</th>'
  '<th>胜率</th><th>平均浮亏</th></tr></thead><tbody>')
for x in bull:
    cls = ' class="hl"' if '10%' in x['lab'] else ''
    P(f'<tr{cls}><td>{x["lab"]}</td><td>{x["n"]}</td><td><b>{x["mean"]:+.1f}%</b></td>'
      f'<td>{x["win"]:.0f}%</td><td>{x["low"]:+.1f}%</td></tr>')
P('</tbody></table></div>')
P('<div class="verdict"><b>牛市里随便哪天买，之后 250 日 +5.5%；'
  '非要等到"跌 10%"才买，只有 +0.6%，胜率还从 45% 掉到 38%，浮亏反而更深（−15.2%）。</b><br><br>'
  '这是本轮最干净的反证：<b>"等回调再买"在牛市里是负价值。</b>'
  '因为在上升趋势里，"跌 10%"这个信号既可能是洗盘，也可能是趋势断裂，它不筛选好坏，只筛掉时间 —— '
  '而时间在这段行情里是赚钱的。<br>'
  '顺带说明：牛市里跌 15% 只发生过 2 次，样本根本不够，所以"牛市深回调"这件事在 A 股是<b>稀有事件</b>，'
  '不是一个可以反复做的套路。</div>')

# ---------- 07 ----------
P('<h2><span class="no">07</span>换成年化：加条件一律降收益、降回撤</h2>')
P('<p class="note">单笔收益会骗人。真正的问法是：<b>把现金空转的 2.5% 也算进去，全期年化多少、'
  '最大回撤多少。</b>日频资金曲线，满仓进、持满 250 日出、其余时间生息：</p>')
P('<div class="card"><table><thead><tr><th>策略</th><th>年化</th><th>最大回撤</th>'
  '<th>期末净值</th><th>Calmar</th></tr></thead><tbody>')
for x in eqs:
    calmar = abs(x['cagr'] / x['dd']) if x['dd'] < 0 else None
    cls = ' class="hl2"' if '骨架：' in x['lab'] else ''
    c_s = f'{calmar:.2f}' if calmar else '—'
    P(f'<tr{cls}><td>{x["lab"]}</td><td><b>{x["cagr"]:+.2f}%</b></td><td>{x["dd"]:+.1f}%</td>'
      f'<td>{x["nav"]:.2f}</td><td>{c_s}</td></tr>')
P('</tbody></table></div>')
P('<div class="card"><table><thead><tr><th>加的条件</th><th>年化</th><th>回撤</th></tr></thead><tbody>'
  '<tr class="hl2"><td>骨架：进带即买</td><td><b>+5.53%</b></td><td>−43.3%</td></tr>'
  '<tr><td>+ 上月收阴才买</td><td>+4.49%</td><td>−28.6%</td></tr>'
  '<tr><td>+ 连续 2 月收阴才买</td><td>+4.20%</td><td>−29.3%</td></tr>'
  '<tr><td>+ 跌 15% 才买</td><td>+4.85%</td><td>−25.7%</td></tr>'
  '<tr><td>+ 否决（前一年涨 &gt;40% 不买）</td><td>+5.53%</td><td>−43.3%</td></tr>'
  '</tbody></table></div>')
P('<p class="note"><b>规律很清楚：所有"加条件"都在降年化（5.53% → 4.20%~4.85%），'
  '同时降回撤（−43.3% → −25.7%~−29.3%）。</b>'
  '没有一条既能加收益又能降回撤 —— 这跟 v1.4 分档的结论一模一样：'
  '<b>精细化的价值全在"能不能拿住"，不在"多赚多少"。</b></p>')
P('<div class="card"><div class="chartbox tall"><canvas id="c6"></canvas></div></div>')

P('<h3>而且那 −43.3% 的回撤，是运气还是必然</h3>')
P('<div class="warn"><b>必须说清：加条件后回撤变小，主要不是因为规则更聪明，'
  '而是因为它们恰好错过了同一次入场。</b><br><br>'
  f'骨架 11 次触发里有 <b>2014-10-20</b> 这一次：买在 {case14["buy"]:.0f}，'
  f'一路涨到 {case14["peak"]:.0f}（{case14["peak_r"]:+.1f}%），'
  f'然后回吐到 {case14["low"]:.0f}（从高点 <b>{case14["giveback"]:+.1f}%</b>），'
  f'持满 250 日时是 {case14["end_r"]:+.1f}%。<br>'
  '组合净值从 3.55 掉到 2.01，就是这一笔。<b>"跌 15% 才买"和"等月线收阴"这两条规则，'
  '恰好在 2014-10-20 那天都不成立，于是绕过了它 —— 这是样本运气，不是规则优势。</b></div>')
P('<p class="note">反过来也说明了另一件事：<b>骨架的 −43.3% 不是入场的问题，是出场的问题。</b>'
  '2014-10 买入后涨了 119%，然后回吐 55.7%，规则却一直没让卖（v1.3 起出场规则就是"待重做"状态）。'
  '把出场补上的诊断：</p>')
P('<div class="card"><table><thead><tr><th>出场规则</th><th>年化</th><th>最大回撤</th><th>净值</th></tr></thead><tbody>'
  '<tr class="hl"><td>只持 250 日（v1.3 现状）</td><td>+5.53%</td><td>−43.3%</td><td>4.98</td></tr>'
  '<tr><td>+ 峰回 20% 出场</td><td>+4.23%</td><td>−38.5%</td><td>3.44</td></tr>'
  '<tr><td>+ 峰回 25% 出场</td><td>+3.99%</td><td>−36.8%</td><td>3.22</td></tr>'
  '<tr><td>+ 峰回 30% 出场</td><td>+5.69%</td><td>−32.3%</td><td>5.22</td></tr>'
  '</tbody></table></div>')
P('<p class="note">20% 和 25% 都让年化变差，30% 变好 —— <b>又是"测不出最优"的形状</b>，'
  '和 v1.3 测八种出场方案时一模一样。所以这里不下结论，只把它记成下一件事的输入：'
  '<b>出场规则必须重做，而且不能靠单一阈值。</b></p>')

# ---------- 08 ----------
P('<h2><span class="no">08</span>唯一的真增量：一条否决权</h2>')
P('<p class="note">上面全是坏消息，这一节是好消息。在 33 次回撤触发里，'
  '<b>前 250 日涨幅 &gt;40%</b> 的那几次，是另一种东西：</p>')
P('<div class="card"><table><thead><tr><th>分组</th><th>次数</th><th>之后 250 日均值</th>'
  '<th>胜率</th><th>平均浮亏</th></tr></thead><tbody>')
for x in veto:
    cls = ' class="hl"' if '否决掉' in x['lab'] else (' class="hl2"' if '保留' in x['lab'] else '')
    P(f'<tr{cls}><td>{x["lab"]}</td><td>{x["n"]}</td><td><b>{x["mean"]:+.1f}%</b></td>'
      f'<td>{x["win"]:.0f}%</td><td>{x["low"]:+.1f}%</td></tr>')
P('</tbody></table></div>')
P(f'<p class="note">被否决掉的那几次是：<b>{"、".join(veto_dates)}</b> —— '
  '全部是泡沫顶刚破、或者反弹诱多的时候（2007-06 是 5,300 点下跌的第一波，2007-11 是顶部，'
  '2015-06 是 5,166 的第一根大阴线）。<b>四次全部为负，均值 −26.9%，胜率 0%，浮亏 −34.9%，'
  '最差那次 −61.5%。</b><br>'
  '保留 vs 否决的差异 <b>+37.7pp，t=2.72，p=0.052</b> —— 卡在 0.05 门口，差一点点。</p>')
P('<div class="warn"><b>诚实标注：n=4，p=0.052 刚好不显著。</b>这个 p 值是靠"4 个样本全是负的"堆出来的，'
  '不满足正态假设，<b>不能当成严格的显著性</b>。但它有常识支撑：'
  '"一年翻倍之后的第一波下跌"和"跌了三年之后的横盘"是两种完全不同的东西，'
  '把它们归成同一个"跌 15%"本身就是错的。<b>所以我建议采纳，但标注为"方向明确、样本不足"。</b></div>')
P('<div class="ok"><b>这条否决权和骨架不冲突。</b>骨架的出手区（离底 ≤10%）历史上'
  '"前一年涨幅"全部在 −17.7%~+6.9% 之间，<b>从来没有触发过这条否决</b>。'
  '所以它是一条纯保险 —— 不改变骨架行为，只防止你在骨架之外被"跌了很多"诱惑进去。</div>')

# ---------- 09 ----------
P('<h2><span class="no">09</span>多重检验：把"最好"这条打回原形</h2>')
P('<p class="note">先把入场规则那 7 条拿出来校正（出场规则和持有期的 p 值更差，不列了）。'
  '在这么多组合里，"最好的那个"很容易是噪声。做一次校正：</p>')
P('<div class="card"><table><thead><tr><th>排名</th><th>原始 p</th><th>Bonferroni</th>'
  '<th>Holm</th><th>BH (FDR)</th></tr></thead><tbody>'
  '<tr class="hl"><td>1</td><td>0.048</td><td>0.336</td><td>0.336</td><td>0.336</td></tr>'
  '<tr><td>2</td><td>0.048</td><td>0.336</td><td>0.288</td><td>0.168</td></tr>'
  '<tr><td>3</td><td>0.086</td><td>0.602</td><td>0.430</td><td>0.201</td></tr>'
  '<tr><td>4</td><td>0.149</td><td>1.000</td><td>0.596</td><td>0.261</td></tr>'
  '<tr><td>5</td><td>0.183</td><td>1.000</td><td>0.549</td><td>0.256</td></tr>'
  '<tr><td>6</td><td>0.537</td><td>1.000</td><td>1.000</td><td>0.627</td></tr>'
  '<tr><td>7</td><td>0.569</td><td>1.000</td><td>0.569</td><td>0.569</td></tr>'
  '</tbody></table></div>')
P('<div class="verdict"><b>校正后最小 p = 0.336，一个都不显著。</b><br><br>'
  '这句话必须原样留着：<b>"离底决定一切"这个结论，统计上只能算"强线索"，不算"已证明"。</b>'
  '它之所以还能用，是因为它和骨架的底层逻辑一致（买得便宜才赚钱），'
  '而不是因为它过了 p&lt;0.05。<b>如果你只记一句话：月线回调这一层，'
  '没有任何一条规则经得起多重检验。</b></div>')

# ---------- 10 ----------
P('<h2><span class="no">10</span>判决</h2>')
P('<div class="verdict"><b>月线级回调不是一个独立战场。它是骨架同一个战场里的"入场时点"问题，'
  '而且加它只会降收益。</b></div>')
P('<div class="card"><table class="tleft"><thead><tr><th>问题</th><th>答案</th><th>依据</th></tr></thead><tbody>'
  '<tr><td>月线回调存在吗</td><td><b>存在</b>，15% 回撤 2.0 次/年（去重后 1.1 次）</td>'
  '<td>33 次 / 29.9 年</td></tr>'
  '<tr><td>不看位置能吃吗</td><td><b>不能</b>，中位 −0.6%、胜率 48%</td><td>§01</td></tr>'
  '<tr><td>看位置呢</td><td><b>能，但只有离底 ≤10% 那一段</b>（+20.2%、83%）</td><td>§02，p=0.025</td></tr>'
  '<tr><td>骨架不买的区间能吃吗</td><td><b>不能</b>，+3.16% vs 生息 2.57%，p=0.926</td><td>§03</td></tr>'
  '<tr><td>"吃一口就跑"行不行</td><td><b>最差的选择</b>，+10% 止盈比不动少赚 11.7pp</td><td>§04</td></tr>'
  '<tr><td>月 K 连阴有没有用</td><td><b>单独没用</b>（+1.3%）；<b>叠骨架也不增年化</b></td><td>§05、§07</td></tr>'
  '<tr><td>牛市里等回调再买</td><td><b>负价值</b>，+0.6% vs 随便买 +5.5%</td><td>§06</td></tr>'
  '<tr><td>有什么能加进骨架</td><td><b>只有一条否决权</b>：前一年涨 &gt;40% 不做回调</td><td>§08</td></tr>'
  '</tbody></table></div>')

P('<h3>v1.5 的改动（只有两条）</h3>')
P('<div class="card"><table class="tleft"><thead><tr><th>位置</th><th>改动</th></tr></thead><tbody>'
  '<tr><td>§3 出手规则</td><td>新增否决条款：<b>若前 250 日涨幅 &gt;40%，暂停出手（等回到 &lt;40% 或跌幅确认）</b>。'
  '历史上在出手区从未触发，是纯保险</td></tr>'
  '<tr><td>§7 缺陷清单</td><td>新增 <b>#14：月线级回调不可作为独立战场</b>'
  '（不加筛选中位 −0.6%；骨架区间外 p=0.926；吃一口砍掉全部右尾）</td></tr>'
  '<tr><td>§6.2 等待</td><td>不改。本轮反而<b>加强</b>了它：'
  '骨架不买的地方买进去 = 白折腾，这正是"空仓是成本、不是浪费"的实证</td></tr>'
  '</tbody></table></div>')

P('<div class="ok"><b>对你那句话的直接回答：</b>'
  '"月线级别的回调能不能吃一口" —— <b>能吃，但不是"一口"，是"一整段"</b>；'
  '而且吃的地方必须是骨架划好的那一片。<br>'
  '如果改成"跌 15% 就抄一把，涨 10% 就跑"，历史上 11 次里赚 7 次、每次 7.5%，'
  '看起来挺美，但同期把 4 次 +30% 以上的行情全部砍掉，'
  '算上 62% 的时间空仓，年化还不如一直拿着。'
  '<b>"吃一口"的诱惑在于它让你感觉每次都在赢，代价是赢的都很小。</b></div>')

P('<h3>本轮作废与翻车清单</h3>')
P('<div class="card"><table class="tleft"><thead><tr><th>说过的话</th><th>为什么错</th></tr></thead><tbody>'
  '<tr><td>"回撤 25% 以上是深水区，之后 250 日 +2.7%"</td>'
  '<td>均值被右尾拉动，中位是 −1.6%。<b>看均值会得出相反的结论</b></td></tr>'
  '<tr><td>"月 K 连阴 + 离底≤10% 能到 +25%，比骨架好"</td>'
  '<td>单笔收益，不是年化。换成资金曲线后 5.53% → 4.49%，<b>是变差</b></td></tr>'
  '<tr><td>"加条件后回撤从 −43.3% 降到 −25.7%，规则更稳"</td>'
  '<td>那是因为绕过了 2014-10-20 这一笔，<b>样本运气</b>，不是规则优势</td></tr>'
  '<tr><td>第一版把全期年化算成 +1.79%（跑输空仓）</td>'
  '<td>用了线性加权。正确是几何加权，实际 <b>+5.53%</b>。差 3 倍，结论完全反转</td></tr>'
  '<tr><td>"去重后 33 次 = 1.1 次/年，样本够"</td>'
  '<td>分层后每档只有 6~7 次，<b>ANOVA 靠 4 个 6 样本撑起来</b>，功效很低</td></tr>'
  '</tbody></table></div>')

P('<div class="foot">')
P(f'数据：上证综指日收盘 {d.d.iloc[0].date()} ~ {d.d.iloc[-1].date()}（{n:,} 根，{yrs:.1f} 年），'
  'local parquet；CPI 口径与 v1.4 一致（2008-01=100，"全国-当月"开 12 次方当环比）。<br>')
P('口径：<b>不折 CPI</b>（持有期 1~3 年，通胀累计 1%~4%，会略微高估绝对收益；横向比较不受影响）。'
  '现金空转按 2.5%/年日复利。买入价用信号日收盘（保守：收盘才知道触发）。<br>')
P('脚本：<code>d140_monthly_corr.py</code>（阈值扫描/事件明细）、'
  '<code>d141_eat_bite.py</code>（出场网格/分层）、<code>d142_dedup.py</code>（去重/增量检验）、'
  '<code>d143_decisive.py</code>（规则对比/分批止盈/持有期）、'
  '<code>d144_monthK.py</code>（月K连阴/牛市回调/否决权/多重检验）、'
  '<code>d145_equity.py</code>（资金曲线）。<br>')
P('局限：①离底分层只有 2008 年后样本；②分层每档 6~7 次，统计功效低，'
  '经 Bonferroni 校正后全部不显著；③持有 750 日的样本只剩 3~4 笔，有幸存偏差；'
  '④所有结论不含交易成本（ETF 双边约 0.1%）。<b>不构成投资建议。</b>')
P('</div>')
P('</div>')

BODY = '\n'.join(H)

# 图表脚本
JS = """
<script>
const G={color:'rgba(0,0,0,.06)'};
new Chart(document.getElementById('c1'),{type:'line',
 data:{labels:@@L1@@,datasets:[
  {label:'之后 250 日中位',data:@@D1A@@,borderColor:'#A32D2D',borderWidth:2,
   pointRadius:4,pointBackgroundColor:'#A32D2D',tension:0.2},
  {label:'之后 250 日均值',data:@@D1B@@,borderColor:'#5f5e5a',borderWidth:1.5,
   borderDash:[5,4],pointRadius:3,pointBackgroundColor:'#5f5e5a',tension:0.2}]},
 options:{responsive:true,maintainAspectRatio:false,animation:false,
  plugins:{legend:{position:'top',labels:{boxWidth:14,font:{size:11}}},
   tooltip:{callbacks:{label:c=>c.dataset.label+' '+c.parsed.y.toFixed(1)+'%'}}},
  scales:{y:{title:{display:true,text:'之后 250 日收益'},grid:G,
    ticks:{font:{size:11},callback:v=>v+'%'}},
   x:{title:{display:true,text:'从 250 日高点的回撤阈值'},grid:{display:false},
    ticks:{font:{size:11}}}}}});

new Chart(document.getElementById('c2'),{data:{labels:@@L2@@,datasets:[
  {type:'bar',label:'之后 250 日均值',data:@@D2A@@,backgroundColor:@@C2@@,
   borderRadius:3,borderSkipped:false,order:2,yAxisID:'y'},
  {type:'line',label:'胜率（右轴）',data:@@D2B@@,borderColor:'#854F0B',borderWidth:2,
   pointRadius:4,pointBackgroundColor:'#854F0B',tension:0.15,order:1,yAxisID:'y1'}]},
 options:{responsive:true,maintainAspectRatio:false,animation:false,
  plugins:{legend:{position:'top',labels:{boxWidth:14,font:{size:11}}},
   tooltip:{callbacks:{label:c=>c.dataset.type==='bar'?c.parsed.y.toFixed(1)+'%':'胜率 '+c.parsed.y.toFixed(0)+'%'}}},
  scales:{y:{position:'left',title:{display:true,text:'之后 250 日均值'},grid:G,
    ticks:{font:{size:11},callback:v=>v+'%'}},
   y1:{position:'right',title:{display:true,text:'胜率'},min:0,max:100,
    grid:{display:false},ticks:{font:{size:11},callback:v=>v+'%'}},
   x:{grid:{display:false},ticks:{font:{size:11}}}}}});

new Chart(document.getElementById('c3'),{type:'bar',
 data:{labels:@@L3@@,datasets:[{data:@@D3@@,backgroundColor:@@C3@@,
   borderRadius:3,borderSkipped:false}]},
 options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,animation:false,
  plugins:{legend:{display:false},
   tooltip:{callbacks:{label:c=>c.parsed.x.toFixed(1)+'%'}}},
  scales:{x:{title:{display:true,text:'持有 250 日的平均收益'},grid:G,
    ticks:{font:{size:11},callback:v=>v+'%'}},
   y:{grid:{display:false},ticks:{font:{size:11}}}}}});

new Chart(document.getElementById('c4'),{type:'line',
 data:{labels:@@L4@@,datasets:[
  {label:'平均收益',data:@@D4@@,borderColor:'#A32D2D',borderWidth:2,
   pointRadius:5,pointBackgroundColor:'#A32D2D',tension:0.2,fill:false}]},
 options:{responsive:true,maintainAspectRatio:false,animation:false,
  plugins:{legend:{display:false},
   tooltip:{callbacks:{label:c=>c.parsed.y.toFixed(1)+'%'}}},
  scales:{y:{title:{display:true,text:'平均收益'},grid:G,
    ticks:{font:{size:11},callback:v=>v+'%'}},
   x:{title:{display:true,text:'持有期（交易日）'},grid:{display:false},
    ticks:{font:{size:11}}}}}});

new Chart(document.getElementById('c6'),{type:'line',
 data:{labels:@@L6@@,datasets:[
  {label:'一直空仓生息 2.49%',data:@@D6A@@,borderColor:'#8c8b86',borderWidth:1.5,
   borderDash:[4,3],pointRadius:0,tension:0.1},
  {label:'买入持有 4.76%',data:@@D6B@@,borderColor:'#1a1a1a',borderWidth:1.8,
   pointRadius:0,tension:0.1},
  {label:'骨架：进带即买 5.53%',data:@@D6C@@,borderColor:'#A32D2D',borderWidth:2.2,
   pointRadius:0,tension:0.1},
  {label:'骨架 + 跌15% 4.85%',data:@@D6D@@,borderColor:'#854F0B',borderWidth:1.6,
   borderDash:[6,4],pointRadius:0,tension:0.1}]},
 options:{responsive:true,maintainAspectRatio:false,animation:false,
  plugins:{legend:{position:'top',labels:{boxWidth:14,font:{size:11}}}},
  scales:{y:{title:{display:true,text:'净值（起点=1）'},grid:G,min:0.8,
    ticks:{font:{size:11},stepSize:1}},
   x:{grid:{display:false},ticks:{font:{size:10},maxTicksLimit:10}}}}});
</script>
"""

col2 = []
for x in layers:
    col2.append('#1C6B3C' if '≤10' in x['lab'] else
                ('#B5B3AC' if '10~30' in x['lab'] else
                 ('#C98A3E' if '30~60' in x['lab'] else '#A32D2D')))
col3 = ['#C98A3E' if '全跑' in x['lab'] else ('#8c8b86' if '一半' in x['lab'] else '#1C6B3C')
        for x in exits]

HEAD = (f'<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>月线回调能不能吃一口</title>{CSS}</head><body>')
FOOT = f'<script>{CHARTJS}</script>' + JS + '</body></html>'
HTML = HEAD + BODY + FOOT

repl = {
    '@@L1@@': json.dumps([f"{x['thr']:.0f}%" for x in scan]),
    '@@D1A@@': json.dumps([round(x['med250'], 1) for x in scan]),
    '@@D1B@@': json.dumps([round(x['mean250'], 1) for x in scan]),
    '@@L2@@': json.dumps([x['lab'] for x in layers]),
    '@@D2A@@': json.dumps([round(x['mean'], 1) for x in layers]),
    '@@D2B@@': json.dumps([round(x['win'], 0) for x in layers]),
    '@@C2@@': json.dumps(col2),
    '@@L3@@': json.dumps([x['lab'] for x in exits]),
    '@@D3@@': json.dumps([round(x['mean'], 1) for x in exits]),
    '@@C3@@': json.dumps(col3),
    '@@L4@@': json.dumps([f"{x['h']} 日" for x in holds]),
    '@@D4@@': json.dumps([round(x['mean'], 1) for x in holds]),
    '@@L6@@': json.dumps(L6),
    '@@D6A@@': json.dumps(D6[0]),
    '@@D6B@@': json.dumps(D6[1]),
    '@@D6C@@': json.dumps(D6[2]),
    '@@D6D@@': json.dumps(D6[3]),
}
for k, v in repl.items():
    HTML = HTML.replace(k, v)

open(OUT, 'w', encoding='utf-8').write(HTML)
print('written', OUT, len(HTML), 'chars')
i = HTML.find('<script>')
j = HTML.find('</script>', i)
body = HTML[:i] + HTML[j:]
print('正文 NaN:', body.count('NaN'), '| undefined:', body.count('undefined'))
print('canvas:', HTML.count('<canvas'))
