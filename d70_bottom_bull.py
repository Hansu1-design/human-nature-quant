# -*- coding: utf-8 -*-
"""
d70: 检验用户三条新论点
  A. 下有底 —— 上证跌破 L 后的连续区段，最长多久收回
  B. 牛市回吐 —— 每轮牛市顶后，跌掉牛市涨幅的百分之几
  C. 2016-03 那次误判的真实原因（三种锚读数对照）
"""
import json
import numpy as np
import pandas as pd

OUT = {}

d = pd.read_parquet('idx_sse_all.parquet')
d['close'] = pd.to_numeric(d['close'], errors='coerce')
d['d'] = pd.to_datetime(d['date'])
d = d.dropna(subset=['close']).sort_values('d').reset_index(drop=True)
d = d[d['d'] >= '1996-01-01'].reset_index(drop=True)
n = len(d)
C = d['close'].values
print(f"样本 {d['d'].iloc[0].date()} ~ {d['d'].iloc[-1].date()}  n={n}\n")

print("=" * 84)
print("A. 低于某点位的连续区段：几次、最长多久收回（gap 20 日内的反复算同一段）")
print("=" * 84)

def episodes(L, gap=20):
    below = C < L
    eps, i = [], 0
    while i < n:
        if below[i]:
            j, k = i, i
            while k < n:
                if below[k]:
                    j, k = k, k + 1
                else:
                    k += 1
                    if k >= n or k - j > gap:
                        break
            eps.append((i, j))
            i = j + 1
        else:
            i += 1
    return eps

A = []
for L in [2000, 2200, 2400, 2500, 2700, 2800, 3000, 3200]:
    eps = episodes(L)
    recs = []
    for i, j in eps:
        r = j + 1
        while r < n and C[r] < L:
            r += 1
        recs.append(dict(start=str(d['d'].iloc[i].date()), end=str(d['d'].iloc[j].date()),
                         dur=int(j - i + 1), low=round(float(C[i:j + 1].min()), 0),
                         recover=None if r >= n else str(d['d'].iloc[r].date()),
                         wait=None if r >= n else int(r - j)))
    never = [x for x in recs if x['recover'] is None]
    A.append(dict(L=L, n=len(recs), never=len(never), recs=recs))
    durs = [x['dur'] for x in recs]
    print(f"\nL={L}  共 {len(recs)} 段  未收回 {len(never)} 段  "
          f"最长在下方 {max(durs)} 交易日 ({max(durs)/244:.1f} 年)")
    for x in recs:
        if x['dur'] >= 100 or x['recover'] is None:
            print(f"   {x['start']} -> {x['end']}  在下方 {x['dur']:4d} 日 ({x['dur']/244:4.1f}年) "
                  f"最低 {x['low']:.0f}  收回 {x['recover'] or '未收回'}")
OUT['A_bottom'] = A

print("\n  各点位最后一次低于，距今：")
for L in [2000, 2200, 2400, 2500, 2800, 3000]:
    eps = episodes(L)
    if not eps:
        print(f"   L={L}: 从未低于")
        continue
    i, j = eps[-1]
    print(f"   L={L}: 最后一段 {d['d'].iloc[i].date()} ~ {d['d'].iloc[j].date()}，"
          f"距今 {(d['d'].iloc[-1] - d['d'].iloc[j]).days} 自然日")

print("\n" + "=" * 84)
print("B. 牛市顶之后回吐了多少（收盘价口径）")
print("=" * 84)

cycles = [
    ('1996-01-02', '1997-05-12', '1999-05-18'),
    ('2005-06-06', '2007-10-16', '2008-10-28'),
    ('2008-10-28', '2009-08-04', '2013-06-25'),
    ('2013-06-25', '2015-06-12', '2016-01-28'),
    ('2019-01-04', '2021-02-18', '2024-02-05'),
    ('2024-09-18', '2024-10-08', '2025-04-07'),
]
B = []
print(f"{'启动':>12} {'点位':>7} {'顶部':>12} {'点位':>7} {'熊底':>12} {'点位':>7} "
      f"{'涨幅':>7} {'回吐%':>6} {'剩%':>5} {'底/启动':>7}")
for s, p, e in cycles:
    i_s = int(np.argmin(np.abs((d['d'] - pd.Timestamp(s)).values)))
    i_p = int(np.argmin(np.abs((d['d'] - pd.Timestamp(p)).values)))
    i_e = int(np.argmin(np.abs((d['d'] - pd.Timestamp(e)).values)))
    cs, cp, ce = C[i_s], C[i_p], C[i_e]
    up = cp - cs
    ratio = (cp - ce) / up if up > 0 else np.nan
    B.append(dict(start=s, s_lvl=round(float(cs), 0), peak=p, p_lvl=round(float(cp), 0),
                  trough=e, e_lvl=round(float(ce), 0),
                  up=round(float(up / cs * 100), 1),
                  give_pct=round(float(ratio * 100), 1),
                  remain_pct=round(float((1 - ratio) * 100), 1),
                  trough_vs_start=round(float(ce / cs), 2)))
    print(f"{s:>12} {cs:7.0f} {p:>12} {cp:7.0f} {e:>12} {ce:7.0f} "
          f"{up/cs*100:6.0f}% {ratio*100:5.0f}% {(1-ratio)*100:4.0f}% {ce/cs:7.2f}")
OUT['B_bull_giveback'] = B
g = [r['give_pct'] for r in B if r['give_pct'] == r['give_pct']]
print(f"\n  回吐比例 {sorted(g)}  中位 {np.median(g):.0f}%  范围 {min(g):.0f}~{max(g):.0f}%")
print(f"  底/启动 {[r['trough_vs_start'] for r in B]}  全部 >= 0.90")

print("\n  若规则是「顶后回吐 >= 60% 则买」，历史触发位置与实际最低：")
for r in B:
    cp, cs, ce = r['p_lvl'], r['s_lvl'], r['e_lvl']
    trig = cp - 0.60 * (cp - cs)
    gap = (ce / trig - 1) * 100
    print(f"   {r['peak']} 顶 {cp:.0f}（启动 {cs:.0f}）触发点 {trig:.0f}  "
          f"实际最低 {ce:.0f}  " + (f"触发后还跌 {gap:.0f}%" if ce < trig else "未跌到触发点"))

print("\n" + "=" * 84)
print("C. 2016-03-02 那一笔：为什么分位说贵、回吐说刚好")
print("=" * 84)
t0 = pd.Timestamp('2016-03-02')
i0 = int(np.argmin(np.abs((d['d'] - t0).values)))
c0 = C[i0]
print(f"当日上证 {c0:.0f}")
for yrs in [3, 5, 7, 10]:
    w = C[max(0, i0 - int(244 * yrs)):i0 + 1]
    print(f"  过去 {yrs:2d} 年价格分位 {float((w < c0).mean()):.3f}   窗口 {w[0]:.0f}~{w.max():.0f}")
rm = np.maximum.accumulate(C)
print(f"  距历史最高({rm[i0]:.0f}) 回撤 {(c0/rm[i0]-1)*100:.1f}%")
i15 = int(np.argmin(np.abs((d['d'] - pd.Timestamp('2015-06-12')).values)))
i13 = int(np.argmin(np.abs((d['d'] - pd.Timestamp('2013-06-25')).values)))
up = C[i15] - C[i13]
gb = (C[i15] - c0) / up * 100
print(f"  距 2015 顶({C[i15]:.0f}) 回撤 {(c0/C[i15]-1)*100:.1f}%")
print(f"  回吐 2013-06({C[i13]:.0f}) 至 2015-06 涨幅的比例 = {gb:.1f}%")
OUT['C_anchor'] = dict(date='2016-03-02', lvl=round(float(c0), 0),
                       q3=round(float((C[i0-732:i0+1] < c0).mean()), 3),
                       q5=round(float((C[i0-1220:i0+1] < c0).mean()), 3),
                       q10=round(float((C[i0-2440:i0+1] < c0).mean()), 3),
                       dd_from_hist_peak=round(float(c0 / rm[i0] - 1) * 100, 1),
                       dd_from_15peak=round(float(c0 / C[i15] - 1) * 100, 1),
                       giveback_15=round(float(gb), 1))

json.dump(OUT, open('d70_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print("\n[saved] d70_result.json")
