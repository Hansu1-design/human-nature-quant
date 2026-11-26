# -*- coding: utf-8 -*-
"""
d99: 去重后的独立样本 —— 加 12 个月冷却期
     并检验两段式：准备区（离底 20%~10%）→ 出手区（离底 ≤10%）
"""
import os, json
import numpy as np
import pandas as pd

B = os.path.dirname(os.path.abspath(__file__))
OUT = {}
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
d['real'] = d['close'] * 100.0 / c['cpi_idx'].iloc[-1] * 100 / 100
d['real'] = d['close'] * 100.0 / d['cpi_idx']
d = d.dropna(subset=['real']).reset_index(drop=True)
N = len(d); C = d['close'].values
ANCHOR = 1880.0
d['rA'] = d['real'] / ANCHOR
LOCK = 244          # 12 个月冷却

def entries(k, lock=LOCK, start=0):
    m = (d['rA'] <= 1 + k).values
    out, last = [], -10**9
    for i in range(max(1, start), N):
        if m[i] and not m[i - 1] and i - last >= lock:
            out.append(i); last = i
    return out

def stats(es, tag):
    r1 = [C[i+244]/C[i]-1 for i in es if i+244 < N]
    r2 = [C[i+488]/C[i]-1 for i in es if i+488 < N]
    r20 = [C[min(N-1,i+407)]/C[i]-1 for i in es if i+407 < N]
    f = lambda v: (round(float(np.mean(v))*100,1), round(float((np.array(v)>0).mean())*100,0)) if v else (None,None)
    a1,w1 = f(r1); a2,w2 = f(r2); a3,w3 = f(r20)
    print(f'{tag:<22}{len(es):>4} 笔  1年 {str(a1):>6}%/{str(w1):>3}%   2年 {str(a2):>6}%/{str(w2):>3}%   20月 {str(a3):>6}%/{str(w3):>3}%   最差1年 {round(min(r1)*100,1) if r1 else None}')
    return dict(tag=tag, n=len(es), m1=a1, w1=w1, m2=a2, w2=w2, m20=a3, w20=w3,
                worst=round(float(min(r1))*100,1) if r1 else None,
                dates=[str(d.d.iloc[i].date()) for i in es],
                closes=[int(round(C[i])) for i in es])

print('=== 去重后（12 个月冷却）===')
res = []
for k in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
    es = entries(k)
    res.append(stats(es, f'离底 {int(k*100)}% 内'))
OUT['dedupe'] = res

base1 = [(C[i+244]/C[i]-1) for i in range(N-244)]
base2 = [(C[i+488]/C[i]-1) for i in range(N-488)]
print(f"{'基准(任意一天)':<22}{N-244:>4} 天  1年 {np.mean(base1)*100:>6.1f}%/{ (np.array(base1)>0).mean()*100:>3.0f}%   2年 {np.mean(base2)*100:>6.1f}%/{ (np.array(base2)>0).mean()*100:>3.0f}%")
OUT['base'] = dict(m1=round(float(np.mean(base1))*100,1), w1=round(float((np.array(base1)>0).mean())*100,0),
                   m2=round(float(np.mean(base2))*100,1), w2=round(float((np.array(base2)>0).mean())*100,0))

# 两段式：先"准备"(<=20%)，再在"出手区"(<=10%) 下单
print('\n=== 两段式：准备区 20% → 出手区 10% ===')
for kk in [0.05, 0.10, 0.15]:
    es = entries(kk)
    print(f'  只在离底 {int(kk*100)}% 内出手（12月冷却）：')
    print('   ', [f'{str(d.d.iloc[i].date())}({int(C[i])})' for i in es])
    OUT[f'entry_{int(kk*100)}'] = [str(d.d.iloc[i].date()) for i in es]

# 逐年：13 年里有多少年出现过"出手区"
yrs = sorted(set(d.d.dt.year))
print('\n=== 逐年：最低离底倍数 / 是否进入出手区(<=10%) ===')
yr_rows = []
for y in yrs:
    s = d[d.d.dt.year == y]
    lo = float(s['rA'].min())
    yr_rows.append(dict(year=int(y), min_rA=round(lo,3), in_zone=lo <= 1.10, deep=lo <= 1.05,
                        low_close=int(s['close'].min()), low_date=str(s.loc[s['rA'].idxmin(),'d'].date())))
    print(f"  {y}  最低 {lo:.3f}（离底 {(lo-1)*100:+.1f}%）  最低点 {int(s['close'].min())} @ {yr_rows[-1]['low_date']}  {'✓出手区' if lo<=1.10 else '—'}")
OUT['by_year'] = yr_rows
nz = sum(1 for r in yr_rows if r['in_zone'])
print(f'\n  13 年里 {nz} 年进入过 10% 出手区；{sum(1 for r in yr_rows if r["deep"])} 年进入过 5% 深水区')

json.dump(OUT, open(os.path.join(B, 'd99_result.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nsaved')
