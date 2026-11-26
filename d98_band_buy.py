# -*- coding: utf-8 -*-
"""
d98: 触带即买、不做择时出场 —— 检验"接近底就是出手信号"能不能单独成立
      以及：段数/日期，看看"十年三次"和"一年一两次"差在哪
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
d['real'] = d['close'] * 100.0 / d['cpi_idx']
d = d.dropna(subset=['real']).reset_index(drop=True)
N = len(d)
ANCHOR = 1880.0
d['rA'] = d['real'] / ANCHOR
C = d['close'].values

def eps(k, min_gap=20):
    m = (d['rA'] <= 1 + k).values
    segs, i = [], 0
    while i < N:
        if m[i]:
            j = i
            while j + 1 < N and m[j + 1]:
                j += 1
            segs.append([i, j]); i = j + 1
        else:
            i += 1
    mg = []
    for s in segs:
        if mg and s[0] - mg[-1][1] <= min_gap:
            mg[-1][1] = s[1]
        else:
            mg.append(s)
    return mg

# 转成「首次下穿 1+k」的入场：从 >1+k 变成 <=1+k 的第一天
def entries(k):
    m = (d['rA'] <= 1 + k).values
    out = []
    for i in range(1, N):
        if m[i] and not m[i - 1]:
            out.append(i)
    return out

print('=== 各带宽：段数与入场日期 ===')
rows = []
for k in [0.0, 0.05, 0.10, 0.15, 0.20]:
    es = entries(k)
    ds = [str(d.d.iloc[i].date()) for i in es]
    ns = [round(float(d.close.iloc[i])) for i in es]
    print(f'  离底 {int(k*100):>2}% 内：{len(es)} 次  ' + ' | '.join(f'{a}({b})' for a, b in zip(ds, ns)))
    rows.append(dict(k=int(k * 100), n=len(es), dates=ds, closes=ns))
OUT['entries'] = rows

print('\n=== 触带即买、固定持有（等权，不做任何出场规则）===')
print(f"{'带宽':>6}{'笔数':>6}{'持1年':>9}{'胜率':>7}{'持2年':>9}{'胜率':>7}{'持20月':>9}{'胜率':>7}{'最差':>8}")
res = []
for k in [0.0, 0.05, 0.10, 0.15, 0.20]:
    es = entries(k)
    r1, r2, r20 = [], [], []
    for i in es:
        if i + 244 < N: r1.append(C[i + 244] / C[i] - 1)
        if i + 488 < N: r2.append(C[i + 488] / C[i] - 1)
        j = min(N - 1, i + 407)                       # 20 个月
        if i + 407 < N: r20.append(C[j] / C[i] - 1)
    def st(v):
        if not v: return (None, None)
        return (round(float(np.mean(v)) * 100, 1), round(float((np.array(v) > 0).mean()) * 100, 0))
    a1, w1 = st(r1); a2, w2 = st(r2); a3, w3 = st(r20)
    worst = round(float(min(r1)) * 100, 1) if r1 else None
    print(f'  {int(k*100):>4}%{len(es):>6}{str(a1):>9}{str(w1):>7}{str(a2):>9}{str(w2):>7}{str(a3):>9}{str(w3):>7}{str(worst):>8}')
    res.append(dict(k=int(k * 100), n=len(es), m1=a1, w1=w1, m2=a2, w2=w2, m20=a3, w20=w3, worst=worst))
OUT['hold'] = res

# 同期基准：任意一天买入持 1/2 年
base1 = [(C[i + 244] / C[i] - 1) for i in range(N - 244)]
base2 = [(C[i + 488] / C[i] - 1) for i in range(N - 488)]
print(f'\n基准（任意一天买入）：1年 均 {np.mean(base1)*100:.1f}% 胜率 {(np.array(base1)>0).mean()*100:.0f}%'
      f' | 2年 均 {np.mean(base2)*100:.1f}% 胜率 {(np.array(base2)>0).mean()*100:.0f}%')
OUT['base'] = dict(m1=round(float(np.mean(base1)) * 100, 1), w1=round(float((np.array(base1) > 0).mean()) * 100, 0),
                   m2=round(float(np.mean(base2)) * 100, 1), w2=round(float((np.array(base2) > 0).mean()) * 100, 0))

# 「准备期」有多长：从第一次触 15% 带，到真正见底（实际值最低）隔多久
print('\n=== 从「进入 20% 区」到「真底」隔多久 ===')
es20 = entries(0.20)
gap = []
for i in es20:
    j = i
    while j + 1 < N and d['real'].iloc[j + 1] <= d['real'].iloc[i] * 1.0 and False:
        j += 1
    # 找 i 之后 500 个交易日内的实际值最低点
    hi = min(N, i + 500)
    lo = i + int(np.argmin(d['real'].values[i:hi]))
    gap.append(dict(enter=str(d.d.iloc[i].date()), low=str(d.d.iloc[lo].date()), days=int(lo - i),
                    drop=round(float(d.real.iloc[lo] / d.real.iloc[i] - 1) * 100, 1)))
    print(f"  {gap[-1]['enter']} → 底 {gap[-1]['low']}  隔 {gap[-1]['days']:>3} 交易日  再跌 {gap[-1]['drop']:>6}%")
OUT['gap'] = gap
print(f"  中位 {np.median([g['days'] for g in gap]):.0f} 交易日，中位再跌 {np.median([g['drop'] for g in gap]):.1f}%")

json.dump(OUT, open(os.path.join(B, 'd98_result.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nsaved')
