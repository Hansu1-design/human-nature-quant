# -*- coding: utf-8 -*-
"""d100: 出手区买入 × 三种出场，看"吃肥"能不能留住"""
import os, json
import numpy as np
import pandas as pd

B = os.path.dirname(os.path.abspath(__file__))
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
N = len(d); C = d['close'].values; E = d['etf'].values
d['rA'] = d['real'] / 1880.0

def entries(k, lock=244):
    m = (d['rA'] <= 1 + k).values
    out, last = [], -10**9
    for i in range(1, N):
        if m[i] and not m[i-1] and i - last >= lock:
            out.append(i); last = i
    return out

def run(i0, tp=0.50, trail=0.20, maxm=20, use_tp=True, use_trail=True):
    """从 i0 买入；+50% 止盈 / 从峰回撤 20% 止盈 / 20个月到期"""
    end = min(N - 1, i0 + int(maxm * 21))
    peak = C[i0]
    for j in range(i0 + 1, end + 1):
        peak = max(peak, C[j])
        if use_tp and C[j] / C[i0] - 1 >= tp:
            return C[j] / C[i0] - 1, j - i0, '止盈+50%'
        if use_trail and peak > C[i0] * 1.15 and C[j] / peak - 1 <= -trail:
            return C[j] / C[i0] - 1, j - i0, '回撤-20%'
    return C[end] / C[i0] - 1, end - i0, '到期'

print(f"{'入场':<12}{'点数':>6}{'纯持有20月':>11}{'仅止盈50%':>11}{'仅回撤20%':>11}{'两条都要':>11}")
rows = []
for k in [0.05, 0.10, 0.15]:
    es = entries(k)
    print(f'--- 离底 {int(k*100)}% 内 ---')
    acc = {kk: [] for kk in ['pure', 'tp', 'tr', 'both']}
    for i in es:
        if i + 407 >= N: 
            continue
        pure = (C[min(N-1,i+407)]/C[i]-1, 407, '到期')
        tp   = run(i, use_tp=True, use_trail=False)
        tr   = run(i, use_tp=False, use_trail=True)
        bo   = run(i, use_tp=True, use_trail=True)
        for kk, v in zip(['pure','tp','tr','both'], [pure, tp, tr, bo]):
            acc[kk].append(v[0])
        print(f"  {str(d.d.iloc[i].date()):<12}{int(C[i]):>6}{pure[0]:>10.1%}{tp[0]:>10.1%}({tp[2][:2]}){tr[0]:>9.1%}({tr[2][:2]}){bo[0]:>10.1%}({bo[2][:2]})")
    m = {kk: (np.mean(v)*100 if v else None) for kk, v in acc.items()}
    print(f"  {'均值':<10}{'':>6}{m['pure']:>10.1f}%{m['tp']:>11.1f}%{m['tr']:>11.1f}%{m['both']:>11.1f}%")
    rows.append(dict(k=int(k*100), n=len(acc['pure']), pure=round(m['pure'],1),
                     tp=round(m['tp'],1), tr=round(m['tr'],1), both=round(m['both'],1)))
    print()

# 现在买入的期望（离底 1.36）
i_now = N - 1
print('现在（离底 36%）不构成入场，仅列参考分档：1.3~1.4 → 1年均 -4.6%，胜率 32.1%')
# 横盘磨底：需要多少年 CPI 才把底抬到 1.10 倍
close_now = C[-1]; cpi_now = float(d['cpi_idx'].iloc[-1])
need_cpi = close_now / 1.10 / 1880 * 100
for g in [0.015, 0.02, 0.03]:
    yrs = np.log(need_cpi / cpi_now) / np.log(1 + g)
    print(f'  CPI {g*100:.1f}%/yr、上证横盘 {close_now:.0f} 不动：{yrs:.1f} 年后才进 10% 出手区')
json.dump(rows, open(os.path.join(B, 'd100_result.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
