# -*- coding: utf-8 -*-
"""
d102: 修掉「起点在带内」被漏掉的 bug，重做全部
      入场 = 从带外进入带内的第一天（含样本首日已在带内）
      去重 = 12 个月冷却
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
d = d.dropna(subset=['cpi_idx']).reset_index(drop=True)
d['real'] = d['close'] * 100.0 / d['cpi_idx']
N = len(d); C = d['close'].values
ANCHOR = 1880.0
d['rA'] = d['real'] / ANCHOR
print(f'样本 {N} 根 {d.d.iloc[0].date()} ~ {d.d.iloc[-1].date()}')

def entries(k, lock=244):
    m = (d['rA'] <= 1 + k).values
    out, last = [], -10**9
    for i in range(N):
        prev = m[i-1] if i > 0 else False          # 首日视为「刚进入」
        if m[i] and not prev and i - last >= lock:
            out.append(i); last = i
    return out

def summ(k):
    es = entries(k)
    r1 = [C[i+244]/C[i]-1 for i in es if i+244 < N]
    r2 = [C[i+488]/C[i]-1 for i in es if i+488 < N]
    r20 = [C[min(N-1, i+407)]/C[i]-1 for i in es if i+407 < N]
    pk, pine = [], []
    for i in es:
        e = min(N-1, i+5*244); seg = C[i:e+1]
        p = i + int(np.argmax(seg)); pk.append(seg.max()/C[i]-1); pine.append((p-i)/21)
    f = lambda v: (round(float(np.mean(v))*100, 1), round(float((np.array(v) > 0).mean())*100, 0)) if v else (None, None)
    a1, w1 = f(r1); a2, w2 = f(r2); a3, w3 = f(r20)
    return dict(k=int(k*100), n=len(es), m1=a1, w1=w1, m2=a2, w2=w2, m20=a3, w20=w3,
                peak=round(float(np.mean(pk))*100, 1) if pk else None,
                peak_med=round(float(np.median(pk))*100, 1) if pk else None,
                peak_mo=round(float(np.median(pine)), 1) if pine else None,
                h50=int(sum(1 for x in pk if x >= .5)), h100=int(sum(1 for x in pk if x >= 1.0)),
                worst=round(float(min(r1))*100, 1) if r1 else None,
                dates=[str(d.d.iloc[i].date()) for i in es],
                closes=[int(round(C[i])) for i in es])

print(f"\n{'带宽':>6}{'笔数':>5}{'1年':>8}{'胜率':>6}{'2年':>8}{'20月':>8}{'5年最高(中位)':>13}{'到顶月':>7}{'≥50%':>7}{'≥100%':>7}{'最差1年':>8}")
rows = []
for k in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
    r = summ(k); rows.append(r)
    print(f"  {r['k']:>4}%{r['n']:>5}{str(r['m1']):>8}{str(r['w1']):>6}{str(r['m2']):>8}{str(r['m20']):>8}"
          f"{str(r['peak_med']):>13}{str(r['peak_mo']):>7}{r['h50']:>7}{r['h100']:>7}{str(r['worst']):>8}")
OUT['bands'] = rows

b1 = [(C[i+244]/C[i]-1) for i in range(N-244)]
b2 = [(C[i+488]/C[i]-1) for i in range(N-488)]
print(f"  {'基准':>5}{N-244:>5}{np.mean(b1)*100:>8.1f}{(np.array(b1)>0).mean()*100:>6.0f}{np.mean(b2)*100:>8.1f}")
OUT['base'] = dict(m1=round(float(np.mean(b1))*100,1), w1=round(float((np.array(b1)>0).mean())*100,0), m2=round(float(np.mean(b2))*100,1))

print('\n=== 各带宽入场明细 ===')
for r in rows:
    print(f"  离底 {r['k']:>2}%：{r['n']} 笔  " + ' | '.join(f'{a}({b})' for a, b in zip(r['dates'], r['closes'])))

json.dump(OUT, open(os.path.join(B, 'd102_result.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved')
