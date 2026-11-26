# -*- coding: utf-8 -*-
"""
d94: 两条线的交叉 —— "固定3000点" vs "随CPI上调的等值线"
+ 空仓的真实成本（含理财收益）
+ 买入日明细
"""
import pandas as pd, numpy as np, json, os

B = os.path.dirname(os.path.abspath(__file__))
OUT = {}

cpi = pd.read_parquet('G:/AStockData/supplement/macro/cpi.parquet')
cpi['ym'] = cpi['月份'].str.replace('年', '-').str.replace('月份', '')
cpi['d'] = pd.to_datetime(cpi['ym'], format='%Y-%m')
cpi = cpi.sort_values('d').reset_index(drop=True)
cpi['mom'] = (pd.to_numeric(cpi['全国-当月'], errors='coerce') / 100.0) ** (1 / 12.0)
ix = [100.0]
for i in range(1, len(cpi)): ix.append(ix[-1] * cpi['mom'].iloc[i])
cpi['cpi_idx'] = ix
CPI_T = cpi['cpi_idx'].iloc[-1]

print('=== 两条买入线：固定 3,000 vs "实际 2,100" 等值线（随 CPI 上调）===')
print(f"{'年':<6}{'CPI定基':>9}{'固定3000的实际值':>16}{'实际2100→名义':>15}{'差':>8}")
rows = []
for y in range(2015, 2027):
    sub = cpi[cpi['d'].dt.year == y]
    if len(sub) == 0: continue
    ci = float(sub['cpi_idx'].iloc[-1])
    real_of_3000 = 3000 * 100.0 / ci
    nom_of_2100 = 2100 * ci / 100.0
    rows.append(dict(year=y, cpi=round(ci, 1), real3000=round(real_of_3000), nom2100=round(nom_of_2100),
                     diff=round(nom_of_2100 - 3000)))
    print(f"{y:<6}{ci:>9.1f}{real_of_3000:>16.0f}{nom_of_2100:>15.0f}{nom_of_2100-3000:>+8.0f}")
OUT['two_lines'] = rows

# 交叉点
print('\n=== 交叉点："实际2,100等值线"何时越过固定3,000 ===')
for r in rows:
    if r['nom2100'] > 3000:
        print(f"  {r['year']} 年：等值线 {r['nom2100']:.0f} 首次超过 3,000（固定线此时只值实际 {r['real3000']:.0f}）")
        OUT['cross'] = dict(year=r['year'], nom2100=r['nom2100'], real3000=r['real3000'])
        break

# 未来
print('\n=== 未来：守住"实际 2,100"需要把线抬到哪（CPI 假设 1.5%/yr）===')
fut = []
for y in range(2026, 2041, 2):
    k = 1.015 ** (y - 2026)
    nom = 2100 * CPI_T / 100.0 * k
    real_of_3000 = 3000 * 100.0 / (CPI_T * k)
    fut.append(dict(year=y, line=round(nom), real3000=round(real_of_3000)))
    print(f"  {y}: 线在 {nom:>5.0f} 点 | 若你仍用 3,000，等于实际 {real_of_3000:>5.0f}（比 2,100 {'高' if real_of_3000>2100 else '低'} {abs(real_of_3000-2100):.0f}）")
OUT['future'] = fut

# 空仓成本
print('\n=== 空仓的真实成本（每年）===')
m2 = pd.read_parquet('G:/AStockData/supplement/macro/money_supply.parquet')
m2['ym'] = m2['月份'].str.replace('年', '-').str.replace('月份', '')
m2['d'] = pd.to_datetime(m2['ym'], format='%Y-%m')
m2 = m2.sort_values('d')
v0 = float(m2['货币和准货币(M2)-数量(亿元)'].iloc[0]); v1 = float(m2['货币和准货币(M2)-数量(亿元)'].iloc[-1])
yrs = (m2['d'].iloc[-1] - m2['d'].iloc[0]).days / 365.25
m2c = (v1 / v0) ** (1 / yrs) - 1
cpic = CPI_T ** (1 / yrs * 1)  # 不对，重算下面
cpic = (CPI_T / 100.0) ** (1 / yrs) - 1
print(f"M2 CAGR {m2c*100:.2f}%/yr | CPI CAGR {cpic*100:.2f}%/yr")
for dep in [0.0, 0.015, 0.025, 0.035]:
    print(f"  空仓拿 {dep*100:.1f}% 理财：相对M2 {(dep-m2c)*100:+.2f}%/yr，10年剩 {((1+dep)/(1+m2c))**10*100:.0f}% | 相对CPI {(dep-cpic)*100:+.2f}%/yr，10年剩 {((1+dep)/(1+cpic))**10*100:.0f}%")
OUT['cash'] = dict(m2=round(m2c*100, 2), cpi=round(cpic*100, 2))

# 买入日明细（改造 backtest 记录买入日）
m = pd.read_parquet(os.path.join(B, 'd91_daily.parquet')).sort_values('d').reset_index(drop=True)
m = pd.merge_asof(m, cpi[['d', 'cpi_idx']].assign(avail=cpi['d'] + pd.offsets.MonthBegin(1) + pd.Timedelta(days=14))[['avail', 'cpi_idx']].sort_values('avail'),
                  left_on='d', right_on='avail', direction='backward')
m['real'] = m['close'] * 100.0 / m['cpi_idx']

def bt(gate, th, start='2015-01-05', up_cap=3900):
    i0 = int(np.argmin(np.abs((m['d'] - pd.Timestamp(start)).values)))
    pos, half, peak, peak_i, eq, ent, entd = 0.0, False, 0.0, 0, 1.0, 0.0, None
    out = []
    for i in range(i0, len(m)):
        c = m['etf'].iloc[i]
        if pos > 0:
            eq *= c / m['etf'].iloc[i-1]
            if c >= peak: peak, peak_i = c, i
        cheep = (not np.isnan(m[gate].iloc[i])) and m[gate].iloc[i] <= th
        cross = m['close'].iloc[i] > m['ma20'].iloc[i] and m['close'].iloc[i-1] <= m['ma20'].iloc[i-1]
        if pos == 0:
            if cheep and cross and m['close'].iloc[i] < up_cap:
                pos, half, peak, peak_i, ent, entd = 1.0, False, c, i, c, str(m['d'].iloc[i].date())
        else:
            why = None
            if (peak-c)/peak >= .10 and i-peak_i >= 20: why = '回撤'
            elif m['amt_q'].iloc[i] >= .95 and m['close'].iloc[i] > 3500: why = '热度'
            if why and not half: half = True
            elif why and half:
                out.append((entd, str(m['d'].iloc[i].date()), round(float(c/ent-1)*100, 1), why,
                            round(float(m['real'].iloc[i]), 0)))
                pos, half = 0.0, False
    return out, round(float((eq-1)*100), 1)

print('\n=== 5 笔（实际<=2100）的买入日与卖出日 ===')
o, tot = bt('real', 2100)
for x in o: print(f"  买 {x[0]} → 卖 {x[1]}  {x[2]:+6.1f}%  {x[3]}  卖时实际值 {x[4]:.0f}")
print(f"  合计 {tot:.1f}%")
OUT['trades_2100'] = o

print('\n=== 8 笔（名义<=3000）的买入日与卖出日 ===')
o2, tot2 = bt('close', 3000)
for x in o2: print(f"  买 {x[0]} → 卖 {x[1]}  {x[2]:+6.1f}%  {x[3]}  卖时实际值 {x[4]:.0f}")
print(f"  合计 {tot2:.1f}%")
OUT['trades_3000'] = o2

with open(os.path.join(B, 'd94_result.json'), 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print('\nsaved d94_result.json')
