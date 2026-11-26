# -*- coding: utf-8 -*-
"""
d92: 把"等值"算回来 —— 熊底/牛顶在 CPI 口径下的等值点位，以及 CPI 分位闸门回测
"""
import pandas as pd, numpy as np, json, os

B = os.path.dirname(os.path.abspath(__file__))
OUT = {}
m = pd.read_parquet(os.path.join(B, 'd91_daily.parquet'))
m = m.sort_values('d').reset_index(drop=True)
df = pd.read_parquet(os.path.join(B, 'd90_daily.parquet')).sort_values('d').reset_index(drop=True)

# 重新把 CPI 定基接到 df 上
cpi = pd.read_parquet('G:/AStockData/supplement/macro/cpi.parquet')
cpi['ym'] = cpi['月份'].str.replace('年', '-').str.replace('月份', '')
cpi['d'] = pd.to_datetime(cpi['ym'], format='%Y-%m')
cpi = cpi.sort_values('d').reset_index(drop=True)
cpi['mom'] = (pd.to_numeric(cpi['全国-当月'], errors='coerce') / 100.0) ** (1 / 12.0)
idx = [100.0]
for i in range(1, len(cpi)):
    idx.append(idx[-1] * cpi['mom'].iloc[i])
cpi['cpi_idx'] = idx
cpi['avail'] = cpi['d'] + pd.offsets.MonthBegin(1) + pd.Timedelta(days=14)
df = pd.merge_asof(df.sort_values('d'), cpi[['avail', 'cpi_idx']].sort_values('avail'),
                   left_on='d', right_on='avail', direction='backward')
df = df.dropna(subset=['cpi_idx']).reset_index(drop=True)
df['real'] = df['close'] * 100.0 / df['cpi_idx']
CPI_T = df['cpi_idx'].iloc[-1]
print(f'CPI 定基(2008-01=100) 末值 {CPI_T:.1f}')

# ---------- 1. 熊底 / 牛顶 的等值点位表 ----------
bottoms = ['2008-11-04', '2013-06-25', '2016-01-28', '2019-01-03', '2024-02-05']
tops = ['2009-08-04', '2015-06-12', '2018-01-24', '2021-09-13', '2026-05-13']

def snap(d0):
    i = int(np.argmin(np.abs((df['d'] - pd.Timestamp(d0)).values)))
    r = df.iloc[i]
    return dict(date=d0, nom=round(float(r['close'])), cpi_idx=round(float(r['cpi_idx']), 1),
                real=round(float(r['real'])), m2=round(float(r['eq'])), ratio=round(float(r['ratio']), 2),
                nom_today=round(float(r['real']) * CPI_T / 100.0))

print('\n=== 熊底：把当时的"实际底"折算成今天的名义点位 ===')
print(f"{'日期':<12}{'名义':>7}{'CPI定基':>9}{'实际值':>8}{'→今日名义':>11}{'M2等价':>8}")
bt = []
for d0 in bottoms:
    x = snap(d0); bt.append(x)
    print(f"{x['date']:<12}{x['nom']:>7}{x['cpi_idx']:>9.1f}{x['real']:>8}{x['nom_today']:>11}{x['m2']:>8}")
OUT['bottoms'] = bt
rv = [x['real'] for x in bt]; nt = [x['nom_today'] for x in bt]
print(f"实际底：均值 {np.mean(rv):.0f} | 中位 {np.median(rv):.0f} | 范围 {min(rv)}~{max(rv)} | 变异系数 {np.std(rv)/np.mean(rv)*100:.1f}%")
print(f"→ 折算今日名义：均值 {np.mean(nt):.0f} | 中位 {np.median(nt):.0f} | 范围 {min(nt)}~{max(nt)}")
OUT['bottom_stat'] = dict(real_mean=round(float(np.mean(rv))), real_med=round(float(np.median(rv))),
                          real_min=min(rv), real_max=max(rv), cv=round(float(np.std(rv)/np.mean(rv)*100), 1),
                          nom_today_mean=round(float(np.mean(nt))), nom_today_med=round(float(np.median(nt))),
                          nom_today_min=min(nt), nom_today_max=max(nt))

print('\n=== 牛顶：把当时的"实际顶"折算成今天的名义点位 ===')
print(f"{'日期':<12}{'名义':>7}{'CPI定基':>9}{'实际值':>8}{'→今日名义':>11}{'M2等价':>8}")
tp = []
for d0 in tops:
    x = snap(d0); tp.append(x)
    print(f"{x['date']:<12}{x['nom']:>7}{x['cpi_idx']:>9.1f}{x['real']:>8}{x['nom_today']:>11}{x['m2']:>8}")
OUT['tops'] = tp
tv = [x['real'] for x in tp]
print(f"实际顶：均值 {np.mean(tv):.0f} | 中位 {np.median(tv):.0f} | 范围 {min(tv)}~{max(tv)}")
print(f"→ 折算今日名义：{min(x['nom_today'] for x in tp)}~{max(x['nom_today'] for x in tp)}")

# ---------- 2. 3000 点的"实际含量"随时间变化 ----------
print('\n=== 同样 3,000 点，在不同年份等于多少"2008 年不变价" ===')
rows3k = []
for y in [2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024, 2026]:
    sub = df[df['d'].dt.year == y]
    if len(sub) == 0: continue
    ci = float(sub['cpi_idx'].mean())
    real = 3000 * 100.0 / ci
    rows3k.append(dict(year=y, cpi=round(ci, 1), real=round(real)))
    print(f"  {y}: CPI定基 {ci:>6.1f}  →  3,000 点 = 实际 {real:>5.0f}")
OUT['three_k'] = rows3k

# ---------- 3. 现在的位置：实际值 vs 五轮熊底 ----------
last = df.iloc[-1]
print(f"\n=== 当前（{last['d'].date()}）===")
print(f"名义 {last['close']:.0f} | CPI实际 {last['real']:.0f} | 五轮熊底中位 {np.median(rv):.0f}")
print(f"  → 当前实际值比熊底中位高 {(last['real']/np.median(rv)-1)*100:+.0f}%")
print(f"  → 距最深的 2008 底（{min(rv)}）还高 {(last['real']/min(rv)-1)*100:+.0f}%")
OUT['now'] = dict(date=str(last['d'].date()), nom=round(float(last['close'])), real=round(float(last['real'])),
                  vs_med=round(float(last['real']/np.median(rv)-1)*100), vs_min=round(float(last['real']/min(rv)-1)*100))

# ---------- 4. 闸门对比：CPI 实际分位 vs 名义分位 ----------
print('\n=== 闸门对比（同出场规则，11 个起点）===')
def backtest(gate_col, gate_th, start, hold_floor=20, dd_stop=0.10, exit_heat=0.95):
    i0 = int(np.argmin(np.abs((m['d'] - pd.Timestamp(start)).values)))
    pos, half, peak, peak_i, entry = 0.0, False, 0.0, 0, 0.0
    eq = 1.0
    trades = []
    for i in range(i0, len(m)):
        c = m['etf'].iloc[i]
        if pos > 0:
            eq *= c / m['etf'].iloc[i - 1]
            if c >= peak: peak, peak_i = c, i
        cheep = (not np.isnan(m[gate_col].iloc[i])) and m[gate_col].iloc[i] <= gate_th
        cross = m['close'].iloc[i] > m['ma20'].iloc[i] and m['close'].iloc[i-1] <= m['ma20'].iloc[i-1]
        if pos == 0:
            if cheep and cross and m['close'].iloc[i] < 3900:
                pos, half, peak, peak_i, entry = 1.0, False, c, i, c
                eq = eq if eq else 1.0
        else:
            why = None
            if (peak - c) / peak >= dd_stop and i - peak_i >= hold_floor: why = '回撤'
            elif m['amt_q'].iloc[i] >= exit_heat and m['close'].iloc[i] > 3500: why = '热度'
            if why and not half: half = True
            elif why and half:
                trades.append((str(m['d'].iloc[i].date()), round(float(c/entry-1)*100, 1), why))
                pos, half = 0.0, False
    return dict(total=round(float((eq-1)*100), 1), n=len(trades), trades=trades)

starts = ['2015-01-05', '2015-06-12', '2016-01-04', '2017-01-03', '2018-01-02', '2019-01-02',
          '2020-01-02', '2021-01-04', '2022-01-04', '2023-01-03', '2024-01-02']
res = {}
for nm, col, th in [('名义分位≤0.25', 'q_nom', 0.25), ('CPI实际分位≤0.25', 'q_cpi', 0.25),
                    ('CPI实际分位≤0.20', 'q_cpi', 0.20), ('ratio分位≤0.25', 'q_ratio', 0.25)]:
    outs = []
    for st in starts:
        r = backtest(col, th, st)
        i0 = int(np.argmin(np.abs((m['d'] - pd.Timestamp(st)).values)))
        bh = (m['etf'].iloc[-1] / m['etf'].iloc[i0] - 1) * 100
        outs.append(dict(start=st, total=r['total'], n=r['n'], bh=round(float(bh), 1), win=bool(r['total'] > bh)))
    w = sum(1 for x in outs if x['win'])
    med = float(np.median([x['total'] for x in outs]))
    res[nm] = dict(rows=outs, win=f'{w}/11', median=round(med, 1))
    print(f"\n-- {nm}  跑赢 {w}/11 | 11起点中位收益 {med:.1f}%")
    for x in outs:
        print(f"   {x['start']}  {x['total']:>7.1f}% ({x['n']}笔)  BH {x['bh']:>6.1f}%  {'赢' if x['win'] else '输'}")
OUT['gates'] = res

with open(os.path.join(B, 'd92_result.json'), 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print('\nsaved d92_result.json')
