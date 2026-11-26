# -*- coding: utf-8 -*-
"""
d95: 核实 Hermes 的四处质疑
A. 2024 年上证最低收盘到底在哪天（五轮熊底取点是否有偏）
B. 实际<=2000 与 名义<=2707 的 2 笔是否同一组（重复计数？）
C. "最差 +12.2%" 三行相同 -> 该起点是否全程空仓
D. 笔数 vs 在场时间 vs 收益：收益到底由什么驱动
E. 2016-02-16 那笔，CPI 实际值是多少（为什么"实际2100"漏掉）
"""
import pandas as pd, numpy as np, json, os

B = os.path.dirname(os.path.abspath(__file__))
OUT = {}

sse = pd.read_parquet(os.path.join(B, 'idx_sse_all.parquet'))
sse['close'] = pd.to_numeric(sse['close'], errors='coerce')
sse['low'] = pd.to_numeric(sse['low'], errors='coerce')
sse['d'] = pd.to_datetime(sse['date'])
sse = sse.dropna(subset=['close']).sort_values('d').reset_index(drop=True)

# ---------- A. 每年最低收盘 ----------
print('=== A. 逐年上证最低收盘（核对熊底取点）===')
for y in range(2008, 2027):
    s = sse[sse['d'].dt.year == y]
    if len(s) == 0: continue
    i = s['close'].idxmin()
    il = s['low'].idxmin() if s['low'].notna().any() else i
    print(f"  {y}: 最低收盘 {s.loc[i,'close']:.0f} @ {s.loc[i,'d'].date()} | 最低盘中 {s.loc[il,'low']:.0f} @ {s.loc[il,'d'].date()}")
print('  → 2024 年两次探底：')
s24 = sse[(sse['d'] >= '2024-01-01') & (sse['d'] <= '2024-12-31')]
print(s24.nsmallest(5, 'close')[['d', 'close', 'low']].to_string(index=False))

# ---------- B/C/D/E ----------
m = pd.read_parquet(os.path.join(B, 'd91_daily.parquet')).sort_values('d').reset_index(drop=True)
cpi = pd.read_parquet('G:/AStockData/supplement/macro/cpi.parquet')
cpi['ym'] = cpi['月份'].str.replace('年', '-').str.replace('月份', '')
cpi['d'] = pd.to_datetime(cpi['ym'], format='%Y-%m')
cpi = cpi.sort_values('d').reset_index(drop=True)
cpi['mo'] = (pd.to_numeric(cpi['全国-当月'], errors='coerce') / 100.0) ** (1 / 12.0)
ix = [100.0]
for i in range(1, len(cpi)): ix.append(ix[-1] * cpi['mo'].iloc[i])
cpi['cpi_idx'] = ix
cpi['avail'] = cpi['d'] + pd.offsets.MonthBegin(1) + pd.Timedelta(days=14)
m = pd.merge_asof(m, cpi[['avail', 'cpi_idx']].sort_values('avail'), left_on='d', right_on='avail', direction='backward')
m['real'] = m['close'] * 100.0 / m['cpi_idx']
m = m.dropna(subset=['real']).reset_index(drop=True)

def bt(gate, th, start='2015-01-05', up_cap=3900):
    i0 = int(np.argmin(np.abs((m['d'] - pd.Timestamp(start)).values)))
    pos, half, peak, peak_i, eq, ent, entd = 0.0, False, 0.0, 0, 1.0, 0.0, None
    out, in_days, last_i = [], 0, i0
    for i in range(i0, len(m)):
        c = m['etf'].iloc[i]
        if pos > 0:
            eq *= c / m['etf'].iloc[i-1]
            if c >= peak: peak, peak_i = c, i
        if pos > 0: in_days += 1
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
                out.append(dict(buy=entd, sell=str(m['d'].iloc[i].date()), ret=round(float(c/ent-1)*100, 1),
                                why=why, real_buy=round(float(m['real'].iloc[m.index[m['d']==pd.Timestamp(entd)][0]]), 0) if entd else None,
                                real_sell=round(float(m['real'].iloc[i]), 0)))
                pos, half = 0.0, False
    tot = (eq - 1) * 100
    ndays = len(m) - i0
    return dict(trades=out, total=round(float(tot), 1), n=len(out), in_days=in_days, ndays=ndays,
                expo=round(in_days / ndays * 100, 1),
                cagr=round(float((eq) ** (250.0 / ndays) - 1) * 100, 2))

print('\n=== B/E. 逐笔（含买入时的 CPI 实际值）===')
for gate, th, nm in [('real', 2000, '实际<=2000'), ('close', 2707, '名义<=2707'), ('real', 2100, '实际<=2100'), ('close', 3000, '名义<=3000')]:
    r = bt(gate, th)
    print(f"\n-- {nm}: {r['n']} 笔, 合计 {r['total']}%, 在场 {r['expo']}%, 在场年化 {r['cagr']}%")
    for x in r['trades']:
        print(f"   买 {x['buy']}(实际{x['real_buy']}) → 卖 {x['sell']}(实际{x['real_sell']})  {x['ret']:+.1f}%  {x['why']}")

# ---------- C. 各起点的空仓比例 ----------
print('\n=== C. 各起点的在场比例（检验"最差+12.2%"是否=全程空仓）===')
starts = ['2015-01-05', '2015-06-12', '2016-01-04', '2017-01-03', '2018-01-02', '2019-01-02',
          '2020-01-02', '2021-01-04', '2022-01-04', '2023-01-03', '2024-01-02']
tbl = {}
for gate, th, nm in [('real', 2100, '实际<=2100'), ('close', 3000, '名义<=3000'), ('q_nom', 0.25, '名义分位<=0.25')]:
    rows = []
    for st in starts:
        r = bt(gate, th, st)
        rows.append(dict(start=st, total=r['total'], n=r['n'], expo=r['expo'], cagr=r['cagr']))
    tbl[nm] = rows
    print(f"\n-- {nm}")
    for x in rows:
        print(f"   {x['start']}  {x['total']:>7.1f}%  {x['n']}笔  在场{x['expo']:>5.1f}%  在场年化 {x['cagr']:>5.2f}%")
OUT['expo'] = tbl

# ---------- D. 笔数/在场 vs 收益 ----------
print('\n=== D. 收益由什么驱动：笔数 vs 在场比例 vs 总收益（2015-01 起点，多档闸门）===')
drive = []
for gate, th, nm in [('real', 2000, '实际2000'), ('real', 2100, '实际2100'), ('close', 2707, '名义2707'),
                     ('close', 3000, '名义3000'), ('close', 3200, '名义3200'), ('close', 3400, '名义3400'),
                     ('q_nom', 0.25, '名义分位25'), ('q_nom', 0.50, '名义分位50')]:
    r = bt(gate, th)
    drive.append(dict(gate=nm, n=r['n'], expo=r['expo'], total=r['total'], cagr=r['cagr']))
    print(f"  {nm:<12} {r['n']:>2}笔  在场 {r['expo']:>5.1f}%  总收益 {r['total']:>7.1f}%  在场年化 {r['cagr']:>5.2f}%")
OUT['drive'] = drive

with open(os.path.join(B, 'd95_result.json'), 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print('\nsaved d95_result.json')
