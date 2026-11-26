# -*- coding: utf-8 -*-
"""
d93: ratio 为什么失效 + 用"实际值绝对水平"当闸门（用户"算回来"的本意）
1) ratio 的时间序列性质：趋势、滚动分位分布（是否恒为0）
2) 闸门改成"CPI 实际值 <= 常数"（等价于"名义 <= k × CPI定基"，每年自动上调）
3) 敏感性：1,800 / 1,900 / 2,000 / 2,100
4) M2 口径的另一个用途：持有现金相对 M2 的年度缩水
"""
import pandas as pd, numpy as np, json, os

B = os.path.dirname(os.path.abspath(__file__))
OUT = {}
m = pd.read_parquet(os.path.join(B, 'd91_daily.parquet')).sort_values('d').reset_index(drop=True)
CPI_T = 144.0

# 补 CPI 定基并算实际值
_cpi = pd.read_parquet('G:/AStockData/supplement/macro/cpi.parquet')
_cpi['ym'] = _cpi['月份'].str.replace('年', '-').str.replace('月份', '')
_cpi['d'] = pd.to_datetime(_cpi['ym'], format='%Y-%m')
_cpi = _cpi.sort_values('d').reset_index(drop=True)
_cpi['mom'] = (pd.to_numeric(_cpi['全国-当月'], errors='coerce') / 100.0) ** (1 / 12.0)
_ix = [100.0]
for i in range(1, len(_cpi)): _ix.append(_ix[-1] * _cpi['mom'].iloc[i])
_cpi['cpi_idx'] = _ix
_cpi['avail'] = _cpi['d'] + pd.offsets.MonthBegin(1) + pd.Timedelta(days=14)
m = pd.merge_asof(m.sort_values('d'), _cpi[['avail', 'cpi_idx']].sort_values('avail'), left_on='d', right_on='avail', direction='backward')
m['real_cpi'] = m['close'] * 100.0 / m['cpi_idx']
m = m.dropna(subset=['real_cpi']).reset_index(drop=True)
print(f'补入 real_cpi，{len(m)} 行，当前实际值 {m["real_cpi"].iloc[-1]:.0f}')

# ---------- 1. ratio 的序列性质 ----------
print('=== ratio 的分布与趋势 ===')
print(f"ratio 首 {m['ratio'].iloc[0]:.1f} → 末 {m['ratio'].iloc[-1]:.1f}")
ln = np.log(m['ratio'].values)
x = np.arange(len(ln))
sl, ic = np.polyfit(x, ln, 1)
print(f"log 线性趋势：年化斜率 {(np.exp(sl*250)-1)*100:+.1f}%/yr  （即 ratio 每年{'下降' if sl<0 else '上升'} {abs((np.exp(sl*250)-1)*100):.1f}%）")
q = m['q_ratio'].dropna()
print(f"滚动5年分位的分布：=0 占比 {(q<=0.001).mean()*100:.1f}% | <=0.1 占比 {(q<=0.1).mean()*100:.1f}% | 中位 {q.median():.3f}")
qn = m['q_nom'].dropna()
print(f"对照 名义分位：=0 占比 {(qn<=0.001).mean()*100:.1f}% | <=0.1 占比 {(qn<=0.1).mean()*100:.1f}% | 中位 {qn.median():.3f}")
OUT['ratio_trend'] = dict(ann=round(float((np.exp(sl*250)-1)*100), 1),
                          zero_pct=round(float((q<=0.001).mean()*100), 1),
                          p10=round(float((q<=0.1).mean()*100), 1),
                          nom_zero_pct=round(float((qn<=0.001).mean()*100), 1),
                          nom_p10=round(float((qn<=0.1).mean()*100), 1))

# 每轮 ratio 顶底
print('\n=== ratio 的顶与底（每轮）===')
z = pd.read_json(os.path.join(B, 'd90_result.json'), typ='series')
print('（见 d90_result.json pivots）')

# ---------- 2. 实际值绝对闸门 ----------
def backtest(gate_col, gate_th, start, hold_floor=20, dd_stop=0.10, exit_heat=0.95, up_cap=3900):
    i0 = int(np.argmin(np.abs((m['d'] - pd.Timestamp(start)).values)))
    pos, half, peak, peak_i, entry, eq = 0.0, False, 0.0, 0, 0.0, 1.0
    trades = []
    for i in range(i0, len(m)):
        c = m['etf'].iloc[i]
        if pos > 0:
            eq *= c / m['etf'].iloc[i-1]
            if c >= peak: peak, peak_i = c, i
        cheep = (not np.isnan(m[gate_col].iloc[i])) and m[gate_col].iloc[i] <= gate_th
        cross = m['close'].iloc[i] > m['ma20'].iloc[i] and m['close'].iloc[i-1] <= m['ma20'].iloc[i-1]
        if pos == 0:
            if cheep and cross and m['close'].iloc[i] < up_cap:
                pos, half, peak, peak_i, entry = 1.0, False, c, i, c
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

print('\n=== 闸门：CPI 实际值 <= 常数（自动随通胀上调）===')
res = {}
for th in [1800, 1900, 2000, 2100]:
    outs = []
    for st in starts:
        r = backtest('real_cpi', th, st)
        bh = (m['etf'].iloc[-1] / m['etf'].iloc[int(np.argmin(np.abs((m['d'] - pd.Timestamp(st)).values)))] - 1) * 100
        outs.append(dict(start=st, total=r['total'], n=r['n'], bh=round(float(bh), 1), win=bool(r['total'] > bh)))
    w = sum(1 for x in outs if x['win']); md = float(np.median([x['total'] for x in outs]))
    res[f'real<={th}'] = dict(rows=outs, win=f'{w}/11', median=round(md, 1))
    print(f"  实际<={th}  跑赢 {w}/11 | 中位 {md:>6.1f}% | 笔数 {[x['n'] for x in outs]}")

# 对照
for nm, col, th in [('名义<=3000', 'close', 3000), ('名义分位<=0.25', 'q_nom', 0.25), ('ratio分位<=0.25', 'q_ratio', 0.25)]:
    outs = []
    for st in starts:
        r = backtest(col, th, st)
        bh = (m['etf'].iloc[-1] / m['etf'].iloc[int(np.argmin(np.abs((m['d'] - pd.Timestamp(st)).values)))] - 1) * 100
        outs.append(dict(start=st, total=r['total'], n=r['n'], bh=round(float(bh), 1), win=bool(r['total'] > bh)))
    w = sum(1 for x in outs if x['win']); md = float(np.median([x['total'] for x in outs]))
    res[nm] = dict(rows=outs, win=f'{w}/11', median=round(md, 1))
    print(f"  {nm:<16} 跑赢 {w}/11 | 中位 {md:>6.1f}% | 笔数 {[x['n'] for x in outs]}")
OUT['gates'] = res

print('\n--- 逐笔明细 ---')
for th in [1900, 2000, 2100]:
    r = backtest('real_cpi', th, '2015-01-05')
    print(f"实际<={th}: {r['n']} 笔 {r['trades']}")
r = backtest('q_nom', 0.25, '2015-01-05')
print(f"名义分位<=0.25: {r['n']} 笔 {r['trades']}")
r = backtest('close', 3000, '2015-01-05')
print(f"名义<=3000: {r['n']} 笔 {r['trades']}")
r = backtest('close', 2707, '2015-01-05')
print(f"名义<=2707: {r['n']} 笔 {r['trades']}")
# 补一个 名义<=2707 的 11 起点
outs=[]
for st in starts:
    rr = backtest('close', 2707, st)
    i0 = int(np.argmin(np.abs((m['d'] - pd.Timestamp(st)).values)))
    bh = (m['etf'].iloc[-1] / m['etf'].iloc[i0] - 1) * 100
    outs.append(dict(start=st, total=rr['total'], n=rr['n'], bh=round(float(bh),1), win=bool(rr['total']>bh)))
w=sum(1 for x in outs if x['win']); md=float(np.median([x['total'] for x in outs]))
OUT['gates']['名义<=2707']=dict(rows=outs, win=f'{w}/11', median=round(md,1))
print(f"  名义<=2707（实际底折算值）跑赢 {w}/11 | 中位 {md:.1f}% | 笔数 {[x['n'] for x in outs]}")

# ---------- 3. 买入线随时间的值 ----------
print('\n=== "实际底 1,880" 折算成各年的名义买入线 ===')
line = []
for y in [2026, 2028, 2030, 2033, 2036, 2040]:
    for cpi_assume in [1.015]:  # 假设 CPI 年均 1.5%
        k = cpi_assume ** (y - 2026)
        nom = 1880 * (CPI_T / 100.0) * k
        line.append(dict(year=y, nom=round(nom)))
        print(f"  {y}: 约 {nom:>5.0f} 点（假设 CPI 年均 +1.5%）")
OUT['line'] = line

# ---------- 4. M2 口径的另一个用途 ----------
print('\n=== 持有现金相对 M2 的缩水（这才是 M2 该用的地方）===')
m2 = pd.read_parquet('G:/AStockData/supplement/macro/money_supply.parquet')
m2['ym'] = m2['月份'].str.replace('年', '-').str.replace('月份', '')
m2['d'] = pd.to_datetime(m2['ym'], format='%Y-%m')
m2 = m2.sort_values('d')
v0 = float(m2['货币和准货币(M2)-数量(亿元)'].iloc[0]); v1 = float(m2['货币和准货币(M2)-数量(亿元)'].iloc[-1])
yrs = (m2['d'].iloc[-1] - m2['d'].iloc[0]).days / 365.25
cagr = (v1 / v0) ** (1 / yrs) - 1
print(f"M2 {v0/1e4:.1f} → {v1/1e4:.1f} 万亿，{yrs:.1f} 年，CAGR {cagr*100:.2f}%")
print(f"→ 持有现金（0 利息）相对 M2 池，每年缩水 {cagr*100:.2f}%")
print(f"→ 10 年购买力剩 {(1/(1+cagr)**10)*100:.0f}%（以 M2 为池）")
print(f"→ 但以 CPI 为池：10 年剩 {(1/1.015**10)*100:.0f}%")
OUT['m2_cash'] = dict(cagr=round(cagr*100, 2), ten_y_m2=round((1/(1+cagr)**10)*100),
                      ten_y_cpi=round((1/1.015**10)*100))

with open(os.path.join(B, 'd93_result.json'), 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print('\nsaved d93_result.json')
