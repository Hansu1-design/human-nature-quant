# -*- coding: utf-8 -*-
"""
d91: 三口径平减 + ratio 分位的预测力检验
1) 名义 / CPI平减 / M2平减 三条长期序列，各自的中枢
2) ratio 滚动分位 vs 名义点位滚动分位 -> 前瞻收益单调性
3) 单点诊断：2016-03-02 (名义分位0.717判贵) 在 ratio 口径下是多少
4) 用 ratio 分位做买点，替换"上证<3000"，同出场规则跑 11 个起点
"""
import pandas as pd, numpy as np, json, os
from scipy import stats

B = os.path.dirname(os.path.abspath(__file__))
OUT = {}

# ---------- 数据 ----------
df = pd.read_parquet(os.path.join(B, 'd90_daily.parquet'))
df = df.sort_values('d').reset_index(drop=True)

# CPI 定基（2008-01 = 100），按"上年同月=100"连乘
cpi = pd.read_parquet('G:/AStockData/supplement/macro/cpi.parquet')
cpi['ym'] = cpi['月份'].str.replace('年', '-').str.replace('月份', '')
cpi['d'] = pd.to_datetime(cpi['ym'], format='%Y-%m')
cpi = cpi.sort_values('d').reset_index(drop=True)
# '全国-当月' 是同比指数(上年同月=100)，不是环比 -> 必须开 12 次方当环比
cpi['mom'] = (pd.to_numeric(cpi['全国-当月'], errors='coerce') / 100.0) ** (1 / 12.0)
idx = [100.0]
for i in range(1, len(cpi)):
    idx.append(idx[-1] * cpi['mom'].iloc[i])
cpi['cpi_idx'] = idx
cpi['avail'] = cpi['d'] + pd.offsets.MonthBegin(1) + pd.Timedelta(days=14)

df = pd.merge_asof(df.sort_values('d'), cpi[['avail', 'cpi_idx']].dropna().sort_values('avail'),
                   left_on='d', right_on='avail', direction='backward')
df = df.dropna(subset=['cpi_idx']).reset_index(drop=True)

CPI_T = df['cpi_idx'].iloc[-1]
M2_T = df['m2_wy'].iloc[-1]
df['real_cpi'] = df['close'] * 100.0 / df['cpi_idx']         # 2008-01 不变价
df['eq_m2'] = df['close'] * M2_T / df['m2_wy']               # 末日货币水平
df['ratio'] = df['close'] / df['m2_wy']
print(f'区间 {df["d"].iloc[0].date()} → {df["d"].iloc[-1].date()}  n={len(df)}')
print(f'CPI 累计 {CPI_T:.1f}（2008-01=100，即涨 {CPI_T-100:.1f}%）；M2 {M2_T:.1f} 万亿')

# ---------- 1. 三口径中枢 ----------
print('\n=== 三种口径的长期中枢（2008-2026，18.7年）===')
res = {}
for nm, col in [('名义点位', 'close'), ('CPI平减(2008年不变价)', 'real_cpi'), ('M2平减(2026货币水平)', 'eq_m2')]:
    s = df[col]
    res[nm] = dict(mean=round(float(s.mean())), med=round(float(s.median())),
                   p20=round(float(s.quantile(.2))), p80=round(float(s.quantile(.8))),
                   first=round(float(s.iloc[0])), last=round(float(s.iloc[-1])))
    print(f'{nm:<22} 均值 {s.mean():>7.0f} | 中位 {s.median():>7.0f} | P20 {s.quantile(.2):>7.0f} | P80 {s.quantile(.8):>7.0f} | 首 {s.iloc[0]:>7.0f} → 末 {s.iloc[-1]:>7.0f}')
OUT['center3'] = res

# 三种口径下，每轮熊底
print('\n=== 每轮熊底在三种口径下的数值 ===')
bottoms = [('2008-11-04', 1707), ('2013-06-25', 1950), ('2016-01-28', 2656), ('2019-01-03', 2464), ('2024-02-05', 2702)]
bt = []
for d0, p in bottoms:
    i = int(np.argmin(np.abs((df['d'] - pd.Timestamp(d0)).values)))
    r = df.iloc[i]
    bt.append(dict(date=d0, nom=int(r['close']), cpi=round(float(r['real_cpi'])),
                   m2=round(float(r['eq_m2'])), ratio=round(float(r['ratio']), 2)))
    print(f"{d0}  名义 {r['close']:>5.0f} | CPI不变价 {r['real_cpi']:>5.0f} | M2等价 {r['eq_m2']:>6.0f} | ratio {r['ratio']:.2f}")
OUT['bottoms3'] = bt
b0, b1 = bt[0], bt[-1]
print(f'\n从 2008 底到 2024 底：名义 {(b1["nom"]/b0["nom"]-1)*100:+.1f}% | CPI口径 {(b1["cpi"]/b0["cpi"]-1)*100:+.1f}% | M2口径 {(b1["m2"]/b0["m2"]-1)*100:+.1f}%')
OUT['bottom_change'] = dict(nom=round((b1['nom']/b0['nom']-1)*100, 1),
                            cpi=round((b1['cpi']/b0['cpi']-1)*100, 1),
                            m2=round((b1['m2']/b0['m2']-1)*100, 1))

# ---------- 2. 分位与前瞻收益 ----------
df['ret1y'] = df['close'].shift(-250) / df['close'] - 1
df['ret3y'] = df['close'].shift(-750) / df['close'] - 1

def rolling_q(s, win=1250):
    return s.rolling(win, min_periods=250).apply(lambda x: (x[:-1] < x[-1]).mean() if len(x) > 1 else np.nan, raw=True)

df['q_nom'] = rolling_q(df['close'])
df['q_ratio'] = rolling_q(df['ratio'])
df['q_cpi'] = rolling_q(df['real_cpi'])

print('\n=== 滚动 5 年分位 → 未来 1 年收益（五分位）===')
tbl = {}
for nm, qc in [('名义点位分位', 'q_nom'), ('ratio分位', 'q_ratio'), ('CPI实际分位', 'q_cpi')]:
    sub = df.dropna(subset=[qc, 'ret1y']).copy()
    sub['bin'] = pd.cut(sub[qc], [0, .2, .4, .6, .8, 1.001], labels=['Q1最低', 'Q2', 'Q3', 'Q4', 'Q5最高'])
    g = sub.groupby('bin', observed=True)['ret1y'].agg(['mean', 'median', 'count'])
    tbl[nm] = {str(k): dict(mean=round(float(v['mean'])*100, 2), med=round(float(v['median'])*100, 2), n=int(v['count'])) for k, v in g.iterrows()}
    print(f'\n-- {nm}（n={len(sub)}）')
    print(f"{'档':<8}{'均值':>9}{'中位':>9}{'样本':>7}")
    for k, v in g.iterrows():
        print(f"{k:<8}{v['mean']*100:>8.1f}%{v['median']*100:>8.1f}%{int(v['count']):>7}")
    q1 = g['mean'].iloc[0]; q5 = g['mean'].iloc[-1]
    monot = stats.spearmanr(range(len(g)), g['mean'].values)
    print(f"   Q1−Q5 差 {(q1-q5)*100:+.1f}pt | 单调性 Spearman {monot.statistic:+.2f} (p={monot.pvalue:.3f})")
    OUT.setdefault('quint', {})[nm] = dict(tbl=tbl[nm], gap=round(float(q1-q5)*100, 1), rho=round(float(monot.statistic), 3))

# 3年窗口
print('\n=== 滚动 5 年分位 → 未来 3 年收益（Q1 vs Q5）===')
for nm, qc in [('名义点位分位', 'q_nom'), ('ratio分位', 'q_ratio'), ('CPI实际分位', 'q_cpi')]:
    sub = df.dropna(subset=[qc, 'ret3y']).copy()
    sub['bin'] = pd.cut(sub[qc], [0, .2, .4, .6, .8, 1.001], labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    g = sub.groupby('bin', observed=True)['ret3y'].mean()
    print(f"{nm:<14} Q1 {g.iloc[0]*100:+6.1f}%  Q5 {g.iloc[-1]*100:+6.1f}%  差 {(g.iloc[0]-g.iloc[-1])*100:+.1f}pt  n={len(sub)}")
    OUT.setdefault('q3y', {})[nm] = dict(q1=round(float(g.iloc[0])*100, 1), q5=round(float(g.iloc[-1])*100, 1))

# ---------- 3. 单点诊断 ----------
print('\n=== 单点诊断：2016-03-02 上证 2,850（名义 5 年分位 0.717 判"贵"）===')
i = int(np.argmin(np.abs((df['d'] - pd.Timestamp('2016-03-02')).values)))
r = df.iloc[i]
print(f"当日：名义 {r['close']:.0f} | ratio {r['ratio']:.2f} | M2 {r['m2_wy']:.1f}万亿")
print(f"  名义5年分位 {r['q_nom']:.3f} | ratio5年分位 {r['q_ratio']:.3f} | CPI实际5年分位 {r['q_cpi']:.3f}")
lo = df[df['d'] < pd.Timestamp('2016-03-02')]
print(f"  ratio 历史分位(全样本至当日) {(lo['ratio'] < r['ratio']).mean():.3f}")
print(f"  后续：1年 {df['close'].iloc[i+250]/r['close']-1:+.1%} | 到 2018-01-24 高点 {3559/r['close']-1:+.1%}")
OUT['case2016'] = dict(close=round(float(r['close'])), ratio=round(float(r['ratio']), 2),
                       q_nom=round(float(r['q_nom']), 3), q_ratio=round(float(r['q_ratio']), 3),
                       q_cpi=round(float(r['q_cpi']), 3), fwd1y=round(float(df['close'].iloc[i+250]/r['close']-1)*100, 1))

# ---------- 4. ratio 分位做买点，同出场规则 ----------
etf = pd.read_parquet('G:/AStockData/supplement/etf_history/510300_沪深300ETF.parquet')
etf = etf.rename(columns={'日期': 'd', '收盘': 'close'})
etf['d'] = pd.to_datetime(etf['d']); etf['close'] = pd.to_numeric(etf['close'], errors='coerce')
etf = etf.dropna(subset=['close']).sort_values('d')[['d', 'close']].rename(columns={'close': 'etf'}).reset_index(drop=True)

m = pd.merge_asof(df[['d', 'close', 'ratio', 'q_nom', 'q_ratio', 'q_cpi', 'm2_wy']],
                  etf, on='d', direction='backward').dropna(subset=['etf']).reset_index(drop=True)
m['ma20'] = m['close'].rolling(20).mean()
m['ma250'] = m['close'].rolling(250).mean()
# 热度：成交额分位（用上证 amount 的 250 日分位代理）
sse_full = pd.read_parquet(os.path.join(B, 'idx_sse_all.parquet'))
sse_full['amount'] = pd.to_numeric(sse_full['amount'], errors='coerce')
sse_full['d'] = pd.to_datetime(sse_full['date'])
sse_full = sse_full.dropna(subset=['amount']).sort_values('d')[['d', 'amount']]
m = pd.merge_asof(m, sse_full, on='d', direction='backward')
m['amt_q'] = m['amount'].rolling(250).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
m = m.dropna(subset=['ma20', 'ma250', 'amt_q', 'q_ratio', 'q_cpi']).reset_index(drop=True)
print(f'\n回测数据 {m["d"].iloc[0].date()} → {m["d"].iloc[-1].date()}  n={len(m)}')

def backtest(m, gate_col, gate_th, start, hold_floor=20, dd_stop=0.10, cost=0.0005, exit_heat=0.95):
    """gate: 该列 <= gate_th 视为'便宜'，且上穿 MA20 磨底确认入场
       出场：回撤10%且距顶20日，或 热度>=0.95 且上证>3500"""
    i0 = int(np.argmin(np.abs((m['d'] - pd.Timestamp(start)).values)))
    pos, half, peak, peak_i, entry, below = 0.0, False, 0.0, 0, 0.0, 0
    nav, eq = 1.0, 1.0
    trades = []
    for i in range(i0, len(m)):
        c = m['etf'].iloc[i]
        if pos > 0:
            eq *= c / m['etf'].iloc[i - 1]
            peak = max(peak, c)
            if c >= peak: peak_i = i
        cheep = (not np.isnan(m[gate_col].iloc[i])) and m[gate_col].iloc[i] <= gate_th
        cross = m['close'].iloc[i] > m['ma20'].iloc[i] and m['close'].iloc[i - 1] <= m['ma20'].iloc[i - 1]
        if pos == 0:
            if cheep and cross and m['close'].iloc[i] < 3600:
                pos, half, peak, peak_i = 1.0, False, c, i
                entry = c
                below = 1.0
        else:
            why = None
            if (peak - c) / peak >= dd_stop and i - peak_i >= hold_floor:
                why = '回撤'
            elif m['amt_q'].iloc[i] >= exit_heat and m['close'].iloc[i] > 3500:
                why = '热度'
            if why and not half:
                pos, half = 0.5, True
            elif why and half:
                trades.append((str(m['d'].iloc[i].date()), round(float(c / entry - 1) * 100, 1), why))
                pos, half = 0.0, False
    tr = 1.0 if pos == 0 else eq
    tot = (eq - 1) * 100
    # MDD
    navs, e2, p2, pk = [], 1.0, 0.0, 0.0
    return dict(total=round(float(tot), 1), trades=trades, n=len(trades))

# 起点集合
starts = ['2015-01-05', '2015-06-12', '2016-01-04', '2017-01-03', '2018-01-02', '2019-01-02', '2020-01-02', '2021-01-04', '2022-01-04', '2023-01-03', '2024-01-02']
print('\n=== ratio 分位 25 做买点（替换"上证<3000"），同出场规则 ===')
rows = []
for st in starts:
    r = backtest(m, 'q_ratio', 0.25, st)
    # 同期买入持有
    i0 = int(np.argmin(np.abs((m['d'] - pd.Timestamp(st)).values)))
    bh = (m['etf'].iloc[-1] / m['etf'].iloc[i0] - 1) * 100
    rows.append(dict(start=st, ratio=r['total'], n=r['n'], bh=round(float(bh), 1)))
    print(f"{st}  ratio买点 {r['total']:>7.1f}%  ({r['n']}笔)   买入持有 {bh:>7.1f}%   {'赢' if r['total']>bh else '输'}")
OUT['ratio_gate'] = rows

print('\n--- 2015-01 起的逐笔 ---')
print(backtest(m, 'q_ratio', 0.25, '2015-01-05')['trades'])

# 对照：名义<3000
print('\n=== 对照：名义点位做买点 ===')
rows2 = []
m['nom_gate'] = m['close']
for st in starts:
    r = backtest(m, 'nom_gate', 3000, st)
    i0 = int(np.argmin(np.abs((m['d'] - pd.Timestamp(st)).values)))
    bh = (m['etf'].iloc[-1] / m['etf'].iloc[i0] - 1) * 100
    rows2.append(dict(start=st, nom=r['total'], n=r['n'], bh=round(float(bh), 1)))
    print(f"{st}  名义买点 {r['total']:>7.1f}%  ({r['n']}笔)   买入持有 {bh:>7.1f}%   {'赢' if r['total']>bh else '输'}")
OUT['nom_gate'] = rows2

w1 = sum(1 for x in rows if x['ratio'] > x['bh'])
w2 = sum(1 for x in rows2 if x['nom'] > x['bh'])
print(f'\n跑赢买入持有：ratio买点 {w1}/{len(rows)} | 名义买点 {w2}/{len(rows2)}')
OUT['win'] = dict(ratio=f'{w1}/{len(rows)}', nom=f'{w2}/{len(rows2)}')

with open(os.path.join(B, 'd91_result.json'), 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
m.to_parquet(os.path.join(B, 'd91_daily.parquet'), index=False)
print('\nsaved d91_result.json')
