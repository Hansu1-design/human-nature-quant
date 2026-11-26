# -*- coding: utf-8 -*-
"""
d60 — 用户"骨架"验证：指数分档（3000以下买 / 3500以上卖 + 趋势/热度清仓）
只做骨架级验证：存不存在、对不对、有没有价值。
"""
import os, json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = {}

# ---------------- 数据 ----------------
idx = pd.read_parquet(os.path.join(BASE, 'idx_sse.parquet'))
idx['d'] = pd.to_datetime(idx['date'])
for c in ['open', 'high', 'low', 'close', 'volume', 'amount']:
    idx[c] = idx[c].astype(float)
idx = idx.sort_values('d').reset_index(drop=True)

etf = pd.read_parquet(os.path.join('G:/AStockData/supplement/etf_history', '510300_沪深300ETF.parquet'))
etf['d'] = pd.to_datetime(etf['日期'])
etf = etf[['d', '收盘']].rename(columns={'收盘': 'etf'}).sort_values('d').reset_index(drop=True)

df = idx[['d', 'close', 'amount']].merge(etf, on='d', how='left')
df['etf'] = df['etf'].ffill()
df = df[df['d'] >= '2015-01-01'].reset_index(drop=True)

df['ma20'] = df['close'].rolling(20).mean()
df['ma60'] = df['close'].rolling(60).mean()
df['ma250'] = df['close'].rolling(250).mean()
df['ma750'] = df['close'].rolling(750).mean()
# 热度：上证成交额在滚动250日中的分位
df['heat'] = df['amount'].rolling(250).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)

OUT['meta'] = {
    'start': str(df['d'].iloc[0].date()), 'end': str(df['d'].iloc[-1].date()),
    'n': int(len(df)), 'etf_end': str(etf['d'].max().date()),
    'idx_first': float(df['close'].iloc[0]), 'idx_last': float(df['close'].iloc[-1]),
}

# ---------------- 测试A：指数点位 → 前瞻收益（分档） ----------------
def fwd_table():
    rows = []
    for h, lab in [(250, '1年'), (500, '2年'), (750, '3年')]:
        f = df['close'].shift(-h) / df['close'] - 1
        for lo, hi in [(0, 2500), (2500, 2800), (2800, 3000), (3000, 3200),
                       (3200, 3500), (3500, 3800), (3800, 10000)]:
            m = (df['close'] >= lo) & (df['close'] < hi) & f.notna()
            if m.sum() < 20:
                rows.append({'h': lab, 'band': f'{lo}-{hi}', 'n': int(m.sum()),
                             'mean': None, 'win': None})
                continue
            rows.append({'h': lab, 'band': f'{lo}-{hi}', 'n': int(m.sum()),
                         'mean': round(float(f[m].mean()) * 100, 2),
                         'win': round(float((f[m] > 0).mean()) * 100, 1)})
    return rows

OUT['A_band_fwd'] = fwd_table()

# 归因：3000点这个"绝对点位"是不是真变量？对比"相对自身3年均值偏离"
df['dev3y'] = df['close'] / df['ma750'] - 1
t = []
for h, lab in [(250, '1年'), (500, '2年')]:
    f = df['close'].shift(-h) / df['close'] - 1
    for lo, hi in [(-1, -0.25), (-0.25, -0.10), (-0.10, 0.0), (0.0, 0.15), (0.15, 1)]:
        m = (df['dev3y'] >= lo) & (df['dev3y'] < hi) & f.notna()
        if m.sum() < 40:
            continue
        t.append({'h': lab, 'dev': f'{lo:+.2f}~{hi:+.2f}', 'n': int(m.sum()),
                  'mean': round(float(f[m].mean()) * 100, 2),
                  'win': round(float((f[m] > 0).mean()) * 100, 1)})
OUT['A2_dev_fwd'] = t

# ---------------- 策略引擎 ----------------
def run(buy_th=3000, sell_th=3500, entry='E1', dd=0.10, dd_days=20,
        heat_q=0.95, half=True, warm=20, cost=0.0005):
    pos, half_sold = 0.0, False
    peak, peak_i = 0.0, 0
    below = 0
    eq = []
    eps, cur = [], None
    for i in range(len(df)):
        c = df['close'].iloc[i]
        e = df['etf'].iloc[i]
        below = below + 1 if c < buy_th else 0

        # ---------- 卖出 ----------
        if pos > 0:
            if c > peak:
                peak, peak_i = c, i
            dd_hit = (c <= peak * (1 - dd)) and (i - peak_i >= dd_days)
            h = df['heat'].iloc[i]
            heat_hit = (not np.isnan(h)) and h >= heat_q and c > sell_th
            if dd_hit or heat_hit:
                ret = e / cur['entry_idx'] - 1
                cur.update(exit=str(df['d'].iloc[i].date()), exit_lv=c,
                           exit_idx=e, ret=round(ret * 100, 2),
                           reason=('回撤%.0f%%且距顶%d日' % (dd * 100, dd_days)) if dd_hit
                           else ('热度分位%.2f' % h),
                           hold_d=int(i - cur['i']))
                eps.append(cur)
                cur, pos, half_sold = None, 0.0, False
            elif half and not half_sold and c > sell_th:
                pos = pos / 2
                half_sold = True

        # ---------- 买入 ----------
        if pos == 0 and i > 60:
            prev_c = df['close'].iloc[i - 1]
            ok = False
            if entry == 'E0':
                ok = c < buy_th
            elif entry == 'E1':
                ok = (below >= warm and c < buy_th and c > df['ma20'].iloc[i]
                      and prev_c <= df['ma20'].iloc[i - 1])
            elif entry == 'E2':
                ok = (below >= warm and c < buy_th and c > df['ma60'].iloc[i]
                      and prev_c <= df['ma60'].iloc[i - 1])
            if ok:
                pos, half_sold, peak, peak_i = 1.0, False, c, i
                cur = {'entry': str(df['d'].iloc[i].date()), 'entry_lv': c,
                       'entry_idx': e, 'i': i}

        eq.append(pos if not np.isnan(e) else 0.0)

    df['pos'] = eq
    r = df['etf'].pct_change().fillna(0)
    df['strat'] = df['pos'].shift(1).fillna(0) * r
    # 交易成本（每次调仓按比例收）
    turn = df['pos'].diff().abs().fillna(0)
    df['strat'] = df['strat'] - turn * cost
    df['eqs'] = (1 + df['strat']).cumprod()
    df['eqb'] = (1 + r).cumprod()
    return df, eps

def stats(eqs, days_per_year=243):
    n = len(eqs)
    yrs = n / days_per_year
    total = eqs.iloc[-1] - 1
    cagr = eqs.iloc[-1] ** (1 / yrs) - 1
    dd = (eqs / eqs.cummax() - 1).min()
    ret = eqs.pct_change().fillna(0)
    sharpe = ret.mean() / ret.std() * np.sqrt(days_per_year) if ret.std() > 0 else 0
    return {'total_%': round(total * 100, 1), 'cagr_%': round(cagr * 100, 2),
            'mdd_%': round(dd * 100, 1), 'sharpe': round(float(sharpe), 2), 'yrs': round(yrs, 1)}

RES = {}
variants = {
    'E0_跌破即买': dict(entry='E0'),
    'E1_磨底20日+上穿MA20': dict(entry='E1', warm=20),
    'E2_磨底60日+上穿MA60': dict(entry='E2', warm=60),
}
for name, kw in variants.items():
    d2, eps = run(**kw)
    RES[name] = {'stats': stats(d2['eqs']), 'n_trades': len(eps), 'episodes': eps,
                 'exposure_%': round(float((d2['pos'] > 0).mean()) * 100, 1)}
    df['eq_' + name] = d2['eqs']

d_bh = df['eqb']
RES['买入持有510300'] = {'stats': stats(d_bh), 'n_trades': 1,
                        'exposure_%': 100.0}
OUT['B_strategy'] = RES

# 定投（每月首个交易日等额买入，不计息）
df['ym'] = df['d'].dt.to_period('M')
first = df.groupby('ym').head(1).index
shares = 0.0
cash_in = 0.0
vals = []
for i in range(len(df)):
    if i in set(first):
        shares += 1000 / df['etf'].iloc[i]
        cash_in += 1000
    vals.append(shares * df['etf'].iloc[i] / cash_in)
OUT['C_dca'] = {'total_%': round((vals[-1] - 1) * 100, 1),
                'cagr_%': round((vals[-1] ** (243 / len(df)) - 1) * 100, 2),
                'note': '每月1000元等额，2015-01起'}

# ---------------- 阈值敏感性 ----------------
grid = []
for bt in [2600, 2800, 3000, 3200, 3400]:
    row = {'buy': bt}
    for st in [3200, 3500, 3800, 4100, 4400]:
        try:
            d3, ep = run(buy_th=bt, sell_th=st, entry='E1', warm=20)
            row[str(st)] = round((d3['eqs'].iloc[-1] - 1) * 100, 1)
        except Exception:
            row[str(st)] = None
    grid.append(row)
OUT['D_grid'] = grid

# ---------------- 分段 ----------------
segs = [('2015-01', '2016-12'), ('2017-01', '2018-12'), ('2019-01', '2020-12'),
        ('2021-01', '2022-12'), ('2023-01', '2024-12'), ('2025-01', '2026-12')]
sg = []
for a, b in segs:
    m = (df['d'] >= a) & (df['d'] < b)
    if m.sum() < 50:
        continue
    sub = df[m]
    r = sub['etf'].pct_change().fillna(0)
    st = sub['pos'].shift(1).fillna(0) * r
    sg.append({'seg': f'{a}~{b}',
               'idx': f"{sub['close'].iloc[0]:.0f}→{sub['close'].iloc[-1]:.0f}",
               'bh_%': round((np.prod(1 + r) - 1) * 100, 1),
               'strat_%': round((np.prod(1 + st) - 1) * 100, 1),
               'expo_%': round(float((sub['pos'] > 0).mean()) * 100, 1)})
OUT['E_segments'] = sg

# ---------------- 热度检验 ----------------
f60 = df['close'].shift(-60) / df['close'] - 1
q = pd.qcut(df['heat'].dropna(), 5, labels=False)
heat_tab = []
for k in range(5):
    idxs = q[q == k].index
    heat_tab.append({'heat_q': k + 1,
                     'n': int(len(idxs)),
                     'fwd60_%': round(float(f60.loc[idxs].dropna().mean()) * 100, 2),
                     'lvl_mean': round(float(df.loc[idxs, 'close'].mean()), 0)})
OUT['F_heat'] = heat_tab

# 箱体统计：时间分布
OUT['G_box'] = {
    '<2800_%': round(float((df['close'] < 2800).mean()) * 100, 1),
    '2800-3500_%': round(float(((df['close'] >= 2800) & (df['close'] <= 3500)).mean()) * 100, 1),
    '>3500_%': round(float((df['close'] > 3500).mean()) * 100, 1),
    'days': int(len(df)),
}

df[['d', 'close', 'etf', 'pos', 'eqs', 'eqb']].to_parquet(os.path.join(BASE, 'd60_path.parquet'), index=False)
with open(os.path.join(BASE, 'd60_result.json'), 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)

# ---------------- 打印 ----------------
print('== meta ==', OUT['meta'])
print('\n== A 分档前瞻收益 ==')
for r in OUT['A_band_fwd']:
    print(r)
print('\n== A2 相对3年均值偏离 ==')
for r in OUT['A2_dev_fwd']:
    print(r)
print('\n== B 策略 ==')
for k, v in RES.items():
    print(k, v['stats'], 'trades', v['n_trades'], 'expo', v['exposure_%'])
print('\n== 交易明细 (E1) ==')
for e in RES['E1_磨底20日+上穿MA20']['episodes']:
    print(e)
print('\n== C 定投 ==', OUT['C_dca'])
print('\n== D 阈值网格 (期末总收益%) ==')
for r in OUT['D_grid']:
    print(r)
print('\n== E 分段 ==')
for r in OUT['E_segments']:
    print(r)
print('\n== F 热度 ==')
for r in OUT['F_heat']:
    print(r)
print('\n== G 箱体 ==', OUT['G_box'])
