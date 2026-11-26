# -*- coding: utf-8 -*-
"""
d64 — 决定性对照：思路 vs 数字
A. 原骨架（绝对点位 3000/3500）
B. 同骨架，但把 3000/3500 换成"过去5年相对位置分位"（自适应）
C. 加回补（趋势出场后站回年线可买回）
D. 起点稳健性 + 出场归因
"""
import os, json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
idx = pd.read_parquet(os.path.join(BASE, 'idx_sse_long.parquet'))
idx['d'] = pd.to_datetime(idx['date'])
for c in ['close', 'amount']:
    idx[c] = idx[c].astype(float)
idx = idx.sort_values('d').reset_index(drop=True)
etf = pd.read_parquet(os.path.join('G:/AStockData/supplement/etf_history', '510300_沪深300ETF.parquet'))
etf['d'] = pd.to_datetime(etf['日期'])
etf = etf[['d', '收盘']].rename(columns={'收盘': 'etf'}).sort_values('d').reset_index(drop=True)
df = idx[['d', 'close', 'amount']].merge(etf, on='d', how='left')
df['etf'] = df['etf'].ffill()
df['ma20'] = df['close'].rolling(20).mean()
df['ma250'] = df['close'].rolling(250).mean()
df['heat'] = df['amount'].rolling(250).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
# 5年相对位置分位（1250个交易日）
df['pct5y'] = df['close'].rolling(1250).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
df = df.dropna(subset=['heat', 'ma250']).reset_index(drop=True)

D = df['d'].values
C = df['close'].values
E = df['etf'].values
MA20 = df['ma20'].values
MA250 = df['ma250'].values
HEAT = df['heat'].values
PCT = df['pct5y'].values
R = np.r_[0, np.diff(E) / E[:-1]]
N = len(C)


def sim(i0=0, mode='abs', buy_th=3000, sell_th=3500, buy_pct=0.25, sell_pct=0.75,
        warm=20, dd=0.10, dd_days=20, heat_q=0.95, half=True, reenter=False, cost=0.0005):
    pos, half_sold, peak, peak_i, below = 0.0, False, 0.0, 0, 0
    last_exit, need_re = 0.0, False
    ps = np.zeros(N)
    eps, cur = [], None
    for i in range(N):
        c = C[i]
        below = below + 1 if c < buy_th else 0
        in_zone = (c < buy_th) if mode == 'abs' else (not np.isnan(PCT[i]) and PCT[i] < buy_pct)
        out_zone = (c > sell_th) if mode == 'abs' else (not np.isnan(PCT[i]) and PCT[i] > sell_pct)

        if i >= i0 and pos > 0:
            if c > peak:
                peak, peak_i = c, i
            hit, why = False, ''
            if c <= peak * (1 - dd) and (i - peak_i) >= dd_days:
                hit, why = True, f'回撤{dd*100:.0f}%+距顶{dd_days}日'
            if HEAT[i] >= heat_q and out_zone:
                hit, why = True, f'热度{HEAT[i]:.2f}'
            if hit:
                cur.update(exit=str(D[i])[:10], exit_lv=round(c, 0),
                           ret=round((E[i] / cur['ei'] - 1) * 100, 1), reason=why,
                           hold=int(i - cur['i']))
                eps.append(cur)
                last_exit, need_re = c, ('回撤' in why)
                cur, pos, half_sold = None, 0.0, False
            elif half and not half_sold and out_zone:
                pos, half_sold = 0.5, True

        if i >= i0 and pos == 0 and i > 60:
            ok = False
            if need_re and reenter and c > MA250[i] and c > last_exit:
                ok = True
            elif in_zone and (below >= warm if mode == 'abs' else True):
                # 磨底确认：上穿MA20
                ok = c > MA20[i] and C[i - 1] <= MA20[i - 1]
            if ok:
                need_re = False
                pos, half_sold, peak, peak_i = 1.0, False, c, i
                cur = {'entry': str(D[i])[:10], 'entry_lv': round(c, 0), 'ei': E[i], 'i': i,
                       'pct': round(float(PCT[i]), 2) if not np.isnan(PCT[i]) else None}
        ps[i] = pos
    s = np.zeros(N)
    s[1:] = ps[:-1] * R[1:]
    s -= np.abs(np.diff(np.r_[0, ps])) * cost
    eq = np.cumprod(1 + s[i0:])
    ret = np.diff(eq) / eq[:-1]
    yrs = len(eq) / 243
    cagr = eq[-1] ** (1 / yrs) - 1
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    return {'eq': eq, 'eps': eps, 'total_%': round((eq[-1] - 1) * 100, 1),
            'cagr_%': round(cagr * 100, 2), 'mdd_%': round(mdd * 100, 1),
            'calmar': round(float(cagr / abs(mdd)) if mdd < 0 else 0, 2),
            'flat_%': round(float((ps[i0:] == 0).mean() * 100), 1), 'n': len(eps),
            'yrs': round(yrs, 1), 'pos': ps[i0:]}


def bh(i0=0):
    eq = np.cumprod(1 + R[i0:])
    ret = np.diff(eq) / eq[:-1]
    yrs = len(eq) / 243
    cagr = eq[-1] ** (1 / yrs) - 1
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    return {'total_%': round((eq[-1] - 1) * 100, 1), 'cagr_%': round(cagr * 100, 2),
            'mdd_%': round(mdd * 100, 1), 'calmar': round(float(cagr / abs(mdd)) if mdd < 0 else 0, 2), 'eq': eq}


def i_at(s):
    return int(np.argmax(D >= np.datetime64(s)))

OUT = {}
i15 = i_at('2015-01-05')

variants = {
    'A_绝对点位3000/3500': dict(mode='abs'),
    'A2_绝对点位+回补': dict(mode='abs', reenter=True),
    'B_自适应分位25/75': dict(mode='pct', buy_pct=0.25, sell_pct=0.75),
    'B2_自适应分位30/70': dict(mode='pct', buy_pct=0.30, sell_pct=0.70),
    'B3_自适应+回补': dict(mode='pct', buy_pct=0.25, sell_pct=0.75, reenter=True),
}
OUT['main'] = {}
eqs = {}
for k, kw in variants.items():
    s = sim(i0=i15, **kw)
    eqs[k] = s['eq']
    s.pop('eq'); s.pop('pos')
    OUT['main'][k] = s
OUT['bh'] = {k: v for k, v in bh(i15).items() if k != 'eq'}

# 起点稳健性：每个起点跑 A 和 B
rows = []
for s0 in ['2015-01-05', '2015-06-12', '2016-01-04', '2017-01-03', '2018-01-02',
           '2019-01-02', '2020-01-02', '2021-01-04', '2022-01-04', '2023-01-03', '2024-01-02']:
    i0 = i_at(s0)
    a = sim(i0=i0, mode='abs')
    b = sim(i0=i0, mode='pct')
    m = bh(i0)
    rows.append({'start': s0, 'lvl': round(float(C[i0]), 0), 'bh_%': m['total_%'],
                 'bh_mdd_%': m['mdd_%'], 'A_%': a['total_%'], 'A_mdd_%': a['mdd_%'],
                 'B_%': b['total_%'], 'B_mdd_%': b['mdd_%'],
                 'A_beat': 'Y' if a['total_%'] > m['total_%'] else 'N',
                 'B_beat': 'Y' if b['total_%'] > m['total_%'] else 'N'})
OUT['starts'] = rows

# 交易明细（A / B）
OUT['eps_A'] = sim(i0=i15, mode='abs')['eps']
OUT['eps_B'] = sim(i0=i15, mode='pct')['eps']

# 分档（相对分位）
t = []
for h, lab in [(250, '1年'), (500, '2年')]:
    f = pd.Series(C).shift(-h) / pd.Series(C) - 1
    for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.001)]:
        m = (PCT >= lo) & (PCT < hi) & f.notna().values
        if m.sum() < 40:
            continue
        t.append({'h': lab, 'pct5y': f'{lo:.1f}-{hi:.1f}', 'n': int(m.sum()),
                  'mean_%': round(float(f.values[m].mean()) * 100, 2),
                  'win_%': round(float((f.values[m] > 0).mean()) * 100, 1)})
OUT['pct_table'] = t

with open(os.path.join(BASE, 'd64_result.json'), 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1, default=float)

pd.DataFrame({'d': D[i15:], 'close': C[i15:], 'etf': E[i15:],
              **{k: v for k, v in eqs.items()},
              'bh': bh(i15)['eq']}).to_parquet(os.path.join(BASE, 'd64_curves.parquet'), index=False)

print('== 主结果 (2015-01起) ==')
print('  买入持有   ', OUT['bh'])
for k, v in OUT['main'].items():
    print(f'  {k:<22} {v}')
print('\n== 起点稳健性 ==')
for x in rows:
    print('  ', x)
print('\n== A 交易明细 ==')
for e in OUT['eps_A']:
    print('  ', e)
print('\n== B 交易明细 ==')
for e in OUT['eps_B']:
    print('  ', e)
print('\n== 相对分位表 ==')
for x in OUT['pct_table']:
    print('  ', x)
