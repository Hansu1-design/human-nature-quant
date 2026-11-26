# -*- coding: utf-8 -*-
"""
d62 — 骨架验证 v2：起点稳健性 + 规则归因
关键问题：
  Q1 3000以下买 这根梁是真的，还是"任何时候买都行"？
  Q2 3500以上卖 / 热度清仓 / 回撤止损，哪一根才真正贡献收益？
  Q3 换个起点（2015-06 高位、2021-01 高位）还成立吗？
"""
import os, json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
idx = pd.read_parquet(os.path.join(BASE, 'idx_sse.parquet'))
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
df['ma60'] = df['close'].rolling(60).mean()
df['heat'] = df['amount'].rolling(250).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
df = df.dropna(subset=['heat', 'ma60']).reset_index(drop=True)
D = df['d'].values
C = df['close'].values
E = df['etf'].values
MA20 = df['ma20'].values
MA60 = df['ma60'].values
HEAT = df['heat'].values
R = np.r_[0, np.diff(E) / E[:-1]]


def sim(i0=0, buy_th=3000, sell_th=3500, entry='E1', warm=20, dd=0.10, dd_days=20,
        heat_q=0.95, half=True, ex_dd=True, ex_heat=True, cost=0.0005):
    n = len(C)
    pos, half_sold, peak, peak_i, below = 0.0, False, 0.0, 0, 0
    ps = np.zeros(n)
    eps, cur = [], None
    for i in range(n):
        c = C[i]
        below = below + 1 if c < buy_th else 0
        if i >= i0 and pos > 0:
            if c > peak:
                peak, peak_i = c, i
            hit, why = False, ''
            if ex_dd and c <= peak * (1 - dd) and (i - peak_i) >= dd_days:
                hit, why = True, f'回撤{dd*100:.0f}%+距顶{dd_days}日'
            if ex_heat and HEAT[i] >= heat_q and c > sell_th:
                hit, why = True, f'热度{HEAT[i]:.2f}'
            if hit:
                cur.update(exit=str(D[i])[:10], exit_lv=round(c, 0),
                           ret=round((E[i] / cur['ei'] - 1) * 100, 1), reason=why,
                           hold=int(i - cur['i']))
                eps.append(cur)
                cur, pos, half_sold = None, 0.0, False
            elif half and not half_sold and c > sell_th:
                pos, half_sold = 0.5, True
        if i >= i0 and pos == 0 and i > 60:
            ok = False
            if entry == 'E0':
                ok = c < buy_th
            elif entry == 'always':
                ok = True
            elif entry == 'E1':
                ok = below >= warm and c < buy_th and c > MA20[i] and C[i - 1] <= MA20[i - 1]
            elif entry == 'E2':
                ok = below >= warm and c < buy_th and c > MA60[i] and C[i - 1] <= MA60[i - 1]
            if ok:
                pos, half_sold, peak, peak_i = 1.0, False, c, i
                cur = {'entry': str(D[i])[:10], 'entry_lv': round(c, 0), 'ei': E[i], 'i': i}
        ps[i] = pos
    s = np.zeros(n)
    s[1:] = ps[:-1] * R[1:]
    turn = np.abs(np.diff(np.r_[0, ps]))
    s = s - turn * cost
    eq = np.cumprod(1 + s[i0:])
    ret = np.diff(eq) / eq[:-1]
    yrs = len(eq) / 243
    cagr = eq[-1] ** (1 / yrs) - 1
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    sh = ret.mean() / ret.std() * np.sqrt(243) if ret.std() > 1e-12 else 0
    # 空仓
    p = ps[i0:]
    flat = float((p == 0).mean() * 100)
    run_len, mx = 0, 0
    for x in p:
        run_len = run_len + 1 if x == 0 else 0
        mx = max(mx, run_len)
    return {'eq': eq, 'eps': eps, 'total_%': round((eq[-1] - 1) * 100, 1),
            'cagr_%': round(cagr * 100, 2), 'mdd_%': round(mdd * 100, 1),
            'sharpe': round(float(sh), 2), 'calmar': round(float(cagr / abs(mdd)) if mdd < 0 else 0, 2),
            'flat_%': round(flat, 1), 'max_flat_d': int(mx), 'n': len(eps),
            'yrs': round(yrs, 1)}


def bh(i0=0):
    eq = np.cumprod(1 + R[i0:])
    ret = np.diff(eq) / eq[:-1]
    yrs = len(eq) / 243
    cagr = eq[-1] ** (1 / yrs) - 1
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    return {'total_%': round((eq[-1] - 1) * 100, 1), 'cagr_%': round(cagr * 100, 2),
            'mdd_%': round(mdd * 100, 1), 'calmar': round(float(cagr / abs(mdd)) if mdd < 0 else 0, 2),
            'eq': eq}


def i_at(s):
    return int(np.argmax(D >= np.datetime64(s)))


OUT = {}

# ---------- Q1: 买点是不是装饰 ----------
q1 = {}
for lab, kw in [('3000以下+磨底20日', dict(entry='E1', buy_th=3000)),
                ('3000以下+跌破即买', dict(entry='E0', buy_th=3000)),
                ('任何时候都买(无买点)', dict(entry='always', buy_th=99999))]:
    q1[lab] = sim(i0=i_at('2015-01-05'), **kw)
    q1[lab].pop('eq')
OUT['Q1_buypoint'] = q1

# ---------- Q2: 出场规则归因（买点固定为3000以下E0，避免交互） ----------
q2 = {}
combos = [('只有3500减半,无清仓', dict(ex_dd=False, ex_heat=False)),
          ('+热度清仓', dict(ex_dd=False, ex_heat=True)),
          ('+回撤止损', dict(ex_dd=True, ex_heat=False)),
          ('两者都要(原骨架)', dict(ex_dd=True, ex_heat=True))]
for lab, kw in combos:
    s = sim(i0=i_at('2015-01-05'), entry='E0', buy_th=3000, **kw)
    s.pop('eq')
    q2[lab] = s
OUT['Q2_exit'] = q2

# ---------- Q3: 起点稳健性 ----------
starts = ['2015-01-05', '2015-06-12', '2016-01-04', '2018-01-02', '2019-01-02',
          '2020-01-02', '2021-01-04', '2022-01-04', '2023-01-03', '2024-01-02']
rows = []
for s0 in starts:
    i0 = i_at(s0)
    a = sim(i0=i0, entry='E1', buy_th=3000)
    b = sim(i0=i0, entry='E0', buy_th=3000)
    m = bh(i0)
    rows.append({'start': s0, 'lvl': round(C[i0], 0),
                 'bh_%': m['total_%'], 'bh_mdd': m['mdd_%'],
                 's_E1_%': a['total_%'], 's_E1_mdd': a['mdd_%'], 's_E1_cagr': a['cagr_%'],
                 's_E0_%': b['total_%'], 'E0_flat_%': b['flat_%'],
                 'win': 'Y' if a['total_%'] > m['total_%'] else 'N'})
OUT['Q3_starts'] = rows

# ---------- Q4: 2015-06 高位起点特写 ----------
i0 = i_at('2015-06-12')
OUT['Q4_2015top'] = {'idx_at_start': round(C[i0], 0), 'bh': {k: v for k, v in bh(i0).items() if k != 'eq'},
                     'strat_E1': {k: v for k, v in sim(i0=i0, entry='E1').items() if k != 'eq'},
                     'strat_E1_eps': sim(i0=i0, entry='E1')['eps']}

# ---------- Q5: 现状态 ----------
OUT['Q5_now'] = {'idx': round(float(C[-1]), 0), 'date': str(D[-1])[:10],
                 'state': '3500以上=逐步卖出区,且已空仓',
                 'last_signal': sim(i0=i_at('2015-01-05'), entry='E1')['eps'][-1]}

# ---------- Q6: 3000这个数在别的市场还成立吗（对照：中证500/1000） ----------
OUT['Q6_note'] = '见文末说明：换标的需重定阈值'

with open(os.path.join(BASE, 'd62_result.json'), 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1, default=float)

print('== Q1 买点是否装饰 ==')
for k, v in OUT['Q1_buypoint'].items():
    print(f'  {k:<22} 总收益{v["total_%"]:>7}% CAGR{v["cagr_%"]:>6}% MDD{v["mdd_%"]:>6}% calmar{v["calmar"]:>5} 空仓{v["flat_%"]:>5}% 笔数{v["n"]}')
print('\n== Q2 出场归因 ==')
for k, v in OUT['Q2_exit'].items():
    print(f'  {k:<22} 总收益{v["total_%"]:>7}% CAGR{v["cagr_%"]:>6}% MDD{v["mdd_%"]:>6}% calmar{v["calmar"]:>5} 笔数{v["n"]}')
print('\n== Q3 起点稳健性 ==')
for x in rows:
    print('  ', x)
print('\n== Q4 2015-06 高位 ==')
print('  start lvl', OUT['Q4_2015top']['idx_at_start'], 'bh', OUT['Q4_2015top']['bh'])
print('  strat', OUT['Q4_2015top']['strat_E1'])
for e in OUT['Q4_2015top']['strat_E1_eps']:
    print('    ', e)
print('\n== Q5 现在 ==', OUT['Q5_now'])
