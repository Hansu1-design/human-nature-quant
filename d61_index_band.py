# -*- coding: utf-8 -*-
"""
d61 — 用户骨架验证（修正版）
规则骨架：
  1) 指数<3000 → 熊，开始关注（不买）
  2) 熊市里磨底/筹码震荡结束后 → 重仓买入长期持有
  3) 指数>3500 → 逐步卖出
  4) 下跌趋势满1个月 或 全民热度高涨 → 全部清仓
验证：存不存在 / 对不对 / 有没有价值（vs 买入持有、定投、均线择时）
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
df = df[df['d'] >= '2015-01-01'].reset_index(drop=True)
df['ma20'] = df['close'].rolling(20).mean()
df['ma60'] = df['close'].rolling(60).mean()
df['ma120'] = df['close'].rolling(120).mean()
df['ma250'] = df['close'].rolling(250).mean()
df['ma750'] = df['close'].rolling(750).mean()
df['heat'] = df['amount'].rolling(250).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
N = len(df)
r = df['etf'].pct_change().fillna(0).values
dv = df['d'].values

OUT = {'meta': {'start': str(df['d'].iloc[0].date()), 'end': str(df['d'].iloc[-1].date()),
                'n': N, 'etf_end': str(etf['d'].max().date()),
                'idx_first': round(float(df['close'].iloc[0]), 0),
                'idx_last': round(float(df['close'].iloc[-1]), 0)}}

# ================= 测试A：点位分档 → 前瞻收益 =================
A = []
for h, lab in [(250, '1年'), (500, '2年'), (750, '3年')]:
    f = df['close'].shift(-h) / df['close'] - 1
    for lo, hi in [(0, 2500), (2500, 2800), (2800, 3000), (3000, 3200),
                   (3200, 3500), (3500, 3800), (3800, 10000)]:
        m = (df['close'] >= lo) & (df['close'] < hi) & f.notna()
        if m.sum() < 30:
            continue
        A.append({'h': lab, 'band': f'{lo}~{hi}', 'n': int(m.sum()),
                  'mean_%': round(float(f[m].mean()) * 100, 2),
                  'win_%': round(float((f[m] > 0).mean()) * 100, 1),
                  'worst_%': round(float(f[m].min()) * 100, 1)})
OUT['A_band'] = A

A2 = []
df['dev3y'] = df['close'] / df['ma750'] - 1
for h, lab in [(250, '1年'), (500, '2年')]:
    f = df['close'].shift(-h) / df['close'] - 1
    for lo, hi in [(-1, -0.20), (-0.20, -0.10), (-0.10, 0.0), (0.0, 0.15), (0.15, 1)]:
        m = (df['dev3y'] >= lo) & (df['dev3y'] < hi) & f.notna()
        if m.sum() < 40:
            continue
        A2.append({'h': lab, 'dev_vs_3yma': f'{lo:+.2f}~{hi:+.2f}', 'n': int(m.sum()),
                   'mean_%': round(float(f[m].mean()) * 100, 2),
                   'win_%': round(float((f[m] > 0).mean()) * 100, 1)})
OUT['A2_dev'] = A2

# ================= 策略引擎 =================
def stats(eq):
    eq = np.asarray(eq, dtype=float)
    yrs = len(eq) / 243
    cagr = eq[-1] ** (1 / yrs) - 1
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    ret = np.diff(eq) / eq[:-1]
    sh = ret.mean() / ret.std() * np.sqrt(243) if ret.std() > 0 else 0.0
    return {'total_%': round((eq[-1] - 1) * 100, 1), 'cagr_%': round(cagr * 100, 2),
            'mdd_%': round(mdd * 100, 1), 'sharpe': round(float(sh), 2),
            'calmar': round((cagr / abs(mdd)) if mdd < 0 else 0, 2)}


def run(buy_th=3000, sell_th=3500, entry='E1', dd=0.10, dd_days=20, heat_q=0.95,
        half=True, warm=20, exit_mode='all', cost=0.0005):
    pos, half_sold, peak, peak_i, below = 0.0, False, 0.0, 0, 0
    ps = np.zeros(N)
    eps, cur = [], None
    for i in range(N):
        c = df['close'].iloc[i]
        e = df['etf'].iloc[i]
        below = below + 1 if c < buy_th else 0

        if pos > 0:
            if c > peak:
                peak, peak_i = c, i
            dd_hit = (c <= peak * (1 - dd)) and (i - peak_i >= dd_days)
            h = df['heat'].iloc[i]
            heat_hit = (not np.isnan(h)) and h >= heat_q and c > sell_th
            if exit_mode == 'all':
                hit = dd_hit or heat_hit
            elif exit_mode == 'dd':
                hit = dd_hit
            elif exit_mode == 'heat':
                hit = heat_hit
            elif exit_mode == 'none':
                hit = False
            else:
                hit = False
            if hit:
                cur.update(exit=str(dv[i])[:10], exit_lv=round(c, 0), exit_idx=e,
                           ret=round((e / cur['entry_idx'] - 1) * 100, 1),
                           reason=('回撤%.0f%%+距顶%d日' % (dd * 100, dd_days)) if dd_hit else ('热度%.2f' % h),
                           hold=int(i - cur['i']))
                eps.append(cur)
                cur, pos, half_sold = None, 0.0, False
            elif half and not half_sold and c > sell_th:
                pos, half_sold = 0.5, True

        if pos == 0 and i > 60:
            prev = df['close'].iloc[i - 1]
            ok = False
            if entry == 'E0':
                ok = c < buy_th
            elif entry == 'E1':
                ok = below >= warm and c < buy_th and c > df['ma20'].iloc[i] and prev <= df['ma20'].iloc[i - 1]
            elif entry == 'E2':
                ok = below >= warm and c < buy_th and c > df['ma60'].iloc[i] and prev <= df['ma60'].iloc[i - 1]
            if ok:
                pos, half_sold, peak, peak_i = 1.0, False, c, i
                cur = {'entry': str(dv[i])[:10], 'entry_lv': round(c, 0),
                       'entry_idx': e, 'i': i}
        ps[i] = pos

    sr = pos_returns(ps)
    eq = np.cumprod(1 + sr)
    return {'pos': ps, 'eq': eq, 'stats': stats(eq), 'eps': eps,
            'expo_%': round(float((ps > 0).mean()) * 100, 1), 'n': len(eps)}


def pos_returns(ps):
    s = np.zeros(N)
    s[1:] = ps[:-1] * r[1:]
    turn = np.abs(np.diff(np.concatenate([[0], ps])))
    return s - turn * 0.0005


MAIN = run(entry='E1', warm=20)
MAIN['pos'] = MAIN['pos']
OUT['B_main'] = {'stats': MAIN['stats'], 'expo_%': MAIN['expo_%'],
                 'n_trades': MAIN['n'], 'episodes': MAIN['eps']}

bh = np.cumprod(1 + r)
OUT['B_bh'] = stats(bh)

# 3000以下买入后永不卖
ps_hold = (pd.Series(df['close'] < 3000).astype(int)).values
first = np.argmax(ps_hold == 1)
ps_h2 = np.zeros(N); ps_h2[first:] = 1
OUT['B_buy_below3000_hold'] = stats(np.cumprod(1 + pos_returns(ps_h2)))

# 均线择时对照组
for w in [120, 250]:
    ps_m = (df['close'] > df[f'ma{w}']).astype(float).values
    ps_m = pd.Series(ps_m).shift(1).fillna(0).values
    OUT[f'B_ma{w}_timing'] = stats(np.cumprod(1 + pos_returns(ps_m)))

# 定投（正确 IRR）
sh, cash = 0.0, 0.0
first_of_month = set(df.groupby(df['d'].dt.to_period('M')).head(1).index)
mon_cf = {}
for i in range(N):
    per = df['d'].dt.to_period('M').iloc[i]
    mon_cf.setdefault(per, 0.0)
    if i in first_of_month:
        sh += 1000 / df['etf'].iloc[i]
        cash += 1000
        mon_cf[per] -= 1000
last_per = df['d'].dt.to_period('M').iloc[-1]
mon_cf[last_per] += sh * df['etf'].iloc[-1]
cfs = [mon_cf[k] for k in sorted(mon_cf)]

def irr_monthly(cf):
    lo, hi = -0.9, 0.5
    for _ in range(300):
        mid = (lo + hi) / 2
        npv = sum(c / (1 + mid) ** i for i, c in enumerate(cf) if c)
        if npv > 0:
            lo = mid
        else:
            hi = mid
    return mid

OUT['C_dca'] = {'total_value': round(sh * df['etf'].iloc[-1], 0), 'cash_in': round(cash, 0),
                'gain_%': round((sh * df['etf'].iloc[-1] / cash - 1) * 100, 1),
                'irr_%': round(((1 + irr_monthly(cfs)) ** 12 - 1) * 100, 2)}

# ================= 规则拆解：价值来自哪一条 =================
DEC = {}
for mode, lab in [('all', '完整骨架(回撤+热度)'), ('dd', '只留回撤止损'), ('heat', '只留热度清仓'),
                  ('none', '不止损,只在3500减半')]:
    d = run(entry='E1', warm=20, exit_mode=mode)
    DEC[lab] = {'stats': d['stats'], 'expo_%': d['expo_%'], 'n': d['n']}
for entry, lab in [('E0', '跌破3000当天买')]:
    d = run(entry=entry)
    DEC[lab] = {'stats': d['stats'], 'expo_%': d['expo_%'], 'n': d['n']}
d = run(entry='E1', warm=20, half=False)
DEC['3500不逐步卖(死拿)'] = {'stats': d['stats'], 'expo_%': d['expo_%'], 'n': d['n']}
OUT['D_decomp'] = DEC

# ================= 阈值敏感性 =================
G = []
for bt in [2600, 2800, 3000, 3200, 3400]:
    row = {'buy': bt}
    for st in [3200, 3500, 3800, 4100]:
        d = run(buy_th=bt, sell_th=st, entry='E1', warm=20)
        row[str(st)] = round((d['eq'][-1] - 1) * 100, 1)
    G.append(row)
OUT['E_grid'] = G

# ================= 分段（用主策略自己的pos） =================
ps = MAIN['pos']
sr = pos_returns(ps)
seg = []
for a, b in [('2015-01', '2016-12'), ('2017-01', '2018-12'), ('2019-01', '2020-12'),
             ('2021-01', '2022-12'), ('2023-01', '2024-12'), ('2025-01', '2026-12')]:
    m = (df['d'] >= a) & (df['d'] < b)
    if m.sum() < 50:
        continue
    i0 = np.where(m.values)[0][0]
    seg.append({'seg': f'{a}~{b}',
                'idx': f"{df['close'].values[m.values][0]:.0f}→{df['close'].values[m.values][-1]:.0f}",
                'bh_%': round((np.prod(1 + r[m.values]) - 1) * 100, 1),
                'strat_%': round((np.prod(1 + sr[m.values]) - 1) * 100, 1),
                'expo_%': round(float((ps[m.values] > 0).mean()) * 100, 1)})
OUT['F_seg'] = seg

# 空仓时长统计
out_days, run_len = [], 0
for p in ps:
    if p == 0:
        run_len += 1
    elif run_len:
        out_days.append(run_len); run_len = 0
if run_len:
    out_days.append(run_len)
OUT['G_flat'] = {'max_flat_days': int(max(out_days)), 'mean_flat_days': round(float(np.mean(out_days)), 0),
                 'n_flat_periods': len(out_days), 'flat_pct': round(float((ps == 0).mean()) * 100, 1)}

# 热度
f60 = df['close'].shift(-60) / df['close'] - 1
q = pd.qcut(df['heat'].dropna(), 5, labels=False)
OUT['H_heat'] = [{'quintile': k + 1, 'n': int((q == k).sum()),
                  'fwd60_%': round(float(f60.loc[q[q == k].index].dropna().mean()) * 100, 2),
                  'lvl': round(float(df.loc[q[q == k].index, 'close'].mean()), 0)} for k in range(5)]

OUT['I_box'] = {'<2800': round(float((df['close'] < 2800).mean()) * 100, 1),
                '2800-3500': round(float(((df['close'] >= 2800) & (df['close'] <= 3500)).mean()) * 100, 1),
                '>3500': round(float((df['close'] > 3500).mean()) * 100, 1)}

pd.DataFrame({'d': df['d'], 'close': df['close'], 'etf': df['etf'],
              'pos': ps, 'eqs': MAIN['eq'], 'eqb': bh}).to_parquet(
    os.path.join(BASE, 'd61_path.parquet'), index=False)
with open(os.path.join(BASE, 'd61_result.json'), 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1, default=float)

# ================= 打印 =================
print('meta', OUT['meta'])
print('\n[A] 点位分档 → 前瞻收益')
for x in A:
    print(f"  {x['h']:>3} {x['band']:>10} n={x['n']:>4} 均值{x['mean_%']:>7}% 胜率{x['win_%']:>5}% 最差{x['worst_%']:>7}%")
print('\n[A2] 相对3年均值偏离')
for x in A2:
    print(f"  {x['h']:>3} {x['dev_vs_3yma']:>14} n={x['n']:>4} 均值{x['mean_%']:>7}% 胜率{x['win_%']:>5}%")
print('\n[B] 主策略', OUT['B_main']['stats'], 'expo', OUT['B_main']['expo_%'], 'n', OUT['B_main']['n_trades'])
for e in OUT['B_main']['episodes']:
    print('   ', e)
print('    买入持有', OUT['B_bh'])
print('    3000下买永不卖', OUT['B_buy_below3000_hold'])
print('    MA120择时', OUT['B_ma120_timing'])
print('    MA250择时', OUT['B_ma250_timing'])
print('    定投', OUT['C_dca'])
print('\n[D] 规则拆解')
for k, v in DEC.items():
    print(f"  {k:<22} {v['stats']} expo={v['expo_%']} n={v['n']}")
print('\n[E] 阈值网格 期末总收益%')
for x in G:
    print('  ', x)
print('\n[F] 分段')
for x in seg:
    print('  ', x)
print('\n[G] 空仓', OUT['G_flat'])
print('\n[H] 热度', OUT['H_heat'])
print('\n[I] 箱体', OUT['I_box'])
