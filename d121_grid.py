# -*- coding: utf-8 -*-
"""
d121: 仓位间距数值化 —— 全样本扫描
样本：2008-02 ~ 2026-11（CPI 可得区间，18.7 年，4555 交易日）

扫什么：档数 n × 确认信号 confirm × 最小间隔 gap
评价：一次下跌（事件）内，用该规则建仓，到"离底回到 30% 以上"（或 6 年截断）时的账户总值
      —— 出场不参与，隔离建仓变量

账户口径：初始 1.0，等额 n 笔，现金按 2.5%/年计息，标的用上证（无分红，所有策略一致）
"""
import os, json, itertools
import numpy as np
import pandas as pd

B = os.path.dirname(os.path.abspath(__file__))
ANCHOR = 1880.0
CASH_Y = 0.025
DAILY_CASH = (1 + CASH_Y) ** (1 / 243.0) - 1


def load():
    a = pd.read_parquet(os.path.join(B, 'idx_sse_all.parquet'))
    a['d'] = pd.to_datetime(a['date'])
    a['close'] = pd.to_numeric(a['close'], errors='coerce')
    a = a[['d', 'close', 'high', 'low', 'amount']].dropna().sort_values('d').reset_index(drop=True)

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

    d = pd.merge_asof(a, c[['avail', 'cpi_idx']].sort_values('avail'),
                      left_on='d', right_on='avail', direction='backward')
    d = d.dropna(subset=['cpi_idx']).reset_index(drop=True)
    d['real'] = d['close'] * 100.0 / d['cpi_idx']
    d['rA'] = d['real'] / ANCHOR
    d['peak'] = d['close'].cummax()
    d['dd_peak'] = d['close'] / d['peak'] - 1.0
    d['ma20'] = d['close'].rolling(20).mean()
    d['ma20_up'] = d['ma20'] > d['ma20'].shift(5)
    d['low20'] = d['close'].rolling(20).min()
    d['reb8'] = d['close'] / d['low20'] - 1.0 >= 0.08
    # 月线收阳：每月最后交易日 close > 上月最后交易日 close
    m = d.set_index('d')['close'].resample('ME').last()
    mup = (m > m.shift(1)).astype(int)
    mup.index = mup.index + pd.offsets.MonthBegin(1)  # 次月 1 日才可知
    d['mup'] = 0
    idx = pd.merge_asof(d[['d']], pd.DataFrame({'avail_m': mup.index, 'v': mup.values}).sort_values('avail_m'),
                        left_on='d', right_on='avail_m', direction='backward')['v'].fillna(0).astype(int).values
    d['mup'] = idx
    return d


def find_events(d, k_enter=0.10, k_exit=0.20, max_wait=1600):
    """事件 = 从 >1+k_enter 首次下穿，到 rA 回到 >1+k_exit（离开观察区）为止"""
    N = len(d)
    r = d['rA'].values
    inb = r <= 1 + k_enter
    out = r > 1 + k_exit
    ev, i = [], 0
    while i < N:
        if inb[i] and (i == 0 or not inb[i - 1]):
            j = i
            k = i + 1
            while k < N and not (out[k]):
                k += 1
            end = k if k < N else N - 1
            if end - i >= 15:
                ev.append((i, end))
            i = end + 1
        else:
            i += 1
    return ev


def regime(d, i):
    return 'burst' if d['dd_peak'].iloc[i] <= -0.50 else 'box'


def confirm_ok(d, i, kind):
    if kind == 'none':
        return True
    if kind == 'ma20':
        v = d['ma20'].iloc[i]
        return (not np.isnan(v)) and d['close'].iloc[i] > v and bool(d['ma20_up'].iloc[i])
    if kind == 'reb8':
        v = d['low20'].iloc[i]
        return (not np.isnan(v)) and d['close'].iloc[i] / v - 1 >= 0.08
    if kind == 'mup':
        return bool(d['mup'].iloc[i])
    raise ValueError(kind)


def run_event(d, i0, i1, n, confirm, gap_m, ladder=None):
    """在事件 [i0,i1] 内按规则建仓，返回明细"""
    if ladder is None:
        if n == 1:
            ladder = [0.10]
        else:
            ladder = [0.10 * (1 - k / (n - 1)) for k in range(n)]
    lot = 1.0 / n
    cash, shares, invested = 1.0, 0.0, 0.0
    buys = []
    leg = 0                      # 已买档数
    last_buy = None
    reb_confirm_cnt = 0
    r = d['rA'].values
    lo_real = np.inf
    end_i = i1

    for i in range(i0, i1 + 1):
        lo_real = min(lo_real, d['real'].iloc[i])
        if leg >= n:
            pass
        else:
            k_i = ladder[leg]
            # 距离线触发
            by_depth = r[i] <= 1 + k_i
            # 确认解锁：已在带内 + 确认成立 + 已有 1 档在手（第 1 档不需要确认）
            by_conf = False
            if leg >= 1 and r[i] <= 1 + 0.10 and confirm_ok(d, i, confirm):
                if gap_m == 0:
                    by_conf = True
                elif last_buy is None:
                    by_conf = True
                else:
                    months = (d['d'].iloc[i].year - d['d'].iloc[last_buy].year) * 12 + \
                             (d['d'].iloc[i].month - d['d'].iloc[last_buy].month)
                    by_conf = months >= gap_m
            # 间隔约束对深度触发同样适用
            gap_ok = True
            if last_buy is not None:
                months = (d['d'].iloc[i].year - d['d'].iloc[last_buy].year) * 12 + \
                         (d['d'].iloc[i].month - d['d'].iloc[last_buy].month)
                gap_ok = months >= max(1, gap_m)
            if by_depth and gap_ok:
                take = True
            elif by_conf and gap_ok:
                take = True
            else:
                take = False
            if take:
                amt = min(lot, cash)
                if amt > 1e-9:
                    px = d['close'].iloc[i]
                    shares += amt / px
                    cash -= amt
                    invested += amt
                    buys.append(dict(i=i, d=str(d['d'].iloc[i].date()), px=float(px),
                                     real=float(d['real'].iloc[i]), leg=leg + 1))
                    last_buy = i
                    leg += 1
        cash *= (1 + DAILY_CASH)

    nav_end = cash + shares * d['close'].iloc[i1]
    nav_low = cash + shares * d['close'].iloc[i1]  # 占位
    cost = (invested / shares) if shares > 0 else np.nan
    return dict(i0=i0, i1=i1, d0=str(d['d'].iloc[i0].date()), d1=str(d['d'].iloc[i1].date()),
                regime=regime(d, i0), n_buys=len(buys), maxpos=invested,
                cost_real=float(cost) if shares > 0 else None,
                low_real=float(lo_real),
                cost_prem=(float(cost) / lo_real - 1) if shares > 0 else None,
                nav=float(nav_end), ret=float(nav_end - 1),
                buys=buys)


def main():
    d = load()
    ev = find_events(d)
    print(f'样本 {d.d.iloc[0].date()} ~ {d.d.iloc[-1].date()}  {len(d)} 日')
    print(f'独立事件 {len(ev)} 个：')
    for i0, i1 in ev:
        print(f"  {d.d.iloc[i0].date()} → {d.d.iloc[i1].date()}  "
              f"{regime(d,i0):5s}  入场 {d.close.iloc[i0]:6.0f}  离底 {d.rA.iloc[i0]*100-100:+5.1f}%  "
              f"距顶 {d.dd_peak.iloc[i0]*100:+5.1f}%  区间最低实际 {d.real.iloc[i0:i1+1].min():6.0f}")
    print()

    # 事件内上涨幅度（满仓在最低点）
    print('=== 每个事件的理论上下限 ===')
    for i0, i1 in ev:
        lo = d.real.iloc[i0:i1 + 1].min()
        lo_i = i0 + int(np.argmin(d.real.values[i0:i1 + 1]))
        up = d.close.iloc[i1] / d.close.iloc[lo_i] - 1
        allin = d.close.iloc[i1] / d.close.iloc[i0] - 1
        print(f"  {d.d.iloc[i0].date()}  等最低全仓 {up*100:+6.1f}%  "
              f"（底 {d.d.iloc[lo_i].date()}）| 进带即满仓 {allin*100:+6.1f}%")
    print()

    combos = []
    for n in [2, 3, 4, 5]:
        for cf in ['none', 'ma20', 'reb8', 'mup']:
            for g in [1, 2, 3]:
                combos.append((n, cf, g))
    print(f'扫描 {len(combos)} 组')

    rows = []
    for n, cf, g in combos:
        per = [run_event(d, i0, i1, n, cf, g) for i0, i1 in ev]
        rets = [p['ret'] for p in per]
        prems = [p['cost_prem'] for p in per if p['cost_prem'] is not None]
        pos = [p['maxpos'] for p in per]
        rows.append(dict(n=n, cf=cf, g=g,
                         ret_mean=float(np.mean(rets)), ret_min=float(min(rets)),
                         prem_mean=float(np.mean(prems)) if prems else None,
                         pos_mean=float(np.mean(pos)),
                         rets=[round(x * 100, 1) for x in rets]))
    df = pd.DataFrame(rows).sort_values('ret_mean', ascending=False)
    print('\n=== 按平均收益排序（前 15）===')
    print(df.head(15).to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\n=== 后 8 ===')
    print(df.tail(8).to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    # 对照
    print('\n=== 对照 ===')
    ctrl = {}
    for nm, fn in [
        ('A 单笔全仓（进带即满仓）', lambda d, a, b: run_event(d, a, b, 1, 'none', 1, ladder=[0.10])),
        ('B 只在 rA≤1.0 全仓（等真底）', lambda d, a, b: run_event(d, a, b, 1, 'none', 1, ladder=[0.0])),
        ('C v1.0 匀速加仓 每跌3%', None),
    ]:
        if fn is None:
            continue
        per = [fn(d, i0, i1) for i0, i1 in ev]
        ctrl[nm] = dict(ret_mean=float(np.mean([p['ret'] for p in per])),
                        pos_mean=float(np.mean([p['maxpos'] for p in per])),
                        prems=[p['cost_prem'] for p in per],
                        rets=[round(p['ret'] * 100, 1) for p in per])
        print(f"  {nm}: 平均 {ctrl[nm]['ret_mean']*100:6.2f}%  平均仓位 {ctrl[nm]['pos_mean']*100:5.1f}%  各事件 {ctrl[nm]['rets']}")

    # v1.0 匀速加仓：从进带起每跌 3%（按 rA 算 3% 相对值）加一笔，共 5 笔
    def uni(d, i0, i1):
        lot = 1.0 / 5
        cash, sh, inv = 1.0, 0.0, 0.0
        last = None
        for i in range(i0, i1 + 1):
            go = False
            if last is None:
                go = True
            elif d['rA'].iloc[i] <= d['rA'].iloc[last] * 0.97:
                go = True
            if go and inv < 1 - 1e-9:
                amt = min(lot, cash)
                sh += amt / d['close'].iloc[i]; cash -= amt; inv += amt; last = i
            cash *= (1 + DAILY_CASH)
        nav = cash + sh * d['close'].iloc[i1]
        return dict(ret=nav - 1, maxpos=inv, cost_prem=(inv / sh) / d['real'].iloc[i0:i1 + 1].min() - 1 if sh > 0 else None)
    per = [uni(d, i0, i1) for i0, i1 in ev]
    print(f"  C v1.0 匀速加仓 每跌3%: 平均 {np.mean([p['ret'] for p in per])*100:6.2f}%  "
          f"平均仓位 {np.mean([p['maxpos'] for p in per])*100:5.1f}%  "
          f"各事件 {[round(p['ret']*100,1) for p in per]}")
    ctrl['C v1.0 匀速加仓 每跌3%'] = dict(ret_mean=float(np.mean([p['ret'] for p in per])),
                                          pos_mean=float(np.mean([p['maxpos'] for p in per])),
                                          rets=[round(p['ret'] * 100, 1) for p in per])

    # 事件明细（用最优参数）
    best = df.iloc[0]
    print(f"\n=== 最优组合明细 n={best.n} cf={best.cf} g={best.g} ===")
    detail = []
    for i0, i1 in ev:
        p = run_event(d, i0, i1, int(best.n), best.cf, int(best.g))
        detail.append({k: v for k, v in p.items() if k != 'buys'})
        print(f"  {p['d0']}  {p['regime']:5s}  {p['n_buys']}/{best.n} 笔  "
              f"仓位 {p['maxpos']*100:5.1f}%  成本溢价 {p['cost_prem']*100:+6.1f}%  收益 {p['ret']*100:+7.1f}%")

    out = dict(events=[dict(d0=str(d.d.iloc[i0].date()), d1=str(d.d.iloc[i1].date()),
                            rA0=round(float(d.rA.iloc[i0]), 4), regime=regime(d, i0))
                       for i0, i1 in ev],
               grid=rows, ctrl=ctrl, best_detail=detail)
    json.dump(out, open(os.path.join(B, 'd121_result.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\nsaved d121_result.json')


if __name__ == '__main__':
    main()
