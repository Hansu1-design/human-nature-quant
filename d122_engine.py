# -*- coding: utf-8 -*-
"""
d122: 仓位间距数值化（v2）—— 全链条回测
样本：2008-02 ~ 2026-11（CPI 可得区间，18.7 年）

修正 d121 的两个致命问题：
  ① 事件被"rA 回到 20% 就结束"切碎 → 2009 年那波 +62% 完全没吃到
     改为：6 轮独立熊市（12 月冷却 + 中途 rA 须曾 >1.25 才算新一轮）
  ② 没有出场规则 → 只能说"涨了多少"，不能说"拿到多少"
     改为：v1.2 的体制分流出场（勃发段趋势 / 箱体 +50% 止盈）
"""
import os, json
import numpy as np
import pandas as pd

B = os.path.dirname(os.path.abspath(__file__))
ANCHOR = 1880.0
CASH_Y = 0.025
DC = (1 + CASH_Y) ** (1 / 243.0) - 1


def load():
    a = pd.read_parquet(os.path.join(B, 'idx_sse_all.parquet'))
    a['d'] = pd.to_datetime(a['date'])
    a['close'] = pd.to_numeric(a['close'], errors='coerce')
    a = a[['d', 'close']].dropna().sort_values('d').reset_index(drop=True)

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
    d['ma120'] = d['close'].rolling(120).mean()
    d['ma120_down'] = d['ma120'] <= d['ma120'].shift(20) * 1.0
    m = d.set_index('d')['close'].resample('ME').last()
    mup = (m > m.shift(1)).astype(int)
    mup.index = mup.index + pd.offsets.MonthBegin(1)
    d['mup'] = pd.merge_asof(
        d[['d']], pd.DataFrame({'avail_m': mup.index, 'v': mup.values}).sort_values('avail_m'),
        left_on='d', right_on='avail_m', direction='backward')['v'].fillna(0).astype(int).values
    return d


def find_events(d, k=0.10, cool_m=12, reset_k=0.25):
    """6 轮独立熊市：首次下穿 ≤1+k，12 月冷却，且中间 rA 须曾回到 >1+reset_k"""
    N = len(d)
    r = d['rA'].values
    inb = r <= 1 + k
    cand = [i for i in range(1, N) if inb[i] and not inb[i - 1]]
    ev, last = [], None
    for i in cand:
        if last is None:
            ev.append(i); last = i; continue
        months = (d['d'].iloc[i].year - d['d'].iloc[last].year) * 12 + (d['d'].iloc[i].month - d['d'].iloc[last].month)
        if months < cool_m:
            continue
        if r[last:i].max() <= 1 + reset_k:
            continue
        ev.append(i); last = i
    return ev


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


def backtest(d, i0, n=3, confirm='ma20', gap_m=1, ladder=None, max_hold=1800, calib=False):
    """从 i0 进场，按规则建仓，按 v1.2 分流出场。返回完整轨迹

    calib=True 时，在**首次建仓当天**打印 §7.2 校准卡三行（band_calib.card）。
    该开关只写 stdout，不参与任何 if / 仓位 / 出场判断；默认 False ⇒ 基线逐位不变。
    """
    N = len(d)
    if ladder is None:
        ladder = [0.10] if n == 1 else [0.10 * (1 - k / (n - 1)) for k in range(n)]
    lot = 1.0 / n
    regime = 'burst' if d['dd_peak'].iloc[i0] <= -0.50 else 'box'

    cash, sh, inv = 1.0, 0.0, 0.0
    leg, last_buy = 0, None
    buys = []
    peak_px = 0.0
    touched_ma120 = False      # 持仓期内是否曾站上 MA120（反转确认）
    peak_gain = -9.9           # 持仓期最高浮盈
    exited, exit_i, exit_why = False, N - 1, '未出场'
    max_dd_acct = 0.0
    nav_peak = 1.0
    months_in = 0

    for i in range(i0, min(N, i0 + max_hold)):
        px = d['close'].iloc[i]
        # 先记净值再操作（当日收盘价成交）
        # ---- 建仓 ----
        if not exited and leg < n:
            by_depth = d['rA'].iloc[i] <= 1 + ladder[leg]
            by_conf = False
            if leg >= 1 and inv > 0 and d['rA'].iloc[i] <= 1.10 and confirm_ok(d, i, confirm):
                by_conf = True
            gap_ok = True
            if last_buy is not None:
                mm = (d['d'].iloc[i].year - d['d'].iloc[last_buy].year) * 12 + \
                     (d['d'].iloc[i].month - d['d'].iloc[last_buy].month)
                gap_ok = mm >= max(1, gap_m)
            if (by_depth or by_conf) and gap_ok:
                amt = min(lot, cash)
                if amt > 1e-9:
                    sh += amt / px; cash -= amt; inv += amt
                    buys.append(dict(i=i, d=str(d['d'].iloc[i].date()), px=float(px),
                                     rA=float(d['rA'].iloc[i]), leg=len(buys) + 1))
                    last_buy = i; leg += 1
                    if calib and len(buys) == 1:
                        # §7.2 校准卡：只打印，不改变任何决策（返回值不接任何 if）
                        # 懒加载：calib=False 时连 import 都不发生，基线零扰动
                        import band_calib
                        for _ln in band_calib.card(float(d['rA'].iloc[i]),
                                                   when=str(d['d'].iloc[i].date())):
                            print(_ln)
                    if regime == 'box':
                        pass
        # ---- 出场判定 ----
        if not exited and inv > 0:
            peak_px = max(peak_px, px)
            months_in = (d['d'].iloc[i].year - d['d'].iloc[i0].year) * 12 + (d['d'].iloc[i].month - d['d'].iloc[i0].month)
            avg_cost = inv / sh
            gain = px / avg_cost - 1
            peak_gain = max(peak_gain, gain)
            ma120 = d['ma120'].iloc[i]
            if (not np.isnan(ma120)) and px > ma120:
                touched_ma120 = True
            # 趋势出场只在「已经站上过 MA120」之后才启用；否则熊市入场当天就会误触发
            trend_out = touched_ma120 and (not np.isnan(ma120)) and px < ma120 and bool(d['ma120_down'].iloc[i])
            drop_out = peak_gain > 0.15 and px / peak_px - 1 <= -0.20

            why = None
            if regime == 'burst':
                if trend_out:
                    why = '破MA120且MA120下行'
                elif drop_out:
                    why = '峰回20%'
            else:
                if gain >= 0.50:
                    why = '+50%止盈'
                elif months_in >= 20 and trend_out:
                    why = '20月后趋势结束'
                elif drop_out:
                    why = '浮盈后峰回20%'
            if why:
                cash += sh * px; sh = 0.0; exited = True; exit_i = i; exit_why = why
        # ---- 现金计息 ----
        cash *= (1 + DC)
        nav = cash + sh * px
        nav_peak = max(nav_peak, nav)
        max_dd_acct = min(max_dd_acct, nav / nav_peak - 1)

    px_end = d['close'].iloc[exit_i]
    nav = cash + sh * px_end
    low_real = float(d['real'].iloc[i0:exit_i + 1].min())
    cost_real = None
    if buys:
        rp = [b['px'] * 100.0 / d['cpi_idx'].iloc[b['i']] for b in buys]
        cost_real = float(len(rp) / sum(1.0 / x for x in rp))   # 等金额 → 调和平均
    return dict(i0=i0, d0=str(d['d'].iloc[i0].date()), regime=regime,
                n_buys=len(buys), maxpos=float(inv), nav=float(nav), ret=float(nav - 1),
                exit_i=exit_i, d_exit=str(d['d'].iloc[exit_i].date()), exit_why=exit_why,
                hold_m=months_in, cost_real=cost_real, low_real=low_real,
                cost_prem=(cost_real / low_real - 1) if cost_real else None,
                max_dd_acct=float(max_dd_acct), buys=buys)


def main():
    d = load()
    ev = find_events(d)
    print(f'样本 {d.d.iloc[0].date()} ~ {d.d.iloc[-1].date()}  {len(d)} 日  {(d.d.iloc[-1]-d.d.iloc[0]).days/365.25:.1f} 年')
    print(f'独立熊市 {len(ev)} 轮：')
    for i0 in ev:
        print(f"  {d.d.iloc[i0].date()}  入场 {d.close.iloc[i0]:6.0f}  实际 {d.real.iloc[i0]:6.0f}  "
              f"离底 {d.rA.iloc[i0]*100-100:+5.1f}%  距顶 {d.dd_peak.iloc[i0]*100:+5.1f}%  "
              f"{'勃发' if d.dd_peak.iloc[i0]<=-0.5 else '箱体'}")
    print()

    # 基准：进带即满仓 + 同一出场规则
    print('=== 基准 A：进带即满仓（1 笔），同一出场规则 ===')
    base = [backtest(d, i0, n=1, confirm='none', gap_m=1) for i0 in ev]
    for p in base:
        print(f"  {p['d0']}  {p['regime']:5s}  {p['d_exit']}  {p['exit_why']:12s}  "
              f"持 {p['hold_m']:>2} 月  收益 {p['ret']*100:+7.1f}%  最大浮亏 {p['max_dd_acct']*100:+6.1f}%")
    print(f"  平均 {np.mean([p['ret'] for p in base])*100:+.2f}%")
    print()

    combos = [(n, cf, g) for n in [2, 3, 4, 5] for cf in ['none', 'ma20', 'reb8', 'mup'] for g in [1, 2, 3]]
    rows = []
    for n, cf, g in combos:
        per = [backtest(d, i0, n=n, confirm=cf, gap_m=g) for i0 in ev]
        rets = [p['ret'] for p in per]
        prems = [p['cost_prem'] for p in per if p['cost_prem'] is not None]
        rows.append(dict(n=n, cf=cf, g=g,
                         ret_mean=float(np.mean(rets)), ret_min=float(np.min(rets)),
                         prem=float(np.mean(prems)) if prems else None,
                         pos=float(np.mean([p['maxpos'] for p in per])),
                         dd=float(np.mean([p['max_dd_acct'] for p in per])),
                         rets=[round(x * 100, 1) for x in rets],
                         dds=[round(p['max_dd_acct'] * 100, 1) for p in per]))
    df = pd.DataFrame(rows)
    print('=== 全网格：平均收益 ===')
    piv = df.pivot_table(index=['n', 'g'], columns='cf', values='ret_mean')
    print((piv * 100).round(2).to_string())
    print()
    print('=== 全网格：平均成本溢价（越低越好）===')
    piv2 = df.pivot_table(index=['n', 'g'], columns='cf', values='prem')
    print((piv2 * 100).round(1).to_string())
    print()
    print('=== 全网格：平均最大浮亏（越接近 0 越好）===')
    piv3 = df.pivot_table(index=['n', 'g'], columns='cf', values='dd')
    print((piv3 * 100).round(1).to_string())
    print()
    print('=== 全网格：最差单次收益 ===')
    piv4 = df.pivot_table(index=['n', 'g'], columns='cf', values='ret_min')
    print((piv4 * 100).round(1).to_string())
    print()

    df = df.sort_values('ret_mean', ascending=False)
    print('=== 前 10 ===')
    print(df.head(10).to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\n=== 后 5 ===')
    print(df.tail(5).to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    # v1.0 匀速加仓对照
    print('\n=== 基准 C：v1.0 匀速加仓（进带后每跌 3% 加一笔，共 5 笔）===')
    def uni(i0):
        lot = 1.0 / 5
        cash, sh, inv = 1.0, 0.0, 0.0
        last = None
        for i in range(i0, min(len(d), i0 + 1800)):
            px = d['close'].iloc[i]
            go = last is None or d['rA'].iloc[i] <= d['rA'].iloc[last] * 0.97
            if go and inv < 1 - 1e-9:
                amt = min(lot, cash); sh += amt / px; cash -= amt; inv += amt; last = i
            cash *= (1 + DC)
        return dict(ret=cash + sh * d['close'].iloc[-1] - 1)
    print('  （不做，匀速加仓与出场规则耦合太强，另在 d123 单独说）')

    best = df.iloc[0]
    bestt = (int(best.n), best.cf, int(best.g))
    print(f"\n=== 最优组合明细 n={bestt[0]} cf={bestt[1]} g={bestt[2]} ===")
    det = []
    for i0 in ev:
        p = backtest(d, i0, n=bestt[0], confirm=bestt[1], gap_m=bestt[2])
        det.append({k: v for k, v in p.items() if k != 'buys'})
        print(f"  {p['d0']}  {p['regime']:5s}  {p['n_buys']}/{bestt[0]} 笔  "
              f"仓位 {p['maxpos']*100:5.1f}%  成本溢价 {(p['cost_prem'] or 0)*100:+6.1f}%  "
              f"→ {p['d_exit']}  {p['exit_why']:12s}  收益 {p['ret']*100:+7.1f}%")

    json.dump(dict(events=[dict(i=int(i), d=str(d.d.iloc[i].date()),
                                rA=round(float(d.rA.iloc[i]), 4),
                                dd=round(float(d.dd_peak.iloc[i]), 4),
                                close=round(float(d.close.iloc[i]), 1),
                                real=round(float(d.real.iloc[i]), 1),
                                regime='burst' if d.dd_peak.iloc[i] <= -0.5 else 'box') for i in ev],
                   base=[{k: v for k, v in p.items() if k != 'buys'} for p in base],
                   grid=rows, best_detail=det, best=list(bestt)),
              open(os.path.join(B, 'd122_result.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\nsaved d122_result.json')


if __name__ == '__main__':
    main()
