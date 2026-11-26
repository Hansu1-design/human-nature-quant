# -*- coding: utf-8 -*-
"""
d124: 三个补充诊断
  ① 匀速加仓（v1.0）与确认加仓（v1.1）的风险对照 —— 上一版只比了收益，不公平
  ② 从 2013-06-06 进场（v1.2 的勃发段案例）用同一出场规则能拿多少 —— 交叉验证 v1.2 的 2.65 倍
  ③ 每轮"进带后还要再跌多少" —— 解释为什么分档的收益增益有限
"""
import os, json
import numpy as np
import pandas as pd

import importlib.util
spec = importlib.util.spec_from_file_location('eng', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'd122_engine.py'))
eng = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eng)

B = os.path.dirname(os.path.abspath(__file__))
d = eng.load()
ev = eng.find_events(d)
OUT = {}


def bt_full(dd, i0, n, confirm, gap_m, exit_mode='v12', max_hold=756):
    """完整回测，返回净值曲线与最大浮亏（账户级）"""
    N = len(dd)
    ladder = [0.10] if n == 1 else [0.10 * (1 - k / (n - 1)) for k in range(n)]
    lot = 1.0 / n
    regime = 'burst' if dd['dd_peak'].iloc[i0] <= -0.50 else 'box'
    cash, sh, inv, leg, last = 1.0, 0.0, 0.0, 0, None
    touched, pk, peak_gain = False, 0.0, -9.9
    navs, exit_i, why = [], N - 1, '未出场'
    for i in range(i0, min(N, i0 + max_hold)):
        px = dd['close'].iloc[i]
        if exit_i == N - 1 and leg < n:
            by_depth = dd['rA'].iloc[i] <= 1 + ladder[leg]
            by_conf = False
            if leg >= 1 and inv > 0 and dd['rA'].iloc[i] <= 1.10 and eng.confirm_ok(dd, i, confirm):
                by_conf = True
            go = True
            if last is not None:
                mm = (dd['d'].iloc[i].year - dd['d'].iloc[last].year) * 12 + (dd['d'].iloc[i].month - dd['d'].iloc[last].month)
                go = mm >= max(1, gap_m)
            if (by_depth or by_conf) and go:
                amt = min(lot, cash)
                if amt > 1e-9:
                    sh += amt / px; cash -= amt; inv += amt; last = i; leg += 1
        if exit_i == N - 1 and inv > 0:
            pk = max(pk, px); peak_gain = max(peak_gain, px / (inv / sh) - 1)
            ma = dd['ma120'].iloc[i] if exit_mode == 'v12' else dd['ma250'].iloc[i]
            if not np.isnan(ma) and px > ma: touched = True
            months = (dd['d'].iloc[i].year - dd['d'].iloc[i0].year) * 12 + (dd['d'].iloc[i].month - dd['d'].iloc[i0].month)
            tout = touched and not np.isnan(ma) and px < ma and (
                bool(dd['ma120_down'].iloc[i]) if exit_mode == 'v12' else dd['ma250'].iloc[i] <= dd['ma250'].shift(20).iloc[i])
            dout = peak_gain > 0.15 and px / pk - 1 <= -0.20
            g = px / (inv / sh) - 1
            hit = None
            if regime == 'burst':
                if tout: hit = 'trend'
                elif dout: hit = 'drop'
            else:
                if g >= .5: hit = 'tp50'
                elif months >= 20 and tout: hit = 'trend20'
                elif dout: hit = 'drop'
            if hit:
                cash += sh * px; sh = 0.0; exit_i = i; why = hit
        cash *= (1 + eng.DC)
        navs.append(cash + sh * px)
    nav_end = cash + sh * dd['close'].iloc[exit_i]
    navs = np.array(navs)
    dd_acct = float((navs / np.maximum.accumulate(navs) - 1).min())
    return dict(ret=nav_end - 1, dd=dd_acct, exit_i=exit_i, why=why,
                d_exit=str(dd['d'].iloc[exit_i].date()), navs=navs, i0=i0)


def main():
    d2 = d.copy()
    d2['ma250'] = d2['close'].rolling(250).mean()

    print('=== ① 匀速加仓 vs 确认加仓：收益 + 浮亏 一起看 ===')
    print(f"{'策略':<26}{'收益':>8}{'最大浮亏':>10}{'收益/浮亏':>10}   各次收益")
    rows = []
    # 确认加仓（v1.1）
    for n, cf, g in [(2, 'reb8', 1), (3, 'reb8', 1), (3, 'reb8', 2), (3, 'ma20', 2), (4, 'reb8', 2), (5, 'reb8', 2)]:
        per = [bt_full(d2, i0, n, cf, g) for i0 in ev]
        m = np.mean([p['ret'] for p in per]); dd = np.mean([p['dd'] for p in per])
        print(f"确认加仓 n={n} {cf} g={g}{'':<8}{m*100:7.2f}%{dd*100:9.2f}%{m/abs(dd):10.2f}   {[round(p['ret']*100,1) for p in per]}")
        rows.append(dict(kind='confirm', n=n, cf=cf, g=g, ret=float(m), dd=float(dd), rets=[float(p['ret']) for p in per]))

    # 匀速加仓（v1.0）：进带后每跌 x% 加一笔
    def uniform(i0, drop, legs):
        N = len(d2)
        lot = 1.0 / legs
        cash, sh, inv, last = 1.0, 0.0, 0.0, None
        regime = 'burst' if d2['dd_peak'].iloc[i0] <= -0.50 else 'box'
        touched, pk, peak_gain = False, 0.0, -9.9
        navs, exit_i, why = [], N - 1, '未出场'
        for i in range(i0, min(N, i0 + 756)):
            px = d2['close'].iloc[i]
            if exit_i == N - 1 and inv < 1 - 1e-9:
                go = last is None or d2['rA'].iloc[i] <= d2['rA'].iloc[last] * (1 - drop)
                if go:
                    amt = min(lot, cash); sh += amt / px; cash -= amt; inv += amt; last = i
            if exit_i == N - 1 and inv > 0:
                pk = max(pk, px); peak_gain = max(peak_gain, px / (inv / sh) - 1)
                ma = d2['ma120'].iloc[i]
                if not np.isnan(ma) and px > ma: touched = True
                months = (d2['d'].iloc[i].year - d2['d'].iloc[i0].year) * 12 + (d2['d'].iloc[i].month - d2['d'].iloc[i0].month)
                tout = touched and not np.isnan(ma) and px < ma and bool(d2['ma120_down'].iloc[i])
                dout = peak_gain > 0.15 and px / pk - 1 <= -0.20
                g = px / (inv / sh) - 1
                hit = None
                if regime == 'burst':
                    if tout: hit = 'trend'
                    elif dout: hit = 'drop'
                else:
                    if g >= .5: hit = 'tp50'
                    elif months >= 20 and tout: hit = 'trend20'
                    elif dout: hit = 'drop'
                if hit:
                    cash += sh * px; sh = 0.0; exit_i = i; why = hit
            cash *= (1 + eng.DC)
            navs.append(cash + sh * px)
        navs = np.array(navs)
        return dict(ret=cash + sh * d2['close'].iloc[exit_i] - 1,
                    dd=float((navs / np.maximum.accumulate(navs) - 1).min()),
                    d_exit=str(d2['d'].iloc[exit_i].date()), why=why)

    for drop, legs in [(0.02, 5), (0.03, 5), (0.05, 4), (0.08, 3), (0.10, 3)]:
        per = [uniform(i0, drop, legs) for i0 in ev]
        m = np.mean([p['ret'] for p in per]); dd = np.mean([p['dd'] for p in per])
        print(f"匀速加仓 每跌{int(drop*100)}% 共{legs}笔{'':<4}{m*100:7.2f}%{dd*100:9.2f}%{m/abs(dd):10.2f}   {[round(p['ret']*100,1) for p in per]}")
        rows.append(dict(kind='uniform', drop=drop, legs=legs, ret=float(m), dd=float(dd),
                         rets=[float(p['ret']) for p in per]))
    OUT['compare'] = rows
    print()

    print('=== ② 从 2013-06-06 进场（v1.2 案例）：同一出场规则能拿多少 ===')
    i13 = int(d2.index[d2['d'] == pd.Timestamp('2013-06-06')][0])
    print(f"  进场 {d2.d.iloc[i13].date()} 收 {d2.close.iloc[i13]:.0f} 实际 {d2.real.iloc[i13]:.0f} "
          f"离底 {d2.rA.iloc[i13]*100-100:+.1f}% 距顶 {d2.dd_peak.iloc[i13]*100:+.1f}%")
    for n, cf, g in [(1, 'none', 1), (3, 'reb8', 2), (3, 'ma20', 2)]:
        p = bt_full(d2, i13, n, cf, g)
        # 持有到 2015-06-12 的收益
        i15 = int(d2.index[d2['d'] <= pd.Timestamp('2015-06-12')][-1])
        hold = d2.close.iloc[i15] / d2.close.iloc[i13] - 1
        print(f"  n={n} {cf} g={g}: 出场 {p['d_exit']} ({p['why']}) 收益 {p['ret']*100:+7.1f}%  "
              f"最大浮亏 {p['dd']*100:+6.1f}%  | 持到 2015-06-12 为 {hold*100:+7.1f}%")
    OUT['case2013'] = dict(i=int(i13), d=str(d2.d.iloc[i13].date()),
                           close=float(d2.close.iloc[i13]), rA=float(d2.rA.iloc[i13]))

    print('\n  对照 2011-10-21 进场（同一轮熊市，早 20 个月）：')
    p = bt_full(d2, ev[1], 3, 'reb8', 2)
    print(f"    出场 {p['d_exit']} ({p['why']}) 收益 {p['ret']*100:+.1f}%  最大浮亏 {p['dd']*100:+.1f}%")
    print('    → 早进 20 个月，被 2012 年的假突破震出，错过 2014-2015')
    print()

    print('=== ③ 每轮：进带后还要再跌多少（名义，买到的价）===')
    rows3 = []
    for i0 in ev:
        seg = d2.iloc[i0:i0 + 500]
        lo_i = int(np.argmin(seg['close'].values))
        lo_px = seg['close'].iloc[lo_i]
        lo_d = seg['d'].iloc[lo_i]
        drop = lo_px / d2.close.iloc[i0] - 1
        days = lo_i
        rows3.append(dict(d0=str(d2.d.iloc[i0].date()), lo=str(lo_d.date()), drop=float(drop), days=int(days)))
        print(f"  {d2.d.iloc[i0].date()} 进 {d2.close.iloc[i0]:6.0f} → {lo_d.date()} 低 {lo_px:6.0f} "
              f"再跌 {drop*100:+6.1f}%（{days} 交易日）")
    drops = np.array([r['drop'] for r in rows3])
    print(f'  中位 {np.median(drops)*100:+.1f}%  均值 {drops.mean()*100:+.1f}%  最深 {drops.min()*100:+.1f}%')
    print(f'  → 分 3 笔，第 2 笔在 -{abs(np.median(drops))*100/2:.1f}%，第 3 笔在 -{abs(np.median(drops))*100:.1f}%；'
          f'能摊的空间就这么大')
    OUT['post_entry_drop'] = rows3

    json.dump(OUT, open(os.path.join(B, 'd124_result.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=float)
    print('\nsaved d124_result.json')


if __name__ == '__main__':
    main()
