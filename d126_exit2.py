# -*- coding: utf-8 -*-
"""
d126: 出场规则手术（口径修正版）

d125 的问题：max_hold=1800 天（7.4 年）让所有方案都横跨了 2015 股灾，
评估的不是"这一波拿了多少"，而是"七年后还剩多少"。

本版统一：进场后 36 个自然月结算（所有方案同一窗口，公平）。
并加入 E7「+50% 止盈」—— v1.0 的方案，直接与 v1.2 的趋势出场对撞。
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
d['ma250'] = d['close'].rolling(250).mean()
ev = eng.find_events(d)
N = len(d)
HOLD = 36 * 21          # 36 个自然月 ≈ 756 交易日
OUT = {}

MODES = {
    'e0_hold':   'E0 不择时（36月）',
    'e1_v12':    'E1 v1.2 趋势出场+回补',
    'e1n_v12':   'E1n v1.2 趋势出场（不回补）',
    'e2_ma120':  'E2 只破 MA120+回补',
    'e3_ma250':  'E3 破 MA250↓+回补',
    'e4_drop25': 'E4 峰回 25%',
    'e6_ma3d':   'E6 破MA120 连续3日',
    'e7_tp50':   'E7 +50% 止盈（v1.0）',
    'e8_mix':    'E8 +50%止盈 或 峰回25%',
}


def simulate(i0, mode, n=3, confirm='reb8', gap_m=2):
    ladder = [0.10] if n == 1 else [0.10 * (1 - k / (n - 1)) for k in range(n)]
    lot = 1.0 / n
    regime = 'burst' if d['dd_peak'].iloc[i0] <= -0.50 else 'box'
    cash, sh, inv, leg, last = 1.0, 0.0, 0.0, 0, None
    touched, pk, peak_gain = False, 0.0, -9.9
    navs, log = [], []
    sold_px, n_sell, n_rebuy, pending = None, 0, 0, False
    end = min(N - 1, i0 + HOLD)

    for i in range(i0, end + 1):
        px = d['close'].iloc[i]
        ma120, ma250 = d['ma120'].iloc[i], d['ma250'].iloc[i]
        months = (d['d'].iloc[i].year - d['d'].iloc[i0].year) * 12 + (d['d'].iloc[i].month - d['d'].iloc[i0].month)
        ma120_dn = (not np.isnan(ma120)) and d['ma120'].iloc[i] <= d['ma120'].iloc[i - 20]

        if leg < n and not pending:
            by_depth = d['rA'].iloc[i] <= 1 + ladder[leg]
            by_conf = leg >= 1 and inv > 0 and d['rA'].iloc[i] <= 1.10 and eng.confirm_ok(d, i, confirm)
            go = True
            if last is not None:
                mm = (d['d'].iloc[i].year - d['d'].iloc[last].year) * 12 + (d['d'].iloc[i].month - d['d'].iloc[last].month)
                go = mm >= max(1, gap_m)
            if (by_depth or by_conf) and go:
                amt = min(lot, cash)
                if amt > 1e-9:
                    sh += amt / px; cash -= amt; inv += amt; last = i; leg += 1
                    log.append(('买', str(d['d'].iloc[i].date()), round(px), f'{inv*100:.0f}%'))

        if inv > 0 and sh > 0:
            pk = max(pk, px); peak_gain = max(peak_gain, px / (inv / sh) - 1)
            if not np.isnan(ma120) and px > ma120:
                touched = True
            g = px / (inv / sh) - 1
            hit = None
            if mode == 'e1_v12' or mode == 'e1n_v12':
                if regime == 'burst':
                    if touched and not np.isnan(ma120) and px < ma120 and ma120_dn:
                        hit = '破MA120↓'
                    elif peak_gain > 0.15 and px / pk - 1 <= -0.20:
                        hit = '峰回20%'
                else:
                    if g >= 0.50:
                        hit = '+50%'
                    elif months >= 20 and touched and not np.isnan(ma120) and px < ma120 and ma120_dn:
                        hit = '20月后趋势'
                    elif peak_gain > 0.15 and px / pk - 1 <= -0.20:
                        hit = '峰回20%'
            elif mode == 'e2_ma120':
                if touched and not np.isnan(ma120) and px < ma120:
                    hit = '破MA120'
            elif mode == 'e3_ma250':
                if not np.isnan(ma250) and px < ma250 and d['ma250'].iloc[i] <= d['ma250'].iloc[i - 20]:
                    hit = '破MA250↓'
            elif mode == 'e4_drop25':
                if peak_gain > 0.15 and pk > 0 and px / pk - 1 <= -0.25:
                    hit = '峰回25%'
            elif mode == 'e8_mix':
                if g >= 0.50:
                    hit = '+50%止盈'
                elif peak_gain > 0.15 and pk > 0 and px / pk - 1 <= -0.25:
                    hit = '峰回25%'
            elif mode == 'e6_ma3d':
                if touched and not np.isnan(ma120):
                    w = d['close'].iloc[max(0, i - 2):i + 1]
                    wm = d['ma120'].iloc[max(0, i - 2):i + 1]
                    if len(w) == 3 and (w.values < wm.values).all():
                        hit = '破MA120三日'
            elif mode == 'e7_tp50':
                if g >= 0.50:
                    hit = '+50%止盈'
            if hit:
                cash += sh * px; sh = 0.0; sold_px = px; n_sell += 1
                log.append(('卖', str(d['d'].iloc[i].date()), round(px), hit))
                pending = True
                touched, pk, peak_gain = False, 0.0, -9.9

        if pending and sh == 0 and mode in ('e1_v12', 'e2_ma120', 'e3_ma250'):
            if (not np.isnan(ma250)) and px > ma250 and (sold_px is None or px > sold_px):
                amt = cash
                sh += amt / px; cash -= amt; inv += amt; n_rebuy += 1
                log.append(('回补', str(d['d'].iloc[i].date()), round(px), '现金全入'))
                pending = False

        cash *= (1 + eng.DC)
        navs.append(cash + sh * px)

    nav_end = cash + sh * d['close'].iloc[end]
    navs = np.array(navs)
    return dict(ret=nav_end - 1, dd=float((navs / np.maximum.accumulate(navs) - 1).min()),
                n_sell=n_sell, n_rebuy=n_rebuy, log=log, end=str(d['d'].iloc[end].date()),
                navs=navs)


def main():
    print(f'统一窗口：进场后 36 个自然月（{HOLD} 交易日）\n')
    hdr = f"{'方案':<24}" + ''.join(f'{str(d.d.iloc[i0].date())[2:]:>9}' for i0 in ev) + f"{'均值':>9}{'浮亏':>8}{'卖次':>6}"
    print(hdr)
    table = {}
    for m, lab in MODES.items():
        per = [simulate(i0, m) for i0 in ev]
        rets = [p['ret'] for p in per]
        print(f"{lab:<24}" + ''.join(f'{x*100:8.1f}%' for x in rets) +
              f"{np.mean(rets)*100:8.1f}%{np.mean([p['dd'] for p in per])*100:7.1f}%{np.mean([p['n_sell'] for p in per]):5.1f}")
        table[m] = dict(label=lab, rets=[round(x * 100, 1) for x in rets],
                        mean=round(float(np.mean(rets)) * 100, 2),
                        dd=round(float(np.mean([p['dd'] for p in per])) * 100, 2),
                        n_sell=float(np.mean([p['n_sell'] for p in per])))
    OUT['table'] = table
    print()

    print('=== 2013-06-06 案例（v1.2 §4.2 引用的那一轮）===')
    i13 = int(d.index[d['d'] == pd.Timestamp('2013-06-06')][0])
    i15 = int(d.index[d['d'] <= pd.Timestamp('2015-06-12')][-1])
    print(f"  买入持有 24 个月（到 2015-06）= {(d.close.iloc[i15]/d.close.iloc[i13]-1)*100:+.1f}%")
    print(f"  v1.2 说的「2.65 倍」= 5,166 / 1,950（从底部算的段涨幅）= {(5166/1950-1)*100:+.0f}%，不是任何规则能拿到的收益\n")
    case = {}
    for m, lab in MODES.items():
        p = simulate(i13, m)
        case[m] = dict(ret=round(p['ret'] * 100, 1), dd=round(p['dd'] * 100, 1),
                       n_sell=p['n_sell'], end=p['end'])
        print(f"  {lab:<24} 36月收益 {p['ret']*100:+7.1f}%  浮亏 {p['dd']*100:+6.1f}%  卖 {p['n_sell']} 次")
    p = simulate(i13, 'e1_v12')
    print('\n  E1 轨迹：')
    for x in p['log']:
        print(f"    {x[0]:<3} {x[1]}  {x[2]:>6}  {x[3]}")
    OUT['case2013'] = case

    print('\n=== 3 轮勃发段单独看 ===')
    bi = [j for j, i0 in enumerate(ev) if d.dd_peak.iloc[i0] <= -0.50]
    b = {}
    for m, lab in MODES.items():
        per = [simulate(ev[j], m) for j in bi]
        b[m] = [round(p['ret'] * 100, 1) for p in per]
        print(f"  {lab:<24} {b[m]}  均值 {np.mean([p['ret'] for p in per])*100:.1f}%")
    OUT['burst'] = b

    print('\n=== 3 轮箱体段单独看 ===')
    xi = [j for j, i0 in enumerate(ev) if d.dd_peak.iloc[i0] > -0.50]
    x = {}
    for m, lab in MODES.items():
        per = [simulate(ev[j], m) for j in xi]
        x[m] = [round(p['ret'] * 100, 1) for p in per]
        print(f"  {lab:<24} {x[m]}  均值 {np.mean([p['ret'] for p in per])*100:.1f}%")
    OUT['box'] = x

    json.dump(OUT, open(os.path.join(B, 'd126_result.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=float)
    print('\nsaved d126_result.json')


if __name__ == '__main__':
    main()
