# -*- coding: utf-8 -*-
"""
d125: 出场规则手术 —— v1.2 §4.2 的「趋势出场能拿到 2.65 倍」到底成不成立

2013-06-06 进场，2015-06-12 收 5,166，从进场算 +130.4%。
但「破 MA120 且 MA120 下行即清仓」这条规则，在这 2 年里触发了 7 次，
每次都在 2,000~2,140 之间 —— 也就是说，按规则执行会被震出至少 7 回。

本脚本把「出场」拆成 8 种方案，6 轮事件全跑，看哪种真的能吃到肉。
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
OUT = {}


def simulate(i0, mode, n=3, confirm='reb8', gap_m=2, max_hold=1800, verbose=False):
    """n 档确认加仓 + 可切换的出场方案。返回净值、最大浮亏、卖出次数、日志"""
    ladder = [0.10] if n == 1 else [0.10 * (1 - k / (n - 1)) for k in range(n)]
    lot = 1.0 / n
    regime = 'burst' if d['dd_peak'].iloc[i0] <= -0.50 else 'box'
    cash, sh, inv, leg, last = 1.0, 0.0, 0.0, 0, None
    touched, pk, peak_gain = False, 0.0, -9.9
    navs, log = [], []
    sold_px = None
    n_sell, n_rebuy = 0, 0
    last_sell_i = -999
    pending_rebuy = False

    for i in range(i0, min(N, i0 + max_hold)):
        px = d['close'].iloc[i]
        ma120, ma250 = d['ma120'].iloc[i], d['ma250'].iloc[i]
        months = (d['d'].iloc[i].year - d['d'].iloc[i0].year) * 12 + (d['d'].iloc[i].month - d['d'].iloc[i0].month)

        # ---------- 建仓 ----------
        if leg < n and not pending_rebuy:
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

        # ---------- 出场 ----------
        if inv > 0 and sh > 0:
            pk = max(pk, px); peak_gain = max(peak_gain, px / (inv / sh) - 1)
            if not np.isnan(ma120) and px > ma120:
                touched = True
            g = px / (inv / sh) - 1
            hit = None
            if mode == 'none':
                pass
            elif mode == 'v12':
                if regime == 'burst':
                    if touched and not np.isnan(ma120) and px < ma120 and d['ma120'].iloc[i] <= d['ma120'].iloc[i - 20]:
                        hit = '破MA120↓'
                    elif peak_gain > 0.15 and px / pk - 1 <= -0.20:
                        hit = '峰回20%'
                else:
                    if g >= 0.50:
                        hit = '+50%'
                    elif months >= 20 and touched and not np.isnan(ma120) and px < ma120 and d['ma120'].iloc[i] <= d['ma120'].iloc[i - 20]:
                        hit = '20月后趋势'
                    elif peak_gain > 0.15 and px / pk - 1 <= -0.20:
                        hit = '峰回20%'
            elif mode == 'ma120_only':
                if touched and not np.isnan(ma120) and px < ma120:
                    hit = '破MA120'
            elif mode == 'ma250':
                if not np.isnan(ma250) and px < ma250 and d['ma250'].iloc[i] <= d['ma250'].iloc[i - 20]:
                    hit = '破MA250↓'
            elif mode == 'drop25':
                if pk > 0 and px / pk - 1 <= -0.25:
                    hit = '峰回25%'
            elif mode == 'ma120_depth':
                if touched and not np.isnan(ma120) and px < ma120 and d['ma120'].iloc[i] <= d['ma120'].iloc[i - 20] and pk / px - 1 >= 0.10:
                    hit = '破MA120↓且距峰10%'
            elif mode == 'ma120_3d':
                if touched and not np.isnan(ma120):
                    lo3 = d['close'].iloc[max(0, i - 2):i + 1]
                    if len(lo3) == 3 and (lo3 < d['ma120'].iloc[max(0, i - 2):i + 1]).all():
                        hit = '破MA120连续3日'
            if hit:
                cash += sh * px; sh = 0.0; sold_px = px; n_sell += 1; last_sell_i = i
                log.append(('卖', str(d['d'].iloc[i].date()), round(px), hit))
                pending_rebuy = True
                touched = False; pk = 0.0; peak_gain = -9.9

        # ---------- 回补 ----------
        if pending_rebuy and sh == 0 and mode in ('v12', 'ma120_only', 'ma250', 'ma120_depth', 'v12_rebuy'):
            ok = (not np.isnan(ma250)) and px > ma250 and (sold_px is None or px > sold_px)
            if ok:
                amt = cash
                sh += amt / px; cash -= amt; inv += amt; n_rebuy += 1
                log.append(('回补', str(d['d'].iloc[i].date()), round(px), f'现金全入'))
                pending_rebuy = False

        cash *= (1 + eng.DC)
        navs.append(cash + sh * px)

    nav_end = cash + sh * d['close'].iloc[min(N - 1, i0 + max_hold - 1)]
    navs = np.array(navs)
    return dict(ret=nav_end - 1, dd=float((navs / np.maximum.accumulate(navs) - 1).min()),
                n_sell=n_sell, n_rebuy=n_rebuy, log=log)


def main():
    modes = ['none', 'v12', 'ma120_only', 'ma250', 'drop25', 'ma120_depth', 'ma120_3d']
    label = {'none': 'E0 不择时（持到底）', 'v12': 'E1 v1.2 原版（清仓即止，不回补）',
             'ma120_only': 'E2 只破 MA120（不限方向）', 'ma250': 'E3 破 MA250↓',
             'drop25': 'E4 峰回 25%', 'ma120_depth': 'E5 破MA120↓且距峰≥10%',
             'ma120_3d': 'E6 破MA120 连续3日'}

    print('=== 6 轮事件 × 7 种出场（建仓固定 n=3 reb8 g=2）===\n')
    hdr = f"{'方案':<26}" + ''.join(f'{str(d.d.iloc[i0].date())[2:] :>9}' for i0 in ev) + f"{'均值':>9}{'浮亏':>8}{'卖次':>6}"
    print(hdr)
    table = {}
    for m in modes:
        per = [simulate(i0, m) for i0 in ev]
        rets = [p['ret'] for p in per]
        print(f"{label[m]:<26}" + ''.join(f'{x*100:8.1f}%' for x in rets) +
              f"{np.mean(rets)*100:8.1f}%{np.mean([p['dd'] for p in per])*100:7.1f}%{np.mean([p['n_sell'] for p in per]):5.1f}")
        table[m] = dict(rets=[round(x * 100, 1) for x in rets], mean=round(float(np.mean(rets)) * 100, 2),
                        dd=round(float(np.mean([p['dd'] for p in per])) * 100, 2),
                        n_sell=float(np.mean([p['n_sell'] for p in per])))
    OUT['table'] = table
    print()

    print('=== 2013-06-06 单独看（v1.2 的案例）===\n')
    i13 = int(d.index[d['d'] == pd.Timestamp('2013-06-06')][0])
    i15 = int(d.index[d['d'] <= pd.Timestamp('2015-06-12')][-1])
    hold = d.close.iloc[i15] / d.close.iloc[i13] - 1
    print(f"  基准：买入持有 2013-06-06 → 2015-06-12 = {hold*100:+.1f}%（段内最高 {d.close.iloc[i15]:.0f}）")
    print(f"  从底部 1,950 算的段涨幅 = {(5166/1950-1)*100:+.0f}% ← v1.2 的「2.65 倍」是这么来的\n")
    for m in modes:
        p = simulate(i13, m)
        print(f"  {label[m]:<26} 收益 {p['ret']*100:+7.1f}%  最大浮亏 {p['dd']*100:+6.1f}%  "
              f"卖 {p['n_sell']} 次 回补 {p['n_rebuy']} 次")
        if m == 'v12':
            print('    轨迹：', ' | '.join(f"{a} {b} {c}({e})" for a, b, c, e in p['log'][:12]))
    OUT['case2013'] = dict(hold=round(hold * 100, 1))

    print('\n=== 各方案在 3 轮勃发段的单独表现 ===')
    burst_idx = [j for j, i0 in enumerate(ev) if d.dd_peak.iloc[i0] <= -0.50]
    for m in modes:
        per = [simulate(ev[j], m) for j in burst_idx]
        print(f"  {label[m]:<26} {[round(p['ret']*100,1) for p in per]}  均值 {np.mean([p['ret'] for p in per])*100:.1f}%")

    json.dump(OUT, open(os.path.join(B, 'd125_result.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=float)
    print('\nsaved d125_result.json')


if __name__ == '__main__':
    main()
