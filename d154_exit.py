# -*- coding: utf-8 -*-
"""
d154 卖出扳机对决
先生：这套规则说的其实是"怎么卖"，强势转弱就是卖点。
所以把"卖出信号"单独拿出来比，入场与回补完全统一：
  · 入场：骨架（离底 rA<=1.10），关熊市接刀，不做波段
  · 卖出后：按骨架条件重新入场（ra<=1.10 且冷却 180 日）—— 不让"回补设计"污染结论
  · 变量只有一个：卖出扳机
"""
import numpy as np
import pandas as pd
from d141_eat_bite import load
from d150_weekwave import run, m

pd.set_option('display.width', 240)


def main():
    d = load()
    yrs = (d['d'].iloc[-1] - d['d'].iloc[0]).days / 365.25
    COMMON = dict(wave=False, sell_frac=1.0, cost=0.001,
                  cfg=dict(bear=False, cool=180))
    print(f'样本：{d.d.iloc[0].date()} ~ {d.d.iloc[-1].date()}，{yrs:.1f} 年')
    print('口径：骨架入场 / 关接刀 / 关波段 / 卖出后按骨架条件重新入场 / 成本单边0.1% / 现金2.5%')

    rows = []

    def add(name, **kw):
        c2 = dict(COMMON)
        cfg = dict(c2.pop('cfg'))
        for k in list(kw.keys()):
            if k == 'cfg':
                cfg.update(kw.pop('cfg'))
        c2.update(kw)
        c2['cfg'] = cfg
        nav, tr, occ, nw, wv = run(d, **c2)
        g, dd = m(nav, yrs)
        n_sell = sum(1 for t in tr if t['t'] == '卖')
        rows.append((name, n_sell, g, dd, g / abs(dd) if dd else np.nan, occ / len(d)))
        return g, dd

    print('\n' + '=' * 112)
    print('一、固定期限卖出（不管行情，到点就走）')
    print('=' * 112)
    add('买入持有（永不卖）', cfg=dict(force_days=0, hold_days=99999, drop_after1y=-0.99))
    for lbl, days in [('满 8 周（2个月）', 56), ('满 12 周（3个月）', 84),
                      ('满 16 周（4个月）', 112), ('满 26 周（半年）', 182),
                      ('满 250 交易日（骨架原版）', 365), ('满 2 年', 730)]:
        add(lbl, cfg=dict(force_days=days, hold_days=99999, drop_after1y=-0.99))

    print('\n' + '=' * 112)
    print('二、强弱信号卖出（先生说的"强转弱"）')
    print('=' * 112)
    add('月线收阴即卖', cfg=dict(exit_month=True, hold_days=99999, drop_after1y=-0.99))
    add('1234周线转弱即卖', wave=True, rebuy_wk=99999, cfg=dict(hold_days=99999, drop_after1y=-0.99))
    add('1234周线转弱·宽判据1.02', wave=True, rebuy_wk=99999,
        cfg=dict(hold_days=99999, drop_after1y=-0.99, wk_ratio=1.02))

    print('\n' + '=' * 112)
    print('三、回落清仓（v1.6 候选的方向）')
    print('=' * 112)
    add('满1年+回落 10%', cfg=dict(hold_days=365, drop_after1y=-0.10))
    add('满1年+回落 15%', cfg=dict(hold_days=365, drop_after1y=-0.15))
    add('满1年+回落 20%', cfg=dict(hold_days=365, drop_after1y=-0.20))
    add('回落 15%（不设满期）', cfg=dict(hold_days=0, drop_after1y=-0.15))

    print('\n' + '=' * 112)
    print(f'{"卖出扳机":<26} {"卖出次数":>8} {"年化":>9} {"最大回撤":>10} {"Calmar":>8} {"在场率":>8}')
    print('=' * 112)
    for name, ns, g, dd, cal, occ in rows:
        star = ' ←' if name in ('买入持有（永不卖）',) else ''
        print(f'{name:<26} {ns:>8} {g*100:>+8.2f}% {dd*100:>+9.1f}% {cal:>8.3f} {occ*100:>7.1f}%{star}')

    # 逐笔：月线收阴 vs 满1年+回落15%
    print('\n' + '=' * 112)
    print('四、逐笔明细（看每次卖出赚了多少、卖后行情怎么走）')
    print('=' * 112)
    for tag, kw in [('月线收阴即卖', dict(exit_month=True, hold_days=99999, drop_after1y=-0.99)),
                    ('满1年+回落15%', dict(hold_days=365, drop_after1y=-0.15))]:
        c2 = dict(wave=False, sell_frac=1.0, cost=0.001)
        cfg = dict(bear=False, cool=180, **kw)
        c2['cfg'] = cfg
        nav, tr, occ, nw, wv = run(d, **c2)
        c = d['close'].values
        print(f'\n[{tag}]')
        print(f'  {"动作":<4} {"日期":<12} {"价格":>9} {"本轮":>9} {"卖后1年":>9} {"卖后2年":>9}  原因')
        buy_px, bi = None, None
        for t in tr:
            idx = d.index[d.d.astype(str).str[:10] == t['d']]
            i = int(idx[0]) if len(idx) else None
            if t['t'] == '买':
                buy_px, bi = t['px'], i
                print(f'  买   {t["d"]:<12} {t["px"]:>9.2f} {"":>9} {"":>9} {"":>9}  {t["why"]}')
            else:
                cur = t['px'] / buy_px - 1 if buy_px else np.nan
                a1 = c[min(i + 250, len(c) - 1)] / t['px'] - 1 if i is not None else np.nan
                a2 = c[min(i + 500, len(c) - 1)] / t['px'] - 1 if i is not None else np.nan
                print(f'  卖   {t["d"]:<12} {t["px"]:>9.2f} {cur*100:>+8.1f}% {a1*100:>+8.1f}% {a2*100:>+8.1f}%  {t["why"]}')


if __name__ == '__main__':
    main()
