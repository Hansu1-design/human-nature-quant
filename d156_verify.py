# -*- coding: utf-8 -*-
"""
d156 逐笔验证：26 周卖出是不是只靠某一笔走运？
上一轮的教训：回撤从 -43.3% 降到 -25.7% 可能只是绕过了 2014-10-20 那一次。
本脚本：
  1) 逐笔对比 持250日 / 持26周 / 满6月后回落10%
  2) 持有期参数面（20~40 周）：是平台还是尖峰
  3) 加成本（单边 0.1%）后的排序
  4) 去掉最好的一笔后重算
"""
import numpy as np
import pandas as pd
from d141_eat_bite import load
from d155_exit2 import equity, stats, build

pd.set_option('display.width', 250)


def main():
    d = load()
    c = d['close'].values
    n = len(c)
    yrs = (d['d'].iloc[-1] - d['d'].iloc[0]).days / 365.25
    B = build(d)
    sig = B['sig']

    def run(hold=250, exit_fn=None, cost=0.0):
        nav, tr = equity(d, sig, hold=hold, exit_fn=exit_fn, cost=cost)
        g, dd = stats(nav, yrs)
        return nav, tr, g, dd

    # ---------- 1. 逐笔 ----------
    print('=' * 128)
    print('一、逐笔对比（每笔：买入日/价 → 卖出日/价，本轮收益，卖后 1 年、2 年）')
    print('=' * 128)
    cases = [('持250日（基准）', 250, None),
             ('持26周（半年）', 125, None),
             ('满6月后回落10%', 99999, lambda i, e, p: (i - e >= 125) and c[i] / p - 1 <= -0.10)]
    detail = {}
    for tag, h, fn in cases:
        nav, tr, g, dd = run(h, fn)
        segs = []
        st = None
        for tt, i, p in tr:
            if tt == '买':
                st = (i, p)
            elif st is not None:
                segs.append((st[0], st[1], i, p))
                st = None
        if st is not None:
            segs.append((st[0], st[1], n - 1, c[-1]))
        detail[tag] = segs
        print(f'\n【{tag}】{len(segs)} 笔，年化 {g*100:+.2f}%，回撤 {dd*100:+.1f}%')
        print(f'  {"买日":<12}{"买价":>9}{"卖日":<13}{"卖价":>9}{"本轮":>9}'
              f'{"持有日":>7}{"期间最高":>9}{"卖后1年":>9}{"卖后2年":>9}')
        for a, pa, b, pb in segs:
            hi = c[a:b + 1].max()
            f1 = c[min(b + 250, n - 1)] / pb - 1
            f2 = c[min(b + 500, n - 1)] / pb - 1
            print(f'  {str(d.d.iloc[a].date()):<12}{pa:>9.2f} {str(d.d.iloc[b].date()):<13}{pb:>9.2f}'
                  f'{(pb/pa-1)*100:>+8.1f}%{b-a:>7d}{(hi/pa-1)*100:>+8.1f}%'
                  f'{f1*100:>+8.1f}%{f2*100:>+8.1f}%')

    # ---------- 2. 参数面 ----------
    print('\n' + '=' * 128)
    print('二、持有期参数面（交易日）：是平台还是尖峰？')
    print('=' * 128)
    print(f'{"持有":>6} {"买入":>5} {"年化":>9} {"回撤":>9} {"Calmar":>8}')
    for h in (100, 110, 115, 120, 125, 130, 140, 150, 175, 200, 250, 300):
        nav, tr, g, dd = run(h)
        nb = sum(1 for t in tr if t[0] == '买')
        print(f'{h:>6} {nb:>5} {g*100:>+8.2f}% {dd*100:>+8.1f}% {g/abs(dd):>8.3f}')

    # ---------- 3. 成本 ----------
    print('\n' + '=' * 128)
    print('三、加成本（单边 0.1%，往返 0.2%）')
    print('=' * 128)
    print(f'{"方案":<22} {"无成本":>22} {"含成本":>22}')
    for tag, h, fn in cases:
        _, _, g0, d0 = run(h, fn, 0.0)
        _, _, g1, d1 = run(h, fn, 0.001)
        print(f'{tag:<22} {g0*100:+7.2f}% / {d0*100:+6.1f}%      {g1*100:+7.2f}% / {d1*100:+6.1f}%')

    # ---------- 4. 去掉最好的一笔 ----------
    print('\n' + '=' * 128)
    print('四、去掉收益最高的那一笔后重算（检验是否靠单笔）')
    print('=' * 128)
    for tag, h, fn in cases:
        segs = detail[tag]
        rets = [(pb / pa - 1, a, b) for a, pa, b, pb in segs]
        rets_sorted = sorted(rets, key=lambda x: -x[0])
        print(f'\n【{tag}】')
        for k in range(0, 3):
            if k >= len(rets_sorted):
                break
            drop = set((a, b) for _, a, b in rets_sorted[:k + 1])
            keep = [s for s in segs if (s[0], s[2]) not in drop]
            # 重建净值：被去掉的笔按现金生息
            nav = np.ones(n)
            cash, sh, entry = 1.0, 0.0, -1
            skip = {(a, b) for _, a, b in rets_sorted[:k + 1]}
            for a, pa, b, pb in segs:
                pass
            # 简化：用 equity 但把那几笔的信号剔除
            sig2 = [s for s in sig]
            risky = [(a, b) for a, pa, b, pb in segs if (a, b) in skip]
            sig2 = [s for s in sig2 if not any(abs(s - a) < 5 for a, b in risky)]
            nav2, tr2 = equity(d, sig2, hold=h, exit_fn=fn)
            g2, dd2 = stats(nav2, yrs)
            nm = '去掉最好的 1 笔' if k == 0 else f'去掉最好的 {k+1} 笔'
            print(f'   {nm}：年化 {g2*100:+7.2f}%，回撤 {dd2*100:+6.1f}%')


if __name__ == '__main__':
    main()
