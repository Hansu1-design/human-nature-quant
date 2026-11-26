# -*- coding: utf-8 -*-
"""
d137 价格间距是不是真优势？逐轮 + 检验
- 6 轮逐轮对照，不只看均值
- 配对检验（时间1月 vs 价格-5%）
- 加一个反例场景：进带后直接涨（6 轮里没有，只能构造）
"""
import numpy as np
import pandas as pd
from scipy import stats
from d130_probe_wave import load

pd.set_option('display.width', 240)


def simulate(c, n, i0, mode, param, cap=756):
    buys = [i0]
    for i in range(i0 + 1, min(n, i0 + cap)):
        if len(buys) >= 3:
            break
        if mode == 'time':
            if i - buys[-1] >= param * 20.3:
                buys.append(i)
        else:
            if c[i] <= c[buys[-1]] * (1 - param):
                buys.append(i)
    end = min(n - 1, i0 + cap)
    shares = sum(1.0 / c[b] for b in buys)
    cost = float(len(buys))
    ret = shares * c[end] / cost - 1
    navs = []
    for i in range(i0, end + 1):
        bs = [b for b in buys if b <= i]
        if bs:
            navs.append(sum(1.0 / c[b] for b in bs) * c[i] / len(bs))
    mdd = min(navs) - 1 if navs else 0.0
    return ret, mdd, len(buys)


def main():
    d = load()
    c = d['close'].values
    n = len(c)
    events = ['2008-09-08', '2011-10-21', '2018-08-17', '2020-02-03',
              '2022-04-26', '2023-10-23']
    idxs = []
    for ev in events:
        k = d.index[d.d >= ev]
        if len(k):
            idxs.append((ev, int(k[0])))

    plans = [('time', 1, '时间1月'), ('time', 2, '时间2月'),
             ('price', 0.03, '价格-3%'), ('price', 0.05, '价格-5%'),
             ('price', 0.08, '价格-8%')]

    print('=' * 110)
    print('逐轮对照（36 个月窗口，收益 / 浮亏 / 买入笔数）')
    print('=' * 110)
    hdr = f'{"进场":>12}'
    for _, _, nm in plans:
        hdr += f' {nm:>20}'
    print(hdr)
    table = {}
    for ev, i0 in idxs:
        line = f'{ev:>12}'
        for mode, param, nm in plans:
            r, m, nb = simulate(c, n, i0, mode, param)
            table.setdefault(nm, []).append(r)
            line += f' {r*100:+7.1f}%/{m*100:6.1f}%/{nb}b'
        print(line)

    print()
    print('-' * 110)
    print('统计检验（配对，n=6）')
    print('-' * 110)
    base = np.array(table['时间1月'])
    for _, _, nm in plans:
        x = np.array(table[nm])
        if nm == '时间1月':
            print(f'  {nm:<10} 基准  均值 {x.mean()*100:5.1f}%')
            continue
        diff = x - base
        t, p = stats.ttest_rel(x, base)
        print(f'  {nm:<10} 均值 {x.mean()*100:5.1f}%  vs 时间1月 差 {diff.mean()*100:+5.2f}pp  '
              f't={t:+5.2f} p={p:.3f}  {"显著" if p < 0.05 else "不显著"}  '
              f'{"全胜" if (diff>0).all() else f"胜{(diff>0).sum()}/6"}')

    print()
    print('=' * 110)
    print('关键：每轮"进带后还会再跌多少"，这决定价格间距能不能触发')
    print('=' * 110)
    print(f'  {"进场":>12} {"之后最低点跌幅":>14} {"价格-3%能买":>12} {"价格-5%能买":>12} {"价格-8%能买":>12}')
    for ev, i0 in idxs:
        seg = c[i0:min(n, i0 + 756)]
        lowdrop = seg.min() / c[i0] - 1
        cells = []
        for p in [0.03, 0.05, 0.08]:
            # 逐级触发：每笔要在上一笔基础上再跌 p
            lvl = c[i0]
            nb = 1
            for x in c[i0 + 1:min(n, i0 + 756)]:
                if nb >= 3:
                    break
                if x <= lvl * (1 - p):
                    nb += 1
                    lvl = x
            cells.append(f'{nb}b')
        print(f'  {ev:>12} {lowdrop*100:13.1f}% {cells[0]:>12} {cells[1]:>12} {cells[2]:>12}')

    print()
    print('=' * 110)
    print('反例构造：如果进带后直接涨（6 轮里一次都没有），价格间距会怎样')
    print('=' * 110)
    print('  用 2013-06 那轮做代理（它进带后也跌了，但跌幅小、后续涨幅大）：')
    k = d.index[d.d >= '2013-06-06']
    if len(k):
        i0 = int(k[0])
        for mode, param, nm in plans:
            r, m, nb = simulate(c, n, i0, mode, param)
            print(f'    {nm:<10} 收益 {r*100:+6.1f}%  浮亏 {m*100:6.1f}%  买入 {nb} 笔')
    print()
    print('  真实"进带即反转"的场景在 6 轮里不存在，所以：')
    print('  价格间距的优势，本质是"赌还会跌"。样本 6/6 都跌了，所以它 6/6 占优。')
    print('  这不是规律，是样本特征。若未来某轮进带后 V 型反转，价格间距只会买到 1 笔。')


if __name__ == '__main__':
    main()
