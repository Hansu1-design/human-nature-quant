# -*- coding: utf-8 -*-
"""
d153 两个月：是"周期"还是"波段长度的统计结果"？
两词的交易含义相反：
  周期  → 可按时卖（知道现在第几周就知道还剩几周）
  段长度 → 只能事后看到（知道现在在涨，不知道还能涨几周）

三层：
  A) 周线条件概率：近 k 周涨/跌 → 未来 4 周方向
  B) 段长度的无记忆性（hazard rate）：已持续 k 周后，本段再延续的概率
  C) 若按"满 N 周就卖"，结果如何（用 d150 资金曲线框架）
"""
import numpy as np
import pandas as pd
from scipy import stats
from d141_eat_bite import load
from d152_period import zigzag

pd.set_option('display.width', 240)


def week(d):
    s = d.set_index('d')['close']
    w = s.resample('W-FRI').last().dropna()
    return w


def main():
    d = load()
    px = d['close'].values
    w = week(d)
    wr = w.pct_change().dropna()
    print(f'样本：{d.d.iloc[0].date()} ~ {d.d.iloc[-1].date()}，周线 {len(w)} 根')

    # ================= A. 周线条件概率 =================
    print('\n' + '=' * 100)
    print('A. 周线条件概率：过去 k 周的方向，能否预告未来 4 周')
    print('=' * 100)
    for H in (4, 8):
        fut = (w.shift(-H) / w - 1).dropna()
        base_down = (fut < 0).mean()
        base_up = (fut > 0).mean()
        print(f'\n  【未来 {H} 周】基准：跌 {base_down*100:.1f}% / 涨 {base_up*100:.1f}%（n={len(fut)}）')
        print(f'  {"条件":<22} {"样本":>6} {"未来跌":>8} {"vs基准":>9} {"p":>7}   '
              f'{"未来涨":>8} {"vs基准":>9} {"p":>7}')
        for k in (2, 4, 6, 8, 12):
            past = (w / w.shift(k) - 1)
            df = pd.concat([past.rename('p'), fut.rename('f')], axis=1).dropna()
            for lab, sub in [('近%d周涨' % k, df[df.p > 0]), ('近%d周跌' % k, df[df.p < 0])]:
                if len(sub) < 15:
                    continue
                dn = (sub.f < 0).mean(); up = (sub.f > 0).mean()
                p_dn = stats.binomtest(int((sub.f < 0).sum()), len(sub), base_down).pvalue
                p_up = stats.binomtest(int((sub.f > 0).sum()), len(sub), base_up).pvalue
                print(f'  {lab:<22} {len(sub):>6} {dn*100:>7.1f}% {(dn-base_down)*100:>+8.1f}pp {p_dn:>7.3f}   '
                      f'{up*100:>7.1f}% {(up-base_up)*100:>+8.1f}pp {p_up:>7.3f}')

    # ================= B. 段长度无记忆性 =================
    print('\n' + '=' * 100)
    print('B. 段长度是否"无记忆"（hazard rate）：已涨了 k 周，本段还会继续的概率')
    print('=' * 100)
    for thr in (0.08, 0.10):
        segs = zigzag(px, thr)
        # 用交易日→周 折算段长
        wlens = np.array([(d['d'].iloc[b] - d['d'].iloc[a]).days / 7.0 for _, a, b in segs])
        wlens = np.maximum(np.round(wlens), 1).astype(int)
        lens = np.array([int(round((d['d'].iloc[b] - d['d'].iloc[a]).days / 7.0)) for _, a, b in segs])
        lens = np.maximum(lens, 1)
        print(f'\n  阈值 {thr*100:.0f}%：{len(lens)} 段，长度（周）中位 {np.median(lens):.0f}，均值 {lens.mean():.1f}，最大 {lens.max()}')
        print(f'  长度分布：' + '  '.join([f'{k}周:{int((lens==k).sum())}' for k in range(1, 13) if (lens == k).sum() > 0]))
        print(f'\n  {"已持续":>6} {"存活段数":>8} {"本段在此结束":>12} {"hazard":>8}  解读')
        for k in range(1, 14):
            alive = (lens >= k).sum()
            if alive < 8:
                break
            died = (lens == k).sum()
            hz = died / alive
            print(f'  {k:>6} {alive:>8} {died:>12} {hz*100:>7.1f}%  '
                  f'{"还能撑中位%.0f周" % (1/hz) if hz > 0 else ""}')
        # 无记忆检验：hazard 是否恒定（用分段卡方）
        print(f'  → 若 hazard 各行接近（不随 k 单调上升），则段长度无记忆：'
              f'知道已涨几周，无法推出还剩几周')

    # ================= C. 满 N 周就卖 =================
    print('\n' + '=' * 100)
    print('C. 若"入场后满 N 周无条件卖"，结果如何（骨架入场，关接刀，不做波段）')
    print('=' * 100)
    from d150_weekwave import run, m
    yrs = (d['d'].iloc[-1] - d['d'].iloc[0]).days / 365.25
    print(f'  {"方案":<26} {"笔数":>5} {"年化":>8} {"最大回撤":>9} {"Calmar":>8}')
    nav, tr, occ, nw, wv = run(d, wave=False, cost=0.001, cfg=dict(bear=False, drop_after1y=-0.99, hold_days=99999))
    g0, dd0 = m(nav, yrs)
    print(f'  {"买入持有（参照）":<26} {len(tr):>5} {g0*100:>+7.2f}% {dd0*100:>+8.1f}% {g0/abs(dd0):>8.3f}')
    nav, tr, occ, nw, wv = run(d, wave=False, cost=0.001)
    g1, dd1 = m(nav, yrs)
    print(f'  {"骨架（满250日卖）":<26} {len(tr):>5} {g1*100:>+7.2f}% {dd1*100:>+8.1f}% {g1/abs(dd1):>8.3f}')
    nav, tr, occ, nw, wv = run(d, wave=False, cost=0.001, cfg=dict(bear=False, drop_after1y=-0.15, hold_days=365))
    g2, dd2 = m(nav, yrs)
    print(f'  {"满1年+回落15%（v1.6候选）":<26} {len(tr):>5} {g2*100:>+7.2f}% {dd2*100:>+8.1f}% {g2/abs(dd2):>8.3f}')
    for wk in (6, 8, 10, 12, 16, 26):
        nav, tr, occ, nw, wv = run(d, wave=False, cost=0.001,
                                   cfg=dict(bear=False, drop_after1y=-0.99, hold_days=wk * 7))
        g, dd = m(nav, yrs)
        print(f'  {"满"+str(wk)+"周无条件卖":<26} {len(tr):>5} {g*100:>+7.2f}% {dd*100:>+8.1f}% {g/abs(dd):>8.3f}')


if __name__ == '__main__':
    main()
