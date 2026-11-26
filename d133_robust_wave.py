# -*- coding: utf-8 -*-
"""
d133 波段规则的稳健性：是规律还是过拟合？
- MA 参数 5~250 全扫
- 分段检验（1997-2010 / 2011-2026 / 每 5 年）
- 加滑点（ETF 折溢价 + 冲击）
- 与骨架策略的资金效率对比
"""
import numpy as np
import pandas as pd
from d130_probe_wave import load
from d132_tradable import run

pd.set_option('display.width', 240)


def ma(x, w):
    return pd.Series(x).rolling(w).mean().values


def main():
    d = load()
    c = d['close'].values
    n = len(c)
    yrs = (d.d.iloc[-1] - d.d.iloc[0]).days / 365.25

    # ---------- 1. MA 参数全扫 ----------
    print('=' * 96)
    print('MA 参数全扫（双边 0.10%）—— 看是不是只有 20 有效')
    print('=' * 96)
    print(f'{"窗口":>5} {"年化":>7} {"最大回撤":>9} {"夏普":>6} {"交易/年":>8} {"持仓":>7} {"vs BH 年化差":>12}')
    bh = run(d, np.ones(n), 0.001, 'BH')
    for w in [5, 8, 10, 12, 15, 18, 20, 22, 25, 30, 35, 40, 50, 60, 80, 100, 120, 150, 200, 250]:
        m = ma(c, w)
        with np.errstate(invalid='ignore'):
            sig = (c > m).astype(float)
        sig[:w] = 0
        s = run(d, sig, 0.001)
        print(f'{w:5d} {s["cagr"]*100:6.2f}% {s["mdd"]*100:8.1f}% {s["sharpe"]:6.2f} '
              f'{s["trades"]/yrs:8.1f} {s["expo"]*100:6.1f}% {(s["cagr"]-bh["cagr"])*100:+11.2f}pp')
    print(f'{"BH":>5} {bh["cagr"]*100:6.2f}% {bh["mdd"]*100:8.1f}% {bh["sharpe"]:6.2f} '
          f'{0:8.1f} {100:6.1f}%')

    # ---------- 2. 分段检验 ----------
    print()
    print('=' * 96)
    print('分段检验：MA20 / MA60 在每 5 年是不稳定还是都有效')
    print('=' * 96)
    bounds = ['1997-01-01', '2001-01-01', '2006-01-01', '2011-01-01',
              '2016-01-01', '2021-01-01', '2026-12-31']
    print(f'{"区间":<22} {"BH年化":>8} {"BH回撤":>8} {"MA20年化":>10} {"MA20回撤":>10} '
          f'{"MA60年化":>10} {"MA60回撤":>10} {"MA20胜BH":>9}')
    for k in range(len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1]
        dd = d[(d.d >= a) & (d.d < b)].reset_index(drop=True)
        if len(dd) < 250:
            continue
        cc = dd['close'].values
        nn = len(cc)
        y = (dd.d.iloc[-1] - dd.d.iloc[0]).days / 365.25
        r_bh = run(dd, np.ones(nn), 0.001)
        r20 = run(dd, np.where(np.isnan(ma(cc, 20)), 0, (cc > ma(cc, 20)).astype(float)), 0.001)
        r60 = run(dd, np.where(np.isnan(ma(cc, 60)), 0, (cc > ma(cc, 60)).astype(float)), 0.001)
        win = '✓' if r20['cagr'] > r_bh['cagr'] else '✗'
        print(f'{a[:7]}~{b[:7]} ({y:4.1f}y)  {r_bh["cagr"]*100:7.2f}% {r_bh["mdd"]*100:7.1f}% '
              f'{r20["cagr"]*100:9.2f}% {r20["mdd"]*100:9.1f}% '
              f'{r60["cagr"]*100:9.2f}% {r60["mdd"]*100:9.1f}% {win:>9}')

    # ---------- 3. 加滑点 ----------
    print()
    print('=' * 96)
    print('滑点敏感：ETF 实际成本可能到双边 0.3%（佣金 + 折溢价 + 冲击）')
    print('=' * 96)
    print(f'{"成本":>10} {"MA20年化":>10} {"MA20回撤":>10} {"MA60年化":>10} {"MA60回撤":>10} {"BH年化":>9}')
    for cost in [0.0010, 0.0015, 0.0020, 0.0030, 0.0040]:
        s20 = run(d, np.where(np.isnan(ma(c, 20)), 0, (c > ma(c, 20)).astype(float)), cost)
        s60 = run(d, np.where(np.isnan(ma(c, 60)), 0, (c > ma(c, 60)).astype(float)), cost)
        sb = run(d, np.ones(n), cost)
        print(f'{cost*100:9.2f}% {s20["cagr"]*100:9.2f}% {s20["mdd"]*100:9.1f}% '
              f'{s60["cagr"]*100:9.2f}% {s60["mdd"]*100:9.1f}% {sb["cagr"]*100:8.2f}%')

    # ---------- 4. 关键：这个收益从哪来 ----------
    print()
    print('=' * 96)
    print('收益来源拆解：MA20 波段赚的是"躲过大跌"还是"抓住上涨"？')
    print('=' * 96)
    m20 = ma(c, 20)
    sig = np.zeros(n)
    ok = ~np.isnan(m20)
    sig[ok] = (c[ok] > m20[ok]).astype(float)
    pos = np.zeros(n)
    pos[1:] = sig[:-1]
    r = np.zeros(n)
    r[1:] = c[1:] / c[:-1] - 1
    # 分段统计：持仓日 vs 空仓日
    hold_r = r[pos > 0.5]
    flat_r = r[pos <= 0.5]
    print(f'持仓日 {len(hold_r)} 天：日均 {hold_r.mean()*100:+.4f}%  累计 {(np.prod(1+hold_r)-1)*100:+.0f}%')
    print(f'空仓日 {len(flat_r)} 天：日均 {flat_r.mean()*100:+.4f}%  '
          f'（若在场会累计 {(np.prod(1+flat_r)-1)*100:+.0f}%）')
    print()
    print('  解读：若空仓日的累计收益为负，说明规则的价值是"躲跌"；')
    print('        若为正，说明规则漏掉了本该赚的钱。')

    # 最大单日/单段损失对比
    worst_bh = np.sort(r)[:10]
    print(f'  全样本最差 10 天：{np.round(worst_bh*100,1)}')
    print(f'  这些天 MA20 是否在场：{["是" if pos[np.where(r==x)[0][0]]>0.5 else "否" for x in worst_bh]}')

    # ---------- 5. 与骨架对比：资金效率 ----------
    print()
    print('=' * 96)
    print('和骨架比：骨架大部分时间空仓，年化要按"全资金"算')
    print('=' * 96)
    # 骨架：6 轮，每轮 36 个月窗口，收益取 d126 的实测
    sk = [(0.224, 3.0), (0.382, 3.0), (0.249, 3.0), (0.076, 3.0), (0.126, 3.0), (0.264, 3.0)]
    print('  骨架 6 轮实测（36 个月窗口，v1.3 出场规则）：')
    print('  注：这里用 d126 的 E0 不择时或 E1 的值，需要重查')
    print()
    print('  更公平的比法：同样 10 万本金，30 年，三种玩法')
    for name, sigv, cost in [
        ('买入持有', np.ones(n), 0.001),
        ('MA20 波段（年 12 次）', np.where(np.isnan(m20), 0, (c > m20).astype(float)), 0.0015),
        ('MA60 波段（年 6 次）', np.where(np.isnan(ma(c, 60)), 0, (c > ma(c, 60)).astype(float)), 0.0015),
    ]:
        s = run(d, sigv, cost)
        print(f'  {name:<22} 10 万 → {10*(1+s["total"]):8.1f} 万  年化 {s["cagr"]*100:5.2f}%  '
              f'回撤 {s["mdd"]*100:6.1f}%  夏普 {s["sharpe"]:4.2f}')


if __name__ == '__main__':
    main()
