# -*- coding: utf-8 -*-
"""
d132 两个月波动能不能交易？
- 六种波段规则，全史 30 年，扣成本
- 成本敏感：双边 0.05% / 0.10% / 0.20%
- 对照：买入持有
- 同时算：波动可预测性（动量/自相关）
"""
import numpy as np
import pandas as pd
from d130_probe_wave import load

pd.set_option('display.width', 240)


def stats(nav, trades, yrs):
    nav = np.asarray(nav, dtype=float)
    ret = nav[-1] / nav[0] - 1
    cagr = (nav[-1] / nav[0]) ** (1 / yrs) - 1
    peak = np.maximum.accumulate(nav)
    dd = (nav / peak - 1).min()
    # 年化波动
    r = np.diff(np.log(np.maximum(nav, 1e-9)))
    vol = r.std() * np.sqrt(243) if len(r) > 2 else np.nan
    return dict(total=ret, cagr=cagr, mdd=dd, vol=vol, trades=trades,
                sharpe=(cagr / vol if vol and vol > 0 else np.nan))


def run(d, signal, cost=0.0010, name=''):
    """signal: 每日目标仓位 0/1（用前一日信号决定今日持仓，避免前视）"""
    c = d['close'].values
    n = len(c)
    pos = np.zeros(n)
    tgt = np.asarray(signal, dtype=float)
    pos[1:] = tgt[:-1]           # 今日持仓 = 昨日信号
    # 净值
    r = np.zeros(n)
    r[1:] = c[1:] / c[:-1] - 1
    # 换手成本
    chg = np.abs(np.diff(np.concatenate([[0.0], pos])))
    # 每次仓位变化 |Δpos| -> 成本 = |Δpos| * cost
    net_r = pos * r - chg * cost
    nav = np.cumprod(1 + net_r)
    trades = (chg > 0).sum() / 2.0
    yrs = (d.d.iloc[-1] - d.d.iloc[0]).days / 365.25
    s = stats(nav, trades, yrs)
    s['name'] = name
    s['expo'] = pos.mean()
    return s


def ma(x, w):
    return pd.Series(x).rolling(w).mean().values


def main():
    d = load()
    c = d['close'].values
    n = len(c)
    yrs = (d.d.iloc[-1] - d.d.iloc[0]).days / 365.25
    print(f'样本 {n} 交易日 {d.d.iloc[0].date()} ~ {d.d.iloc[-1].date()} ({yrs:.1f} 年)')
    print()

    ma5, ma10, ma20, ma60, ma120, ma250 = [ma(c, w) for w in (5, 10, 20, 60, 120, 250)]
    with np.errstate(invalid='ignore'):
        sig_ma20 = (c > ma20).astype(float)
        sig_ma60 = (c > ma60).astype(float)
        sig_ma120 = (c > ma120).astype(float)
        sig_cross = ((ma20 > ma60)).astype(float)          # 双均线
        mom20 = np.concatenate([[np.nan] * 20, c[20:] / c[:-20] - 1])
        sig_mom20 = (mom20 > 0).astype(float)
        mom40 = np.concatenate([[np.nan] * 40, c[40:] / c[:-40] - 1])
        sig_mom40 = (mom40 > 0).astype(float)
    # 前置 NaN 填 0（空仓）
    for s in (sig_ma20, sig_ma60, sig_ma120, sig_cross, sig_mom20, sig_mom40):
        s[np.isnan(pd.Series(s).values)] = 0
        s[:60] = 0

    rules = [
        (sig_ma20, 'R1 收盘>MA20 持有'),
        (sig_ma60, 'R2 收盘>MA60 持有'),
        (sig_ma120, 'R3 收盘>MA120 持有'),
        (sig_cross, 'R4 MA20>MA60 持有'),
        (sig_mom20, 'R5 20日动量>0 持有'),
        (sig_mom40, 'R6 40日动量>0 持有'),
        (np.ones(n), 'BH 买入持有'),
    ]

    for cost, lab in [(0.0010, '双边 0.10%'), (0.0005, '双边 0.05%'), (0.0020, '双边 0.20%')]:
        print('=' * 108)
        print(f'成本 {lab}')
        print('=' * 108)
        print(f'{"规则":<24} {"总收益":>9} {"年化":>7} {"最大回撤":>9} {"年化波动":>9} '
              f'{"夏普":>6} {"交易次数":>9} {"持仓占比":>9} {"次数/年":>8}')
        for sig, name in rules:
            s = run(d, sig, cost, name)
            print(f'{name:<24} {s["total"]*100:8.1f}% {s["cagr"]*100:6.2f}% '
                  f'{s["mdd"]*100:8.1f}% {s["vol"]*100:8.2f}% {s["sharpe"]:6.2f} '
                  f'{s["trades"]:9.0f} {s["expo"]*100:8.1f}% {s["trades"]/yrs:8.1f}')
        print()

    # ---------- 可预测性 ----------
    print('=' * 80)
    print('这些波动可预测吗？（用过去推未来的能力）')
    print('=' * 80)
    r = np.diff(np.log(c))
    # 自相关
    print('日收益率自相关：')
    for lag in [1, 2, 5, 10, 20, 40]:
        if len(r) > lag + 10:
            ac = np.corrcoef(r[:-lag], r[lag:])[0, 1]
            se = 1 / np.sqrt(len(r) - lag)
            print(f'  lag={lag:3d} 日: ρ={ac:+.4f}  (2σ={2*se:.4f}) '
                  f'{"显著" if abs(ac) > 2*se else "不显著"}')
    print()
    # 40 日收益对下一个 40 日收益的预测
    print('40 日收益 → 下一个 40 日收益：')
    w = 40
    x = c[w:-w] / c[:-2 * w] - 1
    y = c[2 * w:] / c[w:-w] - 1
    print(f'  ρ={np.corrcoef(x, y)[0,1]:+.4f}   n={len(x)}  '
          f'{"负（均值回复）" if np.corrcoef(x,y)[0,1] < 0 else "正（动量）"}')
    b = np.polyfit(x, y, 1)
    print(f'  斜率 {b[0]:+.4f}  R² {np.corrcoef(x,y)[0,1]**2*100:.2f}%')
    print()
    # 波动率聚集：|r| 的自相关
    print('|日收益| 自相关（波动聚集）：')
    ar = np.abs(r)
    for lag in [1, 5, 20, 40]:
        ac = np.corrcoef(ar[:-lag], ar[lag:])[0, 1]
        print(f'  lag={lag:3d}: ρ={ac:+.4f}')

    # ---------- 两个月波动的分布细节 ----------
    print()
    print('=' * 80)
    print('如果做波段：每笔能赚多少（事后最优，天花板）')
    print('=' * 80)
    from d130_probe_wave import zigzag
    for thr in [0.08, 0.10]:
        segs = zigzag(c, thr)
        segs = [s for s in segs if s[1] - s[0] >= 3]
        amps = np.array([s[3] for s in segs])
        # 假设每次都能吃满一段（不可能，但看上限）
        gross = np.prod(1 + amps[amps > 0])
        print(f'阈值 {thr*100:.0f}%: {len(segs)} 段，其中上涨段 {sum(amps>0)} 个、下跌段 {sum(amps<0)} 个')
        print(f'  上涨段幅度中位 {np.median(amps[amps>0])*100:.1f}%、均值 {amps[amps>0].mean()*100:.1f}%')
        print(f'  若每次精确吃满所有上涨段（不可能）：累计 {(gross-1)*100:,.0f}%')


if __name__ == '__main__':
    main()
