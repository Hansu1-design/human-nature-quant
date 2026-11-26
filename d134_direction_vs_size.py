# -*- coding: utf-8 -*-
"""
d134 两个月波动的真正规律：方向 vs 幅度
- 方向：40 日动量预测力，分段稳定性
- 幅度：波动率聚集 —— 用过去波动预测未来波动
- 对持仓者的意义：进场后 2 个月级别回撤分布
"""
import numpy as np
import pandas as pd
from d130_probe_wave import load, zigzag

pd.set_option('display.width', 240)


def ols_r2(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 10:
        return np.nan, np.nan, 0
    b = np.polyfit(x, y, 1)
    pred = np.polyval(b, x)
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rho = np.corrcoef(x, y)[0, 1]
    return r2, rho, len(x)


def main():
    d = load()
    c = d['close'].values
    n = len(c)
    r = np.zeros(n)
    r[1:] = c[1:] / c[:-1] - 1

    # ================= 1. 方向：动量预测力分段 =================
    print('=' * 96)
    print('一、方向能不能预测？（40 日动量 → 未来 40 日收益）')
    print('=' * 96)
    print('  全样本是非重叠的（不重复计样本），分段看稳定性：')
    w = 40
    x = c[w:-w] / c[:-2 * w] - 1
    y = c[2 * w:] / c[w:-w] - 1
    dd = d['d'].iloc[2 * w:].reset_index(drop=True)
    r2, rho, nn = ols_r2(x, y)
    print(f'  全样本: n={nn}  ρ={rho:+.4f}  R²={r2*100:.2f}%  斜率 {np.polyfit(x,y,1)[0]:+.3f}')
    print()
    print(f'  {"区间":<22} {"n":>5} {"ρ":>8} {"R²":>7} {"判读":>8}')
    for a, b in [('1997-01-01', '2005-01-01'), ('2005-01-01', '2012-01-01'),
                 ('2012-01-01', '2019-01-01'), ('2019-01-01', '2026-12-31')]:
        m = (dd >= a) & (dd < b)
        if m.sum() > 20:
            r2k, rhok, nk = ols_r2(x[m.values], y[m.values])
            print(f'  {a[:7]}~{b[:7]}      {nk:5d} {rhok:+8.4f} {r2k*100:6.2f}% '
                  f'{"动量" if rhok>0.05 else ("反转" if rhok<-0.05 else "无效"):>8}')

    print()
    print('  换窗口（全样本，非重叠）：')
    print(f'  {"窗口":>6} {"n":>6} {"ρ":>8} {"R²":>7}')
    for w in [10, 20, 40, 60, 120]:
        xx = c[w:-w] / c[:-2 * w] - 1
        yy = c[2 * w:] / c[w:-w] - 1
        r2k, rhok, nk = ols_r2(xx, yy)
        print(f'  {w:6d} {nk:6d} {rhok:+8.4f} {r2k*100:6.2f}%')

    # ================= 2. 幅度：波动率聚集 =================
    print()
    print('=' * 96)
    print('二、幅度能不能预测？（过去波动 → 未来波动）')
    print('=' * 96)
    ar = np.abs(r)
    print('  这是"两个月波动"最重要的可用规律：方向随机，但波动的量级有惯性')
    print()
    print(f'  {"预测窗口":>10} {"未来20日波动 ρ":>16} {"R²":>7} {"未来40日波动 ρ":>16} {"R²":>7}')
    for look in [5, 10, 20, 40, 60]:
        past = pd.Series(ar).rolling(look).mean().values
        # 未来 20 日平均波动（不含当天）
        f20 = pd.Series(ar).rolling(20).mean().shift(-20).values
        f40 = pd.Series(ar).rolling(40).mean().shift(-40).values
        r2_20, rho_20, _ = ols_r2(past, f20)
        r2_40, rho_40, _ = ols_r2(past, f40)
        print(f'  过去{look:3d}日 {rho_20:+15.4f} {r2_20*100:6.2f}% {rho_40:+15.4f} {r2_40*100:6.2f}%')

    print()
    print('  分段稳定性（过去20日 → 未来20日波动）：')
    past20 = pd.Series(ar).rolling(20).mean().values
    f20 = pd.Series(ar).rolling(20).mean().shift(-20).values
    for a, b in [('1997-01-01', '2005-01-01'), ('2005-01-01', '2012-01-01'),
                 ('2012-01-01', '2019-01-01'), ('2019-01-01', '2026-12-31')]:
        m = (d.d >= a) & (d.d < b)
        r2k, rhok, nk = ols_r2(past20[m.values], f20[m.values])
        print(f'  {a[:7]}~{b[:7]}  n={nk:5d}  ρ={rhok:+.4f}  R²={r2k*100:5.2f}%')

    # 实用转化：分位数分箱
    print()
    print('  实用形态：按"过去 20 日波动"分四档，看未来 20 日实际波动')
    v = pd.DataFrame({'past': past20, 'fut': f20}).dropna()
    v['bin'] = pd.qcut(v['past'], 4, labels=['低', '偏低', '偏高', '高'])
    g = v.groupby('bin', observed=True).agg(
        n=('fut', 'size'), past_mean=('past', 'mean'), fut_mean=('fut', 'mean'),
        fut_p90=('fut', lambda x: np.percentile(x, 90)))
    print(f'  {"档":>5} {"n":>6} {"过去日均波动":>12} {"未来日均波动":>12} {"未来p90":>9}')
    for idx, row in g.iterrows():
        print(f'  {idx:>5} {row.n:6.0f} {row.past_mean*100:11.3f}% {row.fut_mean*100:11.3f}% '
              f'{row.fut_p90*100:8.3f}%')

    # 转化成年化波动率
    print()
    print('  换成"年化波动率"，更直观：')
    for idx, row in g.iterrows():
        print(f'  {idx:>5}档: 过去 {row.past_mean*np.sqrt(243)*100:5.1f}%/年 '
              f'→ 未来 {row.fut_mean*np.sqrt(243)*100:5.1f}%/年')

    # ================= 3. 进场后的两个月回撤 =================
    print()
    print('=' * 96)
    print('三、对持仓者的意义：买在底部，两个月内会先亏多少')
    print('=' * 96)
    events = ['2008-09-08', '2011-10-21', '2018-08-17', '2020-02-03',
              '2022-04-26', '2023-10-23']
    print(f'  {"进场":>12} {"进场点位":>9} {"20日内最低":>10} {"40日内最低":>10} '
          f'{"60日内最低":>10} {"40日最大回撤":>12}')
    for ev in events:
        i0 = d.index[d.d >= ev]
        if len(i0) == 0:
            continue
        i0 = int(i0[0])
        p0 = c[i0]
        out = [f'{ev:>12}', f'{p0:9.0f}']
        for win in [20, 40, 60]:
            seg = c[i0:i0 + win + 1]
            out.append(f'{(seg.min()/p0-1)*100:9.1f}%')
        # 40 日内从进场的最大回撤（只看最低点）
        seg = c[i0:i0 + 41]
        out.append(f'{(seg.min()/p0-1)*100:11.1f}%')
        print(' '.join(out))

    print()
    print('  另一面：进场后两个月内，最多能浮盈多少（会不会先被震出去）')
    print(f'  {"进场":>12} {"20日最高":>10} {"40日最高":>10} {"60日最高":>10}')
    for ev in events:
        i0 = d.index[d.d >= ev]
        if len(i0) == 0:
            continue
        i0 = int(i0[0])
        p0 = c[i0]
        out = [f'{ev:>12}']
        for win in [20, 40, 60]:
            seg = c[i0:i0 + win + 1]
            out.append(f'{(seg.max()/p0-1)*100:9.1f}%')
        print(' '.join(out))


if __name__ == '__main__':
    main()
