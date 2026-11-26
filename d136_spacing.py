# -*- coding: utf-8 -*-
"""
d136 修正 + 落地
1) 用 CPI 实际点位算"离底 +30~40% 之后多久到 +10%"（修 d135 的口径错误）
2) 建仓间距：时间间距 vs 价格间距，在 6 轮上的实测
3) 波动档位 × 间距的交互
"""
import numpy as np
import pandas as pd
from d130_probe_wave import load

pd.set_option('display.width', 240)


def load_cpi(d):
    """按 d91 同口径自建 CPI 定基（2008-01=100）：'全国-当月' 是同比，开 12 次方当环比"""
    cpi = pd.read_parquet('G:/AStockData/supplement/macro/cpi.parquet')
    cpi['ym'] = cpi['月份'].str.replace('年', '-').str.replace('月份', '')
    cpi['d'] = pd.to_datetime(cpi['ym'], format='%Y-%m').astype('datetime64[ns]')
    cpi = cpi.sort_values('d').reset_index(drop=True)
    cpi['mom'] = (pd.to_numeric(cpi['全国-当月'], errors='coerce') / 100.0) ** (1 / 12.0)
    idx = [100.0]
    for i in range(1, len(cpi)):
        idx.append(idx[-1] * cpi['mom'].iloc[i])
    cpi['cpi_idx'] = idx
    cpi['avail'] = cpi['d'] + pd.offsets.MonthBegin(1) + pd.Timedelta(days=14)
    dd = d[['d']].copy()
    dd['d'] = dd['d'].astype('datetime64[ns]')
    m = pd.merge_asof(dd, cpi[['avail', 'cpi_idx']].dropna().sort_values('avail'),
                      left_on='d', right_on='avail', direction='backward')
    return m['cpi_idx'].values


def main():
    d = load()
    c = d['close'].values
    n = len(c)
    print('=' * 92)
    print('一、用 CPI 实际点位：离底 +30~40% 之后，多久到 +10%（出手区）')
    print('=' * 92)
    cpi_idx = load_cpi(d)
    real = c * 100.0 / cpi_idx
    ANCHOR = 1880.0
    rA = real / ANCHOR
    ok = np.isfinite(rA)
    print(f'  实际点位当前 {real[-1]:.0f}，离底 {rA[-1]*100-100:+.1f}%')
    print(f'  名义 {c[-1]:.0f}，CPI 定基 2008-01=100 现值 {cpi_idx[-1]:.1f}')
    print()
    # 时间/价格线按 2%/年抬（用 CPI 近似）——这里直接看 rA 序列
    waits, drops = [], []
    i = int(np.argmax(ok))          # CPI 起点
    starts = []
    while i < n:
        if ok[i] and 1.30 <= rA[i] <= 1.40:
            starts.append(i)
            j = i + 1
            while j < min(n, i + 1000) and (not ok[j] or rA[j] > 1.10):
                j += 1
            if j < min(n, i + 1000) and ok[j]:
                waits.append(j - i)
                drops.append(real[j] / real[i] - 1)
            i += 20      # 去重：同一次停留不重复计
        else:
            i += 1
    print(f'  样本点 {len(starts)} 个（20 日去重）')
    if waits:
        w = np.array(waits); dr = np.array(drops)
        print(f'  走到 +10% 的样本 {len(w)} 次')
        print(f'    耗时中位 {np.median(w):.0f} 交易日 ({np.median(w)/20.3:.1f} 月)  '
              f'p25 {np.percentile(w,25):.0f}  p75 {np.percentile(w,75):.0f}  最长 {w.max()}')
        print(f'    期间实际点位跌幅中位 {np.median(dr)*100:.1f}%  最深 {dr.min()*100:.1f}%')
        print(f'    1000 交易日内没走到的：{len(starts)-len(w)} 个')
    print()
    print('  当前 rA=%.2f，历史上 rA 从 1.30~1.40 走到 1.10 的中位耗时 = 上面那个数' % rA[-1])
    print('  注意：样本只有 2008 之后（CPI 起点），约 4~6 次独立机会')

    # ============ 2. 建仓间距：时间 vs 价格 ============
    print()
    print('=' * 92)
    print('二、建仓间距：时间间距 vs 价格间距（6 轮，36 个月窗口）')
    print('=' * 92)
    # 复用 d122 的框架：这里简化，只算"三笔等额买入后 36 个月的总收益 + 最大浮亏"
    events = ['2008-09-08', '2011-10-21', '2018-08-17', '2020-02-03',
              '2022-04-26', '2023-10-23']

    def simulate(i0, mode, param, cap=756):
        """mode='time' param=月数; mode='price' param=跌幅(如0.03)"""
        p0 = c[i0]
        buys = [i0]
        last = i0
        for i in range(i0 + 1, min(n, i0 + cap)):
            if len(buys) >= 3:
                break
            if mode == 'time':
                # 按月：20.3 交易日/月
                if i - last >= param * 20.3:
                    buys.append(i); last = i
            else:
                # 按价格：从上一笔跌 param
                if c[i] <= c[buys[-1]] * (1 - param):
                    buys.append(i); last = i
        # 等额买入，算 36 个月后（或数据末尾）的收益
        end = min(n - 1, i0 + cap)
        shares = sum(1.0 / c[b] for b in buys)     # 每笔投 1 元
        cost = len(buys) * 1.0
        nav_end = shares * c[end]
        ret = nav_end / cost - 1
        # 最大浮亏：从首次买入起到结束，账户净值最深
        navs = []
        for i in range(i0, end + 1):
            held = sum(1.0 / c[b] for b in buys if b <= i)
            cash = (3 - sum(1 for b in buys if b <= i)) * 1.0 / 3 * 0  # 假设未买的是现金，不计
            # 按已投入资金算净值
            inv = sum(1 for b in buys if b <= i) * 1.0
            if inv > 0:
                navs.append(held * c[i] / inv)
        mdd = min(navs) - 1 if navs else 0
        return ret, mdd, len(buys), (buys[-1] - i0) if buys else 0

    print(f'  {"方式":<20} {"收益均值":>9} {"收益中位":>9} {"浮亏均值":>9} {"平均买满耗时":>12}')
    plans = [('time', 1, '时间 1 月'), ('time', 2, '时间 2 月'), ('time', 3, '时间 3 月'),
             ('price', 0.02, '价格 -2%'), ('price', 0.03, '价格 -3%'),
             ('price', 0.05, '价格 -5%'), ('price', 0.08, '价格 -8%')]
    for mode, param, nm in plans:
        rets, mdds, nbuy, durs = [], [], [], []
        for ev in events:
            idx = d.index[d.d >= ev]
            if len(idx) == 0:
                continue
            i0 = int(idx[0])
            r, m, nb, dur = simulate(i0, mode, param)
            rets.append(r); mdds.append(m); nbuy.append(nb); durs.append(dur)
        print(f'  {nm:<20} {np.mean(rets)*100:8.1f}% {np.median(rets)*100:8.1f}% '
              f'{np.mean(mdds)*100:8.1f}% {np.mean(durs):11.0f}日')

    # ============ 3. 当前波动档下，各间距会怎样 ============
    print()
    print('=' * 92)
    print('三、当前是低波动档（年化 12.6%），三种间距实际会买在哪')
    print('=' * 92)
    r = np.zeros(n); r[1:] = c[1:] / c[:-1] - 1
    vol20 = pd.Series(np.abs(r)).rolling(20).mean().values * np.sqrt(243)
    print(f'  当前 20 日年化波动 {vol20[-1]*100:.1f}%')
    print()
    print('  按历史同档（低波动）的下跌速度推算，从今天起：')
    med_daily = 0.00747      # d135 算出的低档未来日均波动 0.747%
    for m in [1, 2, 3]:
        d20 = m * 20.3
        # 随机游走：m 个月后典型偏离 = 日均波动 * sqrt(天数) * sqrt(2/pi)
        typ = med_daily * np.sqrt(d20) * np.sqrt(2 / np.pi)
        print(f'    {m} 个月后，典型偏离 ±{typ*100:.1f}%（单日波动累积，非趋势）')
    print()
    print('  对比：中波动档日均 1.018%，高波动档 1.512%')
    for lab, md in [('低', 0.00747), ('中', 0.01018), ('高', 0.01512)]:
        row = []
        for m in [1, 2, 3]:
            typ = md * np.sqrt(m * 20.3) * np.sqrt(2 / np.pi)
            row.append(f'{typ*100:4.1f}%')
        print(f'    {lab}波动档：1月 {row[0]}  2月 {row[1]}  3月 {row[2]}')


if __name__ == '__main__':
    main()
