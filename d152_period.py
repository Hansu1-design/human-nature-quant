# -*- coding: utf-8 -*-
"""
d152 两个月周期存在性检验
先生观察：大盘/个股基本两个月一个周期，强转弱、弱转强。
本脚本只回答一个问题：这个周期是真的，还是"涨跌各半"产生的必然外观？

三层检验：
  1) 段长度分布 —— 实测 vs 打乱顺序（保留收益率分布、破坏时间结构）
  2) 自相关 / 条件概率 —— 连涨 k 月后下月转跌的概率，与基准比
  3) 谱分析 —— 月度收益在 2 个月频率上是否有峰
再加一层：zigzag 波段长度（阈值 5%/8%），直接量"一个波段多久"
"""
import numpy as np
import pandas as pd
from scipy import stats, signal

pd.set_option('display.width', 240)
rng = np.random.default_rng(20261112)


def load():
    a = pd.read_parquet('idx_sse_all.parquet')
    a['d'] = pd.to_datetime(a['date'])
    a = a[['d', 'close']].sort_values('d').reset_index(drop=True)
    a['close'] = pd.to_numeric(a['close'], errors='coerce')
    return a.dropna().reset_index(drop=True)


def month_ret(d):
    s = d.set_index('d')['close']
    m = s.resample('ME').last()
    r = m.pct_change().dropna()
    return m, r


def run_lengths(sign):
    """连续同号段长度"""
    out, cur = [], 1
    for i in range(1, len(sign)):
        if sign[i] == sign[i - 1]:
            cur += 1
        else:
            out.append((sign[i - 1], cur))
            cur = 1
    out.append((sign[-1], cur))
    return out


def zigzag(c, thr=0.05):
    """波段切分：反向幅度须超过 thr 才确认转折。返回 [(方向, 起点i, 终点i, 天数)]"""
    n = len(c)
    segs = []
    # 初始方向由首个 thr 级波动决定
    ext_i, ext_p = 0, c[0]
    dirn = 0
    start_i = 0
    for i in range(1, n):
        if dirn == 0:
            if c[i] / ext_p - 1 >= thr:
                dirn, start_i = 1, ext_i
                ext_i, ext_p = i, c[i]
            elif c[i] / ext_p - 1 <= -thr:
                dirn, start_i = -1, ext_i
                ext_i, ext_p = i, c[i]
            elif c[i] > ext_p:
                ext_i, ext_p = i, c[i]
            elif c[i] < c[ext_i]:
                ext_i, ext_p = i, c[i]
        elif dirn == 1:
            if c[i] > ext_p:
                ext_i, ext_p = i, c[i]
            elif c[i] / ext_p - 1 <= -thr:
                segs.append((1, start_i, ext_i))
                dirn, start_i = -1, ext_i
                ext_i, ext_p = i, c[i]
        else:
            if c[i] < ext_p:
                ext_i, ext_p = i, c[i]
            elif c[i] / ext_p - 1 >= thr:
                segs.append((-1, start_i, ext_i))
                dirn, start_i = 1, ext_i
                ext_i, ext_p = i, c[i]
    segs.append((dirn, start_i, ext_i))
    return segs


def main():
    d = load()
    px = d['close'].values
    n = len(px)
    yrs = (d['d'].iloc[-1] - d['d'].iloc[0]).days / 365.25
    m, r = month_ret(d)
    rr = r.values
    print(f'样本：上证 {d.d.iloc[0].date()} ~ {d.d.iloc[-1].date()}，{yrs:.1f} 年，{n} 交易日，{len(rr)} 个月')

    # ================= 一、月度收益自相关 =================
    print('\n' + '=' * 100)
    print('一、月度收益自相关（负=反转倾向，正=延续倾向）')
    print('=' * 100)
    band = 1.96 / np.sqrt(len(rr))
    print(f'  随机噪声 95% 置信带 ±{band:.3f}')
    print(f'  {"lag":>4} {"自相关":>9} {"p值":>8}  判断')
    for L in range(1, 13):
        x, y = rr[:-L], rr[L:]
        ac = np.corrcoef(x, y)[0, 1]
        # Ljung-Box 单点近似
        p = 2 * (1 - stats.norm.cdf(abs(ac) * np.sqrt(len(rr))))
        flag = '显著' if abs(ac) > band else '噪声内'
        print(f'  {L:>4} {ac:>+9.3f} {p:>8.3f}  {flag}')

    # ================= 二、段长度分布 vs 打乱 =================
    print('\n' + '=' * 100)
    print('二、月线连涨/连跌段长度：实测 vs 打乱顺序（10000 次）')
    print('=' * 100)
    sign = np.sign(rr)
    obs = run_lengths(sign)
    L_obs = np.array([x[1] for x in obs])
    print(f'  实测：{len(obs)} 段，均值 {L_obs.mean():.2f} 月，中位 {np.median(L_obs):.0f} 月，'
          f'分布 {dict(pd.Series(L_obs).value_counts().sort_index().head(8))}')
    sim_means, sim_meds, sim_p2, sim_max = [], [], [], []
    sim_dist = np.zeros(12)
    for _ in range(10000):
        sh = rng.permutation(rr)
        s2 = run_lengths(np.sign(sh))
        L2 = np.array([x[1] for x in s2])
        sim_means.append(L2.mean())
        sim_meds.append(np.median(L2))
        sim_p2.append((L2 == 2).mean())
        sim_max.append(L2.max())
        vc = pd.Series(L2).value_counts()
        for k in range(1, 13):
            sim_dist[k - 1] += vc.get(k, 0) / len(L2)
    sim_dist /= 10000
    print(f'  打乱：均值 {np.mean(sim_means):.2f}±{np.std(sim_means):.2f} 月，'
          f'中位 {np.mean(sim_meds):.1f} 月')
    p_mean = np.mean(np.array(sim_means) >= L_obs.mean())
    print(f'  → 实测均值在打乱分布中的位置：p = {p_mean:.3f}（>.05 即与随机无异）')
    print(f'\n  {"段长":>4} {"实测占比":>10} {"打乱占比":>10} {"差":>8}')
    for k in range(1, 9):
        o = (L_obs == k).mean()
        print(f'  {k:>4} {o*100:>9.1f}% {sim_dist[k-1]*100:>9.1f}% {(o-sim_dist[k-1])*100:>+7.1f}pp')
    ks = stats.ks_2samp(L_obs, rng.permutation(rr)[:0]) if False else None
    # KS：实测 vs 一次大规模打乱拼接
    big = []
    for _ in range(200):
        s2 = run_lengths(np.sign(rng.permutation(rr)))
        big += [x[1] for x in s2]
    ks = stats.ks_2samp(L_obs, np.array(big))
    print(f'  KS 检验（实测 vs 打乱）：D={ks.statistic:.3f}, p={ks.pvalue:.3f}'
          f'{"  → 分布无差异" if ks.pvalue > 0.05 else "  → 分布有差异"}')

    # ================= 三、条件概率 =================
    print('\n' + '=' * 100)
    print('三、连涨 k 月后，下个月转跌的概率（周期若真，应随 k 上升）')
    print('=' * 100)
    base = (rr < 0).mean()
    print(f'  基准：月跌概率 {base*100:.1f}%')
    print(f'  {"连涨k月":>8} {"样本数":>7} {"下月跌概率":>11} {"vs基准":>9} {"二项检验p":>10}')
    for k in range(1, 6):
        cnt, hit = 0, 0
        for i in range(k, len(rr)):
            if all(rr[i - j] > 0 for j in range(1, k + 1)):
                cnt += 1
                if rr[i] < 0:
                    hit += 1
        if cnt < 8:
            print(f'  {k:>8} {cnt:>7}  （样本不足）')
            continue
        pr = hit / cnt
        p = stats.binomtest(hit, cnt, base).pvalue
        print(f'  {k:>8} {cnt:>7} {pr*100:>10.1f}% {(pr-base)*100:>+8.1f}pp {p:>10.3f}')
    print()
    print(f'  {"连跌k月":>8} {"样本数":>7} {"下月涨概率":>11} {"vs基准":>9} {"二项检验p":>10}')
    base_up = (rr > 0).mean()
    for k in range(1, 6):
        cnt, hit = 0, 0
        for i in range(k, len(rr)):
            if all(rr[i - j] < 0 for j in range(1, k + 1)):
                cnt += 1
                if rr[i] > 0:
                    hit += 1
        if cnt < 8:
            print(f'  {k:>8} {cnt:>7}  （样本不足）')
            continue
        pr = hit / cnt
        p = stats.binomtest(hit, cnt, base_up).pvalue
        print(f'  {k:>8} {cnt:>7} {pr*100:>10.1f}% {(pr-base_up)*100:>+8.1f}pp {p:>10.3f}')

    # ================= 四、谱分析 =================
    print('\n' + '=' * 100)
    print('四、月度收益谱分析：2 个月周期对应频率 0.5 次/月')
    print('=' * 100)
    x = rr - rr.mean()
    f, P = signal.periodogram(x, fs=1.0, detrend='linear')
    order = np.argsort(P)[::-1][:5]
    print(f'  {"频率(次/月)":>12} {"对应周期(月)":>12} {"功率占比":>9}')
    for o in order:
        per = 1 / f[o] if f[o] > 0 else np.inf
        print(f'  {f[o]:>12.3f} {per:>12.2f} {P[o]/P.sum()*100:>8.1f}%')
    # 2 个月频率处的功率 vs 全谱
    i2 = np.argmin(abs(f - 0.5))
    print(f'\n  2 个月频率(0.5) 处功率占比 {P[i2]/P.sum()*100:.2f}%；'
          f'随机序列期望占比约 {1/len(f)*100:.2f}%')

    # ================= 五、zigzag 波段长度 =================
    print('\n' + '=' * 100)
    print('五、日线 zigzag 波段长度（一个"波段"实际持续多久）')
    print('=' * 100)
    for thr in (0.05, 0.08, 0.10, 0.15):
        segs = zigzag(px, thr)
        if len(segs) < 6:
            print(f'  阈值 {thr*100:.0f}%：段数太少')
            continue
        lens = np.array([d['d'].iloc[b] for _, a, b in segs])
        dd = np.array([(d['d'].iloc[b] - d['d'].iloc[a]).days for _, a, b in segs])
        amp = np.array([px[b] / px[a] - 1 for _, a, b in segs])
        up = np.array([g for g, _, _ in segs]) == 1
        print(f'  阈值 {thr*100:>2.0f}%：{len(segs)} 段，'
              f'时长 中位 {np.median(dd):>4.0f} 日 / 均值 {dd.mean():>5.0f} 日；'
              f'幅度 中位 {np.median(abs(amp))*100:>4.1f}%')
        print(f'          上涨段中位 {np.median(dd[up]):>4.0f} 日（n={up.sum()}），'
              f'下跌段中位 {np.median(dd[~up]):>4.0f} 日（n={(~up).sum()}）')
        # 落在 40~60 日（2~3 个月）的比例
        near2 = ((dd >= 30) & (dd <= 65)).mean()
        print(f'          时长落在 30~65 日（约 1.5~3 个月）的占比：{near2*100:.1f}%')

    print('\n' + '=' * 100)
    print('结论提示：若"一、自相关全在噪声带内 + 二、KS 不显著 + 三、条件概率不随 k 上升'
          ' + 四、0.5 频率无峰" → 两个月周期不成立')


if __name__ == '__main__':
    main()
