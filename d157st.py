# -*- coding: utf-8 -*-
"""
d157 个股是否也有"两个月周期"
先生：我看大盘、个股，基本两个月一个周期。
数据：本地 G:/AStockData/supplement/daily_tencent_full（3179 只，2010 起日线，不复权）
      不复权会有除权跳空，剔除 |月收益|>60% 的异常月
检验：段长度分布 vs 打乱顺序（KS）、lag1/lag2 自相关、2 个月段占比
"""
import glob
import os
import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(20261112)


def runs_of(sign):
    out, cur = [], 1
    for i in range(1, len(sign)):
        if sign[i] == sign[i - 1]:
            cur += 1
        else:
            out.append(cur)
            cur = 1
    out.append(cur)
    return np.array(out)


def main():
    files = sorted(glob.glob('G:/AStockData/supplement/daily_tencent_full/*.parquet'))
    if not files:
        print('未找到本地个股数据')
        return
    rng.shuffle(files)
    files = files[:400]
    rows, fails = [], []
    for fp in files:
        code = os.path.basename(fp)[:6]
        try:
            df = pd.read_parquet(fp)
            df['date'] = pd.to_datetime(df['date'])
            s = df.set_index('date')['close'].astype(float)
            m = s.resample('ME').last().dropna()
            r = m.pct_change().dropna().values
            r = r[np.abs(r) <= 0.60]
            if len(r) < 100:
                fails.append(code)
                continue
            L = runs_of(np.sign(r))
            ac1 = np.corrcoef(r[:-1], r[1:])[0, 1]
            ac2 = np.corrcoef(r[:-2], r[2:])[0, 1]
            sim = []
            for _ in range(200):
                sim += list(runs_of(np.sign(rng.permutation(r))))
            ks = stats.ks_2samp(L, np.array(sim))
            rows.append(dict(code=code, n=len(r), med=float(np.median(L)),
                             mean=float(L.mean()), p2=float((L == 2).mean()),
                             ac1=ac1, ac2=ac2, ks_p=ks.pvalue, ks_D=ks.statistic))
        except Exception:
            fails.append(code)
    R = pd.DataFrame(rows)
    print(f'成功 {len(R)} 只，失败 {len(fails)} 只')
    if len(R) == 0:
        return
    print(f'样本月数：中位 {R["n"].median():.0f}（最少 {R["n"].min()}，最多 {R["n"].max()}）')

    print('\n' + '=' * 100)
    print('一、个股段长度（连续同向月的段）：是否集中在 2 个月？')
    print('=' * 100)
    print(f'  段长中位 across 个股：{R.med.median():.1f} 月'
          f'（p25 {R.med.quantile(.25):.1f} / p75 {R.med.quantile(.75):.1f}）')
    print(f'  段长均值：{R["mean"].median():.2f} 月')
    print(f'  "2 个月段"占比：中位 {R.p2.median()*100:.1f}%'
          f'（p25 {R.p2.quantile(.25)*100:.1f}% / p75 {R.p2.quantile(.75)*100:.1f}%）')

    print('\n' + '=' * 100)
    print('二、与打乱顺序对照（KS）：p<0.05 表示分布与随机不同')
    print('=' * 100)
    sig = int((R.ks_p < 0.05).sum())
    print(f'  {len(R)} 只中 {sig} 只显著（{sig/len(R)*100:.0f}%），'
          f'随机期望约 5%（{len(R)*0.05:.0f} 只）')
    print(f'  KS D 中位 {R.ks_D.median():.3f}，p 中位 {R.ks_p.median():.3f}')

    print('\n' + '=' * 100)
    print('三、月度自相关：lag2 显著为正才是"两个月周期"的证据')
    print('=' * 100)
    band = 1.96 / np.sqrt(R["n"].median())
    print(f'  噪声带约 ±{band:.3f}')
    print(f'  lag1 中位 {R.ac1.median():+.3f}；lag1 显著为负 {(R.ac1 < -band).sum()} 只，'
          f'显著为正 {(R.ac1 > band).sum()} 只')
    print(f'  lag2 中位 {R.ac2.median():+.3f}；lag2 显著为正 {(R.ac2 > band).sum()} 只，'
          f'显著为负 {(R.ac2 < -band).sum()} 只')

    print('\n' + '=' * 100)
    print('四、逐只明细（按段长中位排序，前 25 + 后 5）')
    print('=' * 100)
    print(f'{"代码":<8}{"月数":>5}{"段长中位":>9}{"均值":>7}{"2月段":>8}'
          f'{"lag1":>8}{"lag2":>8}{"KS_p":>8}')
    for _, r in pd.concat([R.sort_values('med').head(25),
                           R.sort_values('med').tail(5)]).iterrows():
        print(f'{r.code:<8}{r["n"]:>5.0f}{r.med:>9.1f}{r["mean"]:>7.2f}'
              f'{r.p2*100:>7.1f}%{r.ac1:>+8.3f}{r.ac2:>+8.3f}{r.ks_p:>8.3f}')
    R.to_csv('d157_stocks.csv', index=False, encoding='utf-8-sig')
    print(f'\n明细已存 d157_stocks.csv（{len(R)} 只）')


if __name__ == '__main__':
    main()
