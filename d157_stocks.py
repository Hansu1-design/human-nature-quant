# -*- coding: utf-8 -*-
"""
d157 个股是否也有"两个月周期"
先生：我看大盘、个股，基本两个月一个周期。
本脚本对一批个股（月线，后复权）做与指数相同的检验：
  段长度分布 vs 打乱顺序、lag1/lag2 自相关、2 个月段占比
"""
import numpy as np
import pandas as pd
from scipy import stats
import glob, os

rng = np.random.default_rng(20261112)

CODES = ['000001', '000002', '000063', '000333', '000651', '000858', '002415', '002594',
         '300059', '300750', '600000', '600009', '600016', '600028', '600030', '600036',
         '600048', '600050', '600104', '600276', '600309', '600519', '600585', '600690',
         '600809', '600887', '601012', '601088', '601138', '601166', '601318', '601398',
         '601601', '601628', '601668', '601688', '601857', '601888', '601899', '603259']


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
    rows = []
    fails = []
    files = sorted(glob.glob('G:/AStockData/supplement/daily_tencent_full/*.parquet'))
    rng2 = np.random.default_rng(7)
    rng2.shuffle(files)
    files = files[:400]
    for k, fp in enumerate(files):
        code = os.path.basename(fp)[:6]
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="monthly",
                                    start_date="20050101", adjust="hfq")
            if df is None or len(df) < 60:
                fails.append(code)
                continue
            c = pd.to_numeric(df['收盘'], errors='coerce').dropna().values
            r = c[1:] / c[:-1] - 1
            if len(r) < 60:
                fails.append(code)
                continue
            L = runs_of(np.sign(r))
            ac1 = np.corrcoef(r[:-1], r[1:])[0, 1]
            ac2 = np.corrcoef(r[:-2], r[2:])[0, 1]
            # 打乱对照
            sim = []
            for _ in range(300):
                sim += list(runs_of(np.sign(rng.permutation(r))))
            ks = stats.ks_2samp(L, np.array(sim))
            rows.append(dict(code=code, n=len(r), med=np.median(L), mean=L.mean(),
                             p2=(L == 2).mean(), ac1=ac1, ac2=ac2,
                             ks_p=ks.pvalue, ks_D=ks.statistic))
        except Exception as e:
            fails.append(code)
    R = pd.DataFrame(rows)
    print(f'成功 {len(R)} 只，失败 {len(fails)} 只 {fails}')
    print(f'样本月数：中位 {R.n.median():.0f}（最少 {R.n.min()}，最多 {R.n.max()}）')

    print('\n' + '=' * 100)
    print('个股段长度（连续同向月的段）：是否集中在 2 个月？')
    print('=' * 100)
    print(f'  段长中位 across 个股：{R.med.median():.1f} 月'
          f'（p25 {R.med.quantile(.25):.1f} / p75 {R.med.quantile(.75):.1f}）')
    print(f'  段长均值：{R["mean"].median():.2f} 月')
    print(f'  "2 个月段"占比：中位 {R.p2.median()*100:.1f}%'
          f'（p25 {R.p2.quantile(.25)*100:.1f}% / p75 {R.p2.quantile(.75)*100:.1f}%）')

    print('\n' + '=' * 100)
    print('与打乱顺序对照（KS 检验）：p<0.05 表示"分布与随机不同"')
    print('=' * 100)
    sig = (R.ks_p < 0.05).sum()
    print(f'  {len(R)} 只中，{sig} 只显著（{sig/len(R)*100:.0f}%）；'
          f'随机期望 5%（{len(R)*0.05:.1f} 只）')
    print(f'  KS D 中位 {R.ks_D.median():.3f}，p 值中位 {R.ks_p.median():.3f}')

    print('\n' + '=' * 100)
    print('月度自相关：lag1（反转倾向）/ lag2（两个月周期会在 lag2 显正）')
    print('=' * 100)
    band = 1.96 / np.sqrt(R.n.median())
    print(f'  个股 lag1 中位 {R.ac1.median():+.3f}，lag2 中位 {R.ac2.median():+.3f}'
          f'（噪声带约 ±{band:.3f}）')
    print(f'  lag1 显著为负的个股：{(R.ac1 < -band).sum()} 只；'
          f'显著为正：{(R.ac1 > band).sum()} 只')
    print(f'  lag2 显著为正的个股：{(R.ac2 > band).sum()} 只；'
          f'显著为负：{(R.ac2 < -band).sum()} 只')
    print(f'  → 若"两个月周期"普遍存在，lag2 应普遍显著为正')

    print('\n' + '=' * 100)
    print('逐只明细（按段长中位排序）')
    print('=' * 100)
    print(f'{"代码":<8}{"月数":>5}{"段长中位":>9}{"段长均值":>9}{"2月段占比":>10}'
          f'{"lag1":>8}{"lag2":>8}{"KS_p":>8}')
    for _, r in R.sort_values('med').iterrows():
        print(f'{r.code:<8}{r.n:>5.0f}{r.med:>9.1f}{r["mean"]:>9.2f}{r.p2*100:>9.1f}%'
              f'{r.ac1:>+8.3f}{r.ac2:>+8.3f}{r.ks_p:>8.3f}')
    R.to_csv('d157_stocks.csv', index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()        try:
            df = pd.read_parquet(fp)
            df['date'] = pd.to_datetime(df['date'])
            s2 = df.set_index('date')['close'].astype(float)
            m = s2.resample('ME').last().dropna()
            r = m.pct_change().dropna().values
            r = r[np.abs(r) <= 0.60]              # 剔除除权跳空造成的假暴跌
            if len(r) < 100:
                fails.append(code)
                continue

