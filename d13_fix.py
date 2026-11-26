# -*- coding: utf-8 -*-
"""d13 修正判定：以「年」为独立单位做两两检验 + CUSUM 拐点
d12 的错误：用「95% 区间是否重叠」判断两组分布是否相同 —— 重叠≠无差异。
正确做法：以年为独立单位（每年 20 次分裂的均值作为该年估计），做 Welch t。
"""
import json
import numpy as np
from itertools import combinations

samp = json.load(open("d12_subsample_kappa.json", encoding="utf-8"))
years = sorted(int(k) for k in samp)
yb = {y: np.mean(samp[str(y)]) for y in years}          # 年度点估计
ysd = {y: np.std(samp[str(y)], ddof=1) for y in years}  # 年内分裂噪声（不作为独立单位误差）
vals = np.array([yb[y] for y in years])
print("年度 kappa（20 次分裂均值）:")
for y in years:
    print("  %d  %.4f" % (y, yb[y]))

def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return d, se, d / se

def seg(arr, y0, y1):
    return [yb[y] for y in years if y0 <= y <= y1]

print("\n[分段均值]")
segs = {"2011-2013": (2011, 2013), "2014-2018": (2014, 2018),
        "2019-2022": (2019, 2022), "2023-2025": (2023, 2025)}
for k, (a, b) in segs.items():
    v = seg(vals, a, b)
    print("  %-10s n=%d mean=%.4f sd=%.4f" % (k, len(v), np.mean(v), np.std(v, ddof=1)))

print("\n[两两 Welch t（以年为独立单位）]")
res = []
for (k1, (a1, b1)), (k2, (a2, b2)) in combinations(segs.items(), 2):
    v1, v2 = np.asarray(seg(vals, a1, b1), float), np.asarray(seg(vals, a2, b2), float)
    d, se, t = welch(v1, v2)
    # 自由度近似
    df = (v1.var(ddof=1)/len(v1) + v2.var(ddof=1)/len(v2))**2 / (
        (v1.var(ddof=1)/len(v1))**2/(len(v1)-1) + (v2.var(ddof=1)/len(v2))**2/(len(v2)-1))
    res.append((k1, k2, d, se, t, df))
    print("  %-10s vs %-10s  差=%+.4f  SE=%.4f  t=%+.2f  df=%.1f  %s"
          % (k1, k2, d, se, t, df, "显著" if abs(t) > 2 else "不显著"))

print("\n[全序列趋势]")
x = np.array(years, float)
sl, ic = np.polyfit(x, vals, 1)
pred = sl * x + ic
ss_res = ((vals - pred) ** 2).sum()
ss_tot = ((vals - vals.mean()) ** 2).sum()
se_slope = np.sqrt(ss_res / (len(x) - 2) / ((x - x.mean()) ** 2).sum())
print("  线性斜率 = %+.5f /年  (SE=%.5f, t=%.2f)  → 15 年累计 %+.4f"
      % (sl, se_slope, sl / se_slope, sl * 14))
print("  R² = %.3f" % (1 - ss_res / ss_tot))

print("\n[CUSUM 拐点（Pettitt 近似）]")
# Pettitt: 对每个 t，算 U_t = sum_{i<=t} sum_{j>t} sign(x_i - x_j)
n = len(vals)
U = np.zeros(n)
for t in range(n):
    s = 0
    for i in range(t + 1):
        for j in range(t + 1, n):
            s += np.sign(vals[i] - vals[j])
    U[t] = s
k = int(np.argmax(np.abs(U)))
p = 2 * np.exp(-6 * U[k] ** 2 / (n ** 3 + n ** 2))
print("  候选拐点: %d 年  U=%d  p≈%.4f" % (years[k], U[k], p))
print("  拐点前均值 %.4f (n=%d) / 拐点后均值 %.4f (n=%d)"
      % (vals[:k+1].mean(), k+1, vals[k+1:].mean(), n-k-1))

print("\n[判决修正]")
d, se, t = welch(seg(vals, 2011, 2013), seg(vals, 2019, 2022))
print("  2011-13 vs 2019-22: 差=%+.4f  t=%.2f" % (d, t))
print("  d12 用「区间重叠」判为「可由月际波动解释」→ 判据错误，作废。")
print("  正确判据（以年为独立单位）: %s" % ("差异显著" if abs(t) > 2 else "差异不显著"))

json.dump({"year_kappa": {str(y): float(yb[y]) for y in years},
           "slope": float(sl), "slope_t": float(sl / se_slope),
           "cusum_year": int(years[k]), "cusum_p": float(p),
           "pairwise": [(a, b, float(dd), float(tt)) for a, b, dd, _, tt, _ in res]},
          open("d13_result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved d13_result.json")
