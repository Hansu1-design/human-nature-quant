# -*- coding: utf-8 -*-
"""d24 共同因子检验（Hermes R32 Q4 三条 + Q7 gdhs 混杂检验）

① 规模分层：若 R1 与 κ 的驼峰只在小盘出现、大盘没有，共同因子＝散户参与
② 股东户数（gdhs）：散户参与度直接代理，与 R1 是否同向
③ 月度同步：κ 与 R 在月度上是否同相；只在年度聚合相关＝聚合假象

gdhs 用途已按 Hermes 写死：**只能杀不能立**
  |corr| > 0.6  -> R1 被判混杂，作废
  0.3 ~ 0.6     -> 无法判定
  < 0.3         -> 仅「未被混杂」，不得升级为「人性变了」的证据
gdhs 只覆盖 2013Q1 起、季度频率；2011-2012 缺，不许拿它补那两年。
"""
import os, json
import numpy as np, pandas as pd, pyarrow.parquet as pq

OUT = os.path.dirname(os.path.abspath(__file__))
GD = "G:/AStockData/supplement/youzi_history/gdhs.parquet"

def kappa(a, b, K=3):
    M = np.zeros((K, K))
    for i, j in zip(a, b):
        if 0 <= i < K and 0 <= j < K:
            M[i, j] += 1
    n = M.sum()
    if n == 0:
        return np.nan
    Po = np.trace(M) / n
    Pe = (M.sum(1) @ M.sum(0)) / n ** 2
    return (Po - Pe) / (1 - Pe) if Pe < 1 else np.nan

print("=" * 60)
print("【①③】月度 κ（换手态驻留）与月度 R1（处置效应比）")
P = pq.read_table(os.path.join(OUT, "cache_pairs.parquet")).to_pandas()
mk = []
for (y, m), s in P.groupby(["year", "month"]):
    if len(s) < 3000:
        continue
    mk.append((pd.Timestamp(year=int(y), month=int(m), day=1), kappa(s["sa"].values, s["sb"].values)))
MK = pd.DataFrame(mk, columns=["m", "kappa"]).set_index("m").sort_index()

R1 = json.load(open(os.path.join(OUT, "d23_result.json"), encoding="utf-8"))["monthly"]
MR = pd.Series({pd.Timestamp(k + "-01"): v for k, v in R1.items()}).sort_index()
J = pd.DataFrame({"k": MK["kappa"], "r": MR}).dropna()
print("  月度重叠 %d 个月：%s ~ %s" % (len(J), J.index[0].date(), J.index[-1].date()))
print("  月频水平相关 r=%+.3f" % J["k"].corr(J["r"]))
for lag in (1, 2, 3, 6, 12):
    print("  κ(t) vs R(t+%d) r=%+.3f" % (lag, J["k"].corr(J["r"].shift(-lag))))
JA = J.groupby(J.index.year).median()
print("  年度聚合相关 r=%+.3f (n=%d)" % (JA["k"].corr(JA["r"]), len(JA)))
print("  κ 年度序列:", {int(y): round(v, 3) for y, v in JA["k"].items()})
print("  R 年度序列:", {int(y): round(v, 3) for y, v in JA["r"].items()})

print()
print("=" * 60)
print("【①】规模分层：R1 的驼峰在大盘/中盘/小盘里各是什么样")
CS = pq.read_table(os.path.join(OUT, "d23_cross.parquet")).to_pandas()
CS["ter"] = CS.groupby("yr")["mcap"].transform(
    lambda s: pd.qcut(s.rank(method="first"), 3, labels=["小", "中", "大"]))
tab = CS.groupby(["yr", "ter"])["R"].median().unstack()
print(tab.round(3).to_string())
for t in ["小", "中", "大"]:
    v = tab[t].dropna()
    x = v.index.values.astype(float)
    sl = np.polyfit(x, v.values, 1)[0]
    print("  %s盘 斜率 %+.5f/yr  首末 %.3f -> %.3f" % (t, sl, v.iloc[0], v.iloc[-1]))

print()
print("=" * 60)
print("【②】gdhs 股东户数（散户参与代理）")
g = pq.read_table(GD).to_pandas()
g["d"] = pd.to_datetime(g["统计截止日"].astype(str), errors="coerce")
g["hn"] = pd.to_numeric(g["股东户数-增减比例"].astype("float64"), errors="coerce")
g = g[g["hn"].notna() & g["d"].notna()]
g["q"] = g["d"].dt.to_period("Q")
qd = g.groupby("q")["hn"].median()
print("  季度中位户数变动%%, 样本 %d 季度：%s ~ %s" % (len(qd), qd.index[0], qd.index[-1]))
print("  ", {str(k): round(v, 2) for k, v in list(qd.items())[-10:]})
ga = pd.Series({int(str(q)[:4]): v for q, v in qd.groupby(qd.index.year).mean().items()})
print("  年度序列:", {k: round(v, 2) for k, v in ga.items()})

CSy = CS.groupby("yr")["R"].median()
com = sorted(set(CSy.index.astype(int)) & set(ga.index.astype(int)))
r = np.corrcoef([CSy[y] for y in com], [ga[y] for y in com])[0, 1]
print("  R1 年度 vs 户数年度变动 相关 r=%+.3f (n=%d, 年: %s)" % (r, len(com), com))
if abs(r) > 0.6:
    v = "混杂成立 -> R1 作废"
elif abs(r) < 0.3:
    v = "未被混杂（仅此而已，不得升级为『人性变了』的证据）"
else:
    v = "无法判定"
print("  预注册阈值判决:", v)

print()
print("【②补】户数变动与 R1 的同期相关是否只是「户数受送转扰动」——看户均持股是否同向")
gd = g.groupby("d")["hn"].median()
print("  季度口径全样本中位 %.2f%%  正值占比 %.1f%%"
      % (qd.mean(), 100 * (qd > 0).mean()))
