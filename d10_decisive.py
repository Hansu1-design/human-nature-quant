# -*- coding: utf-8 -*-
"""d10 最小可判决实验（Hermes R30 Q8）
指标：分年估计「换手态 → 5 日后换手态」转移矩阵的对角线驻留概率 D_t
统计量：① D_t 的 ACF(1) 及 t 值  ② 安慰剂比值 = 跨年差 / 同年内随机分裂差
判决：ACF(1) 显著 且 比值>2 → 活；ACF≈0 且 比值≈1 → 死
"""
import os, glob, json, sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

R = "G:/AStockData/supplement/daily_tencent_full"
COLS = ["date", "close", "volume", "turnover"]
NSTATE = 3
HORIZON = 5
WARM = 250

files = sorted(glob.glob(R + "/*.parquet"))
print("files=%d" % len(files), flush=True)

# 计数容器：年份 -> 3x3 转移矩阵
years = list(range(2010, 2026))
YI = {y: i for i, y in enumerate(years)}
C = np.zeros((len(years), NSTATE, NSTATE), dtype=np.int64)
# 分半年用（用于安慰剂：年内随机分裂）
CH = np.zeros((len(years), 2, NSTATE, NSTATE), dtype=np.int64)

bad = 0
for k, f in enumerate(files):
    try:
        df = pq.read_table(f, columns=COLS).to_pandas()
    except Exception:
        bad += 1
        continue
    if len(df) < WARM + HORIZON + 40:
        continue
    df = df.sort_values("date")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    t = pd.to_numeric(df["turnover"], errors="coerce").astype("float64") * 100.0
    v = pd.to_numeric(df["volume"], errors="coerce").astype("float64")
    t = t.where(v > 0)
    if t.notna().sum() < WARM + 60:
        continue
    # 自身 250 日分位的三档（用滚动分位阈值，避免逐行 rank）
    q1 = t.rolling(WARM, min_periods=WARM).quantile(1.0 / 3.0)
    q2 = t.rolling(WARM, min_periods=WARM).quantile(2.0 / 3.0)
    st = pd.Series(np.where(t.isna(), np.nan,
                            np.where(t <= q1, 0.0, np.where(t >= q2, 2.0, 1.0))), index=df.index)
    s5 = st.shift(-HORIZON)
    yr = df["date"].dt.year.values
    mon = df["date"].dt.month.values
    a = st.values
    b = s5.values
    m = ~(np.isnan(a) | np.isnan(b))
    m &= (yr >= 2010) & (yr <= 2025)
    if not m.any():
        continue
    ia = a[m].astype(np.int8)
    ib = b[m].astype(np.int8)
    iy = yr[m].astype(np.int16)
    ih = (mon[m] >= 7).astype(np.int8)  # 上/下半年
    for y in np.unique(iy):
        j = YI[int(y)]
        mm = iy == y
        np.add.at(C[j], (ia[mm], ib[mm]), 1)
        sub = ih[mm]
        for h in (0, 1):
            s = sub == h
            if s.any():
                np.add.at(CH[j, h], (ia[mm][s], ib[mm][s]), 1)
    if (k + 1) % 400 == 0:
        print("  %d/%d  bad=%d" % (k + 1, len(files), bad), flush=True)

print("done bad=%d" % bad, flush=True)

# ---- D_t：对角线驻留概率（按行加权 = 条件于出发态的加权平均驻留率）
def diag_stat(M):
    M = M.astype(float)
    n = M.sum()
    if n < 50:
        return np.nan, np.nan, n
    row = M.sum(axis=1)
    ok = row > 0
    if ok.sum() == 0:
        return np.nan, np.nan, n
    stay = (np.diag(M)[ok] / row[ok])
    w = row[ok] / row[ok].sum()
    return float((stay * w).sum()), float(np.diag(M).sum() / n), n

D, Draw, N = [], [], []
for i, y in enumerate(years):
    d, dr, n = diag_stat(C[i])
    D.append(d); Draw.append(dr); N.append(n)
    print("  %d  D=%.4f  raw=%.4f  n=%,d" % (y, d, dr, n) if False else "  %d  D=%.4f  raw=%.4f  n=%d" % (y, d, dr, n))

D = np.array(D, dtype=float)
N = np.array(N, dtype=float)

# 各年转移矩阵（归一化）
P = C.astype(float)
for i in range(len(years)):
    s = P[i].sum(axis=1, keepdims=True)
    P[i] = np.divide(P[i], s, out=np.zeros_like(P[i]), where=s > 0)

def acf1(x):
    x = np.asarray(x, float)
    ok = ~np.isnan(x)
    x = x[ok]
    n = len(x)
    if n < 6:
        return np.nan, np.nan
    x = x - x.mean()
    num = (x[:-1] * x[1:]).sum()
    den = (x * x).sum()
    r = num / den if den > 0 else np.nan
    if np.isnan(r):
        return np.nan, np.nan
    # n-1 个配对
    tstat = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return float(r), float(tstat)

r1, t1 = acf1(D)
print("\n[统计量1] D_t 序列 =", np.round(D, 4).tolist())
print("  ACF(1) = %.4f   t = %.3f   n=%d" % (r1, t1, len(D)))

# ---- 安慰剂：同一年内随机分裂（上下半年各估一个 R，看它们的差）
def dist(A, B):
    """两矩阵的 Frobenius 距离（归一化后，按行等权）"""
    A = np.asarray(A, float); B = np.asarray(B, float)
    if np.isnan(A).any() or np.isnan(B).any():
        return np.nan
    return float(np.sqrt(((A - B) ** 2).mean()))

def norm_rows(M):
    M = M.astype(float)
    s = M.sum(axis=1, keepdims=True)
    return np.divide(M, s, out=np.zeros_like(M), where=s > 0), (M.sum(axis=1) > 0)

placebo = []
for i in range(len(years)):
    A, okA = norm_rows(CH[i, 0])
    B, okB = norm_rows(CH[i, 1])
    ok = okA & okB
    if ok.sum() < 2:
        continue
    d = dist(A[ok], B[ok])
    if not np.isnan(d):
        placebo.append(d)

real = []
for i in range(len(years) - 1):
    ok = (C[i].sum(axis=1) > 0) & (C[i + 1].sum(axis=1) > 0)
    if ok.sum() < 2:
        continue
    d = dist(P[i][ok], P[i + 1][ok])
    if not np.isnan(d):
        real.append(d)

pl = np.array(placebo, float)
re = np.array(real, float)
ratio_mean = re.mean() / pl.mean()
ratio_med = np.median(re) / np.median(pl)
# bootstrap：跨年差是否显著大于同年内分裂差
rng = np.random.default_rng(20261111)
bs = []
for _ in range(5000):
    a = rng.choice(re, len(re), replace=True).mean()
    b = rng.choice(pl, len(pl), replace=True).mean()
    bs.append(a - b)
bs = np.array(bs)
print("\n[统计量2] 安慰剂")
print("  同年内随机分裂差: mean=%.5f med=%.5f  n=%d" % (pl.mean(), np.median(pl), len(pl)))
print("  跨年差          : mean=%.5f med=%.5f  n=%d" % (re.mean(), np.median(re), len(re)))
print("  比值 mean=%.2f  median=%.2f" % (ratio_mean, ratio_med))
print("  bootstrap 跨年-同年: mean=%.5f  95%%CI=[%.5f, %.5f]  P(>0)=%.4f"
      % (bs.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (bs > 0).mean()))

# ---- 判决
print("\n" + "=" * 62)
print("判决（Hermes R30 Q8 判据：ACF(1) 显著 且 比值>2 → 活）")
print("=" * 62)
alive_acf = abs(t1) > 2
alive_ratio = ratio_med > 2
print("  ACF(1)=%.4f  |t|=%.2f  → %s" % (r1, abs(t1), "显著" if alive_acf else "不显著"))
print("  比值(median)=%.2f  → %s" % (ratio_med, ">2 过" if alive_ratio else "≤2 不过"))
if alive_acf and alive_ratio:
    print("  >>> 活：漂移有惯性，值得补血肉")
elif (not alive_acf) and ratio_med < 1.2:
    print("  >>> 死：既无稳定结构也无漂移结构，停手")
else:
    print("  >>> 降级：只拿到基准率表，不许叫理论")

out = {
    "years": years,
    "D": [None if np.isnan(x) else float(x) for x in D],
    "N": [int(x) for x in N],
    "P": P.tolist(),
    "acf1": r1, "t_acf1": t1,
    "placebo_mean": float(pl.mean()), "real_mean": float(re.mean()),
    "ratio_mean": float(ratio_mean), "ratio_median": float(ratio_med),
    "boot_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
    "boot_p": float((bs > 0).mean()),
    "placebo": [float(x) for x in pl],
    "real": [float(x) for x in re],
    "nstate": NSTATE, "horizon": HORIZON, "warm": WARM,
}
json.dump(out, open("d10_result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved d10_result.json")
