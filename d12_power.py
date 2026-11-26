# -*- coding: utf-8 -*-
"""d12 功效与驼峰检验（Hermes R31 Q3）
① 缓存有效样本到 parquet，避免反复重读 3,179 个文件
② 每年 20 次按月随机二分 → 40 个 kappa → 得到该年 kappa 的抽样分布（月际噪声口径）
③ 检验 2011-13 vs 2019-22 的驼峰是否落在同一分布内
④ 算功效：给定实测月际 SE，要多大的年度差才能在 80% 功效下检出
"""
import os, glob, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

R = "G:/AStockData/supplement/daily_tencent_full"
CACHE = "cache_pairs.parquet"
WARM, HOR, K = 250, 5, 3
YEARS = list(range(2011, 2026))
REPEAT = 20
rng = np.random.default_rng(20261111)


def build_cache():
    COLS = ["date", "turnover", "volume"]
    files = sorted(glob.glob(R + "/*.parquet"))
    ys, ms, ds, aa, bb, cc = [], [], [], [], [], []
    for k, f in enumerate(files):
        code = os.path.basename(f).split(".")[0]
        try:
            df = pq.read_table(f, columns=COLS).to_pandas()
        except Exception:
            continue
        if len(df) < WARM + HOR + 60:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        d = pd.to_datetime(df["date"]).dt.normalize()
        t = pd.to_numeric(df["turnover"], errors="coerce").astype("float64") * 100.0
        v = pd.to_numeric(df["volume"], errors="coerce").astype("float64")
        t = t.where(v > 0)
        if t.notna().sum() < WARM + 60:
            continue
        q1 = t.rolling(WARM, min_periods=WARM).quantile(1.0 / K)
        q2 = t.rolling(WARM, min_periods=WARM).quantile(2.0 / K)
        ok = q1.notna() & q2.notna() & t.notna()
        st = pd.Series(np.where(ok, np.where(t <= q1, 0.0, np.where(t >= q2, 2.0, 1.0)), np.nan),
                       index=df.index)
        s5 = st.shift(-HOR)
        a, b = st.values, s5.values
        m = ~(np.isnan(a) | np.isnan(b))
        yr = d.dt.year.values; mo = d.dt.month.values
        m &= (yr >= YEARS[0]) & (yr <= YEARS[-1])
        if not m.any():
            continue
        # 记录日序号（用 date 转 int），用于未来按日重抽样
        dd = (d.values.astype("datetime64[D]").astype(np.int32))[m]
        ys.append(yr[m].astype(np.int16)); ms.append(mo[m].astype(np.int8))
        ds.append(dd); aa.append(a[m].astype(np.int8)); bb.append(b[m].astype(np.int8))
        cc.append(np.full(m.sum(), 1 if d.iloc[0] <= pd.Timestamp("2010-01-31") else 0, dtype=np.int8))
        if (k + 1) % 600 == 0:
            print("  cache %d/%d" % (k + 1, len(files)), flush=True)
    tab = pa.table({
        "year": np.concatenate(ys), "month": np.concatenate(ms),
        "day": np.concatenate(ds), "sa": np.concatenate(aa),
        "sb": np.concatenate(bb), "cohort": np.concatenate(cc),
    })
    pq.write_table(tab, CACHE, compression="zstd")
    print("cached ->", CACHE, tab.num_rows, flush=True)


if not os.path.exists(CACHE):
    build_cache()

df = pq.read_table(CACHE).to_pandas()
print("loaded %d rows" % len(df), flush=True)
yr = df.year.values; mo = df.month.values; day = df.day.values
sa = df.sa.values; sb = df.sb.values; co = df.cohort.values


def mat(a, b):
    M = np.zeros((K, K), dtype=np.float64)
    np.add.at(M, (a, b), 1)
    return M


def kappa(M):
    n = M.sum()
    if n < 200:
        return np.nan
    po = np.trace(M) / n
    r = M.sum(1) / n; c = M.sum(0) / n
    pe = float((r * c).sum())
    return np.nan if pe >= 1 else (po - pe) / (1 - pe)


# ---------- ① 每年 20 次按月随机二分，收集 40 个 kappa ----------
print("\n[①] 年内按月随机二分：每年 40 个子样本 kappa", flush=True)
samp = {}
for y in YEARS:
    m = yr == y
    a, b, mm = sa[m], sb[m], mo[m]
    vals = []
    for _ in range(REPEAT):
        pick = rng.random(12) < 0.5
        if pick.all() or (~pick).all():
            continue
        sel = pick[mm - 1]
        for s in (sel, ~sel):
            if s.sum() >= 200:
                vals.append(kappa(mat(a[s], b[s])))
    vals = [v for v in vals if not np.isnan(v)]
    samp[y] = np.array(vals)
    print("  %d  n_sub=%d  mean=%.4f  sd=%.4f  p2.5=%.4f  p97.5=%.4f"
          % (y, len(vals), np.mean(vals), np.std(vals, ddof=1),
             np.percentile(vals, 2.5), np.percentile(vals, 97.5)), flush=True)

# ---------- ② 驼峰检验：2011-13 vs 2019-22 ----------
gA = np.concatenate([samp[y] for y in (2011, 2012, 2013)])
gB = np.concatenate([samp[y] for y in (2019, 2020, 2021, 2022)])
print("\n[②] 驼峰检验（月际噪声口径）")
print("  2011-13: n=%d mean=%.4f sd=%.4f  95%%=[%.4f, %.4f]"
      % (len(gA), gA.mean(), gA.std(ddof=1), np.percentile(gA, 2.5), np.percentile(gA, 97.5)))
print("  2019-22: n=%d mean=%.4f sd=%.4f  95%%=[%.4f, %.4f]"
      % (len(gB), gB.mean(), gB.std(ddof=1), np.percentile(gB, 2.5), np.percentile(gB, 97.5)))
overlap = not (np.percentile(gB, 2.5) > np.percentile(gA, 97.5) or
               np.percentile(gA, 2.5) > np.percentile(gB, 97.5))
print("  两组 95%% 区间是否重叠: %s" % ("重叠 → 驼峰可由月际波动解释" if overlap else "不重叠 → 驼峰超出月际噪声"))
# 秩检验（不假设正态）
from itertools import product
U = (gB[:, None] > gA[None, :]).mean()
print("  P(2019-22 的某个kappai > 2011-13 的某个kappa) = %.4f  (0.5=无法区分)" % U)

# 差值 t（Welch）
d = gB.mean() - gA.mean()
se = np.sqrt(gA.var(ddof=1) / len(gA) + gB.var(ddof=1) / len(gB))
print("  差=%.4f  SE=%.4f  t=%.2f" % (d, se, d / se))

# ---------- ③ 功效分析 ----------
print("\n[③] 功效分析（给定实测月际 SE）")
sd_all = np.mean([samp[y].std(ddof=1) for y in YEARS])
print("  各年月际口径 sd 的均值 = %.4f（这是「年的 kappa 估计」的抽样误差）" % sd_all)
# 两年比较的 SE
se2 = np.sqrt(2) * sd_all
from math import sqrt
z_a, z_b = 1.959964, 0.8416212  # 0.05 双侧, 80% 功效
mde80 = (z_a + z_b) * se2
print("  两年比较 SE=%.4f  → 80%% 功效可检出的最小差 = %.4f（约 %.2f 个 kappa 点）"
      % (se2, mde80, mde80))
print("  参考：实测 2011-13 与 2019-22 差 = %.4f" % d)
print("  结论：可检出差 %.4f %s 实测差 %.4f"
      % (mde80, "大于" if mde80 > d else "小于", d))
if mde80 > d:
    print("  >>> 当前设计对 %.4f 这个量级的变化【没有分辨力】。判决只能是「样本不足」，不是「无变化」。" % d)
else:
    print("  >>> 当前设计对实测量级的变化有分辨力，可以判「有/无」。")

json.dump({str(y): [float(x) for x in samp[y]] for y in YEARS},
          open("d12_subsample_kappa.json", "w", encoding="utf-8"), indent=1)
print("\nsaved d12_subsample_kappa.json")
