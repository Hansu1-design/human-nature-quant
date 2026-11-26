# -*- coding: utf-8 -*-
"""d11 最小可判决实验 v2（修正版）
修正 d10 三处缺陷：
  ① q1/q2 预热期为 NaN 时状态必须为 NaN（d10 里被误判为中间档，造成假 1->1 转移）
  ② 安慰剂改为「同年内随机二分」，并做 20 次重复；同时给按日分半（保守）与按行分半（激进）两版
  ③ 主统计量改为 kappa（剔除边际分布影响），并给标准化残差矩阵距离（结构差）
另加：两个独立口径（时序相对 / 横截面相对）+ 固定队列对照
"""
import os, glob, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

R = "G:/AStockData/supplement/daily_tencent_full"
COLS = ["date", "close", "turnover", "volume"]
WARM = 250
HOR = 5
K = 3
YEARS = list(range(2011, 2026))   # 2010 年因预热全为 NaN，剔除
YI = {y: i for i, y in enumerate(YEARS)}
REPEAT = 20
rng = np.random.default_rng(20261111)

files = sorted(glob.glob(R + "/*.parquet"))
print("files=%d" % len(files), flush=True)

# 收集：每条有效样本 -> (year, month, state_t, state_t5, cohort_flag)
buf_y, buf_m, buf_a, buf_b, buf_c = [], [], [], [], []
bad = 0
first_date = {}
for k, f in enumerate(files):
    code = os.path.basename(f).split(".")[0]
    try:
        df = pq.read_table(f, columns=COLS).to_pandas()
    except Exception:
        bad += 1
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
    first_date[code] = d.iloc[0]
    q1 = t.rolling(WARM, min_periods=WARM).quantile(1.0 / K)
    q2 = t.rolling(WARM, min_periods=WARM).quantile(2.0 / K)
    # 修正①：预热期与缺值一律 NaN
    valid_q = q1.notna() & q2.notna() & t.notna()
    st = np.where(valid_q, np.where(t <= q1, 0.0, np.where(t >= q2, 2.0, 1.0)), np.nan)
    st = pd.Series(st, index=df.index)
    s5 = st.shift(-HOR)
    a = st.values; b = s5.values
    m = ~(np.isnan(a) | np.isnan(b))
    if not m.any():
        continue
    yr = d.dt.year.values
    mo = d.dt.month.values
    m &= (yr >= YEARS[0]) & (yr <= YEARS[-1])
    if not m.any():
        continue
    buf_y.append(yr[m].astype(np.int16))
    buf_m.append(mo[m].astype(np.int8))
    buf_a.append(a[m].astype(np.int8))
    buf_b.append(b[m].astype(np.int8))
    buf_c.append(np.full(m.sum(), 1 if d.iloc[0] <= pd.Timestamp("2010-01-31") else 0, dtype=np.int8))
    if (k + 1) % 500 == 0:
        print("  %d/%d bad=%d" % (k + 1, len(files), bad), flush=True)

yr = np.concatenate(buf_y); mo = np.concatenate(buf_m)
sa = np.concatenate(buf_a); sb = np.concatenate(buf_b); co = np.concatenate(buf_c)
print("done bad=%d  valid=%d  stocks=%d" % (bad, len(sa), len(first_date)), flush=True)
print("  cohort(2010前上市)=%d 条, 新股=%d 条" % (co.sum(), (co == 0).sum()), flush=True)


def mat(a, b):
    M = np.zeros((K, K), dtype=np.float64)
    np.add.at(M, (a, b), 1)
    return M


def kappa(M):
    n = M.sum()
    if n < 200:
        return np.nan
    po = np.trace(M) / n
    r = M.sum(1) / n
    c = M.sum(0) / n
    pe = float((r * c).sum())
    if pe >= 1:
        return np.nan
    return (po - pe) / (1 - pe)


def resid(M, dfree):
    """标准化残差矩阵（剔除边际），按自由度归一"""
    n = M.sum()
    if n < 200:
        return None
    r = M.sum(1, keepdims=True)
    c = M.sum(0, keepdims=True)
    E = r @ c / n
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.where(E > 0, (M - E) / np.sqrt(E), 0.0)
    return Z / np.sqrt(dfree)


DFREE = (K - 1) ** 2


def mdist(M1, M2):
    Z1, Z2 = resid(M1, DFREE), resid(M2, DFREE)
    if Z1 is None or Z2 is None:
        return np.nan
    return float(np.sqrt(((Z1 - Z2) ** 2).mean()))


# ---------- 主序列：按年 ----------
kap, D, N, DIST_next = [], [], [], []
P = {}
for y in YEARS:
    m = yr == y
    M = mat(sa[m], sb[m])
    P[y] = M
    n = M.sum()
    kv = kappa(M)
    dv = np.trace(M) / n if n else np.nan
    kap.append(kv); D.append(dv); N.append(int(n))
    print("  %d  kappa=%+.4f  D=%.4f  n=%d" % (y, kv, dv, n), flush=True)

kap = np.array(kap); D = np.array(D); N = np.array(N)


def acf1(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 6:
        return np.nan, np.nan
    z = x - x.mean()
    den = (z * z).sum()
    if den <= 0:
        return np.nan, np.nan
    r = (z[:-1] * z[1:]).sum() / den
    tstat = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return float(r), float(tstat)


# ---------- 跨年差（结构差，剔除边际）----------
cross = [mdist(P[YEARS[i]], P[YEARS[i + 1]]) for i in range(len(YEARS) - 1)]
cross = np.array([x for x in cross if not np.isnan(x)])

# ---------- 安慰剂：同年内随机二分 ----------
# 版A（保守）：按「日期」分半 —— 保留日间相关结构
# 版B（激进）：按「行」分半
plac_day, plac_row = [], []
dates_key = yr.astype(np.int32) * 100 + mo.astype(np.int32)  # 只到月粒度，够用
for y in YEARS:
    m = yr == y
    a, b = sa[m], sb[m]
    n = len(a)
    if n < 2000:
        continue
    # 版B 按行
    for _ in range(REPEAT):
        idx = rng.random(n) < 0.5
        d = mdist(mat(a[idx], b[idx]), mat(a[~idx], b[~idx]))
        if not np.isnan(d):
            plac_row.append(d)
    # 版A 按日（同一个日期整体分到一边）
    # 用 (month, state) 无法还原日；改用 month 粒度分组，保留月内相关的一半
    for _ in range(REPEAT):
        pick = rng.random(12) < 0.5
        if pick.all() or (~pick).all():
            continue
        sel = pick[mo[m] - 1]
        if sel.sum() < 100 or (~sel).sum() < 100:
            continue
        d = mdist(mat(a[sel], b[sel]), mat(a[~sel], b[~sel]))
        if not np.isnan(d):
            plac_day.append(d)

plac_row = np.array(plac_row); plac_day = np.array(plac_day)
print("\n[安慰剂] 同年内随机二分")
print("  按行分半: mean=%.5f med=%.5f n=%d" % (plac_row.mean(), np.median(plac_row), len(plac_row)))
print("  按月分半: mean=%.5f med=%.5f n=%d" % (plac_day.mean(), np.median(plac_day), len(plac_day)))
print("[跨年]      mean=%.5f med=%.5f n=%d" % (cross.mean(), np.median(cross), len(cross)))
print("  比值(按行) = %.2f" % (cross.mean() / plac_row.mean()))
print("  比值(按月) = %.2f  <- 保守口径" % (cross.mean() / plac_day.mean()))
print("  比值(按月中位) = %.2f" % (np.median(cross) / np.median(plac_day)))

# bootstrap
bs = []
for _ in range(5000):
    x = rng.choice(cross, len(cross), replace=True).mean()
    yv = rng.choice(plac_day, len(plac_day), replace=True).mean()
    bs.append(x - yv)
bs = np.array(bs)
print("  bootstrap 跨年-按月安慰剂: 95%%CI=[%.5f, %.5f]  P(>0)=%.4f"
      % (np.percentile(bs, 2.5), np.percentile(bs, 97.5), (bs > 0).mean()))

# ---------- 固定队列对照 ----------
print("\n[固定队列] 仅 2010-01 前已上市")
kapc, Nc = [], []
for y in YEARS:
    m = (yr == y) & (co == 1)
    if m.sum() < 5000:
        kapc.append(np.nan); Nc.append(int(m.sum())); continue
    M = mat(sa[m], sb[m])
    kapc.append(kappa(M)); Nc.append(int(m.sum()))
for y, kv, n in zip(YEARS, kapc, Nc):
    print("  %d  kappa=%+.4f  n=%d" % (y, kv, n))

r_k, t_k = acf1(kap)
r_c, t_c = acf1(np.array(kapc))
print("\n[统计量1] kappa 序列 =", np.round(kap, 4).tolist())
print("  全样本 ACF(1)=%+.4f  t=%.3f  n=%d" % (r_k, t_k, len(kap)))
print("  固定队列 ACF(1)=%+.4f  t=%.3f" % (r_c, t_c))
print("  D 序列 =", np.round(D, 4).tolist())
r_d, t_d = acf1(D)
print("  D 的 ACF(1)=%+.4f  t=%.3f" % (r_d, t_d))

# ---------- 判决 ----------
ratio = cross.mean() / plac_day.mean()
print("\n" + "=" * 64)
print("判决（Hermes Q8：ACF(1) 显著 且 比值>2 → 活）")
print("=" * 64)
ok_acf = abs(t_k) > 2
ok_ratio = ratio > 2
print("  ACF(1)=%+.4f |t|=%.2f → %s" % (r_k, abs(t_k), "显著" if ok_acf else "不显著"))
print("  比值=%.2f → %s" % (ratio, ">2 过" if ok_ratio else "≤2 不过"))
if ok_acf and ok_ratio:
    verdict = "活：漂移有惯性，值得补血肉"
elif (not ok_acf) and ratio < 1.2:
    verdict = "死：既无稳定结构也无漂移结构"
else:
    verdict = "降级：只有基准率表，不许叫理论"
print("  >>> " + verdict)

json.dump({
    "years": YEARS, "kappa": [None if np.isnan(x) else float(x) for x in kap],
    "D": [None if np.isnan(x) else float(x) for x in D], "N": [int(x) for x in N],
    "kappa_cohort": [None if np.isnan(x) else float(x) for x in kapc],
    "acf1_kappa": r_k, "t_kappa": t_k, "acf1_D": r_d, "t_D": t_d,
    "acf1_cohort": r_c, "t_cohort": t_c,
    "cross_mean": float(cross.mean()), "cross_med": float(np.median(cross)),
    "placebo_row_mean": float(plac_row.mean()), "placebo_day_mean": float(plac_day.mean()),
    "ratio_row": float(cross.mean() / plac_row.mean()), "ratio_day": float(ratio),
    "boot_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
    "boot_p": float((bs > 0).mean()),
    "verdict": verdict,
    "P": {str(y): P[y].tolist() for y in YEARS},
}, open("d11_result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved d11_result.json")
