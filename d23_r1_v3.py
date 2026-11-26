# -*- coding: utf-8 -*-
"""R1 v3：按 Hermes R32 处方重构

四处修正（对应它的 Q1/Q2/Q5/Q3）：
 1  成本锚换成「换手衰减筹码成本」：RP_t = (1−α)RP_{t-1} + α·P_t，α=turnover_t。
    按个股自身换手衰减，不受全市场换手水平（15 年翻倍）影响。
 2  档宽改用**分位切**：每票每年 g 的 top30% / bottom30%，档宽恒定，
    一次性消掉「固定 ±1% 绝对带宽随波动率变化」的构成漂移。
 3  统计量改为「两档内换手中位之比」，不再用「条件中位 ÷ 年度中位」（后者随换手分布离散度上升而自动上移）。
 4  t 值用 Newey-West HAC（lag=2）+ ACF 有效样本校正，不再用年内分裂安慰剂的 SE 去判斜率。
 5  加 Hermes 的三条可区分检验：
    ① 年度回归加「换手 p75/p50」（离散度协变量），趋势消失即机械
    ② 每票 15 年斜率的符号检验（n≈3000，免构成污染）
    ③ 年内月度斜率 vs 跨年斜率：若 12 个月的斜率就能复现，那条「15 年趋势」只是水平差

另出 ADF + KPSS 组合（Hermes Q3：不许在 level/detrended ACF 里二选一）。
输出 d23_result.json
"""
import os, glob, json, time
import numpy as np, pandas as pd, pyarrow.parquet as pq

R = "G:/AStockData/supplement/daily_tencent_full"
OUT = os.path.dirname(os.path.abspath(__file__))
REPAIR = -0.22
Q_LO, Q_HI = 0.30, 0.70
MIN_DAYS = 40

def cost_anchor(p, alpha):
    """换手衰减筹码成本：每日剩余权重*(1-换手)"""
    rp = np.empty(len(p)); rp[0] = p[0]
    for i in range(1, len(p)):
        a = alpha[i]
        rp[i] = (1 - a) * rp[i - 1] + a * p[i]
    return rp

def load(f):
    code = os.path.basename(f).split(".")[0]
    d = pq.read_table(f, columns=["date", "close", "turnover", "outstanding_share"]).to_pandas()
    if len(d) < 400:
        return None
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d.sort_values("date").reset_index(drop=True)
    c = d["close"].astype(float).values
    t = d["turnover"].astype(float).values
    sh = d["outstanding_share"].astype(float).values
    ret = np.ones(len(c)); ret[1:] = c[1:] / c[:-1]
    bad = np.where(ret < (1 + REPAIR))[0]; bad = bad[bad > 0]
    k = np.ones(len(c))
    for i in bad:
        k[:i] *= ret[i]
    p = c * k
    a = np.clip(np.nan_to_num(t, nan=0.0), 0, 0.5)
    rp = cost_anchor(p, a)
    g = p / np.where(rp > 0, rp, np.nan) - 1
    df = pd.DataFrame({"date": d["date"].values, "code": code, "turn": t,
                       "g": g, "mcap": p * sh})
    df["yr"] = df["date"].dt.year
    df = df[np.isfinite(df["g"]) & (df["turn"] > 0) & (df["mcap"] > 0)]
    return df.iloc[250:]

rows = []
t0 = time.time()
for i, f in enumerate(sorted(glob.glob(R + "/*.parquet"))):
    try:
        o = load(f)
    except Exception:
        continue
    if o is not None and len(o) > 60:
        rows.append(o)
D = pd.concat(rows, ignore_index=True)
print("载入 %d 行 / %d 只 / %.0fs" % (len(D), D["code"].nunique(), time.time() - t0), flush=True)

# 分位切档
def qcut(x, ql=Q_LO, qh=Q_HI):
    return np.quantile(x, ql), np.quantile(x, qh)

recs = []
for (code, yr), s in D.groupby(["code", "yr"]):
    if len(s) < MIN_DAYS * 2:
        continue
    lo, hi = qcut(s["g"].values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        continue
    gt = s.loc[s["g"] >= hi, "turn"]; gb = s.loc[s["g"] <= lo, "turn"]
    if len(gt) < MIN_DAYS // 2 or len(gb) < MIN_DAYS // 2:
        continue
    recs.append(dict(code=code, yr=yr, R=float(gt.median() / gb.median()),
                     tG=float(gt.median()), tB=float(gb.median()),
                     nG=len(gt), nB=len(gb),
                     mcap=float(s["mcap"].median()),
                     disp=float(s["turn"].quantile(0.75) / s["turn"].quantile(0.50))))
CS = pd.DataFrame(recs)
CS.to_parquet(os.path.join(OUT, "d23_cross.parquet"))
print("个股-年度面板 %d 条 / %d 只" % (len(CS), CS["code"].nunique()), flush=True)

def nw_t(y, x, lag=2):
    n = len(x)
    A = np.vstack([x, np.ones(n)]).T
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    e = y - A @ b
    XtX_inv = np.linalg.inv(A.T @ A)
    S = (A * e[:, None]).T @ (A * e[:, None])
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        Gl = (A[l:] * e[l:, None]).T @ (A[:-l] * e[:-l, None])
        S += w * (Gl + Gl.T)
    V = XtX_inv @ S @ XtX_inv
    return b[0], np.sqrt(V[0, 0]), b

def acf1(x):
    x = np.asarray(x, float); n = len(x)
    r = np.corrcoef(x[:-1], x[1:])[0, 1]
    return r, r * np.sqrt(n - 2) / np.sqrt(max(1e-12, 1 - r * r))

def neff(x):
    r, _ = acf1(x); n = len(x)
    return n * (1 - r) / (1 + r) if r > -1 else n

def adf(x, lags=1):
    x = np.asarray(x, float); dx = np.diff(x); rr, yy = [], []
    for i in range(lags, len(dx)):
        rr.append([1.0, x[i]] + list(dx[i - lags:i])); yy.append(dx[i])
    A = np.array(rr); y = np.array(yy)
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    e = y - A @ b; s2 = e @ e / (len(y) - A.shape[1])
    se = np.sqrt((s2 * np.linalg.inv(A.T @ A))[1, 1])
    return b[1] / se

def kpss(x, lags=2):
    x = np.asarray(x, float); n = len(x)
    e = x - x.mean(); S = np.cumsum(e)
    s2 = (e @ e) / n
    for l in range(1, lags + 1):
        s2 += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    return float((S ** 2).sum() / (n ** 2 * s2))

# 年度序列（个股等权中位）
ry = CS.groupby("yr")["R"].median()
ny = CS.groupby("yr")["R"].size()
disp = CS.groupby("yr")["disp"].median()
print("\n年   R(两档中位比)  n票  离散p75/p50")
for y in ry.index:
    print("%d      %8.4f  %5d      %.3f" % (y, ry[y], ny[y], disp[y]))

xs = ry.index.values.astype(float); ys = ry.values.astype(float)
sl, se, b = nw_t(ys, xs)
tt = sl / se
ne = neff(ys); ts = tt * np.sqrt(ne / len(ys))
print("\n斜率 %+.6f/yr  NW t=%+.2f  | ACF有效n=%.1f -> 校正t=%+.2f" % (sl, tt, ne, ts))
print("ADF t=%+.2f (5%%临界≈-3.0，不拒绝=有单位根)  KPSS=%.3f (5%%临界 0.463，拒绝=非平稳)" % (adf(ys), kpss(ys)))

# 检验① 加离散度协变量
A = np.vstack([xs, disp.values, np.ones(len(xs))]).T
bb = np.linalg.lstsq(A, ys, rcond=None)[0]
e = ys - A @ bb; XtX_inv = np.linalg.inv(A.T @ A)
S = (A * e[:, None]).T @ (A * e[:, None])
V = XtX_inv @ S @ XtX_inv
print("检验① 控制离散度后斜率 %+.6f  t=%+.2f（原 %+.6f）" % (bb[0], bb[0] / np.sqrt(V[0, 0]), sl))

# 检验② 每票斜率的符号检验
sign = []
for code, s in CS.groupby("code"):
    if len(s) < 8:
        continue
    k = np.polyfit(s["yr"].values.astype(float), s["R"].values.astype(float), 1)[0]
    sign.append(k)
sign = np.array(sign)
pos = (sign > 0).sum(); nS = len(sign)
z = (pos - nS / 2) / np.sqrt(nS / 4)
print("检验② 个股斜率符号检验：正 %d / %d = %.1f%%  z=%+.2f  （50%% 为无趋势）" % (pos, nS, 100 * pos / nS, z))

# 检验③ 年内月度斜率 vs 跨年
D["ym"] = D["date"].dt.to_period("M")
mrec = []
for (code, ym), s in D.groupby(["code", "ym"]):
    if len(s) < 12:
        continue
    lo, hi = qcut(s["g"].values, 0.25, 0.75)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        continue
    gt = s.loc[s["g"] >= hi, "turn"]; gb = s.loc[s["g"] <= lo, "turn"]
    if len(gt) < 4 or len(gb) < 4:
        continue
    mrec.append((code, ym, float(gt.median() / gb.median())))
MM = pd.DataFrame(mrec, columns=["code", "ym", "R"])
MM["t"] = MM["ym"].astype(str)
MM["tn"] = MM.groupby("ym").cumcount()
mry = MM.groupby("ym")["R"].median()
mry.index = mry.index.to_timestamp()
mt = np.arange(len(mry), dtype=float)
msl = np.polyfit(mt, mry.values.astype(float), 1)[0]
print("检验③ 月度序列 n=%d，月斜率 %+.6f/月 = %+.6f/年（跨年斜率 %+.6f/yr）"
      % (len(mry), msl, msl * 12, sl))
print("        月序列 ACF(1)=%+.3f" % acf1(mry.values)[0])

json.dump(dict(series={int(k): float(v) for k, v in ry.items()},
               n={int(k): int(v) for k, v in ny.items()},
               disp={int(k): float(v) for k, v in disp.items()},
               slope=float(sl), se_nw=float(se), t_nw=float(tt), neff=float(ne), t_corr=float(ts),
               adf=float(adf(ys)), kpss=float(kpss(ys)),
               test1=dict(slope_ctrl=float(bb[0]), t_ctrl=float(bb[0] / np.sqrt(V[0, 0])),
                          coef_disp=float(bb[1])),
               test2=dict(pos=int(pos), n=int(nS), z=float(z)),
               test3=dict(month_slope=float(msl), year_eq=float(msl * 12), n_month=int(len(mry))),
               monthly={str(k)[:7]: float(v) for k, v in mry.items()}),
          open(os.path.join(OUT, "d23_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("耗时 %.0fs" % (time.time() - t0))
