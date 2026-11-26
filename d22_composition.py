# -*- coding: utf-8 -*-
"""d22 混杂检查：R1 的年度上升，是「浮盈日占比变化」机械造成的，还是分档内换手真的变了？

输出三类年度量（均为原值，未归一）：
  shareG   浮盈日占（盈+亏）比
  med_t_G  浮盈日绝对换手中位 %
  med_t_L  浮亏日绝对换手中位 %
  R_abs    med_t_G / med_t_L  （纯绝对口径，不做任何个股归一）
若 R_abs 与 R（归一）同步，说明不是归一化制造的。
"""
import os, glob, json
import numpy as np, pandas as pd, pyarrow.parquet as pq

R = "G:/AStockData/supplement/daily_tencent_full"
OUT = os.path.dirname(os.path.abspath(__file__))
WIN, REPAIR, GAIN, LOSS = 250, -0.22, 0.01, -0.01

rows = []
for f in sorted(glob.glob(R + "/*.parquet")):
    try:
        d = pq.read_table(f, columns=["date", "close", "turnover"]).to_pandas()
    except Exception:
        continue
    if len(d) < WIN + 90:
        continue
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d.sort_values("date").reset_index(drop=True)
    c = d["close"].astype(float).values; t = d["turnover"].astype(float).values
    ret = np.ones(len(c)); ret[1:] = c[1:] / c[:-1]
    bad = np.where(ret < (1 + REPAIR))[0]; bad = bad[bad > 0]
    k = np.ones(len(c))
    for i in bad:
        k[:i] *= ret[i]
    p = c * k
    tw = pd.Series(t).clip(lower=0).fillna(0).values
    num = pd.Series(p * tw).rolling(WIN, min_periods=WIN).sum().values
    den = pd.Series(tw).rolling(WIN, min_periods=WIN).sum().values
    with np.errstate(invalid="ignore", divide="ignore"):
        g = p / np.where(den > 0, num / den, np.nan) - 1
    g = pd.Series(g).shift(1).values
    df = pd.DataFrame({"yr": d["date"].dt.year.values, "turn": t * 100, "g": g})
    df = df[np.isfinite(df["g"]) & (df["turn"] > 0)]
    rows.append(df)

D = pd.concat(rows, ignore_index=True)
print("行数 %d" % len(D))
out = {}
print("%s %7s %9s %9s %9s %9s" % ("年", "盈占比", "盈换手%", "亏换手%", "R_abs", "n盈"))
for y, s in D.groupby("yr"):
    G = s.loc[s["g"] > GAIN, "turn"]; L = s.loc[s["g"] < LOSS, "turn"]
    if len(G) < 1000 or len(L) < 1000:
        continue
    o = dict(shareG=float(len(G) / (len(G) + len(L))),
             medG=float(G.median()), medL=float(L.median()),
             Rabs=float(G.median() / L.median()), nG=int(len(G)), nL=int(len(L)))
    out[int(y)] = o
    print("%d %6.1f%% %9.3f %9.3f %9.3f %9d"
          % (y, 100 * o["shareG"], o["medG"], o["medL"], o["Rabs"], o["nG"]))

json.dump(out, open(os.path.join(OUT, "d22_composition.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

# 与 R1 的相关
R1 = json.load(open(os.path.join(OUT, "d21_result.json"), encoding="utf-8"))["P_A_主"]["series"]
ys = sorted(set(int(k) for k in R1) & set(out))
r = [R1[str(y)] for y in ys]
for key in ("shareG", "medG", "medL", "Rabs"):
    v = [out[y][key] for y in ys]
    print("R(R1) 与 %-7s 相关 %+.3f" % (key, np.corrcoef(r, v)[0, 1]))
print("R_abs 斜率 %+.5f/yr" % float(np.polyfit(np.array(ys, float), [out[y]["Rabs"] for y in ys], 1)[0]))
