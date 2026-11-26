# -*- coding: utf-8 -*-
"""
d51 同一规则跨牛熊起点检验
=================================
d50 只测了 2024-08 一个起点，而 2024-08~2026-07 是大牛市（924 之后），
「中位 +50%」很可能主要是行情，不是规则。

本脚本把完全相同的规则搬到 7 个起点，含 2018（熊）、2021（熊）、2022（熊）：
  起点 D = 各年 8 月最后一个交易日
  规则不变：跌 10% 买入 / +50% 止盈 / 峰值回落 20% 止损 / 20 个月到期
对照不变：C1 不等跌直接买（同退出规则）、C2 D 日买入持有 20 月、
          C3 触发日买入纯持有 20 月
另报：触发率、以及「把没买记 0」的实际体验分布（他一次只能拿一只）

数据：daily_tencent_full（前复权，2010-01 ~ 2025-12-31），3,179 只。
  2024-08 起点的 20 个月落在 2026-04，超出数据 → 该起点会被截断到 2025-12，已标注。
"""
import os, glob, json, re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TF = "G:/AStockData/supplement/daily_tencent_full"
LHB = "G:/AStockData/supplement/youzi_history/lhb.parquet"

STARTS = ["2018-08-31", "2019-08-30", "2020-08-31", "2021-08-31",
          "2022-08-31", "2023-08-31", "2024-08-30"]
DROP, TP, TRAIL, HOLD_M, WAIT_M = 0.10, 1.50, 0.20, 20, 12


def add_months(ts, m):
    y, mo = ts.year, ts.month + m
    y += (mo - 1) // 12
    mo = (mo - 1) % 12 + 1
    return pd.Timestamp(year=y, month=mo, day=min(ts.day, 28))


def load_tf(fp):
    try:
        a = pq.read_table(fp, columns=["date", "open", "high", "low", "close"]).to_pandas()
    except Exception:
        return None
    if len(a) < 400:
        return None
    a["d"] = pd.to_datetime(a["date"]).dt.normalize()
    a = a.sort_values("d").drop_duplicates("d").reset_index(drop=True)
    for c in ("open", "high", "low", "close"):
        a[c] = a[c].astype("float64")
    a = a[(a["close"] > 0) & np.isfinite(a["close"])]
    return a.reset_index(drop=True) if len(a) > 400 else None


def sim(a, d_dec, wait_drop=True):
    i0 = a["d"].searchsorted(d_dec)
    if i0 >= len(a):
        return None
    P = float(a["close"].iloc[i0])
    if not (P > 0):
        return None
    if wait_drop:
        trig = P * (1 - DROP)
        sub = a.iloc[i0 + 1:]
        sub = sub[sub["d"] <= add_months(d_dec, WAIT_M)]
        hit = sub[sub["low"] <= trig]
        if len(hit) == 0:
            return dict(entered=False)
        e = hit.iloc[0]
        entry = min(trig, float(e["open"]))
        i1 = int(e.name)
    else:
        entry, i1 = P, i0
    if not (entry > 0):
        return dict(entered=False)

    end_d = add_months(a["d"].iloc[i1], HOLD_M)
    peak = entry
    for j in range(i1 + 1, len(a)):
        row = a.iloc[j]
        stop = peak * (1 - TRAIL)
        if float(row["low"]) <= stop:
            px = min(stop, float(row["open"]))
            return dict(entered=True, ret=px / entry - 1, reason="止损",
                        d_entry=a["d"].iloc[i1], censored=False)
        if float(row["high"]) >= entry * TP:
            px = max(entry * TP, float(row["open"]))
            return dict(entered=True, ret=px / entry - 1, reason="止盈+50%",
                        d_entry=a["d"].iloc[i1], censored=False)
        peak = max(peak, float(row["high"]))
        if row["d"] >= end_d:
            px = float(row["close"])
            return dict(entered=True, ret=px / entry - 1, reason="满20月",
                        d_entry=a["d"].iloc[i1], censored=False)
    px = float(a["close"].iloc[-1])
    return dict(entered=True, ret=px / entry - 1, reason="数据截止",
                d_entry=a["d"].iloc[i1], censored=True)


def hold_simple(a, d0, months=HOLD_M):
    i = a["d"].searchsorted(d0)
    if i >= len(a):
        return None
    p0 = float(a["close"].iloc[i])
    if not (p0 > 0):
        return None
    end = add_months(a["d"].iloc[i], months)
    sub = a[a["d"] >= end]
    p1 = float(sub["close"].iloc[0]) if len(sub) else float(a["close"].iloc[-1])
    return p1 / p0 - 1


def hold_from(a, d0, months=HOLD_M):
    return hold_simple(a, d0, months)


def agg(v):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return None
    return dict(n=int(len(v)), med=float(np.median(v)), mean=float(v.mean()),
                win=float((v > 0).mean()))


def main():
    lhb = pq.read_table(LHB, columns=["代码", "上榜日"]).to_pandas()
    lhb["d"] = pd.to_datetime(lhb["上榜日"].astype(str).str[:10], errors="coerce")
    lhb["code"] = lhb["代码"].astype(str).str.zfill(6)

    files = sorted(glob.glob(TF + "/*.parquet"))
    series = []
    for f in files:
        code = os.path.basename(f)[:-8]
        if not re.fullmatch(r"\d{6}", code):
            continue
        a = load_tf(f)
        if a is not None:
            series.append((code, a))
    print("载入 %d 只" % len(series), flush=True)

    out = {}
    for D in STARTS:
        d_dec = pd.Timestamp(D)
        i0s = [c for c, a in series if len(a["d"]) and a["d"].iloc[0] <= d_dec < a["d"].iloc[-1]]
        # 故事股：起点前 7 个月内上过龙虎榜 >=2 次
        s0 = d_dec - pd.Timedelta(days=210)
        sub = lhb[(lhb.d >= s0) & (lhb.d <= d_dec)]
        cnt = sub.groupby("code").size()
        story = set(cnt[cnt >= 2].index)

        R, C1, C2, C3, RN = [], [], [], [], []
        Rs, C1s, C2s, C3s, RNs = [], [], [], [], []
        for code, a in series:
            r = sim(a, d_dec)
            if r is None:
                continue
            isS = code in story
            R.append(r)
            if r.get("entered"):
                RN.append(r["ret"])
                c1 = sim(a, d_dec, wait_drop=False)
                C1.append(c1["ret"] if c1 and c1.get("entered") else None)
                C3.append(hold_from(a, pd.Timestamp(r["d_entry"])))
            else:
                RN.append(0.0)
            C2.append(hold_simple(a, d_dec))
            if isS:
                Rs.append(r)
                if r.get("entered"):
                    RNs.append(r["ret"])
                    c1 = sim(a, d_dec, wait_drop=False)
                    C1s.append(c1["ret"] if c1 and c1.get("entered") else None)
                    C3s.append(hold_from(a, pd.Timestamp(r["d_entry"])))
                else:
                    RNs.append(0.0)
                C2s.append(hold_simple(a, d_dec))

        def pack(R, RN, C1, C2, C3):
            ent = [x for x in R if x.get("entered")]
            cens = sum(1 for x in ent if x.get("censored"))
            reasons = pd.Series([x["reason"] for x in ent]).value_counts().to_dict() if ent else {}
            a1 = agg(RN)
            return dict(n_cand=len(R), n_entered=len(ent),
                        trigger_rate=(len(ent) / len(R)) if R else None,
                        entered=agg([x["ret"] for x in ent]),
                        incl_noentry=a1,
                        c1=agg(C1), c2=agg(C2), c3=agg(C3),
                        censored=cens, reasons=reasons)

        out[D] = dict(all=pack(R, RN, C1, C2, C3), story=pack(Rs, RNs, C1s, C2s, C3s))
        o = out[D]["all"]
        print("\n%s  候选 %d  触发 %.1f%%  (%s)" % (
            D, o["n_cand"], 100 * (o["trigger_rate"] or 0), "截断%d" % o["censored"]), flush=True)
        for k, lab in (("entered", "规则(入场后)"), ("incl_noentry", "规则(含没买记0)"),
                       ("c1", "C1不等跌"), ("c2", "C2买入持有"), ("c3", "C3触发后纯持有")):
            v = o[k]
            if v:
                print("   %-16s 中位 %+7.2f%%  均值 %+7.2f%%  胜率 %5.1f%%"
                      % (lab, 100 * v["med"], 100 * v["mean"], 100 * v["win"]), flush=True)
        s = out[D]["story"]
        if s["entered"]:
            print("   故事股(>=2次上榜) n=%d 触发%.1f%% 中位%+.1f%% 均值%+.1f%% | C2 %+.1f%%"
                  % (s["n_entered"], 100 * (s["trigger_rate"] or 0),
                     100 * s["entered"]["med"], 100 * s["entered"]["mean"], 100 * s["c2"]["med"]),
                  flush=True)

    json.dump(out, open("d51_result.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    print("\n已写 d51_result.json")


if __name__ == "__main__":
    main()
