# -*- coding: utf-8 -*-
"""
d50 用户规则的直接回测（不预测，只复现他的操作规则）
=====================================================
用户 2026-11-12 00:16 原话：
「24年8月，随机找一个有故事的股，持有3-20个月。在下定买入后，不买。反而再掉10个点再买入。
  之后一直持有，直到上升到出现回落20个点或者涨幅50个点。为止。
  否则就在成本价20个点以下卖或者20个月结束，收益如何」

规则精确化（跑之前写死，不许事后改）：
  D          = 2024-08-30 收盘（8 月最后一个交易日）为决策日，P_D = 当日收盘
  入场触发   = 之后任一交易日 low <= 0.90 * P_D，成交价 = min(0.90*P_D, open)
  等待上限   = 12 个月（另测 6 / 20 个月）
  持仓退出（每日按序判断，保守口径：同日两触发取先到止损）：
     ① 移动止损：low <= 0.80 * peak_prev  -> 成交 min(0.80*peak_prev, open)
     ② 目标止盈：high >= 1.50 * entry     -> 成交 max(1.50*entry, open)
     ③ 到期      ：持仓满 20 个月          -> 收盘价
     peak 从入场价起算，逐日更新为 max(peak, high)
     注：peak 未上升时 ① 恒等于「成本下方 20% 卖」，两条规则重合，无需单列
  未触发     = 12 个月内从未跌到 -10%，即「这一票他没买」

对照组（都要报，否则收益数字没有意义）：
  C1 不等跌：D 日直接买，退出规则完全相同
  C2 全域买入持有：整个宇宙 D 日等权买入、持到各自 20 个月末
  C3 全部 A 股同口径（不是只有故事股）

数据：market_full（不复权，2023-02~2026-07-03）+ daily_tencent_full（前复权，至 2025-12-31）
拼接：2025-12-31 之后用「前复权/不复权」比值外推，保证跨除权连续
"""
import os, glob, json, random, re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

MF = "G:/AStockData/market_full"
TF = "G:/AStockData/supplement/daily_tencent_full"
LHB = "G:/AStockData/supplement/youzi_history/lhb.parquet"

D_DEC = pd.Timestamp("2024-08-30")
DROP = 0.10
TP = 1.50
TRAIL = 0.20
HOLD_M = 20
WAIT_M = 12
STORY_START = pd.Timestamp("2024-02-01")
STORY_END = pd.Timestamp("2024-08-31")


def add_months(ts, m):
    y, mo = ts.year, ts.month + m
    y += (mo - 1) // 12
    mo = (mo - 1) % 12 + 1
    d = min(ts.day, 28)
    return pd.Timestamp(year=y, month=mo, day=d)


def load_series(code, fp=None):
    if fp is None:
        fp = os.path.join(MF, code + ".parquet")
    if not os.path.exists(fp):
        return None
    try:
        a = pq.read_table(fp, columns=["date", "open", "high", "low", "close"]).to_pandas()
    except Exception:
        return None
    if len(a) < 250:
        return None
    a["d"] = pd.to_datetime(a["date"]).dt.normalize()
    a = a.sort_values("d").drop_duplicates("d").reset_index(drop=True)
    for c in ("open", "high", "low", "close"):
        a[c] = a[c].astype("float64")

    tp = os.path.join(TF, code + ".parquet")
    if os.path.exists(tp):
        try:
            b = pq.read_table(tp, columns=["date", "close"]).to_pandas()
            b["d"] = pd.to_datetime(b["date"]).dt.normalize()
            b = b.sort_values("d").drop_duplicates("d")
            m = a[["d", "close"]].merge(b[["d", "close"]], on="d", suffixes=("", "_adj"))
            m = m[(m.d >= "2024-06-01") & m["close_adj"].notna() & (m["close"] > 0)]
            if len(m) > 20:
                m["r"] = m["close"] / m["close_adj"]          # 不复权 / 前复权
                cut = m[m.d <= "2025-12-31"]
                base_d = cut["d"].iloc[-1] if len(cut) else m["d"].iloc[0]
                base_r = float(m.loc[m.d == base_d, "r"].iloc[0])
                after = a["d"] > base_d
                for c in ("open", "high", "low", "close"):
                    a.loc[after, c] = a.loc[after, c] / base_r
        except Exception:
            pass
    return a


def simulate(a, d_dec=D_DEC, drop=DROP, wait_m=WAIT_M, tp=TP, trail=TRAIL, hold_m=HOLD_M,
             wait_drop=True):
    """返回 dict 或 None。wait_drop=False 时在决策日直接买入（C1 对照）。"""
    if a is None:
        return None
    i0 = a["d"].searchsorted(d_dec)
    if i0 >= len(a) or a["d"].iloc[i0] != d_dec:
        # 决策日停牌：取之后第一个交易日
        i0 = a["d"].searchsorted(d_dec)
        if i0 >= len(a):
            return None
    P = float(a["close"].iloc[i0])

    if wait_drop:
        trig = P * (1 - drop)
        d_wait_end = add_months(d_dec, wait_m)
        sub = a.iloc[i0 + 1:]
        sub = sub[sub["d"] <= d_wait_end]
        hit = sub[sub["low"] <= trig]
        if len(hit) == 0:
            return dict(entered=False, p_dec=P, days_wait=None)
        e = hit.iloc[0]
        entry = min(trig, float(e["open"]))
        i1 = a.index[a["d"] == e["d"]][0]
        days_wait = int(i1 - i0)
    else:
        entry = P
        i1 = i0
        days_wait = 0

    if not (entry > 0):
        return dict(entered=False, p_dec=P, days_wait=None)

    end_d = add_months(a["d"].iloc[i1], hold_m)
    peak = entry
    prev_close = entry
    for j in range(i1 + 1, len(a)):
        row = a.iloc[j]
        stop = peak * (1 - trail)
        if float(row["low"]) <= stop:
            px = min(stop, float(row["open"]))
            return dict(entered=True, entry=entry, exit=px, ret=px / entry - 1,
                        days=int(j - i1), days_wait=days_wait, reason="移动止损",
                        d_entry=str(a["d"].iloc[i1].date()), d_exit=str(row["d"].date),
                        p_dec=P, censored=False)
        tgt = entry * tp
        if float(row["high"]) >= tgt:
            px = max(tgt, float(row["open"]))
            return dict(entered=True, entry=entry, exit=px, ret=px / entry - 1,
                        days=int(j - i1), days_wait=days_wait, reason="止盈+50%",
                        d_entry=str(a["d"].iloc[i1].date()), d_exit=str(row["d"].date),
                        p_dec=P, censored=False)
        peak = max(peak, float(row["high"]))
        if row["d"] >= end_d:
            px = float(row["close"])
            return dict(entered=True, entry=entry, exit=px, ret=px / entry - 1,
                        days=int(j - i1), days_wait=days_wait, reason="满20个月",
                        d_entry=str(a["d"].iloc[i1].date()), d_exit=str(row["d"].date),
                        p_dec=P, censored=False)

    px = float(a["close"].iloc[-1])
    return dict(entered=True, entry=entry, exit=px, ret=px / entry - 1,
                days=int(len(a) - 1 - i1), days_wait=days_wait, reason="数据截止",
                d_entry=str(a["d"].iloc[i1].date()), d_exit=str(a["d"].iloc[-1].date),
                p_dec=P, censored=True)


def hold_ret(a, d0=D_DEC, months=HOLD_M):
    """C2/C3：决策日等权买入持有到 months 个月末"""
    if a is None:
        return None
    i0 = a["d"].searchsorted(d0)
    if i0 >= len(a):
        return None
    p0 = float(a["close"].iloc[i0])
    if not (p0 > 0):
        return None
    end_d = add_months(a["d"].iloc[i0], months)
    sub = a[(a["d"] >= end_d)]
    p1 = float(sub["close"].iloc[0]) if len(sub) else float(a["close"].iloc[-1])
    return dict(ret=p1 / p0 - 1, censored=len(sub) == 0)


def hold_after(a, d_start, months=HOLD_M):
    """从 d_start 当日收盘买入，持有 months 个月，不加任何止损止盈"""
    if a is None:
        return None
    i = a["d"].searchsorted(d_start)
    if i >= len(a):
        return None
    p0 = float(a["close"].iloc[i])
    if not (p0 > 0):
        return None
    end_d = add_months(a["d"].iloc[i], months)
    sub = a[a["d"] >= end_d]
    p1 = float(sub["close"].iloc[0]) if len(sub) else float(a["close"].iloc[-1])
    return p1 / p0 - 1


def stats(v):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return None
    return dict(n=int(len(v)), med=float(np.median(v)), mean=float(v.mean()),
                p10=float(np.percentile(v, 10)), p25=float(np.percentile(v, 25)),
                p75=float(np.percentile(v, 75)), p90=float(np.percentile(v, 90)),
                win=float((v > 0).mean()), gt50=float((v >= 0.5).mean()),
                lt20=float((v <= -0.2).mean()))


def main():
    print("载入龙虎榜…", flush=True)
    lhb = pq.read_table(LHB, columns=["代码", "上榜日", "名称"]).to_pandas()
    lhb["d"] = pd.to_datetime(lhb["上榜日"].astype(str).str[:10], errors="coerce")
    lhb["code"] = lhb["代码"].astype(str).str.zfill(6)
    s = lhb[(lhb.d >= STORY_START) & (lhb.d <= STORY_END)]
    cnt = s.groupby("code").size()
    story1 = set(cnt[cnt >= 1].index)
    story2 = set(cnt[cnt >= 2].index)
    print("  2024-02~08 上榜股票数: 1次+ =%d, 2次+ =%d" % (len(story1), len(story2)), flush=True)

    all_codes = []
    path_of = {}
    for f in glob.glob(MF + "/*.parquet"):
        nm = os.path.basename(f)[:-8]
        m = re.search(r"(\d{6})$", nm)
        c = m.group(1) if m else nm
        if c not in path_of:
            path_of[c] = f
        all_codes.append(c)
    all_codes = sorted(path_of)
    print("  全市场文件数 %d（去重 %d）" % (len(all_codes), len(set(all_codes))), flush=True)

    res = {"story1": [], "story2": [], "all": []}
    hold = {"story1": [], "story2": [], "all": []}
    c1 = {"story1": [], "story2": [], "all": []}
    holdtrig = {"story1": [], "story2": [], "all": []}
    cache = {}

    for k, code in enumerate(all_codes):
        a = load_series(code, path_of.get(code))
        if a is None:
            continue
        grp = "story2" if code in story2 else ("story1" if code in story1 else "all")
        groups = ["all"] + ([grp] if grp != "all" else [])
        cache[code] = a
        r = simulate(a)
        h = hold_ret(a)
        c1r = simulate(a, wait_drop=False) if (r is not None and r.get("entered")) else None
        htrig = None
        if r is not None and r.get("entered") and r.get("d_entry"):
            htrig = hold_after(a, pd.Timestamp(r["d_entry"]))
        for g in groups:
            if r is not None:
                res[g].append(r)
            if c1r is not None:
                c1[g].append(c1r)
            if h is not None:
                hold[g].append(h)
            if htrig is not None:
                holdtrig[g].append(htrig)
        if (k + 1) % 500 == 0:
            print("  ...%d/%d" % (k + 1, len(all_codes)), flush=True)

    out = {}
    for g in ("story1", "story2", "all"):
        rr = res[g]
        ent = [x for x in rr if x.get("entered")]
        neat = len(rr) - len(ent)
        rets = [x["ret"] for x in ent]
        st = stats(rets)
        # 未触发按 0 收益计入（他没买 = 没赚没亏）
        st_all = stats(rets + [0.0] * neat)
        waits = [x["days_wait"] for x in ent if x.get("days_wait") is not None]
        reasons = pd.Series([x["reason"] for x in ent]).value_counts().to_dict()
        out[g] = dict(
            n_total=len(rr), n_entered=len(ent), n_noentry=neat,
            trigger_rate=None if not rr else len(ent) / len(rr),
            wait_med_days=float(np.median(waits)) if waits else None,
            entered=st, include_noentry=st_all, reasons=reasons,
            c1_noWait=stats([x["ret"] for x in c1[g] if x and x.get("entered")]),
            holdFromTrigger=stats(holdtrig[g]),
            hold20m=stats([x["ret"] for x in hold[g]]),
            detail=ent[:40],
        )

    # 阈值敏感性
    sens = {}
    for drop in (0.0, 0.05, 0.10, 0.15, 0.20):
        rr = []
        for code in list(story1)[:400]:
            a = cache.get(code)
            if a is None:
                a = load_series(code)
            r = simulate(a, drop=drop) if a is not None else None
            if r is not None:
                rr.append(r)
        ent = [x for x in rr if x.get("entered")]
        sens["drop%.0f" % (drop * 100)] = dict(
            triggered=len(ent), n=len(rr), stats=stats([x["ret"] for x in ent]))

    out["sens_story1"] = sens
    out["meta"] = dict(D_DEC=str(D_DEC.date()), drop=DROP, tp=TP, trail=TRAIL,
                       hold_m=HOLD_M, wait_m=WAIT_M,
                       story_def="2024-02-01~2024-08-31 上过龙虎榜",
                       data_end="2026-07-03")
    json.dump(out, open("d50_result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n已写 d50_result.json")

    for g in ("story1", "story2", "all"):
        o = out[g]
        print("\n=== %s ===" % g)
        print("  候选 %d 只，触发入场 %d 只（%.1f%%），未触发（没买）%d 只"
              % (o["n_total"], o["n_entered"], 100 * (o["trigger_rate"] or 0), o["n_noentry"]))
        if o["entered"]:
            e = o["entered"]
            print("  入场后收益：中位 %+.2f%%  均值 %+.2f%%  P10 %+.1f%%  P25 %+.1f%%  P75 %+.1f%%  P90 %+.1f%%"
                  % (100 * e["med"], 100 * e["mean"], 100 * e["p10"], 100 * e["p25"], 100 * e["p75"], 100 * e["p90"]))
            print("  胜率 %.1f%%   赚到+50%% %.1f%%   亏20%%+ %.1f%%   等待中位 %.0f 交易日"
                  % (100 * e["win"], 100 * e["gt50"], 100 * e["lt20"], o["wait_med_days"] or -1))
            print("  退出原因:", o["reasons"])
        if o["c1_noWait"]:
            print("  C1 不等跌直接买：中位 %+.2f%% 均值 %+.2f%% 胜率 %.1f%%"
                  % (100 * o["c1_noWait"]["med"], 100 * o["c1_noWait"]["mean"], 100 * o["c1_noWait"]["win"]))
        if o["holdFromTrigger"]:
            ht = o["holdFromTrigger"]
            print("  C3 触发日买入纯持有20月：中位 %+.2f%% 均值 %+.2f%% 胜率 %.1f%%"
                  % (100 * ht["med"], 100 * ht["mean"], 100 * ht["win"]))
        if o["hold20m"]:
            h = o["hold20m"]
            print("  C2 全域买入持有20月：中位 %+.2f%% 均值 %+.2f%% 胜率 %.1f%%"
                  % (100 * h["med"], 100 * h["mean"], 100 * h["win"]))


if __name__ == "__main__":
    main()
