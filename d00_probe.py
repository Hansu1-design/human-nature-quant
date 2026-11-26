# -*- coding: utf-8 -*-
"""本地数据资产盘点：schema / 行数 / 日期范围 / 覆盖股票数"""
import os, sys, glob
import pandas as pd, pyarrow.parquet as pq

R = "G:/AStockData"

def brief(p, n=3):
    try:
        t = pq.read_table(p)
        df = t.to_pandas()
        cols = list(df.columns)
        print("  rows=%-8d cols=%s" % (len(df), cols))
        # 日期列探测
        for c in cols:
            cl = c.lower()
            if cl in ("date", "trade_date", "datetime", "d", "day", "时间", "日期", "end_date", "statdate", "ann_date"):
                s = df[c].astype(str)
                print("    %s: %s ~ %s  n_uniq=%d" % (c, s.min(), s.max(), s.nunique()))
        # 代码列
        for c in cols:
            cl = c.lower()
            if cl in ("code", "ts_code", "symbol", "secid", "stock_code", "代码"):
                print("    %s: n_uniq=%d sample=%s" % (c, df[c].nunique(), list(df[c].unique()[:3])))
        print("    head:")
        print(df.head(n).to_string(max_colwidth=18)[:900])
    except Exception as e:
        print("  ERR", repr(e)[:120])

def probe_dir(name, pattern, limit=None, n=3):
    fs = sorted(glob.glob(os.path.join(R, pattern)))
    print("\n=== %s ===  files=%d" % (name, len(fs)))
    if fs:
        print("  sample file:", fs[0], "size=%d" % os.path.getsize(fs[0]))
        brief(fs[0], n)

# 1. market_full 抽 3 只看一致性
print("### 1. market_full")
fs = sorted(glob.glob(R + "/market_full/*.parquet"))
print("files=%d" % len(fs))
for f in fs[:2]:
    print(" -", os.path.basename(f))
    brief(f, 2)

# 2. recent_2y
print("\n### 2. recent_2y")
fs2 = sorted(glob.glob(R + "/recent_2y/*.parquet"))
print("files=%d  sample=%s" % (len(fs2), os.path.basename(fs2[0]) if fs2 else "-"))
if fs2: brief(fs2[0], 2)

# 3. daily_tencent_full
print("\n### 3. daily_tencent_full")
fs3 = sorted(glob.glob(R + "/supplement/daily_tencent_full/*.parquet"))
print("files=%d  sample=%s" % (len(fs3), os.path.basename(fs3[0]) if fs3 else "-"))
if fs3: brief(fs3[0], 2)

# 4. youzi_history 四表
print("\n### 4. youzi_history")
for fn in ["gdhs", "lhb", "seat_stock", "block_trade"]:
    p = R + "/supplement/youzi_history/%s.parquet" % fn
    print("\n -- %s --" % fn)
    brief(p, 3)

# 5. 其他补充表
print("\n### 5. 其他")
for p in [R+"/supplement/zt_pool/zt_pool_recent.parquet",
          R+"/supplement/northbound/north_hgt.parquet",
          R+"/supplement/northbound/north_sgt.parquet",
          R+"/supplement/valuation/all_a_pe_ttm.parquet",
          R+"/supplement/valuation/all_a_pb.parquet",
          R+"/supplement/macro/cpi.parquet",
          R+"/supplement/macro/lpr.parquet",
          R+"/supplement/sw_industry/sw_l1_day.parquet",
          R+"/supplement/sw_industry/sw_l1_info.parquet",
          R+"/supplement/baostock/industry_classification.parquet"]:
    if not os.path.exists(p):
        print("\n -- MISSING %s" % p); continue
    print("\n -- %s --" % os.path.basename(p))
    brief(p, 2)

# 6. min1
print("\n### 6. min1")
fs6 = sorted(glob.glob(R + "/xiaoshi/min1/*.parquet"))
print("files=%d  sample=%s size=%d" % (len(fs6), os.path.basename(fs6[0]), os.path.getsize(fs6[0])))
if fs6: brief(fs6[0], 3)
