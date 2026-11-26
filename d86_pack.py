# -*- coding: utf-8 -*-
"""
d86: 打包 20 号报告所需全部数据
"""
import json
import numpy as np
import pandas as pd

sse = pd.read_parquet('idx_sse_all.parquet')
sse['close'] = pd.to_numeric(sse['close'], errors='coerce')
sse['d'] = pd.to_datetime(sse['date'])
sse = sse.dropna(subset=['close']).sort_values('d').reset_index(drop=True)

O = {}

# --- 上证周线（2005 起，降采样）---
w = sse[sse['d'] >= '2005-01-01'].set_index('d')['close'].resample('W-FRI').last().dropna()
O['sse_week'] = {'labels': [x.strftime('%Y-%m') for x in w.index],
                 'values': [round(float(v)) for v in w.values]}

# --- 三段 ---
segs = [{'i': 1, 'start': '2005-07-11', 'end': '2007-10-16', 'p0': 1011, 'p1': 6092, 'ret': 502,
         'months': 27.2, 'dd': -54.9, 'mdd': -16.6, 'pe': 25.3},
        {'i': 2, 'start': '2008-11-04', 'end': '2009-08-04', 'p0': 1707, 'p1': 3471, 'ret': 103,
         'months': 9.0, 'dd': -72.0, 'mdd': -13.3, 'pe': 18.4},
        {'i': 3, 'start': '2013-06-27', 'end': '2015-06-12', 'p0': 1950, 'p1': 5166, 'ret': 165,
         'months': 23.5, 'dd': -68.0, 'mdd': -11.7, 'pe': 36.4}]
O['segs'] = segs

# --- 阈值层 ---
O['layers'] = [{'th': 50, 'n': 6, 'share': 42.3, 'comp': 12150},
               {'th': 80, 'n': 3, 'share': 20.6, 'comp': 3129},
               {'th': 100, 'n': 3, 'share': 18.5, 'comp': 3146},
               {'th': 150, 'n': 1, 'share': 5.2, 'comp': 159}]
O['zz_sens'] = [{'zz': 15, 'n100': 3, 'share': 17.1}, {'zz': 20, 'n100': 3, 'share': 20.6},
                {'zz': 25, 'n100': 3, 'share': 18.5}, {'zz': 30, 'n100': 3, 'share': 18.5},
                {'zz': 40, 'n100': 3, 'share': 18.5}]

# --- 顶递降 ---
O['tops'] = [{'date': '2007-10-16', 'lv': 6092, 'pe': 53.6},
             {'date': '2015-06-12', 'lv': 5166, 'pe': 70.0},
             {'date': '2026-05-13', 'lv': 4243, 'pe': 42.1}]

# --- 多起点 ---
O['starts'] = [
    {'s': '2005-06', 'v0': 1081, 'yrs': 21.3, 'nom': 5.80, 'infl': None, 'real': None},
    {'s': '2008-01', 'v0': 4383, 'yrs': 18.7, 'nom': -1.05, 'infl': 2.07, 'real': -3.05},
    {'s': '2014-06', 'v0': 2048, 'yrs': 12.3, 'nom': 4.67, 'infl': 1.36, 'real': 3.27},
    {'s': '2019-01', 'v0': 2585, 'yrs': 7.7, 'nom': 4.37, 'infl': 0.77, 'real': 3.57},
    {'s': '2024-09', 'v0': 3336, 'yrs': 2.1, 'nom': 3.70, 'infl': None, 'real': None},
]

# --- M2 位置 ---
O['m2'] = [{'t': '2008-01', 'sse': 4383, 'm2': 41.8, 'ratio': 104.9},
           {'t': '2012-01', 'sse': 2293, 'm2': 85.6, 'ratio': 26.8},
           {'t': '2015-06', 'sse': 4277, 'm2': 133.3, 'ratio': 32.1},
           {'t': '2019-01', 'sse': 2585, 'm2': 186.6, 'ratio': 13.9},
           {'t': '2024-09', 'sse': 3336, 'm2': 309.5, 'ratio': 10.8},
           {'t': '2026-07', 'sse': 3832, 'm2': 355.5, 'ratio': 10.8}]

# --- 8 笔 ---
O['trades'] = [
    {'t': '2016-03-02', 'sse': 2850, 'cycle_hi': 5166, 'dd': -44.8},
    {'t': '2018-07-06', 'sse': 2747, 'cycle_hi': 5166, 'dd': -46.8},
    {'t': '2018-11-05', 'sse': 2665, 'cycle_hi': 5166, 'dd': -48.4},
    {'t': '2019-06-10', 'sse': 2852, 'cycle_hi': 5166, 'dd': -44.8},
    {'t': '2020-04-01', 'sse': 2735, 'cycle_hi': 5166, 'dd': -47.1},
    {'t': '2024-01-22', 'sse': 2756, 'cycle_hi': 3715, 'dd': -25.8},
    {'t': '2024-08-05', 'sse': 2861, 'cycle_hi': 3715, 'dd': -23.0},
    {'t': '2025-04-09', 'sse': 3187, 'cycle_hi': 3715, 'dd': -14.2},
]

# --- 跌 50% 信号 ---
O['dd_sig'] = [
    {'th': 50, 'n': 9, 'indep': 5, 'further': -15.1, 'r3': 23, 'worst': -25},
    {'th': 60, 'n': 3, 'indep': 3, 'further': -16.3, 'r3': -2, 'worst': -13},
]

# --- 收益对比 ---
O['perf'] = [
    {'case': '只做三段（事后完美）', 'mult': 32.4, 'yrs': 21.3, 'cagr': 17.7, 'note': '上帝视角上界'},
    {'case': '只做第三段 + 空仓 2%', 'mult': 3.32, 'yrs': 13.4, 'cagr': 9.39, 'note': '2013-06 起'},
    {'case': '买入持有上证', 'mult': 1.96, 'yrs': 21.9, 'cagr': 5.09, 'note': '2005-01 起'},
    {'case': '买入持有（第三段同期）', 'mult': 1.89, 'yrs': 13.4, 'cagr': 4.86, 'note': '2013-06 起'},
]

json.dump(O, open('d86_pack.json', 'w', encoding='utf-8'), ensure_ascii=False)
print('packed', {k: len(v) if isinstance(v, list) else 1 for k, v in O.items()})
