# -*- coding: utf-8 -*-
"""d167 外生量初探：两融/开户/基金发行 能否领先于价格"""
import pandas as pd, numpy as np
EX='G:/AStockData/supplement/exogenous/'
d=pd.read_parquet('idx_sse_all.parquet'); d['date']=pd.to_datetime(d['date'])
d['close']=pd.to_numeric(d['close'],errors='coerce'); d=d.dropna(subset=['close']).set_index('date').sort_index()
c=d['close']; amt=pd.to_numeric(d.get('amount',np.nan),errors='coerce')
print('指数',len(c),c.index[0].date(),'→',c.index[-1].date())

# --- 两融：沪+深 合并 ---
def load_m(f):
    x=pd.read_parquet(EX+f); x['日期']=pd.to_datetime(x['日期'])
    return x.set_index('日期').sort_index()
sh=load_m('margin_sh.parquet'); sz=load_m('margin_sz.parquet')
bal=pd.concat([sh['融资融券余额'],sz['融资融券余额']],axis=1,keys=['sh','sz']).sum(axis=1,min_count=1)
fin=pd.concat([sh['融资余额'],sz['融资余额']],axis=1,keys=['sh','sz']).sum(axis=1,min_count=1)
print('两融合计余额 起',bal.dropna().index[0].date(),' 末',bal.dropna().index[-1].date(),
      ' 当前(亿)',round(float(bal.dropna().iloc[-1])/1e8,0))
# 单位：元。检查
print('融资余额当前(万亿)',round(float(fin.dropna().iloc[-1])/1e12,3))

# --- 两融账户（投资者数）---
ma=load_m('margin_account.parquet')
print('两融投资者数 起',ma['个人投资者数量'].first_valid_index().date(),'末',ma.index[-1].date())
print('  当前 个人(万)',ma['个人投资者数量'].iloc[-1],' 有负债者(万)',round(ma['有融资融券负债的投资者数量'].iloc[-1]/1e4,0))

# --- 基金发行（按月汇总募集份额）---
fn=pd.read_parquet(EX+'fund_new.parquet')
fn['成立日期']=pd.to_datetime(fn['成立日期'],errors='coerce')
fn['募集份额']=pd.to_numeric(fn['募集份额'],errors='coerce')
fn=fn.dropna(subset=['成立日期'])
print('基金 起',fn['成立日期'].min().date(),'末',fn['成立日期'].max().date(),' 只数',len(fn))
print('  募集份额分位',fn['募集份额'].quantile([.5,.9,.99]).round(2).tolist(),' 最大',round(float(fn['募集份额'].max()),2))
