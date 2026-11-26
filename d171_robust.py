# -*- coding: utf-8 -*-
"""d171 稳健性：PB 闸门阈值扫描 + 与 rA 重叠度 + 逐轮命中"""
import pandas as pd, numpy as np
RF=0.025
d=pd.read_parquet('idx_sse_all.parquet'); d['date']=pd.to_datetime(d['date'])
for x in ['close','amount']: d[x]=pd.to_numeric(d[x],errors='coerce')
d=d.dropna(subset=['close']).set_index('date').sort_index(); d=d[d.index>='1996-01-01']
c=d['close']; amt=d['amount']; n=len(c); idx=c.index
rf_d=(1+RF)**(1/250)-1
def curve(cond,cost=0.001):
    nav=np.ones(n); pos=0.; cash=1.; on=0; tr=0
    for i in range(n-1):
        w=bool(cond[i])
        if w and pos==0: pos=cash*(1-cost)/c.iloc[i+1]; cash=0.; tr+=1
        elif (not w) and pos>0: cash=pos*c.iloc[i+1]*(1-cost); pos=0.; tr+=1
        if pos>0: nav[i+1]=pos*c.iloc[i+1]; on+=1
        else: cash*=(1+rf_d); nav[i+1]=cash
    return pd.Series(nav,index=idx),on,tr
def met(nv,a='2008-05-01',b='2026-12-31'):
    s=nv[a:b]
    if len(s)<250: return np.nan,np.nan,np.nan
    ann=(s.iloc[-1]/s.iloc[0])**(365.25/(s.index[-1]-s.index[0]).days)-1
    dd=(s/s.cummax()-1).min()
    return ann,dd,ann/abs(dd)
# rA
cpi=pd.read_parquet('G:/AStockData/supplement/macro/cpi.parquet').copy()
cpi['m']=cpi['月份'].apply(lambda s: pd.Timestamp(int(str(s)[:4]),int(str(s)[5:7]),1))
cpi['yoy']=pd.to_numeric(cpi['全国-当月'],errors='coerce')
cpi=cpi.dropna(subset=['yoy']).sort_values('m').reset_index(drop=True)
cpi['mom']=(cpi['yoy']/100.)**(1/12.)
i0=cpi.index[cpi['m']==pd.Timestamp('2008-01-01')][0]
v=[np.nan]*len(cpi); v[i0]=100.
for i in range(i0+1,len(cpi)): v[i]=v[i-1]*cpi.loc[i,'mom']
for i in range(i0-1,-1,-1): v[i]=v[i+1]/cpi.loc[i+1,'mom']
ci=pd.Series(v,index=cpi['m'])
rA=((c/ci.reindex(c.index,method='ffill')*100)/1880.).values
# PB 分位
pb=pd.read_parquet('G:/AStockData/supplement/valuation/index_pb_沪深300.parquet')
pb['日期']=pd.to_datetime(pb['日期']); pb=pb.set_index('日期').sort_index()
arr=pd.to_numeric(pb['市净率'],errors='coerce').dropna().values
pidx=pd.to_numeric(pb['市净率'],errors='coerce').dropna().index
pct=np.array([np.nan]*len(arr))
for i in range(len(arr)):
    if i>=36: pct[i]=(arr[:i+1]<arr[i]).mean()
pctS=pd.Series(pct,index=pidx).reindex(idx,method='ffill').values

START='2008-05-01'   # PB 有效起点（留 36 期窗口）
print('='*82); print('【阈值扫描】PB 分位闸门（统一起点 2008-05，可比）'); print('='*82)
print(f"{'阈值':<8}{'年化':>9}{'回撤':>9}{'Calmar':>8}{'持仓%':>7}{'换手':>6}")
best=None
for th in [0.10,0.15,0.20,0.25,0.30,0.35,0.40]:
    cond=(pctS<=th)
    cond=np.where(np.isnan(cond),False,cond)
    nv,on,tr=curve(cond)
    ann,dd,cal=met(nv,START)
    s=nv[START:]; onr=(s.pct_change().abs()>0)
    print(f'≤{th*100:>3.0f}%   {ann*100:>8.2f}%{dd*100:>8.1f}%{cal:>8.3f}{tr/n*100:>13.0f}{tr:>6d}')
    if best is None or cal>best[1]: best=(th,cal,ann,dd)
print('最优:',best)
print()
print('='*82); print('【对照】rA 闸门同起点 2008-05'); print('='*82)
print(f"{'阈值':<8}{'年化':>9}{'回撤':>9}{'Calmar':>8}{'换手':>6}")
for th in [1.05,1.10,1.20,1.30,1.45,1.60]:
    cond=np.where(np.isnan(rA),False,rA<=th)
    nv,on,tr=curve(cond); ann,dd,cal=met(nv,START)
    print(f'≤{th:<7.2f}{ann*100:>8.2f}%{dd*100:>8.1f}%{cal:>8.3f}{tr:>6d}')
print()
print('='*82); print('【重叠度】PB≤20% 与 rA≤1.20 的命中日'); print('='*82)
a=np.where(np.isnan(pctS),False,pctS<=0.20); b=np.where(np.isnan(rA),False,rA<=1.20)
m=idx>=pd.Timestamp(START); a2,b2=a[m],b[m]
print('PB 命中',a2.sum(),' rA 命中',b2.sum(),' 交集',(a2&b2).sum())
print('PB 命中中 rA 也命中占',round(float((a2&b2).sum()/max(1,a2.sum()))*100,1),'%')
print('rA 命中中 PB 也命中占',round(float((a2&b2).sum()/max(1,b2.sum()))*100,1),'%')
print()
print('【rA 未命中但 PB 命中】的时段（PB 独有的机会）:')
only=a2&(~b2)
runs=[]; st=None
for i in range(len(only)):
    if only[i] and st is None: st=i
    if (not only[i]) and st is not None:
        if i-st>=10: runs.append((idx[m][st],idx[m][i-1]))
        st=None
for s,e in runs[:15]: print('   ',s.date(),'→',e.date(),f'({(e-s).days}天)')
