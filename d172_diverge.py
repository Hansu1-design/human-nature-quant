# -*- coding: utf-8 -*-
"""d172 背离裁决：rA 说贵 而 PB 分位说便宜 时，未来一年谁对？"""
import pandas as pd, numpy as np
d=pd.read_parquet('idx_sse_all.parquet'); d['date']=pd.to_datetime(d['date'])
for x in ['close']: d[x]=pd.to_numeric(d[x],errors='coerce')
d=d.dropna(subset=['close']).set_index('date').sort_index(); d=d[d.index>='1996-01-01']
c=d['close']; idx=c.index
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
rA=((c/ci.reindex(c.index,method='ffill')*100)/1880.)
pb=pd.read_parquet('G:/AStockData/supplement/valuation/index_pb_沪深300.parquet')
pb['日期']=pd.to_datetime(pb['日期']); pb=pb.set_index('日期').sort_index()
arr=pd.to_numeric(pb['市净率'],errors='coerce').dropna().values
pidx=pd.to_numeric(pb['市净率'],errors='coerce').dropna().index
pct=np.array([np.nan]*len(arr))
for i in range(len(arr)):
    if i>=36: pct[i]=(arr[:i+1]<arr[i]).mean()
pctS=pd.Series(pct,index=pidx).reindex(idx,method='ffill')

D=pd.DataFrame({'c':c,'rA':rA,'pb':pctS}).dropna()
D['f1']=D['c'].shift(-250)/D['c']-1
print('样本',len(D),D.index[0].date(),'→',D.index[-1].date())
print('当前: rA',round(float(D['rA'].iloc[-1]),3),' PB分位',round(float(D['pb'].iloc[-1]),3),' 点位',round(float(D['c'].iloc[-1])))
print()
# 状态划分
def st(r,p):
    if r>1.20 and p<0.25: return '背离:rA贵/PB便宜'
    if r<=1.20 and p>=0.25: return '背离:rA便宜/PB贵'
    if r>1.20 and p>=0.25: return '一致:都贵'
    return '一致:都便宜'
D['s']=D.apply(lambda x: st(x.rA,x.pb),axis=1)
D2=D.dropna(subset=['f1'])
g=D2.groupby('s').agg(n=('f1','size'),ret=('f1','median'),win=('f1',lambda x:(x>0).mean()))
g['ret']=(g.ret*100).round(1); g['win']=(g.win*100).round(0)
print('='*70); print('【四种状态】未来 250 日'); print('='*70)
print(g.to_string())
print()
print('='*70); print('【背离:rA贵/PB便宜】出现的所有时段（≥20个交易日）'); print('='*70)
a=(D['s']=='背离:rA贵/PB便宜').values
runs=[];st0=None
for i in range(len(a)):
    if a[i] and st0 is None: st0=i
    if (not a[i]) and st0 is not None:
        if i-st0>=20: runs.append((i,st0,i-1))
        st0=None
if st0 is not None and len(a)-st0>=20: runs.append((0,st0,len(a)-1))
print(f"{'起':<12}{'止':<12}{'天':>5}{'rA':>7}{'PB分位':>8}{'当时点位':>9}{'之后250日':>10}")
for _,s,e in runs:
    r=D.iloc[s:e+1]
    f=D['c'].iloc[min(e+250,len(D)-1)]/D['c'].iloc[e]-1 if e+250<len(D) else np.nan
    print(f"{D.index[s].date()!s:<12}{D.index[e].date()!s:<12}{e-s+1:>5}{r['rA'].median():>7.2f}{r['pb'].median():>8.2f}{r['c'].iloc[-1]:>9.0f}"
          + (f'{f*100:>9.1f}%' if not np.isnan(f) else '        -'))
print()
print('注：背离段一旦结束即为"rA 也进入观察区"或"PB 抬到 25% 以上"')
