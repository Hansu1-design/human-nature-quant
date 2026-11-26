# -*- coding: utf-8 -*-
"""d173 细化：PB 分位亚档 & 那个 98% 胜率到底是什么时候"""
import pandas as pd, numpy as np
d=pd.read_parquet('idx_sse_all.parquet'); d['date']=pd.to_datetime(d['date'])
d['close']=pd.to_numeric(d['close'],errors='coerce'); d=d.dropna(subset=['close']).set_index('date').sort_index()
c=d['close']; idx=c.index
cpi=pd.read_parquet('G:/AStockData/supplement/macro/cpi.parquet').copy()
cpi['m']=cpi['月份'].apply(lambda s: pd.Timestamp(int(str(s)[:4]),int(str(s)[5:7]),1))
cpi['yoy']=pd.to_numeric(cpi['全国-当月'],errors='coerce')
cpi=cpi.dropna(subset=['yoy']).sort_values('m').reset_index(drop=True)
cpi['mom']=(cpi['yoy']/100.)**(1/12.); i0=cpi.index[cpi['m']==pd.Timestamp('2008-01-01')][0]
v=[np.nan]*len(cpi); v[i0]=100.
for i in range(i0+1,len(cpi)): v[i]=v[i-1]*cpi.loc[i,'mom']
for i in range(i0-1,-1,-1): v[i]=v[i+1]/cpi.loc[i+1,'mom']
rA=((c/pd.Series(v,index=cpi['m']).reindex(c.index,method='ffill')*100)/1880.)
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
D2=D.dropna(subset=['f1']).copy()

print('='*72); print('【PB 分位亚档】不看 rA，未来 250 日'); print('='*72)
bins=[0,0.10,0.15,0.20,0.25,0.35,1.0]
D2['pbg']=pd.cut(D2['pb'],bins=bins,labels=['≤10%','10-15%','15-20%','20-25%','25-35%','>35%'])
g=D2.groupby('pbg',observed=True).agg(n=('f1','size'),ret=('f1','median'),
    p25=('f1',lambda x:x.quantile(.25)),win=('f1',lambda x:(x>0).mean()))
g['ret']=(g.ret*100).round(1); g['p25']=(g.p25*100).round(1); g['win']=(g.win*100).round(0)
print(g.to_string())
print()
print('各档出现的年份分布:')
for lab in g.index:
    yr=D2[D2.pbg==lab].index.year
    print(f'  {lab:<8}', dict(pd.Series(yr).value_counts().sort_index().head(12)))

print()
print('='*72); print('【rA便宜/PB贵 那 153 天是什么时候】'); print('='*72)
s=D2[(D2.rA<=1.20)&(D2.pb>=0.25)]
print('时间:',s.index.min().date(),'→',s.index.max().date(),' n=',len(s))
print('年份:',dict(pd.Series(s.index.year).value_counts().sort_index()))
print('rA 中位',round(float(s.rA.median()),2),' PB分位中位',round(float(s.pb.median()),2))
print('未来250日 中位',round(float(s.f1.median())*100,1),'%  最小',round(float(s.f1.min())*100,1),'%  最大',round(float(s.f1.max())*100,1),'%')

print()
print('='*72); print('【当前定位】'); print('='*72)
cur=D.iloc[-1]
print('当前 rA',round(float(cur.rA),3),' PB分位',round(float(cur.pb),3),' 点位',round(float(cur.c)))
# 历史最近邻
D2['dist']=((D2.rA-cur.rA).abs()/0.1+(D2.pb-cur.pb).abs()/0.05)
nb=D2.nsmallest(40,'dist')
print('最接近当前状态的 40 个历史交易日：')
print('  年份分布',dict(pd.Series(nb.index.year).value_counts().sort_index()))
print('  未来250日 中位',round(float(nb.f1.median())*100,1),'%  胜率',round(float((nb.f1>0).mean())*100,0),'%')
print('  列表:',[f"{i.date()}:{v*100:+.0f}%" for i,v in nb['f1'].head(12).items()])
