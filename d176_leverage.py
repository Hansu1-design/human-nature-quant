# -*- coding: utf-8 -*-
"""d176 两融作脆弱性指标（测回撤非收益）+ 换手成本核算"""
import pandas as pd, numpy as np
EX='G:/AStockData/supplement/exogenous/'
d=pd.read_parquet('idx_sse_all.parquet'); d['date']=pd.to_datetime(d['date'])
d['close']=pd.to_numeric(d['close'],errors='coerce'); d=d.dropna(subset=['close']).set_index('date').sort_index()
c=d['close']
def lm(f):
    x=pd.read_parquet(EX+f); x['日期']=pd.to_datetime(x['日期']); return x.set_index('日期').sort_index()
sh=lm('margin_sh.parquet'); sz=lm('margin_sz.parquet')
bal=sh['融资融券余额'].add(sz['融资融券余额'],fill_value=np.nan).dropna()
M=pd.DataFrame({'c':c,'bal':bal}).dropna()
M=M[M.index>='2010-04-01']
# 杠杆强度代理：两融余额/指数点位（市值∝点位），取其 3 年 z-score
lev=M['bal']/M['c']
M['levz']=(lev-lev.rolling(500,min_periods=120).mean())/lev.rolling(500,min_periods=120).std()
# 急涨：过去60日涨幅
M['r60']=M['c'].pct_change(60)
# 未来 250 日 最大回撤 与 收益
n=len(M); cv=M['c'].values
fmd=np.full(n,np.nan); fr=np.full(n,np.nan)
for i in range(n-250):
    s=cv[i+1:i+251]; fmd[i]=s.min()/cv[i]-1; fr[i]=s[-1]/cv[i]-1
M['fmd']=fmd; M['fr']=fr
S=M.dropna(subset=['levz','fmd','fr'])
print('样本',len(S),S.index[0].date(),'→',S.index[-1].date())
print()
print('='*76)
print('【两融杠杆强度 → 未来 250 日 最大回撤】（Hermes 建议：测回撤不是收益）')
print('='*76)
q=S['levz'].quantile([0,.33,.67,1.]).values
S=S.copy(); S['g']=pd.cut(S['levz'],bins=q,labels=['低杠杆','中','高杠杆'],include_lowest=True)
g=S.groupby('g',observed=True).agg(n=('fmd','size'),levz=('levz','median'),
    mdd_med=('fmd','median'),mdd_p25=('fmd',lambda x:x.quantile(.25)),
    ret=('fr','median'),win=('fr',lambda x:(x>0).mean()))
g['levz']=g.levz.round(2); g['mdd_med']=(g.mdd_med*100).round(1)
g['mdd_p25']=(g.mdd_p25*100).round(1); g['ret']=(g.ret*100).round(1); g['win']=(g.win*100).round(0)
print(g.to_string())
print()
print('【高杠杆 × 急涨】双条件（这才是强平踩踏的组合）')
print('='*76)
s2=S.copy()
s2['hl']=s2['levz']>0.5; s2['hot']=s2['r60']>0.15
for hl in [False,True]:
    for hot in [False,True]:
        ss=s2[(s2.hl==hl)&(s2.hot==hot)]
        if len(ss)<20: continue
        print(f'  高杠杆={hl!s:<5} 急涨={hot!s:<5} n={len(ss):4d}  '
              f'回撤中位 {ss.fmd.median()*100:6.1f}%  P25 {ss.fmd.quantile(.25)*100:6.1f}%  收益中位 {ss.fr.median()*100:5.1f}%')
print()
cur_levz=float(M['levz'].iloc[-1]); cur_r60=float(M['r60'].iloc[-1])
print('当前: 杠杆z',round(cur_levz,2),' 60日涨幅',f'{cur_r60*100:+.1f}%')
print('两融余额当前(万亿)',round(float(M['bal'].iloc[-1])/1e12,2))
# 历史分位
print('两融余额/点位 的历史分位:',round(float((lev<lev.iloc[-1]).mean()),3))

print()
print('='*76)
print('【换手成本核算】165 次往返 vs 26 次')
print('='*76)
for tr in [26,32,60,86,164,252]:
    print(f'  {tr:3d} 次单边 → 往返成本 {tr*0.001*100:5.1f}%  年化摊薄(18.5年) {tr*0.001/18.5*100:4.2f}%/年')
