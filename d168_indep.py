# -*- coding: utf-8 -*-
"""d168 独立性检验：外生量是不是价格的影子？
判据：外生量对"同期价格变化"的 R²。
  R² 高 → 它只是价格的函数，仍是共象
  R² 低 → 它有自己的生命，可作根因候选
"""
import pandas as pd, numpy as np
EX='G:/AStockData/supplement/exogenous/'
d=pd.read_parquet('idx_sse_all.parquet'); d['date']=pd.to_datetime(d['date'])
d['close']=pd.to_numeric(d['close'],errors='coerce'); d=d.dropna(subset=['close']).set_index('date').sort_index()
c=d['close']

def load_m(f):
    x=pd.read_parquet(EX+f); x['日期']=pd.to_datetime(x['日期'])
    return x.set_index('日期').sort_index()
sh=load_m('margin_sh.parquet'); sz=load_m('margin_sz.parquet')
bal=(sh['融资融券余额'].add(sz['融资融券余额'],fill_value=np.nan))
fin=(sh['融资余额'].add(sz['融资余额'],fill_value=np.nan))
buy=(sh['融资买入额'].add(sz['融资买入额'],fill_value=np.nan))

# 月末对齐
def m_last(s):
    s=s.dropna()
    return s.resample('ME').last()
M=pd.DataFrame({'c':m_last(c),'bal':m_last(bal),'fin':m_last(fin),'buy':m_last(buy)}).dropna(subset=['c'])
# 基金发行月度
fn=pd.read_parquet(EX+'fund_new.parquet')
fn['成立日期']=pd.to_datetime(fn['成立日期'],errors='coerce'); fn['募集份额']=pd.to_numeric(fn['募集份额'],errors='coerce')
fn=fn.dropna(subset=['成立日期','募集份额'])
fund=fn.set_index('成立日期')['募集份额'].resample('ME').sum()
M['fund']=fund.reindex(M.index)
# 开户
ac=pd.read_parquet(EX+'account_new.parquet')
ac['数据日期']=pd.to_datetime(ac['数据日期']); ac=ac.set_index('数据日期').sort_index()
ac.index=ac.index+pd.offsets.MonthEnd(0)
M['open']=pd.to_numeric(ac['新增投资者-数量'],errors='coerce').reindex(M.index)

M=M[M.index>='2013-12-31'].copy()
M['r1']=M['c'].pct_change()
for k in ['bal','fin','buy','fund']:
    M['g_'+k]=np.log(M[k]).diff()
M['g_open']=np.log(M['open']).diff()

print('样本',len(M),M.index[0].date(),'→',M.index[-1].date())
print()
print('='*74)
print('【检验一】外生量的月度变化 vs 同期指数收益 —— R² 与 corr')
print('='*74)
print(f"{'外生量':<14}{'n':>5}{'corr':>8}{'R²':>8}{'t':>7}   判读")
rows=[]
for k,lab in [('bal','两融余额'),('fin','融资余额'),('buy','融资买入额'),('fund','新基金发行'),('open','新增开户')]:
    x=M['g_'+k]; y=M['r1']; m=pd.concat([x,y],axis=1).dropna()
    if len(m)<20: print(f'{lab:<14}{len(m):>5}  样本不足'); continue
    a,b=m.iloc[:,0].values,m.iloc[:,1].values
    r=float(np.corrcoef(a,b)[0,1]); n=len(m)
    t=r*np.sqrt((n-2)/max(1e-9,(1-r*r)))
    verdict='价格的影子' if r*r>0.5 else ('偏影子' if r*r>0.25 else '有独立性')
    print(f'{lab:<14}{n:>5}{r:>8.3f}{r*r:>8.3f}{t:>7.1f}   {verdict}')
    rows.append((lab,r,r*r,n))

print()
print('='*74)
print('【检验二】领先性：T 月读数 → 未来 12 个月收益/最大回撤')
print('='*74)
fwd_ret=M['c'].shift(-12)/M['c']-1
fwd_mdd=pd.Series(index=M.index,dtype=float)
for i in range(len(M)-12):
    seg=M['c'].values[i+1:i+13]
    fwd_mdd.iloc[i]=seg.min()/M['c'].values[i]-1
M['fret']=fwd_ret; M['fmdd']=fwd_mdd

for k,lab in [('bal','两融余额'),('fin','融资余额'),('buy','融资买入额'),('fund','新基金发行'),('open','新增开户')]:
    x=M['g_'+k]
    if x.notna().sum()<24: continue
    z=(x-x.rolling(36,min_periods=12).mean())/x.rolling(36,min_periods=12).std()
    mm=pd.concat([z,M['fret'],M['fmdd']],axis=1).dropna()
    if len(mm)<24: continue
    q=mm.iloc[:,0].quantile([0,.33,.67,1.0]).values
    mm['g']=pd.cut(mm.iloc[:,0],bins=q,labels=['低','中','高'],include_lowest=True)
    g=mm.groupby('g',observed=True).agg(n=('fret','size'),ret=('fret','median'),win=('fret',lambda s:(s>0).mean()),
        mdd=('fmdd','median'))
    g['ret']=(g.ret*100).round(1); g['win']=(g.win*100).round(0); g['mdd']=(g.mdd*100).round(1)
    print(f'--- {lab}（z 分档，n={len(mm)}）')
    print(g.to_string())
    print()
