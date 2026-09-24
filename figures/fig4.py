import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':6.5,'axes.linewidth':0.5})
P=pd.read_csv('/home/claude/srdmaf/results_policies.csv'); P=P[P.eff_G==0.95]
pols=['B1 naive detect-isolate','B2 uncertainty triage','B3 graduated only','SR-DMAF (full)']
lab=['B1','B2','B3','SR-DMAF']; col=['#b22222','#d98c00','#6d8fbf','#2e7d32']
fig,axs=plt.subplots(1,3,figsize=(4.8,1.75))
def bars(ax,vals,errs,title,fmt='{:.1f}'):
    x=np.arange(len(vals)); ax.bar(x,vals,yerr=errs,color=col,width=0.65,error_kw=dict(lw=0.5,capsize=1.5))
    ax.set_xticks(x); ax.set_xticklabels(lab,rotation=35,ha='right',fontsize=5.8); ax.set_title(title,fontsize=6.5)
    for i,(v,e) in enumerate(zip(vals,errs)): ax.text(i,v+e+max(vals)*0.03,fmt.format(v),ha='center',fontsize=5.2)
    ax.spines[['top','right']].set_visible(False); ax.set_ylim(0,max(vals+errs)*1.25)
u=P[P.scenario=='uniform host']
g=u.groupby('policy'); m=g.mean(numeric_only=True)*100; s=g.std(numeric_only=True)*100
bars(axs[0],m.loc[pols,'containment'].values,s.loc[pols,'containment'].values,'(a) Automated containment (%)')
bars(axs[1],m.loc[pols,'false_action'].values,s.loc[pols,'false_action'].values,'(b) Actions on benign (%)',fmt='{:.2f}')
t=P[P.scenario=='Tier-1 host']; mt=t.groupby('policy').mean(numeric_only=True)*100; st=t.groupby('policy').std(numeric_only=True)*100
bars(axs[2],mt.loc[pols,'CFR'].values,st.loc[pols,'CFR'].values,'(c) CFR, Tier-1 host (%)',fmt='{:.2f}')
fig.tight_layout(pad=0.25,w_pad=0.6); fig.savefig('fig/fig4_tradeoff.png',dpi=400); print('ok')
