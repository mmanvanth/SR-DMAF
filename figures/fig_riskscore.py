import sys; sys.path.insert(0,'/home/claude/srdmaf')
import numpy as np, networkx as nx, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import build_topology, cascade_fraction, blast_radius
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':8,'axes.linewidth':0.6})
R=[];Cc=[];T=[]
for s in range(20):
    G=build_topology(s); btw=nx.betweenness_centrality(G,normalized=True)
    for v in G: R.append(blast_radius(G,v,btw)); Cc.append(100*cascade_fraction(G,v,0.5)); T.append(G.nodes[v]['tier'])
R,Cc,T=map(np.array,(R,Cc,T))
fig,ax=plt.subplots(figsize=(4.8,2.1))
col={1:'#b22222',2:'#d98c00',3:'#2b5797'}; rng=np.random.default_rng(0)
for t in (3,2,1):
    m=T==t; ax.scatter(R[m]+rng.normal(0,0.003,m.sum()),Cc[m],s=6,alpha=0.55,color=col[t],label=f'Tier {t}',lw=0)
ax.axvline(0.40,ls='--',color='k',lw=0.7); ax.text(0.405,85,r'$\tau_d=0.40$',fontsize=7)
ax.set_xlabel(r'Blast-radius risk score $R_{cascade}(v)$'); ax.set_ylabel('Simulated cascade\n(% of other hosts)')
ax.legend(frameon=False,fontsize=7,loc='upper left'); ax.spines[['top','right']].set_visible(False)
fig.tight_layout(pad=0.2); fig.savefig('fig/fig3_blastradius.png',dpi=400); print('ok', len(R))
