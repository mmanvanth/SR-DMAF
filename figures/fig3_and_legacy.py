import sys; sys.path.insert(0,'/home/claude/srdmaf')
import numpy as np, pandas as pd, networkx as nx, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from core import build_topology, cascade_fraction, blast_radius
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':8,'axes.linewidth':0.6})
DPI=400; W=4.8

def box(ax,x,y,w,h,text,fc='#ffffff',ec='#333333',fs=7,bold=False,ls='-'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01,rounding_size=0.02',fc=fc,ec=ec,lw=0.7,ls=ls))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,weight='bold' if bold else 'normal')
def arr(ax,x1,y1,x2,y2,c='#333333',ls='-',lw=0.8):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=7,color=c,lw=lw,ls=ls))

# ---------- Fig 1: naive vs graduated (schematic, no numbers)
fig,axs=plt.subplots(1,2,figsize=(W,2.1))
for k,ax in enumerate(axs):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    naive=k==0
    nodes={'S':(0.18,0.72,'Suspect host'),'DC':(0.5,0.72,'Domain\ncontroller'),'DB':(0.82,0.72,'DB replica'),
           'GW':(0.3,0.25,'API gateway'),'WS':(0.72,0.25,'Workstations')}
    edges=[('DC','S'),('DC','GW'),('S','GW'),('DB','GW'),('GW','WS'),('DC','WS'),('S','DB')]
    lost={'GW','WS'} if naive else set()
    for a,b in edges:
        xa,ya,_=nodes[a]; xb,yb,_=nodes[b]
        cut = naive and 'S' in (a,b)
        ax.plot([xa,xb],[ya,yb],color='#b22222' if cut else '#555555',lw=0.8,ls='--' if cut else '-',zorder=0)
    for n,(x,y,t) in nodes.items():
        if n=='S': fc='#f4c7c3' if naive else '#fde2b8'
        elif n in lost: fc='#e0e0e0'
        else: fc='#dcefe0' if not naive else '#ffffff'
        box(ax,x-0.13,y-0.09,0.26,0.18,t,fc=fc,fs=6.5)
    if naive:
        ax.text(0.5,0.02,'(a) Naive hard isolation: every edge of the suspect\nhost is cut; dependants below threshold fail',ha='center',va='bottom',fontsize=6.5)
        ax.text(0.18,0.93,'VLAN cut',color='#b22222',ha='center',fontsize=6.5,weight='bold')
    else:
        ax.text(0.5,0.02,'(b) Verified graduated containment: process freeze and\nlateral-port blocking; dependency edges preserved',ha='center',va='bottom',fontsize=6.5)
        ax.text(0.18,0.93,'T1 freeze + T2 segment',color='#b86e00',ha='center',fontsize=6.5,weight='bold')
fig.tight_layout(pad=0.2); fig.savefig('fig/fig1_containment.png',dpi=DPI); plt.close(fig)

# ---------- Fig 2: architecture
fig,ax=plt.subplots(figsize=(W,3.0)); ax.set_xlim(0,10); ax.set_ylim(0,6.4); ax.axis('off')
box(ax,0.1,1.2,1.7,4.2,'',fc='#eef3fb',ec='#2b5797')
ax.text(0.95,5.1,'Per-sample\nevidence',ha='center',va='center',fontsize=7,weight='bold')
for i,t in enumerate(['PE header\n(150 feat.)','Sandbox\nbehaviour (14)','Network\nactivity (3)']):
    box(ax,0.25,3.75-i*1.05,1.4,0.8,t,fs=6.3)
ag=[('A1 Static-structure\nagent',4.55),('A2 Behavioural\nagent',3.5),('A3 Network\nagent',2.45)]
for t,y in ag: box(ax,2.3,y-0.35,1.9,0.8,t,fc='#dff1ee',ec='#1f6f63',fs=6.5)
for i in range(3): arr(ax,1.65,4.15-i*1.05,2.3,4.15-i*1.05)
box(ax,2.3,0.95,1.9,1.0,'A4 Attribution agent\n(fused MLP, MC dropout)',fc='#dff1ee',ec='#1f6f63',fs=6.5)
arr(ax,1.8,2.1,2.3,1.5)
box(ax,4.75,1.2,2.4,4.2,'',fc='#fff6d9',ec='#a07800')
ax.text(5.95,5.1,'Self-Reflection &\nVerification Layer',ha='center',va='center',fontsize=7,weight='bold')
pil=['P1 Evidence sufficiency','P2 Epistemic uncertainty','P3 Cross-agent consensus','P4 Cascade bound']
for i,t in enumerate(pil): box(ax,4.9,3.95-i*0.72,2.1,0.55,t,fs=6.3)
for t,y in ag: arr(ax,4.2,y+0.05,4.75,y+0.05)
arr(ax,4.2,1.45,4.75,1.45)
box(ax,2.3,-0.05+0.1,1.9,0.75,'A5 Dependency agent\n(graph, blast radius)',fc='#ece4f5',ec='#5b3a8a',fs=6.3)
arr(ax,4.2,0.45,5.0,1.2,c='#5b3a8a')
box(ax,7.55,3.3,2.35,2.1,'',fc='#e3f2e6',ec='#2e7d32')
ax.text(8.72,5.05,'A6 Response planner',ha='center',fontsize=6.8,weight='bold')
for i,t in enumerate(['T1 process freeze','T2 micro-segmentation','T3 filtered proxy','T4 hard isolation*']):
    ax.text(7.7,4.6-i*0.38,t,fontsize=6.3,va='center')
arr(ax,7.15,4.3,7.55,4.3,c='#2e7d32'); ax.text(7.35,4.45,'HIGH',fontsize=5.5,ha='center',color='#2e7d32')
box(ax,7.55,1.9,2.35,0.9,'MEDIUM: T1 only,\nprobe and re-score',fc='#fdebd3',ec='#b86e00',fs=6.3)
arr(ax,7.15,2.35,7.55,2.35,c='#b86e00')
box(ax,7.55,0.6,2.35,0.95,'LOW: no disruptive\naction, escalate to SOC',fc='#f8d7d4',ec='#b22222',fs=6.3)
arr(ax,7.15,1.07,7.55,1.07,c='#b22222')
ax.text(8.72,0.22,'* only if $R_{cascade}(v)\\leq\\tau_d$',fontsize=6,ha='center')
fig.tight_layout(pad=0.1); fig.savefig('fig/fig2_architecture.png',dpi=DPI); plt.close(fig)

# ---------- Fig 3: blast radius vs simulated cascade (20 topologies)
R=[];Cc=[];T=[]
for s in range(20):
    G=build_topology(s); btw=nx.betweenness_centrality(G,normalized=True)
    for v in G: R.append(blast_radius(G,v,btw)); Cc.append(100*cascade_fraction(G,v,0.5)); T.append(G.nodes[v]['tier'])
R,Cc,T=map(np.array,(R,Cc,T))
fig,ax=plt.subplots(figsize=(W,2.1))
col={1:'#b22222',2:'#d98c00',3:'#2b5797'}
rng=np.random.default_rng(0)
for t in (3,2,1):
    m=T==t; ax.scatter(R[m]+rng.normal(0,0.003,m.sum()),Cc[m],s=6,alpha=0.55,color=col[t],label=f'Tier {t}',lw=0)
ax.axvline(0.40,ls='--',color='k',lw=0.7); ax.text(0.405,85,r'$\tau_d=0.40$',fontsize=7)
ax.set_xlabel(r'Estimated blast radius $R_{cascade}(v)$'); ax.set_ylabel('Simulated cascade\n(% of other hosts)')
ax.legend(frameon=False,fontsize=7,loc='upper left'); ax.spines[['top','right']].set_visible(False)
fig.tight_layout(pad=0.2); fig.savefig('fig/fig3_blastradius.png',dpi=DPI); plt.close(fig)

# ---------- Fig 4: trade-off
P=pd.read_csv('/home/claude/srdmaf/results_policies.csv'); P=P[P.eff_G==0.95]
short={'B1 naive detect-isolate':'B1','B2 uncertainty triage':'B2','B3 graduated only':'B3','SR-DMAF (full)':'SR-DMAF',
       'SR-DMAF w/o P1':'−P1','SR-DMAF w/o P2':'−P2','SR-DMAF w/o P3':'−P3','SR-DMAF w/o P4':'−P4','SR-DMAF no-graduated':'−grad.'}
fig,axs=plt.subplots(1,2,figsize=(W,2.2))
for ax,(metric,lab) in zip(axs,[('false_action','Actions on benign samples (%)'),('CFR','Mean cascading failure rate (%)')]):
    for scen,mk in (('uniform host','o'),('Tier-1 host','s')):
        g=P[P.scenario==scen].groupby('policy')[['containment',metric]].mean()*100
        for pol,row in g.iterrows():
            c='#2e7d32' if pol=='SR-DMAF (full)' else ('#b22222' if pol.startswith('B') else '#777777')
            ax.scatter(row.containment,row[metric],marker=mk,s=18,color=c,lw=0,alpha=0.9)
            if scen=='uniform host' or metric=='CFR':
                if metric=='false_action' and scen=='Tier-1 host': continue
                ax.annotate(short[pol],(row.containment,row[metric]),fontsize=5.5,xytext=(2,2),textcoords='offset points')
    ax.set_xlabel('Automated containment of malware (%)'); ax.set_ylabel(lab,fontsize=7)
    ax.spines[['top','right']].set_visible(False)
axs[1].scatter([],[],marker='o',color='k',s=12,label='uniform host'); axs[1].scatter([],[],marker='s',color='k',s=12,label='Tier-1 host')
axs[1].legend(frameon=False,fontsize=6,loc='upper left')
fig.tight_layout(pad=0.3); fig.savefig('fig/fig4_tradeoff.png',dpi=DPI); plt.close(fig)
print('ok')
