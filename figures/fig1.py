import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif']})
FS=5.4
fig,ax=plt.subplots(figsize=(4.8,2.05)); ax.set_xlim(0,24); ax.set_ylim(0,10); ax.axis('off')
nodes={'S':(2.2,7.2,'Suspect\nhost'),'DC':(6.0,7.2,'Domain\ncontroller'),'DB':(9.8,7.2,'DB\nreplica'),
       'GW':(3.6,3.0,'API\ngateway'),'WS':(8.4,3.0,'Work-\nstations')}
edges=[('S','DC'),('DC','DB'),('S','GW'),('DC','GW'),('DB','GW'),('GW','WS'),('DC','WS')]
for k,off in enumerate((0,12.2)):
    naive=k==0
    for a,b in edges:
        (xa,ya,_),(xb,yb,_)=nodes[a],nodes[b]
        cut=naive and 'S' in (a,b)
        ax.plot([xa+off,xb+off],[ya,yb],color='#b22222' if cut else '#666',lw=0.8,ls=(0,(3,2)) if cut else '-',zorder=0)
        if cut: ax.text((xa+xb)/2+off+0.1,(ya+yb)/2+0.25,'×',color='#b22222',fontsize=7,ha='center',va='center')
    for n,(x,y,t) in nodes.items():
        if n=='S': fc,ec='#f4c7c3' if naive else '#fde2b8','#333'
        elif naive and n in ('GW','WS'): fc,ec='#e6e6e6','#999'
        else: fc,ec='#dcefe0','#333'
        ax.add_patch(FancyBboxPatch((x+off-1.35,y-0.95),2.7,1.9,boxstyle='round,pad=0,rounding_size=0.25',fc=fc,ec=ec,lw=0.6))
        ax.text(x+off,y,t,ha='center',va='center',fontsize=FS,color='#777' if (naive and n in ('GW','WS')) else 'k',linespacing=1.1)
ax.text(6.0,9.5,'(a) Naive hard isolation',ha='center',fontsize=FS+0.6,weight='bold',color='#b22222')
ax.text(6.0,0.75,'All edges of the suspect host are cut; dependants\nthat fall below their threshold stop serving.',ha='center',fontsize=FS-0.2,va='center')
ax.text(18.2,9.5,'(b) Verified graduated containment',ha='center',fontsize=FS+0.6,weight='bold',color='#2e7d32')
ax.text(18.2,0.75,'Malicious process frozen (T1), lateral ports blocked (T2);\nauthentication and replication edges stay up.',ha='center',fontsize=FS-0.2,va='center')
ax.plot([12.0,12.0],[0.2,9.8],color='#bbb',lw=0.5)
fig.subplots_adjust(0.005,0.005,0.995,0.995); fig.savefig('fig/fig1_containment.png',dpi=400); print('ok')
