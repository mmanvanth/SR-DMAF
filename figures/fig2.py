import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'dejavuserif'})
FS=5.2
def box(ax,x,y,w,h,text='',fc='#fff',ec='#333',fs=FS,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=0.08',fc=fc,ec=ec,lw=0.6))
    if text: ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,weight='bold' if bold else 'normal',linespacing=1.15)
def arr(ax,x1,y1,x2,y2,c='#333'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=6,color=c,lw=0.7,shrinkA=0,shrinkB=0))
fig,ax=plt.subplots(figsize=(4.8,2.7)); ax.set_xlim(0,12); ax.set_ylim(0,6.75); ax.axis('off')
# evidence
box(ax,0.05,1.55,2.0,5.1,fc='#eef3fb',ec='#2b5797'); ax.text(1.05,6.1,'Per-sample\nevidence',ha='center',fontsize=FS+0.3,weight='bold')
ev=['PE header\n(150 features)','Sandbox\nbehaviour (14)','Network\nactivity (3)']
ys=[5.0,3.75,2.5]
for t,y in zip(ev,ys): box(ax,0.2,y-0.45,1.7,0.9,t)
# agents
names=['A1 Static-structure\nagent','A2 Behavioural\nagent','A3 Network\nagent']
for t,y in zip(names,ys):
    box(ax,2.55,y-0.45,2.1,0.9,t,fc='#dff1ee',ec='#1f6f63'); arr(ax,1.9,y,2.55,y)
box(ax,2.55,0.95,2.1,1.05,'A4 Attribution\nagent (all views,\nMC dropout)',fc='#dff1ee',ec='#1f6f63')
ax.plot([2.2,2.2],[5.0,1.47],color='#333',lw=0.7); arr(ax,2.2,1.47,2.55,1.47)
for y in ys: ax.plot([2.05,2.2],[y-0.25,y-0.25],color='#333',lw=0.7)
box(ax,2.55,0.05,2.1,0.8,'A5 Dependency agent\n(graph, blast radius)',fc='#ece4f5',ec='#5b3a8a')
# SRVL
box(ax,5.2,0.05,2.75,6.6,fc='#fff6d9',ec='#a07800'); ax.text(6.575,6.2,'Self-Reflection &\nVerification Layer',ha='center',va='center',fontsize=FS+0.3,weight='bold')
pil=['P1 Evidence\nsufficiency','P2 Epistemic\nuncertainty','P3 Cross-agent\nconsensus','P4 Cascade\nbound']
for i,t in enumerate(pil): box(ax,5.4,4.6-i*1.3,2.35,0.95,t)
for y in ys: arr(ax,4.65,y,5.2,y)
arr(ax,4.65,1.47,5.2,1.47); arr(ax,4.65,0.45,5.2,0.45,c='#5b3a8a')
# outcomes
box(ax,8.55,3.7,3.4,2.95,fc='#e3f2e6',ec='#2e7d32'); ax.text(10.25,6.3,'HIGH → A6 response planner',ha='center',fontsize=FS,weight='bold')
for i,t in enumerate(['T1 process freeze','T2 SDN micro-segmentation','T3 filtered proxy tunnel','T4 hard isolation (only if\n     $R_{cascade}(v)\\leq\\tau_d$)']):
    ax.text(8.7,5.85-i*0.52,t,fontsize=FS,va='top',linespacing=1.1)
box(ax,8.55,2.0,3.4,1.3,'MEDIUM → T1 only;\nprobe and re-score',fc='#fdebd3',ec='#b86e00')
box(ax,8.55,0.05,3.4,1.55,'LOW → no disruptive\naction; escalate to SOC\nwith the audit trace',fc='#f8d7d4',ec='#b22222')
arr(ax,7.95,5.1,8.55,5.1,c='#2e7d32'); arr(ax,7.95,2.65,8.55,2.65,c='#b86e00'); arr(ax,7.95,0.8,8.55,0.8,c='#b22222')
fig.subplots_adjust(0.005,0.005,0.995,0.995); fig.savefig('fig/fig2_architecture.png',dpi=400); print('ok')
