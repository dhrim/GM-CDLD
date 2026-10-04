"""GM-CDLD graphical abstract. Run with Python + matplotlib.
Source: submitted manuscript en_v3, Abstract and Sections 3, 5–6.
Vector cells are schematic, not measured values. No generated imagery.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Rectangle

OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(16,8.4))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax.set(xlim=(0,1600),ylim=(0,840)); ax.axis('off')
navy='#163345';teal='#007F83';muted='#647580';line='#CFDCE2';pale='#F2F6F8';orange='#B45E30'
def text(x,y,s,size=15,color=navy,weight='normal',ha='left',va='center'):
 return ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va=va,linespacing=1.45)
def box(x,y,w,h,fc='white',ec=line,lw=1.3,r=12):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',facecolor=fc,edgecolor=ec,linewidth=lw))
def arrow(x1,y1,x2,y2,color=muted,lw=2.1):
 ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=17,color=color,linewidth=lw))
def member(x,y,name,active=False):
 ax.add_patch(Circle((x,y),20,facecolor=teal if active else 'white',edgecolor=teal if active else line,linewidth=1.5))
 text(x,y,name,13,'white' if active else navy,'bold','center')

text(55,785,'GM-CDLD',30,teal,'bold')
text(287,785,'From group interactions to member-specific latents',24,navy,'bold')
text(55,741,'Discover information at the member level; evaluate its use in other group outcomes.',16,muted)
for x,n,title in [(55,'01','Observe group outcomes'),(565,'02','Discover member latents'),(1075,'03','Reuse in other outputs')]:
 box(x,246,470,441,pale,'none')
 text(x+23,653,n,15,teal,'bold');text(x+65,653,title,18,navy,'bold')

text(80,603,'Repeated matches with changing partners',13,muted)
for y,players,result in [(542,'ABCD','Win'),(450,'ACBD','Loss'),(358,'ADBC','Win')]:
 box(78,y-32,129,64);box(255,y-32,129,64)
 for x,name in zip([111,174,288,351],players):member(x,y,name,name=='A')
 text(231,y,'vs',12,muted,ha='center')
 text(410,y,result,15,navy,'bold')
text(80,277,'Doubles tennis shown schematically',12,muted)
arrow(529,470,560,470,teal)

text(591,603,'One persistent vector for each member',13,muted)
colors=['#0D8D91','#51A6AA','#A0CDCE','#D9E9E9','#6BB5B6','#C0DEDF']
for idx,name in enumerate('ABCD'):
 y=548-58*idx
 member(610,y,name,name=='A')
 for j,c in enumerate(colors[idx:]+colors[:idx]):
  ax.add_patch(Rectangle((648+40*j,y-15),33,30,facecolor=c,edgecolor='white'))
 if name=='A':text(916,y,'Update',12,teal,'bold')
 else:text(916,y,'Fixed',12,muted)
text(591,302,'Update one member + active Finder;',13,navy)
text(591,274,'cycle across members and both Finders.',13,navy)
arrow(1039,470,1070,470,teal)

box(1100,548,420,64,'#E2F0EF',teal)
text(1310,580,'Freeze the discovered vectors',16,teal,'bold','center')
arrow(1310,542,1310,510,teal)
text(1310,491,'Train a separate Predictor per output',13,muted,ha='center')
for y,label in [(433,'Team ace rate'),(367,'Team service-point win rate'),(301,'Team double-fault rate')]:
 box(1100,y-25,420,50,'white',line)
 text(1310,y,label,15,navy,ha='center')

ax.plot([55,1545],[211,211],color=line,lw=1.1)
text(55,177,'METHOD',12,teal,'bold')
text(210,177,'Two interacting entities → two interacting groups → individual member latents',16,navy,'bold')
text(55,128,'EVIDENCE',12,teal,'bold')
text(210,130,'Tennis: lower reuse errors than the mean baseline; direct ridge was more accurate.',14,navy)
text(210,94,'Basketball: reuse errors remained near the mean baseline.',14,navy)
text(55,40,'Group-to-Member Cyclic Dual Latent Discovery',11,muted)
text(1545,40,'Dohyoung Rim  |  GM-CDLD',11,muted,ha='right')
fig.savefig(OUT/'GM-CDLD_graphical_abstract_v1.png',dpi=220,facecolor='white')
fig.savefig(OUT/'GM-CDLD_graphical_abstract_v1.svg',facecolor='white')
plt.close(fig)
print(OUT/'GM-CDLD_graphical_abstract_v1.png')
