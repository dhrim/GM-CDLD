"""Rebuild the three published beach source definitions from the pinned CSV files."""
from pathlib import Path
import argparse,csv,json,collections,re
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args();raw=a.raw;w=a.output;w.mkdir(parents=True,exist_ok=False)
p={'targets':['blocks_per_set','digs_per_set']}
records=[];excluded=[]
for year in range(2005,2010):
 for r in csv.DictReader((raw/f'bvb_matches_{year}.csv').open(encoding='latin-1')):
  if r['gender']!='M' or r['circuit']!='AVP':continue
  key='|'.join(r[x] for x in ['year','date','tournament','match_num']);group='|'.join(r[x] for x in ['year','date','tournament'])
  try:
   teams=[[r[f'{s}_player{j}'].strip() for j in [1,2]] for s in ['w','l']];sets=re.findall(r'(\d+)\s*-\s*(\d+)',r['score']);assert len(sets) in (2,3) and len(set(sum(teams,[])))==4
   yy=[]
   for s in ['w','l']:
    sums={x:sum(float(r[f'{s}_p{j}_tot_{x}']) for j in [1,2]) for x in ['attacks','kills','errors','blocks','digs']};assert sums['attacks']>0 and all(v>=0 for v in sums.values()) and sums['kills']+sums['errors']<=sums['attacks']
    yy.append({'shot_success':(sums['kills']-sums['errors'])/sums['attacks'],'blocks_per_set':sums['blocks']/len(sets),'digs_per_set':sums['digs']/len(sets)})
   records.append(dict(key=key,group=group,teams=teams,y=yy))
  except (ValueError,AssertionError):excluded.append(key)
# Exact source match keys must not be silently duplicated.
assert len({r['key'] for r in records})==len(records)
groups=sorted({r['group'] for r in records});order=np.random.default_rng(1262026).permutation(groups);n=len(order);split={g:('fit' if j<int(n*.7) else 'val' if j<int(n*.85) else 'test') for j,g in enumerate(order)}
keep=set(sum([sum(r['teams'],[]) for r in records],[]))
while True:
 fit=[r for r in records if split[r['group']]=='fit' and set(sum(r['teams'],[]))<=keep];cnt=collections.Counter(x for r in fit for x in sum(r['teams'],[]));new={x for x,v in cnt.items() if v>=10}
 if new==keep:break
 keep=new
names=sorted(keep);idx={x:i for i,x in enumerate(names)};parts={s:[r for r in records if split[r['group']]==s and set(sum(r['teams'],[]))<=keep] for s in ['fit','val','test']};arrays={}
for s,rr in parts.items():
 a=[];bb=[];ys={t:[] for t in ['shot_success']+p['targets']};ids=[];clusters=[]
 for r in rr:
  for side in range(2):
   a.append([idx[x] for x in r['teams'][side]]);bb.append([idx[x] for x in r['teams'][1-side]]);ids.append(r['key']+f':{side}');clusters.append(r['group'])
   for t in ys:ys[t].append(r['y'][side][t])
 arrays[s]=dict(a=np.array(a,np.int32),b=np.array(bb,np.int32),context=np.empty((len(a),0),np.float32),row_id=np.array(ids),cluster=np.array(clusters),**{t:np.array(v,np.float32) for t,v in ys.items()})
m=dict(players=len(names),slots=2,context_dim=0,classification=False,targets=['shot_success']+p['targets'],player_names=names,scaling={t:dict(mean=float(arrays['fit'][t].mean()),std=float(arrays['fit'][t].std())) for t in ys})
for kind in ['source','reuse']:
 d=w/'data'/kind;d.mkdir(parents=True,exist_ok=True);(d/'meta.json').write_text(json.dumps(m,indent=2))
 for s,z in arrays.items():np.savez_compressed(d/f'{s}.npz',**z)

summary=dict(players=len(names),matches={s:len(rr) for s,rr in parts.items()},tournaments={s:len({r['group'] for r in rr}) for s,rr in parts.items()},excluded_missing_or_invalid=len(excluded))
(w/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
lookup={}
valid_keys={z['key'] for rr in parts.values() for z in rr}
for year in range(2005,2010):
 for row in csv.DictReader((raw/f'bvb_matches_{year}.csv').open(encoding='latin-1')):
  if row['gender']!='M' or row['circuit']!='AVP':continue
  key='|'.join(row[x] for x in ['year','date','tournament','match_num'])
  if key not in valid_keys:continue
  sets=re.findall(r'(\d+)\s*-\s*(\d+)',row['score'])
  win=sum(int(x) for x,y in sets);lose=sum(int(y) for x,y in sets)
  try:
   rates=[sum(float(row[f'{side}_p{j}_tot_kills']) for j in [1,2])/sum(float(row[f'{side}_p{j}_tot_attacks']) for j in [1,2]) for side in ['w','l']]
  except (ValueError,ZeroDivisionError):continue
  lookup[key]={'point_share':[win/(win+lose),1-win/(win+lose)],'attack_success':rates}
for variant in ['attack_efficiency','point_share','attack_success']:
 for kind in ['source','reuse']:
  out=w/variant/'data'/kind;out.mkdir(parents=True)
  meta=json.loads((w/'data'/kind/'meta.json').read_text())
  for part in ['fit','val','test']:
   z=dict(np.load(w/'data'/kind/f'{part}.npz'))
   if variant!='attack_efficiency':
    z['shot_success']=np.asarray([lookup[str(rid).rsplit(':',1)[0]][variant][int(str(rid).rsplit(':',1)[1])] for rid in z['row_id']],np.float32)
   if part=='fit':meta['scaling']['shot_success']={'mean':float(z['shot_success'].mean()),'std':float(z['shot_success'].std())}
   np.savez_compressed(out/f'{part}.npz',**z)
  if variant=='point_share':meta['source_target_definition']='shot_success is legacy internal key for team_point_share = team total points / both teams total points'
  if variant=='attack_success':meta['source_target_definition']='shot_success is legacy key: team total kills / team total attacks'
  (out/'meta.json').write_text(json.dumps(meta,indent=2))
