"""Prepare tennis arrays using the archived manuscript transformation."""
from pathlib import Path
from collections import Counter
import argparse,csv,json,random
import numpy as np
import pandas as pd
parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
W=args.output;W.mkdir(parents=True,exist_ok=False);(W/'data').mkdir()
P=json.loads((Path(__file__).resolve().parents[1]/'protocol/plan.json').read_text())
def savej(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:int(v) if isinstance(v,np.integer) else str(v)))
def ids_in(f):return {int(i) for col in ['players','opponent_players'] for a in f[col] for i in a}
def subset(f,eligible):return f[[all(int(i) in eligible for a in (r.players,r.opponent_players) for i in a) for r in f.itertuples()]].copy()
def scale(x):return {'mean':float(np.nanmean(x)),'std':max(float(np.nanstd(x)),1e-8)}
def store(domain,fit,test,ids,ctx_cols,targets,classification=False):
 d=W/'data'/domain;d.mkdir(exist_ok=True);lookup={i:k for k,i in enumerate(ids)}
 ss={t:scale(fit[t].to_numpy(float)) for t in targets}
 if classification:ss[targets[0]]={'mean':0.,'std':1.}
 for part,f in [('fit',fit),('test',test)]:
  arrays={'a':np.array([[lookup[int(i)] for i in a] for a in f.players],np.int32),'b':np.array([[lookup[int(i)] for i in a] for a in f.opponent_players],np.int32),'context':f[ctx_cols].to_numpy(np.float32),'cluster':f.cluster.astype(str).to_numpy(dtype=str),'row_id':f.row_id.astype(str).to_numpy(dtype=str)}
  for t in targets:arrays[t]=f[t].to_numpy(np.float32)
  np.savez_compressed(d/f'{part}.npz',**arrays)
  f.to_parquet(d/f'{part}_rows.parquet',index=False)
 np.save(d/'player_ids.npy',ids)
 meta={'domain':domain,'players':len(ids),'slots':len(fit.iloc[0].players),'context_dim':len(ctx_cols),'targets':targets,'classification':classification,'scaling':ss,'fit_rows':len(fit),'test_rows':len(test),'finite_target_rows':{part:{t:int(np.isfinite(f[t]).sum()) for t in targets} for part,f in [('fit',fit),('test',test)]}}
 savej(d/'meta.json',meta);return meta
# Tennis same provisional split, winner-independent orientation locked across model seeds.
base=args.raw;rows=[x for y in [2017,2018,2019] for x in csv.DictReader((base/f'atp_matches_doubles_{y}.csv').open())];rows=[r for r in rows if r['score'].strip() and not any(t in r['score'].upper() for t in ['RET','W/O','DEF','ABD'])];tours=sorted({r['tourney_id'] for r in rows});random.Random(17).shuffle(tours);test_tours=set(tours[:round(.2*len(tours))]);keys=['winner1_id','winner2_id','loser1_id','loser2_id'];counts=Counter(int(r[k]) for r in rows if r['tourney_id'] not in test_tours for k in keys);eligible={i for i,n in counts.items() if n>=20};rng=np.random.default_rng(P['tennis']['orientation_seed']);records=[]
for index,r in enumerate(rows):
 swap=bool(rng.integers(2));s1,s2=('loser','winner') if swap else ('winner','loser');pref='l_' if swap else 'w_';otherpref='w_' if swap else 'l_'
 q={'row_id':f'{r["tourney_id"]}_{r["match_num"]}_{index}','cluster':r['tourney_id'],'test':r['tourney_id'] in test_tours,'players':sorted(int(r[s1+str(i)+'_id']) for i in (1,2)),'opponent_players':sorted(int(r[s2+str(i)+'_id']) for i in (1,2)),'win':float(not swap)}
 for surf in P['tennis']['context']:q['surface_'+surf]=float((r['surface'] if r['surface'] in P['tennis']['context'] else 'Unknown')==surf)
 for suffix,pr in [('',pref),('_other',otherpref)]:
  try:
   den=float(r[pr+'svpt']);assert den>0
   vs=[float(r[pr+'ace'])/den,float(r[pr+'df'])/den,(float(r[pr+'1stWon'])+float(r[pr+'2ndWon']))/den];assert all(np.isfinite(v) for v in vs)
  except (ValueError,KeyError,AssertionError):vs=[np.nan]*3
  for t,v in zip(P['tennis']['targets'][1:],vs):q[t+suffix]=v
 records.append(q)
f=pd.DataFrame(records);fit=subset(f[~f.test],eligible);ids=sorted(ids_in(fit));test=subset(f[f.test],set(ids));targets=P['tennis']['targets'];tennis=store('tennis',fit,test,ids,['surface_'+s for s in P['tennis']['context']],targets,True)
# Preserve opposite-side rate labels for same-game mirrored reuse only.
for part,frame in [('fit',fit),('test',test)]:
 path=W/'data/tennis'/f'{part}.npz';z=dict(np.load(path));z.update({t+'_other':frame[t+'_other'].to_numpy(np.float32) for t in targets[1:]});np.savez_compressed(path,**z)
savej(W/'data/tennis/eligibility.json',{'counts':counts,'threshold':20,'eligible':sorted(eligible),'training_observed':ids,'test_tournaments':sorted(test_tours)})
