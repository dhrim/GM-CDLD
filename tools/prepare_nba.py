"""Rebuild fixed NBA points/reuse arrays from public possession/PBP/boxscore sources.
The committed split metadata preserve the historical game allocation. No outcome arrays
or intermediate research-folder files are required.
"""
from pathlib import Path
from collections import Counter
import argparse,json,re,tarfile
import numpy as np
import pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args();ROOT=Path(__file__).resolve().parents[1];W=a.output;W.mkdir(parents=True,exist_ok=False);(W/'data').mkdir()
P=json.loads((ROOT/'protocol/plan.json').read_text())
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

schedule=pd.read_csv(ROOT/'data/splits/NBA/schedule.csv');D=set(schedule.loc[schedule.candidate_split=='D','GAME_ID']);E=set(schedule.loc[schedule.candidate_split=='E','GAME_ID']);game_ids=D|E
with tarfile.open(a.raw/'cdnnba_2023.tar.xz') as t:
 member=next(x for x in t.getmembers() if x.isfile() and x.name.endswith('.csv'))
 cdn=pd.read_csv(t.extractfile(member),low_memory=False)
box=pd.read_parquet(a.raw/'player_boxscores_2024.parquet');box.game_id=box.game_id.astype(int)
def remaining(clock):
 m,s=re.match(r'PT(\d+)M([\d.]+)S',clock).groups();return round((int(m)*60+float(s))*1000)
intervals=[];failures=[]
for game,frame in cdn[cdn.gameId.isin(game_ids)].groupby('gameId',sort=True):
 local=[]
 try:
  gb=box[box.game_id==game];teams=gb[['team_id','side']].drop_duplicates();assert len(teams)==2
  active={int(r.team_id):set(gb.loc[(gb.team_id==r.team_id)&gb.position.fillna('').ne(''),'person_id'].astype(int)) for r in teams.itertuples()};sides=dict(zip(teams.team_id,teams.side));assert all(len(v)==5 for v in active.values());offset=0
  for period in sorted(frame.period.unique()):
   length=720000 if period<=4 else 300000;subs=frame[(frame.period==period)&(frame.actionType=='substitution')].copy();subs['elapsed_ms']=length-subs.clock.map(remaining);prev=0
   def emit(start,end):
    if start==end:return
    assert 0<=start<end<=length and all(len(v)==5 for v in active.values())
    assert not set.intersection(*active.values())
    for team,players in active.items():
     other=next(t for t in active if t!=team);local.append(dict(game_id=int(game),period=int(period),team_id=team,opponent_team_id=other,home=int(sides[team]=='home'),start_ms=offset+start,end_ms=offset+end,players=sorted(players),opponent_players=sorted(active[other])))
   for moment,changes in subs.groupby('elapsed_ms',sort=True):
    emit(prev,int(moment))
    for ev in changes.sort_values('orderNumber').itertuples():
     team=int(ev.teamId);i=int(ev.personId)
     if ev.subType=='out':assert i in active[team];active[team].remove(i)
     elif ev.subType=='in':assert i not in active[team];active[team].add(i)
     else:raise ValueError(ev.subType)
    prev=int(moment)
   emit(prev,length);offset+=length
  intervals.extend(local)
 except Exception as ex:failures.append({'game':int(game),'error':repr(ex)})
savej(W/'data/lineup_failures.json',failures)
if failures:raise RuntimeError('E lineup reconstruction incomplete; preserve and resolve before training')
lineups=pd.DataFrame(intervals);groups={k:g for k,g in lineups.groupby(['game_id','period','opponent_team_id'])};teams={r.teamTricode:int(r.teamId) for r in cdn[['teamTricode','teamId']].dropna().drop_duplicates().itertuples()}

with tarfile.open(a.raw/'pbpstats_2023.tar.xz') as t:raw=pd.read_csv(t.extractfile('pbpstats_2023.csv'))
def possessions(ids,prefix):
 d=raw[raw.GAMEID.isin(ids)].drop_duplicates([k for k in raw if k not in ['DESCRIPTION','URL']]).reset_index(drop=True)
 final=[];excluded=[]
 for i,r in enumerate(d.itertuples()):
  events=str(r.EVENTS);period=int(r.PERIOD);offset=min(period-1,4)*720+max(period-5,0)*300;length=720 if period<=4 else 300
  def clock(v):m,s=str(v).split(':');return int(m)*60+float(s)
  start=round((offset+length-clock(r.STARTTIME))*1000);end=round((offset+length-clock(r.ENDTIME))*1000);g=groups.get((r.GAMEID,period,teams.get(r.OPPONENT)));reason=None
  if g is None:reason='unmapped_defensive_team_or_period'
  elif end<=start:reason='nonpositive_or_ambiguous_clock_interval'
  elif re.search(r'technical|flagrant|clear path|away.from.play|defensive 3|sub:',events,re.I):reason='special_free_throw_or_substitution_in_summary'
  else:
   lo=max(offset*1000,start-1000);hi=min((offset+length)*1000,end+1000);cand=g[(g.start_ms<=lo)&(g.end_ms>=hi)]
   if len(cand)!=1:reason='lineup_changes_or_second_resolution_boundary_ambiguity'
   else:
    q=cand.iloc[0];ft=sum('Free Throw' in s and not s.lstrip().startswith('MISS ') for s in events.splitlines())
    final.append(dict(row_id=f'{prefix}{i}',game_id=int(r.GAMEID),cluster=str(r.GAMEID),players=q.players,opponent_players=q.opponent_players,home=q.home,period_category=min(period,5),elapsed_minutes_at_start=start/60000,score_difference_at_start=float(r.STARTSCOREDIFFERENTIAL),points=2*int(r.FG2M)+3*int(r.FG3M)+ft,assists=len(re.findall(r'\([^()]*\b\d+ AST\)',events)),turnovers=r.TURNOVERS,offensive_rebounds=r.OFFENSIVEREBOUNDS))
  if reason:excluded.append({'row':i,'game':int(r.GAMEID),'reason':reason})
 savej(W/f'data/{prefix}exclusions.json',excluded)
 return pd.DataFrame(final)

f=possessions(D,'pbpstats_')
split=pd.read_csv(ROOT/'data/splits/NBA/development_split.csv')
f=f.merge(split[['game_id','source_split']],on='game_id',validate='many_to_one')
fit=f[f.source_split=='fit'].copy();final=possessions(E,'E_')
pairs={(int(r.game_id),int(i)) for r in fit.itertuples() for a in (r.players,r.opponent_players) for i in a};counts=Counter(i for _,i in pairs);eligible={i for i,n in counts.items() if n>=10};fit=subset(fit,eligible);ids=sorted(ids_in(fit));test=subset(final,set(ids));fit['cluster']=fit.game_id.astype(str)
ctx=['home']+[f'period{k}' for k in range(1,6)]+['elapsed_scaled','score_scaled'];cscale={c:scale(fit[c]) for c in ['elapsed_minutes_at_start','score_difference_at_start']}
for frame in [fit,test]:
 for k in range(1,6):frame[f'period{k}']=(frame.period_category==k).astype(float)
 for col,new in [('elapsed_minutes_at_start','elapsed_scaled'),('score_difference_at_start','score_scaled')]:frame[new]=(frame[col]-cscale[col]['mean'])/cscale[col]['std']
nba=store('NBA',fit,test,ids,ctx,P['NBA']['targets']);savej(W/'data/NBA/context_scaling.json',cscale);savej(W/'data/NBA/eligibility.json',{'counts':counts,'threshold':10,'eligible':sorted(eligible),'training_observed':ids})
print(json.dumps(nba,indent=2))
