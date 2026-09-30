"""One terminal evaluation pass after every neural training job completes."""
from train import *
from scipy.optimize import minimize
from scipy.special import expit

def metrics(y,p,cls):
 if cls:
  prob=expit(p);return {'loss':float(loss_numpy(y,p,True).mean()),'brier':float(((prob-y)**2).mean()),'accuracy':float(((prob>=.5)==y).mean())}
 return {'loss':float(((p-y)**2).mean())}
def baseline(domain,target,out):
 fit,m=reader_data(domain,target);test,_=reader_data(domain,target,'test');n=m['players'];cls=domain=='tennis' and target=='win'
 def matrix(z):
  x=np.zeros((len(z['a']),2*n+1),np.float64);x[:,0]=1
  for offset,key in [(1,'a'),(n+1,'b')]:
   x[np.arange(len(x))[:,None],z[key]+offset]=1
  return x
 x=matrix(fit);v=matrix(test);y=fit[target].astype(float);yv=test[target].astype(float);lam=.01
 if cls:
  def fun(beta):
   z=x@beta;p=expit(z);val=np.mean(np.logaddexp(0,z)-y*z)+lam*np.sum(beta[1:]**2);g=x.T@(p-y)/len(y);g[1:]+=2*lam*beta[1:];return val,g
  initial_beta=np.zeros(x.shape[1]);initial_beta[0]=np.log(y.mean()/(1-y.mean()));sol=minimize(fun,initial_beta,jac=True,method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-12,'gtol':1e-8})
  if not sol.success:raise RuntimeError('logistic fixed solver did not converge: '+str(sol.message))
  beta=sol.x;constant=np.full(len(yv),initial_beta[0]);prediction=v@beta
 else:
  reg=np.eye(x.shape[1])*lam;reg[0,0]=0;beta=np.linalg.solve(x.T@x/len(y)+reg,x.T@y/len(y));prediction=v@beta;constant=np.full(len(yv),y.mean())
 np.savez_compressed(out/'baseline.npz',y=yv,prediction=prediction,constant=constant,beta=beta,cluster=test['cluster'],row_id=test['row_id']);r={'ID':metrics(yv,prediction,cls),'constant':metrics(yv,constant,cls)};js(out/'baseline.json',r);return r

def main():
 if not (W/'train_complete.json').exists():raise RuntimeError('Final test is locked until all training tasks finish')
 final=W/'final';final.mkdir(exist_ok=True);summary={};finder_summary={}
 for domain in P['domains']:
  m=meta(domain);finder_summary[domain]={}
  for cond in P['conditions']:
   for seed in P['seeds']:
    key=f'{cond}_{seed}';out=final/domain/'finders'/key;out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():finder_summary[domain][key]=json.loads((out/'metrics.json').read_text());continue
    f=Finder(m,cond,seed);state=json.loads((W/'runs'/domain/cond/f'seed_{seed}'/'finder/complete.json').read_text());f.ck.restore(state['checkpoint']).expect_partial();z=load(domain,'test');sc=m['scaling'][m['targets'][0]];y=z[m['targets'][0]];perm=np.random.default_rng(seedval(99,seed)).permutation(m['players']);res={}
    for label,u in [('normal',f.table.numpy()),('zero',np.zeros_like(f.u0)),('shuffle',f.table.numpy()[perm])]:
     pred=f.predict(z,u)*sc['std']+sc['mean'];np.savez_compressed(out/(label+'.npz'),prediction=pred,y=y,cluster=z['cluster'],row_id=z['row_id']);res[label]={role:float(loss_numpy(y[:,None],pred[:,side,:],m['classification']).mean()) for side,role in enumerate('AB')}
    js(out/'metrics.json',res);finder_summary[domain][key]=res
  summary[domain]={}
  for target in m['targets']:
   out=final/domain/target;out.mkdir(parents=True,exist_ok=True);bs=json.loads((out/'baseline.json').read_text()) if (out/'baseline.json').exists() else baseline(domain,target,out);item={'baselines':bs,'conditions':{}};z,_=reader_data(domain,target,'test');cls=domain=='tennis' and target=='win'
   for cond in P['conditions']:
    results=[]
    for seed in P['seeds']:
     d=out/cond/f'seed_{seed}';d.mkdir(parents=True,exist_ok=True)
     if (d/'metrics.json').exists():results.append(json.loads((d/'metrics.json').read_text()));continue
     run=W/'runs'/domain/cond/f'seed_{seed}'/('predictor_'+target);sc=json.loads((run/'scaling.json').read_text());r=Reader(m,latent_for(domain,cond,seed),seed,cls);state=json.loads((run/'complete.json').read_text());r.ck.restore(state['checkpoint']).expect_partial();perm=np.random.default_rng(seedval(99,seed)).permutation(m['players']);res={'seed':seed}
     for label,u in [('normal',r.latent),('zero',tf.zeros_like(r.latent)),('shuffle',tf.gather(r.latent,perm))]:
      pred=r.predict(z,u)*sc['std']+sc['mean'];np.savez_compressed(d/(label+'.npz'),prediction=pred,y=z[target],cluster=z['cluster'],row_id=z['row_id']);res[label]=metrics(z[target],pred,cls)
      if label=='normal':normal=pred
      else:res[label]['max_prediction_change']=float(np.max(np.abs((expit(pred)-expit(normal)) if cls else pred-normal)));res[label]['delta_loss']=res[label]['loss']-res['normal']['loss']
     js(d/'metrics.json',res);results.append(res)
    item['conditions'][cond]={'seeds':results,'mean_loss':float(np.mean([v['normal']['loss'] for v in results]))}
   # cluster paired bootstrap on saved errors, same draw across all contrasts and seeds.
   names,inv=np.unique(z['cluster'],return_inverse=True);sizes=np.bincount(inv);rng=np.random.default_rng(P['bootstrap']['seed']);draws=rng.integers(0,len(names),size=(P['bootstrap']['replicates'],len(names)));errors={}
   for cond in P['conditions']:
    es=[]
    for seed in P['seeds']:
     v=np.load(out/cond/f'seed_{seed}'/'normal.npz');es.append(loss_numpy(v['y'],v['prediction'],cls))
    errors[cond]=np.mean(es,axis=0)
   ci={}
   for a,b in [(a,b) for a,b in [('direct','initial'),('C39','initial'),('C39','direct')] if a in errors and b in errors]:
    delta=errors[a]-errors[b];totals=np.bincount(inv,weights=delta);boot=totals[draws].sum(1)/sizes[draws].sum(1);ci[a+'-'+b]={'difference':float(delta.mean()),'CI95':np.quantile(boot,[.025,.975]).tolist(),'clusters':len(names)}
   item['paired_CI']=ci;js(out/'summary.json',item);summary[domain][target]=item;print(domain,target,{c:item['conditions'][c]['mean_loss'] for c in P['conditions']},flush=True)
 js(final/'summary.json',summary);js(final/'finder_summary.json',finder_summary)
 lines=['# C42 최종 결과','', '모든 학습 완료 후 고정 최종 모델만 test에서 평가했다. NBA는 기존 E, 테니스는 고정 대회 test. 평균은 세 seed 평균. 회귀는 원 정답 단위 MSE, 테니스 승패는 log loss. ID는 직접 감독 기준선이지 성능 천장이 아니다.','']
 for domain,targets in summary.items():
  lines+=['## '+domain,'']
  for target,v in targets.items():
   lines += ['### '+target,'',f"상수 {v['baselines']['constant']['loss']:.9f}, ID {v['baselines']['ID']['loss']:.9f}."]
   for c,s in v['conditions'].items():lines.append(f"- {c}: {s['mean_loss']:.9f}; seed별 "+', '.join(f"{a['normal']['loss']:.9f}" for a in s['seeds']))
   lines+=['', '조건별 paired difference 및 95% CI: '+json.dumps(v['paired_CI'],ensure_ascii=False),'']
 lines+=['Frozen latent reuse; zero/shuffle are input-dependence diagnostics. Bootstrap intervals are conditional on fitted models.']
 (W/'결과보고.md').write_text('\n'.join(lines));js(W/'complete.json',{'state':'complete','time':time.time(),'discovery_runs':len(P['domains'])*len(P['conditions'])*len(P['seeds']),'predictor_runs':sum(len(meta(d)['targets']) for d in P['domains'])*len(P['conditions'])*len(P['seeds']),'final_evaluation_complete':True})
if __name__=='__main__':main()
