from pathlib import Path
import json
import numpy as np
W=Path(__file__).resolve().parent;m=json.loads((W/'data/source/meta.json').read_text());N=m['players'];out=W/'baselines';out.mkdir(exist_ok=True)
def design(z):
 x=np.zeros((len(z['a']),2*N+1));x[:,-1]=1
 for j in range(2):x[np.arange(len(x)),z['a'][:,j]]+=1;x[np.arange(len(x)),N+z['b'][:,j]]+=1
 return x
for target in ['shot_success','blocks_per_set','digs_per_set']:
 kind='source' if target=='shot_success' else 'reuse';f=dict(np.load(W/'data'/kind/'fit.npz'));v=dict(np.load(W/'data'/kind/'val.npz'));x=design(f);xx=design(v);y=f[target].astype(float);mu=y.mean();sd=y.std();z=(y-mu)/sd;pen=np.eye(x.shape[1]);pen[-1,-1]=0;best=None;grid=[]
 for lam in [1e-6,1e-5,1e-4,.001,.01,.1,1.]:
  coef=np.linalg.solve(x.T@x/len(x)+lam*pen,x.T@z/len(x));pred=(xx@coef)*sd+mu;loss=float(np.mean((pred-v[target])**2));grid.append(dict(lambda_=lam,validation_mse=loss))
  if best is None or loss<best['validation_mse']:best=dict(lambda_=lam,validation_mse=loss);bp=pred;bc=coef
 np.savez_compressed(out/f'{target}_ridge.npz',prediction=bp,coef=bc,y=v[target],row_id=v['row_id'],cluster=v['cluster']);best.update(grid=grid,constant_mse=float(np.mean((mu-v[target])**2)),fit_mean=float(mu));(out/f'{target}.json').write_text(json.dumps(best,indent=2));print(target,best['validation_mse'],best['constant_mse'])
