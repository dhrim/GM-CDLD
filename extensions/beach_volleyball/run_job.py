"""C117 orchestration around unchanged C74 Finder/Reader classes.
Source selection and all reports use original-unit squared error.
"""
from pathlib import Path
import argparse,json,time,os,fcntl,traceback,hashlib
import numpy as np
import base_c74 as b
from joint_model import JointFinder
W=Path(__file__).resolve().parent; P=b.P; tf=b.tf

def dataset(kind,part):return dict(np.load(W/'data'/kind/f'{part}.npz'))
def metadata(kind):return json.loads((W/'data'/kind/'meta.json').read_text())
def decode(raw,cls,mu,sd):return b.probability(raw) if cls else raw*sd+mu

def metrics(y,raw,cls,mu,sd):
 y=np.asarray(y,dtype=np.float64);raw=np.asarray(raw,dtype=np.float64);p=decode(raw,cls,mu,sd)
 d={'mse':float(np.mean((p-y)**2)),'prediction_mean':float(p.mean()),'prediction_std':float(p.std()),'outside_01':float(np.mean((p<0)|(p>1)))}
 if cls:d['log_loss']=float(np.mean(np.logaddexp(0.,raw)-y*raw))
 return d

def save_initial_weights(model,out):
 np.savez_compressed(out/'initial_weights.npz',**{f'w{i}':v.numpy() for i,v in enumerate(model.weights)})

def discover(condition,seed,out):
 m=metadata('source');cls=condition=='probability';m['classification']=cls
 fit=dataset('source','fit');val=dataset('source','val');target='shot_success'
 sc=m['scaling'][target];mu=0. if cls else sc['mean'];sd=1. if cls else sc['std']
 b.js(out/'scaling.json',{'mean':mu,'std':sd,'classification':cls})
 f=JointFinder(m,seed) if condition=='joint' else b.Finder(m,'direct',seed);y=((fit[target]-mu)/sd).reshape(-1,1).astype(np.float32)
 pools={side:[np.flatnonzero((fit[k]==i).any(1)) for i in range(m['players'])] for side,k in enumerate(['a','b'])}
 manager=tf.train.CheckpointManager(f.ck,str(out/'checkpoints'),max_to_keep=2);start=0
 def evaluate(z):
  raw=f.predict(z)
  return {letter:metrics(z[target],raw[:,s,0],cls,mu,sd) for s,letter in enumerate('AB')}
 def record(cycle,side=None):
  fm=evaluate(fit);vm=evaluate(val)
  return {'cycle':cycle,'side':side,'fit_loss':{s:d['mse'] for s,d in fm.items()},'validation_loss':{s:d['mse'] for s,d in vm.items()},'fit_metrics':fm,'validation_metrics':vm}
 if (out/'resume.json').exists():
  st=json.loads((out/'resume.json').read_text());f.ck.restore(st['checkpoint']).expect_partial();start=st['cycle'];best=json.loads((out/'best.json').read_text())['validation_mean_AB']
 else:
  np.savez_compressed(out/'initial_latents.npz',latent=f.u0);save_initial_weights(f.nets[0],out)
  rec=record(0);b.js(out/'cycle_000.json',rec);best=float(np.mean(list(rec['validation_loss'].values())))
  b.js(out/'best.json',dict(rec,validation_mean_AB=best));f.ck.write(str(out/'best_state'));np.savez_compressed(out/'best_latents.npz',latent=f.table.numpy())
 for cycle in range(start+1,P['cycles']+1):
  begin=time.time();order=np.random.default_rng(b.seedval(11,seed,cycle)).permutation(m['players'])
  for side in range(2):
   traces=[];presentations=0;started=time.time()
   for turn,i in enumerate(order if condition!='joint' else [0]):
    i=int(i);rows=np.random.default_rng(b.seedval(22,seed,cycle,side,i)).permutation(pools[side][i] if condition!='joint' else np.arange(len(y)));nb=(len(rows)+b.B-1)//b.B
    if nb:
     keys=np.array([b.seedval(33,seed,cycle,side,i,j) for j in range(nb)],np.int32)
     trace=f.visits[side](tf.constant(fit['a'][rows]),tf.constant(fit['b'][rows]),tf.constant(fit['context'][rows]),tf.constant(y[rows]),tf.constant(i),tf.constant(keys)).numpy()
     traces.extend(np.c_[np.full(nb,i),np.arange(nb),trace].tolist());presentations+=len(rows)
    if turn%100==0:b.js(out/'progress.json',{'stage':'Finder','cycle':cycle,'side':'AB'[side],'players_done':turn+1,'pid':os.getpid()})
   rec=record(cycle,'AB'[side]);rec.update(updates=len(traces),presentations=presentations,seconds=time.time()-started)
   ph=(cycle-1)*2+side+1;b.js(out/f'phase_{ph:04d}.json',rec);np.save(out/f'phase_{ph:04d}_loss.npy',np.asarray(traces));np.save(out/f'phase_{ph:04d}_order.npy',order)
  score=float(np.mean(list(rec['validation_loss'].values())))
  if score<best:
   best=score;f.ck.write(str(out/'best_state'));np.savez_compressed(out/'best_latents.npz',latent=f.table.numpy());b.js(out/'best.json',dict(rec,validation_mean_AB=best))
  np.savez_compressed(out/f'latent_cycle_{cycle:03d}.npz',latent=f.table.numpy())
  cp=manager.save(checkpoint_number=cycle);b.js(out/'resume.json',{'cycle':cycle,'checkpoint':cp})
  b.js(out/'progress.json',{'stage':'Finder','cycle':cycle,'best_validation':best,'seconds':time.time()-begin,'pid':os.getpid()})
  print('cycle',cycle,'fit',rec['fit_loss'],'val',rec['validation_loss'],'seconds',round(time.time()-begin,2),flush=True)
  bestcycle=json.loads((out/'best.json').read_text())['cycle']
  if cycle>=P['source_minimum'] and cycle-bestcycle>=P['source_patience']:break
 np.savez_compressed(out/'final_latents.npz',latent=f.table.numpy());f.ck.restore(str(out/'best_state')).expect_partial()
 raw=f.predict(val)[:,:,0];pred=decode(raw,cls,mu,sd)
 np.savez_compressed(out/'validation_predictions.npz',row_id=val['row_id'],cluster=val['cluster'],y=val[target],prediction=pred,raw=raw)
 b.js(out/'complete.json',{'cycles':cycle,'selected':json.loads((out/'best.json').read_text()),'finished':time.time()})

def reader(rep,target,seed,out):
 kind='reuse';fit=dataset(kind,'fit');val=dataset(kind,'val');m=metadata(kind)
 if rep=='initial':latent=b.initial(m['players'],seed);origin='actual seeded source initial latent'
 else:
  source=W/'runs'/f'seed_{seed}'/rep/'finder';assert (source/'complete.json').exists();latent=np.load(source/'best_latents.npz')['latent'];origin=str(source/'best_latents.npz')
 latent_hash=hashlib.sha256(latent.tobytes()).hexdigest();np.savez_compressed(out/'frozen_latent.npz',latent=latent)
 b.js(out/'latent_origin.json',{'source':origin,'array_sha256':latent_hash})
 cls=target=='shot_success';mu=0. if cls else float(fit[target].mean());sd=1. if cls else max(float(fit[target].std()),1e-8)
 b.js(out/'scaling.json',{'mean':mu,'std':sd,'classification':cls});y=((fit[target]-mu)/sd).reshape(-1,1).astype(np.float32)
 r=b.Reader(m,latent,seed,cls);manager=tf.train.CheckpointManager(r.ck,str(out/'checkpoints'),max_to_keep=2);start=0
 def record(epoch):
  fm=metrics(fit[target],r.predict(fit),cls,mu,sd);vm=metrics(val[target],r.predict(val),cls,mu,sd)
  return {'epoch':epoch,'fit_loss':fm['mse'],'validation_loss':vm['mse'],'fit_metrics':fm,'validation_metrics':vm}
 if (out/'resume.json').exists():
  st=json.loads((out/'resume.json').read_text());r.ck.restore(st['checkpoint']).expect_partial();start=st['epoch'];best=json.loads((out/'best.json').read_text())['validation_loss']
 else:
  save_initial_weights(r.model,out);rec=record(0);best=rec['validation_loss'];r.ck.write(str(out/'best_state'));b.js(out/'epoch_000.json',rec);b.js(out/'best.json',rec)
 for epoch in range(start+1,P['epochs']+1):
  begin=time.time();order=np.random.default_rng(b.seedval(71,seed,epoch)).permutation(len(y));keys=np.array([b.seedval(72,seed,epoch,0,0,j) for j in range((len(y)+b.B-1)//b.B)],np.int32)
  trace=r.epoch(tf.constant(fit['a'][order]),tf.constant(fit['b'][order]),tf.constant(y[order]),tf.constant(keys)).numpy();rec=record(epoch);rec['seconds']=time.time()-begin
  if rec['validation_loss']<best:best=rec['validation_loss'];r.ck.write(str(out/'best_state'));b.js(out/'best.json',rec)
  np.save(out/f'epoch_{epoch:03d}_loss.npy',trace);b.js(out/f'epoch_{epoch:03d}.json',rec)
  cp=manager.save(checkpoint_number=epoch);b.js(out/'resume.json',{'epoch':epoch,'checkpoint':cp});b.js(out/'progress.json',dict(stage='Predictor',pid=os.getpid(),**rec))
  if epoch%5==0 or epoch==1:print('epoch',epoch,'fit',rec['fit_loss'],'val',rec['validation_loss'],'seconds',round(rec['seconds'],2),flush=True)
  bestepoch=json.loads((out/'best.json').read_text())['epoch']
  if epoch>=P['reader_minimum'] and epoch-bestepoch>=P['reader_patience']:break
 r.ck.restore(str(out/'best_state')).expect_partial();raw=r.predict(val);pred=decode(raw,cls,mu,sd)
 np.savez_compressed(out/'validation_predictions.npz',row_id=val['row_id'],cluster=val['cluster'],y=val[target],prediction=pred,raw=raw)
 assert hashlib.sha256(r.latent.numpy().tobytes()).hexdigest()==latent_hash
 b.js(out/'complete.json',{'epochs':epoch,'selected':json.loads((out/'best.json').read_text()),'frozen_latent_unchanged':True,'finished':time.time()})

def main():
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['finder','predictor']);a.add_argument('representation');a.add_argument('seed',type=int);a.add_argument('--target');x=a.parse_args()
 out=W/'runs'/f'seed_{x.seed}'/x.representation/('finder' if x.stage=='finder' else 'predictor_'+x.target);out.mkdir(parents=True,exist_ok=True)
 lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (out/'complete.json').exists():return
 b.js(out/'status.json',{'state':'running','pid':os.getpid(),'started':time.time(),'arguments':vars(x)})
 try:
  if x.stage=='finder':discover(x.representation,x.seed,out)
  else:reader(x.representation,x.target,x.seed,out)
  b.js(out/'status.json',{'state':'complete','finished':time.time()})
 except BaseException:
  b.js(out/f'failure_{time.time_ns()}.json',{'error':traceback.format_exc()});b.js(out/'status.json',{'state':'failed','finished':time.time()});raise
if __name__=='__main__':main()
