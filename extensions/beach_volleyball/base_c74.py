"""C74: C70 concat model with persistent per-player Adam states.
Only fit/validation are loaded. Latent optimizer state persists across A/B and cycles.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3');os.environ.setdefault('TF_DETERMINISTIC_OPS','1');os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0')
import argparse,json,time,traceback,fcntl
from pathlib import Path
import numpy as np
import tensorflow as tf
W=Path(__file__).resolve().parent;P=json.loads((W/'plan.json').read_text());D=P['latent_dim'];B=P['batch']
tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.experimental.enable_tensor_float_32_execution(False)
for gpu in tf.config.list_physical_devices('GPU'):tf.config.experimental.set_memory_growth(gpu,True)
def seedval(*v):return int(np.random.SeedSequence(v).generate_state(1)[0]%2147483647)
def js(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)
def load(domain,part='fit'):return dict(np.load(W/'data'/domain/(part+'.npz')))
def meta(domain):return json.loads((W/'data'/domain/'meta.json').read_text())
def initial(n,seed):return np.random.default_rng(seedval(1,seed)).uniform(-np.sqrt(6/(n+D)),np.sqrt(6/(n+D)),(n,D)).astype(np.float32)
def penalty(net):return tf.add_n([tf.reduce_sum(v*v) for v in net.trainable_variables if 'bias' not in v.name])
def net(dim,out,seed,name):
 tf.keras.utils.set_random_seed(seed)
 return tf.keras.Sequential([tf.keras.Input(shape=(dim,)),tf.keras.layers.Dense(64,activation='relu'),tf.keras.layers.Dense(16,activation='relu'),tf.keras.layers.Dense(out)],name=name)
def with_dropout(base,seed,name):
 return tf.keras.Sequential([tf.keras.Input(shape=base.input_shape[1:]),base.layers[0],tf.keras.layers.Dropout(P['dropout'],seed=seedval(seed,P['dropout_repeat'])),*base.layers[1:]],name=name)
def adam(variables):
 o=tf.keras.optimizers.Adam(P['lr']);o.build(variables);return o
def reset(o):
 for v in o.variables:
  if 'learning_rate' not in v.name:v.assign(tf.zeros_like(v))
def objective(y,p,classification):return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=p)) if classification else tf.reduce_mean((y-p)**2)
def loss_numpy(y,p,classification):return np.logaddexp(0.,p)-y*p if classification else (p-y)**2
def probability(logit):return 1/(1+np.exp(-np.clip(logit,-80,80)))
def shuffle(values,key,branch):
 order=tf.argsort(tf.random.stateless_uniform(tf.shape(values)[:2],[key,branch]),axis=1)
 return tf.gather(values,order,batch_dims=1)
class Finder:
 def __init__(self,m,condition,seed):
  self.m=m;self.condition=condition;self.k=m['slots'];self.cls=m['classification'];self.u0=initial(m['players'],seed)
  self.table=tf.Variable(self.u0,trainable=False,name='IndexedValueLayer');self.selected=tf.Variable(self.u0[0],name='selected_latent');k=self.k
  dim=(2*k*34+m['context_dim']) if condition=='direct' else ((2*k-1)*34+2+m['context_dim']);out=1 if condition=='direct' else D+1
  first=net(dim,out,seedval(91 if condition=='C39' else 90,seed),'Finder_A');second=tf.keras.models.clone_model(first);second.set_weights(first.get_weights());first=with_dropout(first,seedval(6301,seed,0),'Finder_A_dropout');second=with_dropout(second,seedval(6301,seed,1),'Finder_B_dropout');self.nets=[first,second];self.opts=[adam(n.trainable_variables) for n in self.nets];self.rowopt=adam([self.selected]);self.ck=tf.train.Checkpoint(table=self.table,selected=self.selected,A=first,B=second,oa=self.opts[0],ob=self.opts[1],row=self.rowopt);self.visits=[self.make_visit(s) for s in range(2)]
  self.input_drops=[[tf.keras.layers.Dropout(.1,seed=seedval(7001,P['dropout_repeat'],side,slot)) for slot in range(2*k)] for side in range(2)]
  self.context_drops=[tf.keras.layers.Dropout(.1,seed=seedval(7002,P['dropout_repeat'],side)) for side in range(2)]
  for side in range(2):
   setattr(self.ck,f'hidden_rng_{side}',self.nets[side].layers[1].seed_generator.state.value)
   setattr(self.ck,f'context_rng_{side}',self.context_drops[side].seed_generator.state.value)
   for slot,layer in enumerate(self.input_drops[side]):setattr(self.ck,f'input_rng_{side}_{slot}',layer.seed_generator.state.value)
  # Adam history belongs to a player, not the reusable selected-latent variable.
  self.row_state_vars=[v for v in self.rowopt.variables if 'learning_rate' not in v.name]
  self.row_banks=[tf.Variable(tf.zeros([m['players']]+list(v.shape),dtype=v.dtype),trainable=False,name=f'player_adam_{j}') for j,v in enumerate(self.row_state_vars)]
  for j,bank in enumerate(self.row_banks):setattr(self.ck,f'player_adam_{j}',bank)
 def features(self,a,b,c,side,player=None,key=None,table=None,training=False):
  table=self.table if table is None else table;own,opp=(a,b) if side==0 else (b,a)
  if self.condition=='C39':
   positions=tf.argsort(tf.cast(own==player,tf.int32),axis=1,stable=True)[:,:self.k-1];own=tf.gather(own,positions,batch_dims=1)
  av=tf.gather(table,own);bv=tf.gather(table,opp)
  if self.condition=='direct' and player is not None:
   av=tf.where((own==player)[...,None],self.selected,av);bv=tf.where((opp==player)[...,None],self.selected,bv)
  if key is not None:av=shuffle(av,key,0);bv=shuffle(bv,key,1)
  av=tf.stack([self.input_drops[side][j](av[:,j,:],training=training) for j in range(self.k)],axis=1)
  bv=tf.stack([self.input_drops[side][self.k+j](bv[:,j,:],training=training) for j in range(self.k)],axis=1)
  c=self.context_drops[side](c,training=training)
  def tag(x,role,n):
   flags=tf.broadcast_to(tf.constant(role,tf.float32),[tf.shape(x)[0],n,2]);return tf.reshape(tf.concat([x,flags],-1),[-1,n*34])
  role=[1.,0.] if side==0 else [0.,1.];other=role[::-1]
  parts=[tag(av,role,self.k-(self.condition=='C39')),tag(bv,other,self.k)]
  if self.condition=='C39':parts.append(tf.broadcast_to(tf.constant(role),[tf.shape(a)[0],2]))
  return tf.concat(parts+[c],1)
 def make_visit(self,side):
  @tf.function(reduce_retracing=True)
  def visit(a,b,c,y,player,keys):
   self.selected.assign(tf.gather(self.table,player))
   for v,bank in zip(self.row_state_vars,self.row_banks):v.assign(tf.gather(bank,player))
   n=tf.shape(a)[0];steps=tf.shape(keys)[0];trace=tf.TensorArray(tf.float32,size=steps)
   for j in tf.range(steps):
    sl=slice(j*B,tf.minimum((j+1)*B,n))
    with tf.GradientTape() as tape:
     raw=self.nets[side](self.features(a[sl],b[sl],c[sl],side,player,keys[j],training=True),training=True)
     pred=raw if self.condition=='direct' else raw[:,:1]+tf.reduce_sum(raw[:,1:]*self.selected,1,keepdims=True)
     data=objective(y[sl],pred,self.cls);reg=penalty(self.nets[side]);loss=data+P['l2']*reg+P['latent_l2']*tf.reduce_sum(self.selected**2)
    variables=[self.selected]+self.nets[side].trainable_variables;grads=tape.gradient(loss,variables)
    tf.debugging.assert_all_finite(loss,'nonfinite training loss')
    self.rowopt.apply_gradients([(grads[0],self.selected)]);self.opts[side].apply_gradients(zip(grads[1:],variables[1:]));self.table.assign(tf.tensor_scatter_nd_update(self.table,tf.reshape(player,[1,1]),self.selected[None]))
    trace=trace.write(j,tf.stack([loss,data,reg]))
   # Persist only the selected player's state; all other rows remain untouched.
   for v,bank in zip(self.row_state_vars,self.row_banks):
    bank.assign(tf.tensor_scatter_nd_update(bank,tf.reshape(player,[1,1]),tf.expand_dims(v,0)))
   return trace.stack()
  return visit
 def predict(self,z,table=None):
  table=self.table if table is None else tf.constant(table);parts=[]
  for start in range(0,len(z['a']),2048):
   a=tf.constant(z['a'][start:start+2048]);b=tf.constant(z['b'][start:start+2048]);c=tf.constant(z['context'][start:start+2048]);sides=[]
   for side in range(2):
    if self.condition=='direct':v=self.nets[side](self.features(a,b,c,side,table=table),training=False).numpy();sides.append(v)
    else:
     own=a if side==0 else b;views=[]
     for slot in range(self.k):
      player=own[:,slot:slot+1];raw=self.nets[side](self.features(a,b,c,side,player=player,table=table),training=False);selected=tf.gather(table,own[:,slot]);views.append((raw[:,0]+tf.reduce_sum(raw[:,1:]*selected,1)).numpy())
     sides.append(np.stack(views,1))
   parts.append(np.stack(sides,1))
  return np.concatenate(parts)
def discover(domain,condition,seed,out):
 m=meta(domain);z0=load(domain);val=load(domain,'val');f=Finder(m,condition,seed);sc=m['scaling'][m['targets'][0]];target=m['targets'][0]
 def evaluate(z):
  yy=(z[target]-sc['mean'])/sc['std'];pr=f.predict(z)
  return {letter:float(loss_numpy(yy[:,None],pr[:,s,:],m['classification']).mean()) for s,letter in enumerate('AB')}
 manager=tf.train.CheckpointManager(f.ck,str(out/'checkpoints'),max_to_keep=2);start=0
 if (out/'resume.json').exists():
  st=json.loads((out/'resume.json').read_text());f.ck.restore(st['checkpoint']).expect_partial();start=st['cycle'];best=json.loads((out/'best.json').read_text())['validation_mean_AB']
 else:
  np.savez_compressed(out/'initial_latents.npz',latent=f.u0)
  initial={'cycle':0,'fit_loss':evaluate(z0),'validation_loss':evaluate(val)};js(out/'cycle_000.json',initial)
  best=float(np.mean(list(initial['validation_loss'].values())));js(out/'best.json',dict(initial,validation_mean_AB=best));f.ck.write(str(out/'best_state'));np.savez_compressed(out/'best_latents.npz',latent=f.table.numpy())
 for cycle in range(start+1,P['cycles']+1):
  begin=time.time();z={k:v.copy() for k,v in z0.items()}
  mask=np.random.default_rng(seedval(5501,seed,cycle)).random(len(z['a']))<.5 if domain=='tennis' else np.zeros(len(z['a']),bool)
  z['a'][mask]=z0['b'][mask];z['b'][mask]=z0['a'][mask]
  if domain=='tennis':z['win'][mask]=1-z0['win'][mask]
  y=((z[target]-sc['mean'])/sc['std'])[:,None];order=np.random.default_rng(seedval(11,seed,cycle)).permutation(m['players'])
  for side in range(2):
   traces=[];nrows=0;started=time.time()
   for turn,i in enumerate(order):
    i=int(i);pool=np.flatnonzero((z['ab'[side]]==i).any(1));rows=np.random.default_rng(seedval(22,seed,cycle,side,i)).permutation(pool);nb=(len(rows)+B-1)//B
    if nb:
     keys=np.array([seedval(33,seed,cycle,side,i,j) for j in range(nb)],np.int32)
     trace=f.visits[side](tf.constant(z['a'][rows]),tf.constant(z['b'][rows]),tf.constant(z['context'][rows]),tf.constant(y[rows],tf.float32),tf.constant(i),tf.constant(keys)).numpy();traces.extend(np.c_[np.full(nb,i),np.arange(nb),trace].tolist());nrows+=len(rows)
    if turn%50==0:js(out/'progress.json',{'stage':'Finder','cycle':cycle,'side':'AB'[side],'players_done':turn+1,'players_total':m['players'],'pid':os.getpid()})
   rec={'cycle':cycle,'side':'AB'[side],'fit_loss':evaluate(z0),'validation_loss':evaluate(val),'updates':len(traces),'presentations':nrows,'swapped_rows':int(mask.sum()),'seconds':time.time()-started}
   ph=(cycle-1)*2+side+1;js(out/f'phase_{ph:04d}.json',rec);np.save(out/f'phase_{ph:04d}_loss.npy',np.asarray(traces));np.save(out/f'phase_{ph:04d}_order.npy',order)
  score=float(np.mean(list(rec['validation_loss'].values())))
  if score<best:
   best=score;f.ck.write(str(out/'best_state'));np.savez_compressed(out/'best_latents.npz',latent=f.table.numpy());js(out/'best.json',dict(rec,validation_mean_AB=best))
  cp=manager.save(checkpoint_number=cycle);js(out/'resume.json',{'cycle':cycle,'checkpoint':cp});js(out/'progress.json',{'stage':'Finder','cycle':cycle,'best_validation':best,'pid':os.getpid()});print('cycle',cycle,'fit',rec['fit_loss'],'val',rec['validation_loss'],'seconds',round(time.time()-begin,2),flush=True)
 np.savez_compressed(out/'final_latents.npz',latent=f.table.numpy());js(out/'complete.json',{'cycles':P['cycles'],'checkpoint':cp})

def reader_data(domain,target,part='fit'):
 z=load(domain,part);m=meta(domain)
 if domain=='tennis' and target!='win':
  z={'a':np.concatenate([z['a'],z['b']]),'b':np.concatenate([z['b'],z['a']]),'cluster':np.tile(z['cluster'],2),'row_id':np.concatenate([np.char.add(z['row_id'],':A'),np.char.add(z['row_id'],':B')]),target:np.concatenate([z[target],z[target+'_other']])}
 mask=np.isfinite(z[target]);return {k:v[mask] for k,v in z.items()},m
class Reader:
 def __init__(self,m,latent,seed,classification):
  self.latent=tf.constant(latent);self.k=m['slots'];self.cls=classification;self.model=with_dropout(net(self.k*2*D,1,seedval(70,seed),'Predictor'),seedval(6302,seed),'Predictor_dropout');self.opt=adam(self.model.trainable_variables);self.ck=tf.train.Checkpoint(model=self.model,opt=self.opt);self.epoch=self.make_epoch()
  self.input_drops=[tf.keras.layers.Dropout(.1,seed=seedval(7003,P['dropout_repeat'],slot)) for slot in range(2*self.k)]
  self.ck.hidden_rng=self.model.layers[1].seed_generator.state.value
  for slot,layer in enumerate(self.input_drops):setattr(self.ck,f'input_rng_{slot}',layer.seed_generator.state.value)
 def features(self,a,b,key=None,table=None,training=False):
  table=self.latent if table is None else table;x=tf.gather(table,a);y=tf.gather(table,b)
  if key is not None:x=shuffle(x,key,0);y=shuffle(y,key,1)
  x=tf.stack([self.input_drops[j](x[:,j,:],training=training) for j in range(self.k)],axis=1)
  y=tf.stack([self.input_drops[self.k+j](y[:,j,:],training=training) for j in range(self.k)],axis=1)
  return tf.concat([tf.reshape(x,[-1,self.k*D]),tf.reshape(y,[-1,self.k*D])],1)
 def make_epoch(self):
  @tf.function(reduce_retracing=True)
  def epoch(a,b,y,keys):
   n=tf.shape(a)[0];steps=tf.shape(keys)[0];trace=tf.TensorArray(tf.float32,size=steps)
   for j in tf.range(steps):
    sl=slice(j*B,tf.minimum((j+1)*B,n))
    with tf.GradientTape() as tape:
     pred=self.model(self.features(a[sl],b[sl],keys[j],training=True),training=True);data=objective(y[sl],pred,self.cls);reg=penalty(self.model);loss=data+P['l2']*reg
    tf.debugging.assert_all_finite(loss,'nonfinite Predictor loss');self.opt.apply_gradients(zip(tape.gradient(loss,self.model.trainable_variables),self.model.trainable_variables));trace=trace.write(j,tf.stack([loss,data,reg]))
   return trace.stack()
  return epoch
 def predict(self,z,table=None):
  return np.concatenate([self.model(self.features(tf.constant(z['a'][i:i+4096]),tf.constant(z['b'][i:i+4096]),table=table),training=False).numpy().ravel() for i in range(0,len(z['a']),4096)])
def latent_for(domain,condition,seed):
 return np.load(Path(P['condition_root'])/'finder/best_latents.npz')['latent']
def train_reader(domain,condition,target,seed,out):
 z,m=reader_data(domain,target);cls=domain=='tennis' and target=='win';mean=0. if cls else float(z[target].mean());std=1. if cls else max(float(z[target].std()),1e-8);js(out/'scaling.json',{'mean':mean,'std':std,'classification':cls});y=((z[target]-mean)/std)[:,None];r=Reader(m,latent_for(domain,condition,seed),seed,cls);manager=tf.train.CheckpointManager(r.ck,str(out/'checkpoints'),max_to_keep=2);start=0
 if (out/'resume.json').exists():
  state=json.loads((out/'resume.json').read_text());r.ck.restore(state['checkpoint']).expect_partial();start=state['epoch']
 val,_=reader_data(domain,target,'val');vy=(val[target]-mean)/std
 if start:
  best=json.loads((out/'best.json').read_text())['validation_loss']
 else:
  best=float(loss_numpy(vy,r.predict(val),cls).mean());fit0=float(loss_numpy(y[:,0],r.predict(z),cls).mean());r.ck.write(str(out/'best_state'));js(out/'best.json',{'epoch':0,'validation_loss':best,'fit_loss':fit0});js(out/'epoch_000.json',{'epoch':0,'fit_loss':fit0,'validation_loss':best,'seconds':0})
 for epoch in range(start,P['epochs']):
  begin=time.time();order=np.random.default_rng(seedval(71,seed,epoch+1)).permutation(len(y));keys=np.array([seedval(72,seed,epoch+1,0,0,j) for j in range((len(y)+B-1)//B)],np.int32)
  loss=r.epoch(tf.constant(z['a'][order]),tf.constant(z['b'][order]),tf.constant(y[order],tf.float32),tf.constant(keys)).numpy();pred=r.predict(z);fit=float(loss_numpy(y[:,0],pred,cls).mean());vl=float(loss_numpy(vy,r.predict(val),cls).mean())
  if vl<best:
   best=vl;r.ck.write(str(out/'best_state'));js(out/'best.json',{'epoch':epoch+1,'validation_loss':best,'fit_loss':fit})
  np.save(out/f'epoch_{epoch+1:02d}_loss.npy',loss);cp=manager.save(checkpoint_number=epoch+1);js(out/'resume.json',{'epoch':epoch+1,'checkpoint':cp});js(out/f'epoch_{epoch+1:02d}.json',{'fit_loss':fit,'validation_loss':vl,'seconds':time.time()-begin});js(out/'progress.json',{'stage':'Predictor','epoch':epoch+1,'fit_loss':fit,'pid':os.getpid()});print('epoch',epoch+1,'fit',fit,'val',vl,flush=True)
 js(out/'complete.json',{'epochs':P['epochs'],'checkpoint':cp})
def main():
 a=argparse.ArgumentParser();a.add_argument('domain');a.add_argument('condition');a.add_argument('seed',type=int);a.add_argument('--target');x=a.parse_args();P.update(P['probe_caps'][x.domain]);out=W/'runs'/x.domain/x.condition/f'seed_{x.seed}'/('predictor_'+x.target if x.target else 'finder');out.mkdir(parents=True,exist_ok=True)
 lock=(out/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (out/'complete.json').exists():return
 js(out/'status.json',{'state':'running','pid':os.getpid(),'started':time.time()})
 try:
  if x.target:train_reader(x.domain,x.condition,x.target,x.seed,out)
  else:discover(x.domain,x.condition,x.seed,out)
  js(out/'status.json',{'state':'complete','finished':time.time()})
 except BaseException:
  js(out/f'failure_{time.time_ns()}.json',{'error':traceback.format_exc()});js(out/'status.json',{'state':'failed'});raise
if __name__=='__main__':main()
