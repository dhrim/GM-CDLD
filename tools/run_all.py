"""Run immutable manuscript jobs in a staged directory; retain individual logs."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--evaluate',action='store_true');p.add_argument('--jobs',type=int,default=1);a=p.parse_args()
 w=Path(__file__).resolve().parent
 if not (w/'train.py').is_file():p.error('Stage a new run directory first.')
 if a.jobs<1:p.error('--jobs must be positive')
 cfg=json.loads((w/'plan.json').read_text());logs=w/'logs';logs.mkdir(exist_ok=True)
 def run(arguments):
  name='_'.join(str(x).replace('--','') for x in arguments);log=logs/(name+'.log')
  print('START',name,flush=True)
  with log.open('a') as f:subprocess.run([sys.executable,str(w/'train.py'),*map(str,arguments)],cwd=w,stdout=f,stderr=subprocess.STDOUT,check=True)
  print('DONE',name,flush=True)
 if a.execute:
  finders=[[d,c,seed] for d in cfg['domains'] for c in cfg['conditions'] for seed in cfg['seeds']]
  readers=[[d,c,seed,'--target',t] for d in cfg['domains'] for c in cfg['conditions'] for seed in cfg['seeds'] for t in json.loads((w/'data'/d/'meta.json').read_text())['targets']]
  with ThreadPoolExecutor(max_workers=a.jobs) as pool:list(pool.map(run,finders))
  with ThreadPoolExecutor(max_workers=a.jobs) as pool:list(pool.map(run,readers))
  (w/'train_complete.json').write_text(json.dumps(dict(discovery_runs=len(finders),predictor_runs=len(readers))))
 if a.evaluate:subprocess.run([sys.executable,str(w/'evaluate.py')],cwd=w,check=True)
 if not(a.execute or a.evaluate):p.error('Specify --execute and/or --evaluate.')
if __name__=='__main__':main()
