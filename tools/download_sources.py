"""Download pinned, hash-checked NBA or beach raw inputs without redistributing them."""
from pathlib import Path
import argparse,hashlib,json,urllib.request,time
ROOT=Path(__file__).resolve().parents[1]
def download(item,dest):
 expected=item['sha256'];dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():
  if hashlib.sha256(dest.read_bytes()).hexdigest()!=expected:raise ValueError(f'Hash mismatch: {dest.name}; do not silently replace source versions')
  print(dest.name,'verified cached file',flush=True);return
 tmp=dest.with_suffix(dest.suffix+'.part')
 for attempt in range(3):
  try:
   req=urllib.request.Request(item['url'],headers={'User-Agent':'GM-CDLD-reproduction'})
   with urllib.request.urlopen(req,timeout=180) as response, tmp.open('wb') as target:
    while True:
     chunk=response.read(1024*1024)
     if not chunk:break
     target.write(chunk)
   if hashlib.sha256(tmp.read_bytes()).hexdigest()!=expected:raise ValueError(f'Hash mismatch for {dest.name}: upstream bytes differ from the study; provide the documented version, not a substitute')
   tmp.replace(dest);print(dest.name,'downloaded, SHA256 verified',flush=True);return
  except Exception:
   if attempt==2:raise
   time.sleep(2*(attempt+1))
def main():
 p=argparse.ArgumentParser();p.add_argument('--domain',choices=['NBA','beach'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.domain=='NBA':items=json.loads((ROOT/'provenance/nba_sources.json').read_text())['files']
 else:
  items=json.loads((ROOT/'extensions/beach_volleyball/source_manifest.json').read_text())['files']
  items=[dict(x,name=Path(x['path']).name) for x in items if Path(x['path']).name in [f'bvb_matches_{y}.csv' for y in range(2005,2010)]]
 for item in items:download(item,a.output/item['name'])
if __name__=='__main__':main()
