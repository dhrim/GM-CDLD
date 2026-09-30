from pathlib import Path
import argparse,json,hashlib,urllib.request
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
manifest=json.loads((Path(__file__).resolve().parents[1]/'provenance/tennis_sources.json').read_text())
for item in manifest:
 dest=a.output/item['file']
 data=dest.read_bytes() if dest.exists() else urllib.request.urlopen(item['url']).read()
 if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Source hash mismatch: '+item['file'])
 dest.write_bytes(data);print(item['file'],len(data),'SHA256 OK')
