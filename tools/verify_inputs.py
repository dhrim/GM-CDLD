"""Verify every prepared array against the study's published content hashes."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def fingerprint(path):
 with np.load(path) as z:
  return {k:{'dtype':str(z[k].dtype),'shape':list(z[k].shape),'sha256':hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest()} for k in z.files}
def main():
 p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);a=p.parse_args()
 expected=json.loads((ROOT/'provenance/prepared_array_hashes.json').read_text());n=0
 for rel,entry in expected.items():
  actual=fingerprint(a.build/rel)
  if actual!=entry:raise ValueError(f'Prepared data mismatch: {rel}; do not train on substituted inputs')
  n+=len(entry);print(rel, len(entry),'arrays verified')
 print('PASS:',n,'arrays match the original study inputs')
if __name__=='__main__':main()
