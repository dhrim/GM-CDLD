"""Stage an independent run directory; never starts model training."""
import argparse
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    required = [args.data / d / f for d in ('NBA','tennis')
                for f in ('fit.npz','test.npz','meta.json','player_ids.npy')]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        parser.error('Missing prepared inputs: ' + ', '.join(missing))
    if args.output.exists():
        parser.error('Output must be a new directory; existing runs are never overwritten.')
    args.output.mkdir(parents=True)
    for name in ('train.py','evaluate.py'):
        shutil.copy2(root/'code'/name,args.output/name)
    shutil.copy2(root/'protocol/plan.json',args.output/'plan.json')
    for domain in ('NBA','tennis'):
        dest = args.output/'data'/domain
        dest.mkdir(parents=True)
        for name in ('fit.npz','test.npz','meta.json','player_ids.npy'):
            shutil.copy2(args.data/domain/name,dest/name)
    shutil.copy2(root/'tools/run_all.py',args.output/'run_all.py')
    print(f'Staged only: {args.output}. Training has not started.')
if __name__ == '__main__':
    main()
