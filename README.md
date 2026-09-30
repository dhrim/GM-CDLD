# GM-CDLD

Reproducible code for discovering member latent values from group outcomes and reusing the fixed values to predict other outcomes.

## Run

Use Python 3.10+ on Linux/macOS:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r environment/requirements.txt
jupyter lab
```

Run these notebooks in order:

1. [Data preparation](notebooks/01_data_preparation.ipynb)
2. [Discovery and frozen-latent predictors](notebooks/02_discovery_and_reuse.ipynb)
3. [Evaluation and detailed results](notebooks/03_evaluation_and_tables.ipynb)
4. [Beach-volleyball exploratory extension](notebooks/04_beach_volleyball_exploratory.ipynb) — separate validation protocol

The default primary workflow uses seeds 17/43/97 and the manuscript's fixed 20-cycle/20-epoch protocol. The supplementary readout variant is optional. Notebook cells show configuration and retain logs, checkpoints, fit losses and detailed predictions. All execution artifacts go under ignored `work/` directories.

- [Method ↔ code mapping](docs/METHOD_MAP.md)
- [Execution and resume](docs/REPRODUCTION.md)
- [Data sources and required inputs](docs/DATA.md)
- [Rights](RIGHTS.md)

## Data availability

Raw outcomes and prepared label arrays are not redistributed. Tennis preparation is included with archived source-hash checks. NBA and beach-volleyball currently require prepared input arrays; their upstream preparation packaging is not yet standalone. See the data guide before attempting those experiments.

This repository contains executable code, not the manuscript or internal research notes. The manuscript was submitted to Applied Intelligence on 30 September 2026. All four notebooks include genuine execution outputs from 30 September 2026 (60 NBA/tennis training jobs, including the supplementary C39 variant, and 9 beach-volleyball jobs). The standalone data-preparation limitations for NBA and beach volleyball are documented above.
