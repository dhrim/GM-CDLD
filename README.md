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
5. [Published supplementary comparisons](notebooks/05_published_supplementary_comparisons.ipynb) — Joint and the other discovery-output rows in submitted Supplement S11

The default primary workflow uses seeds 17/43/97 and the manuscript's fixed 20-cycle/20-epoch protocol. Notebook 03 includes the supplementary C39 readout variant. Notebook cells show configuration and retain logs, checkpoints, fit losses and detailed predictions. All execution artifacts go under ignored `work/` directories.

- [Method ↔ code mapping](docs/METHOD_MAP.md)
- [Execution and resume](docs/REPRODUCTION.md)
- [Data sources and required inputs](docs/DATA.md)
- [Rights](RIGHTS.md)

## Data availability

Raw outcomes and prepared label arrays are not redistributed. Notebook 01 downloads hash-pinned public sources for all three sports and builds the inputs. All 186 arrays are verified against the original study before training; no private research directories or prepared data from the authors are needed. See the data guide for source versions and download requirements.

This repository contains executable code, not the manuscript or internal research notes. The manuscript was submitted to Applied Intelligence on 30 September 2026. The notebooks include genuine execution outputs. The full submitted-results workflow comprises 60 NBA/tennis jobs, 9 main beach-volleyball jobs and 21 additional published supplementary jobs. NBA and beach inputs rebuilt from public sources match the inputs of the completed main runs exactly.
