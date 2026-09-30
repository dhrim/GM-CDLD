# Data preparation from public sources

Run notebook 01 from a fresh checkout. It downloads original source files, verifies SHA256 hashes, builds the model inputs and verifies all 186 arrays against the study's input hashes. No private research folders or previously prepared arrays are required. The committed game/row IDs are frozen split metadata, not outcome records.

## NBA

`tools/download_sources.py --domain NBA` downloads the pinned Shufinskiy `pbpstats_2023.tar.xz` and `cdnnba_2023.tar.xz` archives and SportsDataverse's `player_boxscores_2024.parquet` (NBA Stats starters/team mapping). Exact URLs and hashes are in `provenance/nba_sources.json`.

`tools/prepare_nba.py` reconstructs both five-player lineups from starters and ordered substitutions, maps ordinary possession summaries to stable-lineup intervals, excludes ambiguous boundary/substitution/special-free-throw records, applies the frozen development/final game allocation, selects players using training exposure, and builds the points/assists/turnovers/offensive-rebounds labels and training-standardized context. Frozen schedule and development split files are under `data/splits/NBA/`.

Expected: 425 players; 81,596 training and 18,815 final-evaluation possessions; five players per side; eight Finder context features. This is the manuscript points dataset, not the later C74 shot-success dataset. Every array, player index and scaling value was checked against the submitted-study inputs.

## Tennis

`tools/download_tennis.py` downloads the three 2017–2019 doubles files using `provenance/tennis_sources.json`. `tools/prepare_tennis.py` implements the frozen tournament split, training-only eligibility, fixed winner-independent source orientation and paired target-team reuse labels. Expected: 153 players, 1,962 training and 460 evaluation source matches.

## Beach volleyball

`tools/download_sources.py --domain beach` downloads the five 2005–2009 BigTimeStats CSVs at the recorded commit. `tools/prepare_beach.py` rebuilds the AVP men's records, complete-statistics exclusions, tournament split and training-exposure fixed point. It generates the three source definitions published in Supplement S11: attack efficiency, point share and attack success rate. All use the same groups and downstream blocks/digs labels.

Expected: 94 players; 2,371 training, 501 validation and 490 reserved test matches. Both team orientations are present. Test arrays are rebuilt to verify the split, but notebooks 04–05 stage only fit/validation arrays and never evaluate test outcomes.

## Commands and verification

Notebook 01 runs these commands and stages the resulting data for later notebooks:

```sh
python tools/download_sources.py --domain NBA --output work/raw/NBA
python tools/download_sources.py --domain beach --output work/raw/beach
python tools/download_tennis.py --output work/raw/tennis
python tools/prepare_nba.py --raw work/raw/NBA --output work/raw_build/NBA
python tools/prepare_tennis.py --raw work/raw/tennis --output work/raw_build/tennis
python tools/prepare_beach.py --raw work/raw/beach --output work/raw_build/beach
python tools/verify_inputs.py --build work/raw_build
```

Builders require new output directories; do not overwrite completed runs. Downloads may reuse only hash-matching cached raw files. If a provider removes or changes a pinned file, the downloader fails explicitly. An exact archived copy can be placed at the documented raw path; the same hash check applies. No silent replacement or changed split is used.

Array fingerprints include dtype, shape and content bytes. The verification manifest contains no raw observations or labels. Prepared arrays, checkpoints and predictions stay in ignored `work/`. Original provider terms remain applicable; see [RIGHTS.md](../RIGHTS.md). `provenance/prepare_data_original.py` is the historical preparation snapshot, not the public entry point.
