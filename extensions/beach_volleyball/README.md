# Exploratory beach-volleyball extension

This is the separately selected validation experiment, not the NBA/tennis fixed-budget protocol. Source: team attack success rate; frozen readers: blocks and digs per set. Seeds 17/43/101. Source-selected latent and reader epoch use validation. The test split remains unevaluated.

Notebook 01 builds all source/reuse arrays from hash-pinned public CSVs. Notebook 04 runs the GM attack-success experiment. Notebook 05 reproduces the Joint comparison and the attack-efficiency/point-share discovery-output rows already reported in Supplement S11. `shot_success` is the historical internal target key; its definition depends on the variant. `mse` denotes GM. Initial branches remain historical code but are not run. Training implementations are unchanged. See ../../docs/DATA.md.
