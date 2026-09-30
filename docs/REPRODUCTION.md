# Reproduction workflow

Use notebooks 01–03 for NBA/tennis and 04 for beach volleyball. All run outputs go to ignored `work/` folders. Data preparation is separate from training; final evaluation is a separate explicit cell. Interrupted training can resume by rerunning its training cell without restaging.

NBA/tennis: 32-dimensional member values, 64–16–1 ReLU networks, two Finders, 20 cycles, batch 64, Adam 0.0003, L2 0.001 on selected latent and non-bias Finder weights. Latent Adam resets on each member selection; Finder Adam persists. Readers use frozen latent only, 20 epochs, MSE on training-standardized regression targets or BCE on logits for tennis win. Seeds 17/43/97. Final checkpoint; no early stopping.

Default main notebook: 6 Finders + 24 Predictors. Add C39 to CONDITIONS for the supplementary output-readout variant: 12 + 48. Initial controls are not run by these notebooks. Historical code retains its initial branch for provenance. Training code is byte-identical to the original execution; evaluation orchestration filters out unused conditions without changing normal prediction or loss computations.

The archived environment lock is historical provenance, not a recommended installation recipe for every platform. requirements.txt lists runtime dependencies; hardware/platform can affect exact reproducibility. `fcntl` requires Linux/macOS.

The distributed notebooks do not contain stored execution outputs. Full training was not rerun as part of the GitHub upload. NBA and beach-volleyball upstream data preparation is not yet standalone; see DATA.md.
