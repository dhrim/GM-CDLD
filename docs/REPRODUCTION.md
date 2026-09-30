# Reproduction workflow

Use notebooks 01–03 for NBA/tennis and 04–05 for beach volleyball. All run outputs go to ignored `work/` folders. Data preparation is separate from training; final evaluation is a separate explicit cell. Interrupted training can resume by rerunning its training cell without restaging.

NBA/tennis: 32-dimensional member values, 64–16–1 ReLU networks, two Finders, 20 cycles, batch 64, Adam 0.0003, L2 0.001 on selected latent and non-bias Finder weights. Latent Adam resets on each member selection; Finder Adam persists. Readers use frozen latent only, 20 epochs, MSE on training-standardized regression targets or BCE on logits for tennis win. Seeds 17/43/97. Final checkpoint; no early stopping.

Notebook 02 trains 6 Finders + 24 Predictors. Notebook 03 also trains the supplementary C39 output-readout variant before evaluation, for a total of 12 Finders + 48 Predictors. Initial controls are not run by these notebooks. Historical code retains its initial branch for provenance. Training code is byte-identical to the original execution; evaluation orchestration filters out unused conditions without changing normal prediction or loss computations.

The archived environment lock is historical provenance, not a recommended installation recipe for every platform. requirements.txt lists runtime dependencies; hardware/platform can affect exact reproducibility. `fcntl` requires Linux/macOS.

The first execution ran 69 jobs afresh. Notebook 01 was subsequently extended to rebuild NBA and all three beach source variants from public raw files; all 186 input arrays match the archived study exactly. Notebook 05 adds 21 fresh jobs for the remaining submitted S11 comparisons. Main-analysis outputs in notebooks 02–03 remain the actual completed execution on those identical inputs; they were not copied from historical results. The main-analysis numeric results exactly match the execution used for the submitted manuscript. See DATA.md for the now-complete preparation path.
