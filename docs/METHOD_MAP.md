# Manuscript ↔ implementation

- §3.1: code/train.py `Finder.__init__`, `features`: shared latent, two separately weighted networks, side flags, context.
- §3.2: `Finder.make_visit`, `discover`: selected member only; same pre-update gradient; A then B; per-member Adam reset and persistent Finder Adam. Each cycle presents each row once per member (NBA 10, tennis 4).
- §3.3: `Reader`, `latent_for`, `train_reader`: frozen tf.constant latent; new network per target; no Finder context/weights.
- §4.3: `objective`: regression MSE / sigmoid_cross_entropy_with_logits. `penalty`: non-bias network weights. Finder adds selected latent squared norm; Reader does not optimize latent.
- §4.5: code/evaluate.py `baseline`: role-separated 0/1 player design, intercept, fixed 0.01 regularization, ridge or logistic.
- §5 / tables 2–4: evaluation writes final/summary.json with constant, direct ID and frozen reader metrics.
- Supplementary output-readout variant: internal C39 branch, same original implementation.
- Beach volleyball: separate extensions/beach_volleyball code and notebook 04; validation-selected persistent member Adam protocol, not main-manuscript C42.

Verified in this update: source file hash against archived execution, loss branches, regularization operands, optimizer reset, update variables. The completed main execution matches the archived manuscript results. Raw-data builders verify all 186 input arrays; supplementary notebook 05 covers the remaining beach comparisons.
