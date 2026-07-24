# TimeRAG-LSTM Agent Guide

`AGENTS.md` owns only the compact, durable project handoff and agent policy.
It is not a turn log, monitor, tutorial, experiment history, or artifact index.

## Fast Path

For each repository task:

1. Read `Current Handoff`, `Experiment Contracts`, and `Write Gate`.
2. Inspect only task-relevant source, tests, results, or logs.
3. Run `git status --short` before editing; preserve unrelated user work.
4. Resolve ambiguity in scope, environment, or acceptance criteria before
   non-trivial action.
5. Never claim completion without admissible evidence.

After compaction or context loss, reread this file and the active action's
evidence before modifying, running, or reporting. Briefly state the recovered
goal, open actions, and risks. If the handoff conflicts with the checkout or the
latest user message, stop and report it. Correct it only in a change task or
explicit progress-sync task, never silently during read-only work.

## Current Handoff

- **Last material handoff update:** 2026-07-24 22:59 CST
- **Stage:** Stage 4 - Full M4 Weekly experiment
- **Status:** Completed; the verified full gated-future result is worse than
  the full plain-LSTM baseline on both primary metrics
- **Formal workspace:** AutoDL `/root/autodl-tmp/TimeRAG-LSTM`
  (user-provided remote state)
- **Last run evidence:** The user-provided AutoDL analysis archive was
  independently parsed on 2026-07-24. The matched 353,270-window, 20-epoch
  gated-future CUDA run loaded cache `b644bb412e5f`, exited `0` after `1356s`,
  and scored SMAPE/MASE `8.8535/2.4228` with learned gate `0.674363`.
  JSON, NPZ, checkpoint, TensorBoard events, and both PNGs are readable and
  mutually consistent.
- **Recovery risk:** The formal gated result used CUDA while the verified plain
  baseline used MPS. Their configuration, IDs, targets, and persistence output
  match, but a same-device rerun remains optional robustness evidence. Do not
  tune a new gate on the official-test error tail.
- **Blockers:** None

### Ordered Actions

1. **Active:** Decide whether to report the verified negative Weekly result as
   the experiment conclusion or authorize a sample-adaptive gate follow-up.
2. If authorized, develop and select the adaptive gate only on `train_tail`,
   then freeze it before one official-test evaluation.
3. Optionally rerun plain and gated on the same CUDA device for robustness;
   expand to Daily or Monthly only after the Weekly research decision.

### Verified Milestones

- Weekly data, metrics, plain/history/gated models, retrieval/cache paths, CLI,
  and required outputs are implemented.
- The current suite has 45 tests; recorded local/WSL checks and user-provided
  AutoDL output show passing runs.
- Full plain baseline, 353,270 windows and 20 epochs: SMAPE `8.5460`, MASE
  `2.3428`.
- Train-tail selection chose temperature `0.25`.
- The full 353,270-window v3 strict-episode gated cache was built and verified
  on AutoDL; its arrays are readable and finite, formal config matches, and a
  repeat cache-only run loaded it and exited `0`.
- The full gated run is verified at SMAPE/MASE `8.8535/2.4228`; persistence is
  `9.1613/2.7773`, retrieval prior is `9.3441/2.7277`, and loss decreased
  `0.757354 -> 0.651752` across all 20 epochs.
- Against plain, gated regresses by `+0.3075` SMAPE and `+0.0800` MASE; it wins
  only `169/359` series by SMAPE and `171/359` by MASE, with positive median
  deltas. Paired-series bootstrap 95% intervals include zero.
- Gated-minus-plain and retrieval-prior-minus-plain per-series MASE deltas have
  correlation `0.9439`, so harmful retrieved futures are the main follow-up
  signal; official-test targets cannot be used to design the fix.
- Checkpoint tensors are finite, TensorBoard contains 20 loss points plus all
  declared metrics/images/configuration, and both standalone plots passed
  decode and visual checks.

Canonical evidence:

- `results/weekly_full_lstm/weekly_lstm_metrics.json`
- `results/weekly_full_lstm/weekly_lstm_predictions.npz`
- `/Users/fangyan/Downloads/weekly_full_gated_t025_analysis.tar.gz`
  (SHA-256 `dca7c4dd64c049034e2f3c59e10a87d2ae6e071a0cce0f3c475668ed12f62d8f`)
- AutoDL `cache/weekly_full_gated_t025/weekly_dtw_top5_b644bb412e5f.npz`
- AutoDL `results/weekly_full_gated_t025/weekly_gated_rag_lstm_metrics.json`
- AutoDL `results/weekly_full_gated_t025/weekly_gated_rag_lstm_predictions.npz`

## Experiment Contracts

- Goal: compare plain LSTM with retrieval-enhanced LSTM in a lightweight
  no-LLM TimeRAG reproduction. Finish Weekly before Daily or Monthly.
- Matched full settings: all 353,270 windows (`max_samples` omitted),
  `official_test`, input `26`, horizon `13`, epochs `20`, batch `32`,
  hidden `64`, one layer, dropout `0`, learning rate `0.001`, scale floor
  `0.001`, Smooth L1 beta `1.0`, seed `42`.
- Gated-only settings: `top_k=5`, Euclidean pool `512`, query batch `64`,
  one retrieval worker, temperature `0.25`, initial gate `0.1`. Compare
  against the verified MPS baseline; same-device rerun is optional robustness.
- Training windows, knowledge base, and outcome bank use official-train only.
  Each sample's normalization statistics come only from its input window.
- Select models and hyperparameters on official-train `train_tail`.
  Official-test targets are final evaluation data, never tuning data.
- For same-series gated candidates, enforce
  `candidate_cutoff + horizon <= query_cutoff - input_length`.
- Gated caches are v3, horizon-aware, content-fingerprinted, structurally
  validated, and atomically written. Never reuse v2 input-only causal caches.
- Exact DTW is a small oracle; scalable retrieval is cKDTree Euclidean
  prefiltering followed by exact DTW reranking.
- Formal outputs: effective config, SMAPE/MASE, predictions, checkpoint, log,
  prediction plot, and retrieval plot.
- Report mixed/negative results honestly. Stage 4 ends only after the full
  cache, gated run, required artifacts, and matched comparison are verified.

## Server Safety

- Use local macOS/WSL for development and smoke tests; use current AutoDL for
  the formal run.
- On a shared server, inspect `nvidia-smi`, `who`, and `w`; use one idle
  GPU. Never kill others' processes or reboot/shut down/log out/sleep the host.
- Log long runs and preserve effective configuration. Old terminal output is
  historical evidence, not current remote state.
- Match commands to Linux shell, PowerShell, or CMD. Verify README examples
  match the live platform and model.

## Evidence Routing

| Information | Owner |
|---|---|
| Current objective, state, blocker, next action | `AGENTS.md` |
| Setup and reusable verified commands | `README.md` |
| Design history, diagnoses, superseded metrics | `steps.md` |
| Behavior contracts | relevant `src/` and `tests/` |
| Effective config and metrics | exact `results/` JSON/CSV |
| Runtime output and telemetry | `logs/` or TensorBoard |
| Predictions, caches, checkpoints, plots | their output directories |

Read only the owner(s) needed for the task. Result verification requires the
exact config, completion log, parsed metrics, and readable required artifacts.

## Write Gate

Edit this file only when all are true:

1. The task authorizes repository changes.
2. New direct or user-provided evidence exists.
3. It causes a durable semantic state change.
4. `AGENTS.md` owns that information.

Qualifying changes are limited to:

- active objective, stage, status, blocker, or ordered actions;
- acceptance criteria or formal experiment outcome;
- current-objective readiness changing among unknown/not ready/ready;
- an actual project-level experiment, causality, validation, or leakage
  contract change;
- a canonical artifact needed for the active/next action;
- a contradicted `Current Handoff` field that affects those actions.

Do not write for:

- explanations, walkthroughs, reviews, plans, or command help;
- timestamp-only reviews, unchanged status, or running-job heartbeats;
- PID, utilization, memory, percentage, ETA, or other volatile telemetry;
- expected commands/artifacts, speculation, or unverified claims;
- repeated tests when code/config/data/cache state is unchanged;
- every smoke artifact, checkpoint, raw metric table, or agent narration.

A long job warrants an update only on a durable transition such as
`not started -> running` or `running -> verified complete/failed`. Process
disappearance alone is not terminal evidence.

### Evidence Standard

- **Direct:** current command, file, parsed artifact, or test inspection.
- **User-provided:** only facts explicitly shown in supplied output; label them.
- **Unverified:** inferred, expected, stale, or unsupported; never use for
  `Completed` or `Verified`.

Verification must postdate relevant changes and identify the checkout/content
state plus effective config and data/cache fingerprint when applicable.
`Completed` requires implementation and verification; experiments also require
terminal success, parsed metrics, and readable required artifacts. `Blocked`
requires a safe failed attempt or confirmed missing permission, input,
dependency, or necessary user decision; never manufacture a risky failure.

### Update Rules

1. If the semantic diff is empty, do not edit.
2. Replace state in place; never append a turn log.
3. When action 1 completes, atomically archive its milestone, promote action 2
   with exit criteria, and sync stage, status, blockers, and evidence.
4. Change `Last material handoff update` only with a semantic update.
5. Keep one active action, at most three actions, eight milestones, current
   blockers, and six canonical evidence anchors.
6. On overflow, remove superseded entries, then those unrelated to active/next
   actions; preserve useful history in `steps.md`.
