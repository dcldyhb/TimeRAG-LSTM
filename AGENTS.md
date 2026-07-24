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

- **Last material handoff update:** 2026-07-24 21:49 CST
- **Stage:** Stage 4 - Full M4 Weekly experiment
- **Status:** In progress; full strict-episode cache verified, gated run next
- **Formal workspace:** AutoDL `/root/autodl-tmp/TimeRAG-LSTM`
  (user-provided remote state)
- **Last run evidence:** User-provided AutoDL output verified the full cache at
  `cache/weekly_full_gated_t025/weekly_dtw_top5_b644bb412e5f.npz`: readable
  `(353270, 5)` training and `(359, 5)` evaluation indices/distances, all
  distances finite, formal configuration matched, and a repeat cache-only run
  loaded it and exited `0`. Build time was `10333.17s`; SHA-256 is
  `eb34659ebe01ec77827478f9448c9f558ea41495653dcf1fac6eaca9b1c04347`.
- **Recovery risk:** The verified cache is currently evidenced only in the
  AutoDL data-disk workspace. Run the formal training under `screen`, require
  this exact cache to load, and preserve its completion log and outputs.
- **Blockers:** None

### Ordered Actions

1. **Active:** Run and verify the matched full gated-future Weekly experiment
   at temperature `0.25`, loading the verified `b644bb412e5f` cache. Require
   terminal success, exact formal configuration, metrics, predictions,
   checkpoint, completion log, and readable forecast/retrieval plots.
2. Compare it with the verified full plain-LSTM baseline and inspect the
   per-series error tail.
3. If harmful MASE outliers persist, develop any adaptive gate on `train_tail`,
   never official-test targets.

### Verified Milestones

- Weekly data, metrics, plain/history/gated models, retrieval/cache paths, CLI,
  and required outputs are implemented.
- The current suite has 45 tests; recorded local/WSL checks and user-provided
  AutoDL output show passing runs.
- Full plain baseline, 353,270 windows and 20 epochs: SMAPE `8.5460`, MASE
  `2.3428`.
- Train-tail selection chose temperature `0.25`.
- Matched 10k official result is mixed: gated `8.4361/2.3809` versus plain
  `8.5773/2.3726` SMAPE/MASE.
- WSL CUDA was directly verified; user-provided AutoDL output shows RTX 4090
  dependency, CUDA, test, and 100-window gated smoke checks passed.
- The clean server package was directly verified; user output shows the AutoDL
  tree moved to the faster data disk.
- The full 353,270-window v3 strict-episode gated cache was built and verified
  on AutoDL; its arrays are readable and finite, formal config matches, and a
  repeat cache-only run loaded it and exited `0`.

Canonical evidence:

- `results/weekly_full_lstm/weekly_lstm_metrics.json`
- `results/weekly_gated_validation_comparison.csv`
- `results/weekly_gated_rag_10k_t025/weekly_gated_rag_lstm_metrics.json`
- `results/weekly_gated_10k_comparison.csv`
- AutoDL `cache/weekly_full_gated_t025/weekly_dtw_top5_b644bb412e5f.npz`
- AutoDL `results/weekly_full_gated_t025/weekly_gated_rag_lstm_cache.json`

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
