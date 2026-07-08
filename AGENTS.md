# TimeRAG + LSTM Experiment Agent Guide

## Project Goal

Build a lightweight reproduction of the TimeRAG idea without using an LLM.

The core experiment is:

- Baseline: LSTM time-series forecasting.
- Proposed: TimeRAG-LSTM, using a time-series knowledge base plus DTW top-k retrieval as extra input to LSTM.
- Main question: Does retrieval-enhanced forecasting outperform plain LSTM?

The first target is not to reproduce the full TimeRAG paper table. The first target is to complete a reliable closed-loop experiment on M4 Weekly.

## Available Devices

### 1. Windows 11 Laptop

Hardware:

- CPU: i9-12900H
- GPU: RTX 3060 Laptop
- RAM: 16 GB
- Disk free: about 215 GB
- Has WSL Ubuntu 22.04

Role:

- Main code development.
- Small-sample debugging.
- Optional small GPU tests with RTX 3060.
- Code packaging and synchronization to server.

Use WSL Ubuntu 22.04 as the main development environment. Prefer storing the project under the WSL Linux filesystem, for example:

```bash
~/code/TimeRAG-LSTM
```

Avoid placing the active training project under `/mnt/c/...` because file I/O can be slower.

### 2. MacBook Air

Hardware:

- Apple M4
- RAM: 16 GB
- Disk free: about 74 GB

Role:

- Reading papers.
- Writing reports.
- Reviewing plots and result tables.
- Backup lightweight development.

Do not use it as the main CUDA training device.

### 3. Lab Server via Sunlogin Remote Desktop

Hardware shown by `nvidia-smi`:

- 4 x NVIDIA GeForce RTX 4090
- 24 GB VRAM per GPU
- CUDA 12.7 shown by driver

Role:

- Main formal experiment machine.
- Run full Weekly experiments.
- Run Daily and Monthly extensions.
- Cache DTW retrieval results.

Use only one free RTX 4090 at first. GPU 1 or GPU 2 is preferred if idle.

## Device Workflow

Recommended workflow:

```text
WSL Ubuntu 22.04 on Windows
-> local small-sample debugging
-> upload/sync to lab server
-> run formal experiments on one RTX 4090
-> pull results back to local machine
-> analyze and write report
```

The server should be treated as a compute machine, not the main code editing machine.

## Project Directory

Recommended structure:

```text
TimeRAG-LSTM/
├── data/
│   └── m4/
├── src/
│   ├── data.py
│   ├── metrics.py
│   ├── models.py
│   ├── retrieval.py
│   └── utils.py
├── scripts/
├── configs/
├── checkpoints/
├── results/
├── logs/
├── train.py
├── requirements.txt
└── README.md
```

Large data, checkpoints, logs, and cached retrieval files should not be mixed with source code.

## Experiment Stages

### Stage 0: Preparation

Device:

- Windows WSL
- Lab server via Sunlogin

Tasks:

- Confirm M4 dataset source.
- Confirm server usage rules.
- Check whether the lab server is idle before running.
- Confirm Python, conda, CUDA, and PyTorch availability.
- Create project structure.

Server checks:

```bash
nvidia-smi
```

On Linux server, also check:

```bash
who
w
```

Do not kill other users' processes.

### Stage 1: Local Development

Device:

- Windows WSL Ubuntu 22.04

Tasks:

- Implement M4 data loading.
- Implement sliding-window sample construction.
- Implement LSTM baseline.
- Implement SMAPE and MASE metrics.
- Implement `train.py` with command-line arguments.

Minimum local debug:

- Use M4 Weekly or fake toy data.
- Use about 100 samples.
- Train for 1-2 epochs.
- Confirm loss decreases.
- Confirm metrics and result files are saved.

### Stage 2: Retrieval and RAG-LSTM Debug

Device:

- Windows WSL
- Optional RTX 3060 Laptop GPU

Tasks:

- Implement time-series knowledge base construction.
- Implement DTW top-k retrieval.
- Save retrieval cache, for example `retrieved_indices.npy`.
- Implement RAG-LSTM.

Recommended first design:

- Query length: `L`
- Retrieved sequence count: `top_k = 5`
- Input tensor after retrieval: `[query, retrieved_1, ..., retrieved_5]`
- Treat them as channels: shape `batch_size x L x 6`
- Feed into LSTM and predict future `H` steps.

The goal is correctness, not high accuracy yet.

### Stage 3: Upload to Lab Server

Device:

- Windows WSL -> lab server

Preferred sync method if SSH is available:

```bash
rsync -av ~/TimeRAG-LSTM/ USER@SERVER:/home/USER/TimeRAG-LSTM/
```

If SSH is not convenient, package the project:

```bash
tar -czf TimeRAG-LSTM.tar.gz TimeRAG-LSTM
```

Then transfer it via Sunlogin remote desktop.

On the server:

- Create or activate conda environment.
- Install dependencies.
- Place M4 data under a fixed path.
- Run a small debug job first.

### Stage 4: First Formal Experiment on M4 Weekly

Device:

- Lab server, one RTX 4090

Frequency:

- M4 Weekly

Settings:

- Input length: 26
- Prediction length: 13
- Retrieval top-k: 5

Run order:

1. Train and evaluate LSTM baseline.
2. Build Weekly knowledge base.
3. Compute and cache DTW top-5 retrieval results.
4. Train and evaluate RAG-LSTM.
5. Save metrics, logs, prediction plots, and retrieval example plots.

Required outputs:

```text
results/
├── weekly_lstm_metrics.json
├── weekly_rag_lstm_metrics.json
├── weekly_comparison.csv
├── weekly_prediction_plot.png
└── weekly_retrieval_example.png
```

This stage is the minimum complete reproduction.

### Stage 5: Extension Experiments

Device:

- Lab server, one RTX 4090

Recommended order:

```text
Weekly -> Daily -> Monthly
```

Only expand after Weekly is stable.

For each frequency:

- Run LSTM baseline.
- Build or load knowledge base.
- Cache DTW retrieval.
- Run RAG-LSTM.
- Save metrics and plots.

Do not start with all six M4 frequencies.

### Stage 6: Result Analysis and Report

Device:

- Windows WSL
- Windows desktop
- MacBook Air

Tasks:

- Pull `results/` and `logs/` back from the server.
- Create comparison tables.
- Inspect prediction curves.
- Inspect retrieval examples.
- Write a short experiment report.

The report should clearly state:

- This is a lightweight reproduction of the TimeRAG idea.
- It uses LSTM instead of an LLM.
- The reproduced part is time-series knowledge base plus DTW retrieval augmentation.
- The main comparison is LSTM vs TimeRAG-LSTM.

## Minimum Success Criteria

Minimum:

- M4 Weekly is completed.
- LSTM and RAG-LSTM are both trained and evaluated.
- SMAPE and MASE are reported.
- At least one prediction plot is produced.
- At least one retrieval example plot is produced.

Solid:

- Weekly, Daily, and Monthly are completed.
- Average results show whether retrieval helps.

Full extension:

- All six M4 frequencies are completed.

## Metrics

Required:

- SMAPE
- MASE

Optional:

- MAE
- MSE

When reporting results, keep the table simple:

```text
Dataset | Model | SMAPE | MASE
M4-Weekly | LSTM | ...
M4-Weekly | TimeRAG-LSTM | ...
```

## Server Usage Rules

Before running experiments:

- Ask or confirm whether the server is free.
- Use `nvidia-smi` to check GPU usage.
- Do not use a GPU that is already heavily occupied.
- Do not kill processes that are not yours.
- Do not reboot, shut down, log out, or sleep the server.
- Use logs for long-running experiments.

If using Windows PowerShell on the server:

```powershell
$env:CUDA_VISIBLE_DEVICES="1"
python train.py --freq Weekly --model rag_lstm --top_k 5
```

If using Linux shell:

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --freq Weekly --model rag_lstm --top_k 5
```

Prefer saving logs:

```bash
python train.py --freq Weekly --model rag_lstm --top_k 5 > logs/weekly_rag.log 2>&1
```

## Implementation Notes

- Build a retrieval cache before training RAG-LSTM to avoid recomputing DTW every epoch.
- Start with simple channel fusion for RAG-LSTM.
- Normalize time series before DTW retrieval.
- Ensure the knowledge base is built only from training data.
- Avoid test leakage.
- Fix random seeds for reproducibility.
- Save configuration with every run.

## Current Priority

The next concrete goal is:

```text
Implement and locally debug M4 Weekly LSTM baseline in WSL.
```

After that:

```text
Add DTW retrieval cache and RAG-LSTM.
```

