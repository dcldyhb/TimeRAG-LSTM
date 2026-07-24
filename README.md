# TimeRAG-LSTM

## Environment Setup

Python 3.11 is the verified version. Create a dedicated Conda environment
before installing one platform-specific requirements file.

Windows PowerShell on the RTX 4090 server:

```powershell
conda create -n timerag-lstm python=3.11 -y
conda activate timerag-lstm
python -m pip install -r requirements-windows.txt
```

Ubuntu 22.04 under WSL2:

```bash
conda create -n timerag-lstm python=3.11 -y
conda activate timerag-lstm
python -m pip install -r requirements-wsl.txt
```

macOS or another default development environment:

```bash
python -m pip install -r requirements.txt
```

The Windows server reports driver 565.90 with CUDA compatibility 12.7. The
Windows and WSL files intentionally install PyTorch 2.7.1 built for CUDA 12.6;
the newer driver is backward compatible, and no separate CUDA toolkit is
required for the PyTorch wheel.

Verify the selected environment:

```bash
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Local LSTM Baseline

Run a small M4 Weekly experiment:

```bash
python train.py \
  --freq Weekly \
  --max-samples 100 \
  --epochs 2 \
  --output-dir results/weekly_mps_smoke \
  --checkpoint-dir checkpoints/weekly_mps_smoke \
  --log-dir runs
```

View the event logs:

```bash
tensorboard --logdir runs --port 6006
```

Then open <http://localhost:6006>. TensorBoard shows the epoch loss, SMAPE,
MASE, persistence-baseline metrics, run configuration, and prediction examples.
Each run also saves metrics, predictions, a checkpoint, and a standalone
`weekly_prediction_plot.png` under the selected output directories.

The baseline uses an input-only relative scale floor to stabilize nearly
constant windows and Smooth L1 loss to keep isolated level shifts from
dominating optimization. Both settings are configurable with
`--relative-scale-floor` and `--smooth-l1-beta`.

## Local TimeRAG-LSTM Debug

Run the retrieval-augmented model on a small Weekly subset:

```bash
python train.py \
  --freq Weekly \
  --model rag_lstm \
  --top-k 5 \
  --max-samples 100 \
  --epochs 2 \
  --output-dir results/weekly_rag_smoke \
  --checkpoint-dir checkpoints/weekly_rag_smoke \
  --retrieval-cache-dir cache/weekly_rag_smoke
```

The knowledge base contains normalized input windows from the training split
only. The default scalable strategy uses a SciPy cKDTree to preselect 512
Euclidean neighbors, then reranks only that bounded pool with exact DTW. It
excludes the query itself and requires every same-series candidate to end before
the query window starts. Query and retrieved windows become `top_k + 1` feature
channels for `RAGLSTMForecaster`.

Each RAG run additionally saves a DTW cache and
`weekly_retrieval_example.png`. Cache fingerprints include the strategy,
candidate-pool size, SciPy version, DTW policy, and all query/knowledge-base
content. Cache files are validated after loading and written atomically.

Build the cache separately before a long training run:

```bash
python train.py \
  --freq Weekly \
  --model rag_lstm \
  --top-k 5 \
  --candidate-pool-size 512 \
  --retrieval-query-batch-size 64 \
  --build-cache-only \
  --retrieval-cache-dir cache/weekly_full_rag \
  --output-dir results/weekly_full_rag
```

Run the same command without `--build-cache-only` to load the verified cache
and train. Use `--retrieval-strategy exact` only as an oracle on small subsets.

Compare bounded retrieval against the exact oracle:

```bash
python -m scripts.benchmark_retrieval \
  --knowledge-base-size 10000 \
  --candidate-pool-sizes 256,512,1024
```

## Gated Future Retrieval

The history-channel model above is retained as an ablation. The accuracy-oriented
model retrieves training windows in the same way, then uses their known
training-only futures as a distance-weighted forecast prior. A query-only LSTM
and a learnable convex gate produce the final forecast.

Select retrieval settings without consulting the official M4 test targets:

```bash
python train.py \
  --freq Weekly \
  --model gated_rag_lstm \
  --evaluation-split train_tail \
  --max-samples 10000 \
  --epochs 10 \
  --top-k 5 \
  --candidate-pool-size 512 \
  --retrieval-temperature 0.25 \
  --initial-retrieval-gate 0.1 \
  --retrieval-cache-dir cache/weekly_gated_validation_10k
```

For this mode, a same-series candidate is valid only when its complete input and
future episode ends before the query input begins. Candidate futures are
standardized with their own input-window statistics; neither candidate targets
nor official test targets estimate normalization statistics. The v3 cache key
includes the candidate horizon and invalidates caches built under the older
input-only causal policy.

After selecting settings on `train_tail`, run the chosen configuration once with
`--evaluation-split official_test`. Metrics include the standalone retrieval
prior and learned gate; prediction NPZ files include
`retrieval_prior_predictions`.
