# TimeRAG-LSTM

## Local LSTM Baseline

Activate the project environment and install dependencies:

```bash
conda activate TimeRAG-LSTM
python -m pip install -r requirements.txt
```

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
