"""Train and evaluate the plain LSTM baseline on an M4 frequency."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

from src.config import DEVICE
from src.data import (
    build_evaluation_windows,
    build_training_windows,
    invert_standardization,
    load_m4_frequency,
    standardize_by_input,
)
from src.metric import mase, smape
from src.models import LSTMForecaster


M4_SEASONAL_PERIODS = {
    "Yearly": 1,
    "Quarterly": 4,
    "Monthly": 12,
    "Weekly": 1,
    "Daily": 1,
    "Hourly": 24,
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one plain LSTM baseline run."""

    data_dir: Path = Path("data/m4")
    frequency: str = "Weekly"
    input_length: int = 26
    stride: int = 1
    max_samples: int | None = None
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    relative_scale_floor: float = 1e-3
    smooth_l1_beta: float = 1.0
    seed: int = 42
    output_dir: Path = Path("results")
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("runs")

    def validate(self) -> None:
        positive_values = {
            "input_length": self.input_length,
            "stride": self.stride,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive when provided")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.relative_scale_floor < 0:
            raise ValueError("relative_scale_floor must be non-negative")
        if self.smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible local runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_training_loader(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(inputs, dtype=np.float32)),
        torch.from_numpy(np.asarray(targets, dtype=np.float32)),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )


def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    smooth_l1_beta: float,
    writer: SummaryWriter | None = None,
) -> list[float]:
    """Optimize a forecasting model and return mean loss for each epoch."""
    criterion = nn.SmoothL1Loss(beta=smooth_l1_beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_losses: list[float] = []

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        sample_count = 0

        for batch_inputs, batch_targets in data_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Training produced a non-finite loss")
            loss.backward()
            optimizer.step()

            batch_count = batch_inputs.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_count
            sample_count += batch_count

        mean_loss = total_loss / sample_count
        epoch_losses.append(mean_loss)
        if writer is not None:
            writer.add_scalar("train/loss", mean_loss, epoch + 1)
        print(f"Epoch {epoch + 1:>3}/{epochs}: loss={mean_loss:.6f}")

    return epoch_losses


def predict(
    model: nn.Module,
    inputs: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Run batched inference and return CPU NumPy predictions."""
    input_tensor = torch.from_numpy(np.asarray(inputs, dtype=np.float32))
    predictions: list[Tensor] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(input_tensor), batch_size):
            batch = input_tensor[start : start + batch_size].to(device)
            predictions.append(model(batch).detach().cpu())

    if not predictions:
        raise ValueError("inputs must contain at least one sample")
    return torch.cat(predictions, dim=0).numpy()


def _serializable_config(
    config: ExperimentConfig,
    *,
    frequency: str,
    horizon: int,
) -> dict[str, object]:
    values = asdict(config)
    values["data_dir"] = str(config.data_dir)
    values["output_dir"] = str(config.output_dir)
    values["checkpoint_dir"] = str(config.checkpoint_dir)
    values["log_dir"] = str(config.log_dir)
    values["frequency"] = frequency
    values["horizon"] = horizon
    return values


def _prediction_figure(
    inputs: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    series_id: str,
    persistence_predictions: np.ndarray | None = None,
) -> Figure:
    """Plot one input window followed by its target and forecast."""
    history_steps = np.arange(-len(inputs) + 1, 1)
    forecast_steps = np.arange(1, len(targets) + 1)

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(history_steps, inputs, label="Input history", color="#2563eb")
    axis.plot(forecast_steps, targets, label="Ground truth", color="#15803d")
    axis.plot(
        forecast_steps,
        predictions,
        label="LSTM forecast",
        color="#dc2626",
        linestyle="--",
    )
    if persistence_predictions is not None:
        axis.plot(
            forecast_steps,
            persistence_predictions,
            label="Persistence forecast",
            color="#7c3aed",
            linestyle="-.",
        )
    axis.axvline(0, color="#525252", linewidth=1, linestyle=":")
    axis.set_title(f"M4 forecast: {series_id}")
    axis.set_xlabel("Time step relative to forecast origin")
    axis.set_ylabel("Value")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def run_experiment(
    config: ExperimentConfig,
    *,
    device: torch.device | None = None,
) -> dict[str, object]:
    """Run the complete baseline pipeline and persist its artifacts."""
    config.validate()
    selected_device = DEVICE if device is None else device
    set_random_seed(config.seed)

    data = load_m4_frequency(config.data_dir, config.frequency)
    artifact_stem = f"{data.frequency.lower()}_lstm"
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    tensorboard_dir = config.log_dir / f"{artifact_stem}_{run_timestamp}"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = config.output_dir / f"{artifact_stem}_metrics.json"
    predictions_path = config.output_dir / f"{artifact_stem}_predictions.npz"
    prediction_plot_path = config.output_dir / f"{data.frequency.lower()}_prediction_plot.png"
    checkpoint_path = config.checkpoint_dir / f"{artifact_stem}.pt"
    configuration = _serializable_config(
        config,
        frequency=data.frequency,
        horizon=data.horizon,
    )

    train_samples = build_training_windows(
        data.train,
        input_length=config.input_length,
        horizon=data.horizon,
        stride=config.stride,
        max_samples=config.max_samples,
        seed=config.seed,
    )
    normalized_train_inputs, normalized_train_targets, _, _ = standardize_by_input(
        train_samples.inputs,
        train_samples.targets,
        relative_scale_floor=config.relative_scale_floor,
    )
    if normalized_train_targets is None:
        raise RuntimeError("Training targets were not standardized")

    train_loader = _make_training_loader(
        normalized_train_inputs,
        normalized_train_targets,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    model = LSTMForecaster(
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        horizon=data.horizon,
        dropout=config.dropout,
    ).to(selected_device)

    print(
        f"Training LSTM on M4-{data.frequency}: "
        f"samples={len(train_samples)}, device={selected_device}"
    )
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    try:
        writer.add_text(
            "run/configuration",
            f"```json\n{json.dumps(configuration, indent=2, sort_keys=True)}\n```",
            0,
        )
        epoch_losses = train_model(
            model,
            train_loader,
            device=selected_device,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            smooth_l1_beta=config.smooth_l1_beta,
            writer=writer,
        )

        evaluation_samples = build_evaluation_windows(data, config.input_length)
        normalized_eval_inputs, _, locations, scales = standardize_by_input(
            evaluation_samples.inputs,
            relative_scale_floor=config.relative_scale_floor,
        )
        normalized_predictions = predict(
            model,
            normalized_eval_inputs,
            batch_size=config.batch_size,
            device=selected_device,
        )
        predictions = invert_standardization(
            normalized_predictions, locations, scales
        ).astype(np.float32)

        metric_values = {
            "smape": smape(evaluation_samples.targets, predictions),
            "mase": mase(
                evaluation_samples.targets,
                predictions,
                data.train,
                seasonal_period=M4_SEASONAL_PERIODS[data.frequency],
            ),
        }
        persistence_predictions = np.repeat(
            evaluation_samples.inputs[:, -1:],
            data.horizon,
            axis=1,
        )
        persistence_metric_values = {
            "smape": smape(evaluation_samples.targets, persistence_predictions),
            "mase": mase(
                evaluation_samples.targets,
                persistence_predictions,
                data.train,
                seasonal_period=M4_SEASONAL_PERIODS[data.frequency],
            ),
        }
        writer.add_scalar("eval/smape", metric_values["smape"], config.epochs)
        writer.add_scalar("eval/mase", metric_values["mase"], config.epochs)
        writer.add_scalar(
            "baseline/persistence_smape",
            persistence_metric_values["smape"],
            config.epochs,
        )
        writer.add_scalar(
            "baseline/persistence_mase",
            persistence_metric_values["mase"],
            config.epochs,
        )

        plot_count = min(3, len(evaluation_samples))
        for sample_index in range(plot_count):
            figure = _prediction_figure(
                evaluation_samples.inputs[sample_index],
                evaluation_samples.targets[sample_index],
                predictions[sample_index],
                series_id=data.ids[sample_index],
                persistence_predictions=persistence_predictions[sample_index],
            )
            if sample_index == 0:
                figure.savefig(prediction_plot_path, dpi=150, bbox_inches="tight")
            writer.add_figure(
                f"predictions/{data.ids[sample_index]}",
                figure,
                global_step=config.epochs,
                close=True,
            )
        writer.flush()
    finally:
        writer.close()

    checkpoint_state = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    torch.save(
        {
            "model_state_dict": checkpoint_state,
            "config": configuration,
        },
        checkpoint_path,
    )
    np.savez_compressed(
        predictions_path,
        ids=np.asarray(data.ids),
        targets=evaluation_samples.targets,
        predictions=predictions,
        persistence_predictions=persistence_predictions,
    )

    results: dict[str, object] = {
        "dataset": f"M4-{data.frequency}",
        "model": "LSTM",
        "device": selected_device.type,
        "config": configuration,
        "data": {
            "training_samples": len(train_samples),
            "evaluation_samples": len(evaluation_samples),
            "prediction_shape": list(predictions.shape),
        },
        "training": {
            "loss": "smooth_l1",
            "smooth_l1_beta": config.smooth_l1_beta,
            "epoch_losses": epoch_losses,
        },
        "metrics": metric_values,
        "baselines": {"persistence": persistence_metric_values},
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "predictions": str(predictions_path),
            "prediction_plot": str(prediction_plot_path),
            "metrics": str(metrics_path),
            "tensorboard": str(tensorboard_dir),
        },
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    print(f"SMAPE: {metric_values['smape']:.4f}")
    print(f"MASE:  {metric_values['mase']:.4f}")
    print(
        "Persistence: "
        f"SMAPE={persistence_metric_values['smape']:.4f}, "
        f"MASE={persistence_metric_values['mase']:.4f}"
    )
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved prediction plot to {prediction_plot_path}")
    print(f"Saved TensorBoard logs to {tensorboard_dir}")
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the plain LSTM baseline on M4 data."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/m4"))
    parser.add_argument("--freq", dest="frequency", default="Weekly")
    parser.add_argument("--input-length", type=int, default=26)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--relative-scale-floor", type=float, default=1e-3)
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    return parser


def parse_args(argv: Sequence[str] | None = None) -> ExperimentConfig:
    namespace = build_argument_parser().parse_args(argv)
    return ExperimentConfig(**vars(namespace))


def main(argv: Sequence[str] | None = None) -> None:
    run_experiment(parse_args(argv))


if __name__ == "__main__":
    main()
