"""Train and evaluate LSTM retrieval ablations on an M4 frequency."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
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
    build_temporal_validation_split,
    build_training_windows,
    invert_standardization,
    load_m4_frequency,
    standardize_by_input,
)
from src.metric import mase, smape
from src.models import (
    GatedFutureLSTMForecaster,
    LSTMForecaster,
    RAGLSTMForecaster,
)
from src.retrieval import (
    RetrievalKnowledgeBase,
    RetrievalResult,
    assemble_future_prior,
    assemble_rag_inputs,
    build_knowledge_base,
    build_outcome_bank,
    load_retrieval_cache,
    retrieval_cache_key,
    retrieve_top_k,
    save_retrieval_cache,
    validate_retrieval_result,
)


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
    """Configuration for one LSTM or TimeRAG-LSTM run."""

    data_dir: Path = Path("data/m4")
    frequency: str = "Weekly"
    model: str = "lstm"
    evaluation_split: str = "official_test"
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
    top_k: int = 5
    retrieval_strategy: str = "euclidean_prefilter"
    candidate_pool_size: int = 512
    retrieval_query_batch_size: int = 64
    retrieval_workers: int = 1
    retrieval_temperature: float = 0.25
    initial_retrieval_gate: float = 0.1
    build_cache_only: bool = False
    seed: int = 42
    output_dir: Path = Path("results")
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("runs")
    retrieval_cache_dir: Path = Path("cache")

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
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.retrieval_strategy not in {"exact", "euclidean_prefilter"}:
            raise ValueError("retrieval_strategy must be 'exact' or 'euclidean_prefilter'")
        if (
            self.retrieval_strategy == "euclidean_prefilter"
            and self.candidate_pool_size < self.top_k
        ):
            raise ValueError("candidate_pool_size must be at least top_k")
        if self.retrieval_query_batch_size <= 0:
            raise ValueError("retrieval_query_batch_size must be positive")
        if self.retrieval_workers == 0 or self.retrieval_workers < -1:
            raise ValueError("retrieval_workers must be -1 or a positive integer")
        if (
            not np.isfinite(self.retrieval_temperature)
            or self.retrieval_temperature <= 0
        ):
            raise ValueError("retrieval_temperature must be positive and finite")
        if not 0.0 < self.initial_retrieval_gate < 1.0:
            raise ValueError("initial_retrieval_gate must be strictly between 0 and 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.model not in {"lstm", "rag_lstm", "gated_rag_lstm"}:
            raise ValueError(
                "model must be 'lstm', 'rag_lstm', or 'gated_rag_lstm'"
            )
        if self.evaluation_split not in {"official_test", "train_tail"}:
            raise ValueError(
                "evaluation_split must be 'official_test' or 'train_tail'"
            )
        if self.build_cache_only and self.model not in {
            "rag_lstm",
            "gated_rag_lstm",
        }:
            raise ValueError("build_cache_only requires a retrieval model")


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible local runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _retrieval_progress(label: str):
    next_percentage = 10

    def report(completed: int, total: int) -> None:
        nonlocal next_percentage
        percentage = int(100 * completed / total)
        if completed == total or percentage >= next_percentage:
            print(f"{label} retrieval: {completed}/{total} ({percentage}%)")
            next_percentage = percentage + 10

    return report


def _make_training_loader(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    future_priors: np.ndarray | None = None,
) -> DataLoader:
    tensors = [
        torch.from_numpy(np.asarray(inputs, dtype=np.float32)),
        torch.from_numpy(np.asarray(targets, dtype=np.float32)),
    ]
    if future_priors is not None:
        priors = np.asarray(future_priors, dtype=np.float32)
        if priors.ndim != 2 or priors.shape != targets.shape:
            raise ValueError("future_priors must match targets shape [samples, horizon]")
        tensors.append(torch.from_numpy(priors))
    dataset = TensorDataset(*tensors)
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

        for batch in data_loader:
            batch_inputs, batch_targets = batch[:2]
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            batch_prior = batch[2].to(device) if len(batch) == 3 else None

            optimizer.zero_grad(set_to_none=True)
            predictions = (
                model(batch_inputs)
                if batch_prior is None
                else model(batch_inputs, batch_prior)
            )
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
    future_priors: np.ndarray | None = None,
) -> np.ndarray:
    """Run batched inference and return CPU NumPy predictions."""
    input_tensor = torch.from_numpy(np.asarray(inputs, dtype=np.float32))
    prior_tensor = None
    if future_priors is not None:
        priors = np.asarray(future_priors, dtype=np.float32)
        if priors.ndim != 2 or len(priors) != len(input_tensor):
            raise ValueError("future_priors must have shape [samples, horizon]")
        prior_tensor = torch.from_numpy(priors)
    predictions: list[Tensor] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(input_tensor), batch_size):
            batch = input_tensor[start : start + batch_size].to(device)
            if prior_tensor is None:
                batch_predictions = model(batch)
            else:
                batch_prior = prior_tensor[start : start + batch_size].to(device)
                batch_predictions = model(batch, batch_prior)
            predictions.append(batch_predictions.detach().cpu())

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
    values["retrieval_cache_dir"] = str(config.retrieval_cache_dir)
    values["frequency"] = frequency
    values["horizon"] = horizon
    return values


def _prediction_figure(
    inputs: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    series_id: str,
    model_label: str = "LSTM",
    persistence_predictions: np.ndarray | None = None,
    retrieval_prior: np.ndarray | None = None,
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
        label=f"{model_label} forecast",
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
    if retrieval_prior is not None:
        axis.plot(
            forecast_steps,
            retrieval_prior,
            label="Retrieval future prior",
            color="#a16207",
            linestyle=":",
        )
    axis.axvline(0, color="#525252", linewidth=1, linestyle=":")
    axis.set_title(f"M4 forecast: {series_id}")
    axis.set_xlabel("Time step relative to forecast origin")
    axis.set_ylabel("Value")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def _retrieval_figure(
    query: np.ndarray,
    retrieved: np.ndarray,
    distances: np.ndarray,
    *,
    series_id: str,
) -> Figure:
    """Plot the normalized sequences that were compared by DTW."""
    steps = np.arange(-len(query) + 1, 1)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(steps, query, label=f"Query ({series_id})", linewidth=2.5)
    for rank, (sequence, distance) in enumerate(
        zip(retrieved, distances), start=1
    ):
        axis.plot(
            steps,
            sequence,
            label=f"Top {rank} (DTW={float(distance):.3f})",
            alpha=0.75,
        )
    axis.set_title(f"Training-only normalized DTW retrieval: {series_id}")
    axis.set_xlabel("Time step relative to forecast origin")
    axis.set_ylabel("Input-normalized value")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def run_experiment(
    config: ExperimentConfig,
    *,
    device: torch.device | None = None,
) -> dict[str, object]:
    """Run a plain or retrieval-augmented LSTM experiment."""
    config.validate()
    selected_device = DEVICE if device is None else device
    set_random_seed(config.seed)

    data = load_m4_frequency(config.data_dir, config.frequency)
    if config.evaluation_split == "train_tail":
        validation_split = build_temporal_validation_split(
            data.train,
            input_length=config.input_length,
            horizon=data.horizon,
        )
        fit_series = validation_split.fit_series
        evaluation_samples = validation_split.validation_samples
        split_suffix = "_train_tail"
    else:
        fit_series = data.train
        evaluation_samples = build_evaluation_windows(data, config.input_length)
        split_suffix = ""

    artifact_stem = f"{data.frequency.lower()}_{config.model}{split_suffix}"
    model_label = {
        "lstm": "LSTM",
        "rag_lstm": "TimeRAG-LSTM",
        "gated_rag_lstm": "TimeRAG-LSTM (gated future)",
    }[config.model]
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    tensorboard_dir = config.log_dir / f"{artifact_stem}_{run_timestamp}"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = config.output_dir / f"{artifact_stem}_metrics.json"
    cache_summary_path = config.output_dir / f"{artifact_stem}_cache.json"
    predictions_path = config.output_dir / f"{artifact_stem}_predictions.npz"
    prediction_plot_path = config.output_dir / f"{data.frequency.lower()}_prediction_plot.png"
    retrieval_plot_path = (
        config.output_dir / f"{data.frequency.lower()}_retrieval_example.png"
    )
    checkpoint_path = config.checkpoint_dir / f"{artifact_stem}.pt"
    configuration = _serializable_config(
        config,
        frequency=data.frequency,
        horizon=data.horizon,
    )

    train_samples = build_training_windows(
        fit_series,
        input_length=config.input_length,
        horizon=data.horizon,
        stride=config.stride,
        max_samples=config.max_samples,
        seed=config.seed,
    )
    targets_to_standardize = None if config.build_cache_only else train_samples.targets
    normalized_train_inputs, normalized_train_targets, _, _ = standardize_by_input(
        train_samples.inputs,
        targets_to_standardize,
        relative_scale_floor=config.relative_scale_floor,
    )
    if not config.build_cache_only and normalized_train_targets is None:
        raise RuntimeError("Training targets were not standardized")

    normalized_eval_inputs, _, locations, scales = standardize_by_input(
        evaluation_samples.inputs,
        relative_scale_floor=config.relative_scale_floor,
    )

    knowledge_base: RetrievalKnowledgeBase | None = None
    training_retrieval: RetrievalResult | None = None
    evaluation_retrieval: RetrievalResult | None = None
    retrieval_cache_path: Path | None = None
    cache_was_loaded = False
    retrieval_seconds = 0.0
    train_future_priors: np.ndarray | None = None
    evaluation_future_priors: np.ndarray | None = None
    retrieval_prior_predictions: np.ndarray | None = None
    learned_retrieval_gate: float | None = None
    retrieval_models = {"rag_lstm", "gated_rag_lstm"}
    candidate_horizon = data.horizon if config.model == "gated_rag_lstm" else 0
    same_series_policy = (
        "complete_candidate_episode_before_query_input"
        if candidate_horizon
        else "candidate_input_before_query_input"
    )

    if config.model in retrieval_models:
        knowledge_base = build_knowledge_base(
            train_samples,
            relative_scale_floor=config.relative_scale_floor,
        )
        cache_key = retrieval_cache_key(
            knowledge_base,
            train_queries=normalized_train_inputs,
            train_series_indices=train_samples.series_indices,
            train_cutoffs=train_samples.cutoffs,
            evaluation_queries=normalized_eval_inputs,
            evaluation_series_indices=evaluation_samples.series_indices,
            evaluation_cutoffs=evaluation_samples.cutoffs,
            top_k=config.top_k,
            strategy=config.retrieval_strategy,
            candidate_pool_size=config.candidate_pool_size,
            candidate_horizon=candidate_horizon,
        )
        retrieval_cache_path = config.retrieval_cache_dir / (
            f"{data.frequency.lower()}_dtw_top{config.top_k}_{cache_key[:12]}.npz"
        )
        retrieval_started = perf_counter()
        cached = load_retrieval_cache(
            retrieval_cache_path,
            expected_key=cache_key,
            expected_training_shape=(len(train_samples), config.top_k),
            expected_evaluation_shape=(len(evaluation_samples), config.top_k),
            knowledge_base_size=len(knowledge_base),
        )
        if cached is not None:
            try:
                validate_retrieval_result(
                    cached[0],
                    knowledge_base,
                    query_series_indices=train_samples.series_indices,
                    query_cutoffs=train_samples.cutoffs,
                    exclude_indices=np.arange(len(train_samples), dtype=np.int64),
                    candidate_horizon=candidate_horizon,
                )
                validate_retrieval_result(
                    cached[1],
                    knowledge_base,
                    query_series_indices=evaluation_samples.series_indices,
                    query_cutoffs=evaluation_samples.cutoffs,
                    candidate_horizon=candidate_horizon,
                )
            except ValueError as exc:
                print(f"Ignoring invalid retrieval cache: {exc}")
                cached = None
        if cached is None:
            print(
                f"Computing {config.retrieval_strategy} DTW retrieval: "
                f"queries={len(train_samples)}+{len(evaluation_samples)}, "
                f"knowledge_base={len(knowledge_base)}, "
                f"candidate_pool={config.candidate_pool_size}"
            )
            training_retrieval = retrieve_top_k(
                normalized_train_inputs,
                knowledge_base,
                top_k=config.top_k,
                query_series_indices=train_samples.series_indices,
                query_cutoffs=train_samples.cutoffs,
                exclude_indices=np.arange(len(train_samples), dtype=np.int64),
                strategy=config.retrieval_strategy,
                candidate_pool_size=config.candidate_pool_size,
                query_batch_size=config.retrieval_query_batch_size,
                workers=config.retrieval_workers,
                candidate_horizon=candidate_horizon,
                progress_callback=_retrieval_progress("Training"),
            )
            evaluation_retrieval = retrieve_top_k(
                normalized_eval_inputs,
                knowledge_base,
                top_k=config.top_k,
                query_series_indices=evaluation_samples.series_indices,
                query_cutoffs=evaluation_samples.cutoffs,
                strategy=config.retrieval_strategy,
                candidate_pool_size=config.candidate_pool_size,
                query_batch_size=config.retrieval_query_batch_size,
                workers=config.retrieval_workers,
                candidate_horizon=candidate_horizon,
                progress_callback=_retrieval_progress("Evaluation"),
            )
            validate_retrieval_result(
                training_retrieval,
                knowledge_base,
                query_series_indices=train_samples.series_indices,
                query_cutoffs=train_samples.cutoffs,
                exclude_indices=np.arange(len(train_samples), dtype=np.int64),
                candidate_horizon=candidate_horizon,
            )
            validate_retrieval_result(
                evaluation_retrieval,
                knowledge_base,
                query_series_indices=evaluation_samples.series_indices,
                query_cutoffs=evaluation_samples.cutoffs,
                candidate_horizon=candidate_horizon,
            )
            save_retrieval_cache(
                retrieval_cache_path,
                cache_key=cache_key,
                training=training_retrieval,
                evaluation=evaluation_retrieval,
            )
            print(f"Saved DTW retrieval cache to {retrieval_cache_path}")
        else:
            training_retrieval, evaluation_retrieval = cached
            cache_was_loaded = True
            print(f"Loaded DTW retrieval cache from {retrieval_cache_path}")
        retrieval_seconds = perf_counter() - retrieval_started

        if config.build_cache_only:
            build_elapsed_seconds = retrieval_seconds
            if cache_was_loaded and cache_summary_path.is_file():
                try:
                    with cache_summary_path.open("r", encoding="utf-8") as handle:
                        previous_summary = json.load(handle)
                    previous_retrieval = previous_summary.get("retrieval", {})
                    build_elapsed_seconds = float(
                        previous_retrieval.get(
                            "build_elapsed_seconds",
                            previous_retrieval.get(
                                "elapsed_seconds",
                                retrieval_seconds,
                            ),
                        )
                    )
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    build_elapsed_seconds = retrieval_seconds
            retrieval_figure = _retrieval_figure(
                normalized_eval_inputs[0],
                knowledge_base.inputs[evaluation_retrieval.indices[0]],
                evaluation_retrieval.distances[0],
                series_id=data.ids[0],
            )
            retrieval_figure.savefig(
                retrieval_plot_path,
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(retrieval_figure)
            cache_results: dict[str, object] = {
                "dataset": f"M4-{data.frequency}",
                "model": model_label,
                "evaluation_split": config.evaluation_split,
                "status": "cache_only",
                "config": configuration,
                "data": {
                    "training_samples": len(train_samples),
                    "evaluation_samples": len(evaluation_samples),
                },
                "retrieval": {
                    "strategy": config.retrieval_strategy,
                    "candidate_pool_size": config.candidate_pool_size,
                    "knowledge_base_size": len(knowledge_base),
                    "top_k": config.top_k,
                    "candidate_horizon": candidate_horizon,
                    "cache_loaded": cache_was_loaded,
                    "elapsed_seconds": retrieval_seconds,
                    "build_elapsed_seconds": build_elapsed_seconds,
                    "same_series_policy": same_series_policy,
                },
                "artifacts": {
                    "retrieval_cache": str(retrieval_cache_path),
                    "retrieval_plot": str(retrieval_plot_path),
                    "cache_summary": str(cache_summary_path),
                },
            }
            with cache_summary_path.open("w", encoding="utf-8") as handle:
                json.dump(cache_results, handle, indent=2, sort_keys=True)
                handle.write("\n")
            print(f"Saved retrieval cache summary to {cache_summary_path}")
            return cache_results

        if config.model == "rag_lstm":
            train_model_inputs = assemble_rag_inputs(
                normalized_train_inputs,
                knowledge_base,
                training_retrieval.indices,
            )
            evaluation_model_inputs = assemble_rag_inputs(
                normalized_eval_inputs,
                knowledge_base,
                evaluation_retrieval.indices,
            )
            model = RAGLSTMForecaster(
                top_k=config.top_k,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                horizon=data.horizon,
                dropout=config.dropout,
            ).to(selected_device)
        else:
            outcome_bank = build_outcome_bank(
                train_samples,
                relative_scale_floor=config.relative_scale_floor,
            )
            train_future_priors = assemble_future_prior(
                outcome_bank,
                training_retrieval,
                temperature=config.retrieval_temperature,
            )
            evaluation_future_priors = assemble_future_prior(
                outcome_bank,
                evaluation_retrieval,
                temperature=config.retrieval_temperature,
            )
            train_model_inputs = normalized_train_inputs
            evaluation_model_inputs = normalized_eval_inputs
            model = GatedFutureLSTMForecaster(
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                horizon=data.horizon,
                dropout=config.dropout,
                initial_gate=config.initial_retrieval_gate,
            ).to(selected_device)
    else:
        train_model_inputs = normalized_train_inputs
        evaluation_model_inputs = normalized_eval_inputs
        model = LSTMForecaster(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            horizon=data.horizon,
            dropout=config.dropout,
        ).to(selected_device)

    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    if normalized_train_targets is None:
        raise RuntimeError("Training targets are required outside cache-only mode")
    train_loader = _make_training_loader(
        train_model_inputs,
        normalized_train_targets,
        batch_size=config.batch_size,
        seed=config.seed,
        future_priors=train_future_priors,
    )

    print(
        f"Training {model_label} on M4-{data.frequency}: "
        f"samples={len(train_samples)}, evaluation={config.evaluation_split}, "
        f"device={selected_device}"
    )
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    try:
        writer.add_text(
            "run/configuration",
            f"```json\n{json.dumps(configuration, indent=2, sort_keys=True)}\n```",
            0,
        )
        if (
            knowledge_base is not None
            and evaluation_retrieval is not None
            and len(evaluation_samples) > 0
        ):
            retrieval_figure = _retrieval_figure(
                normalized_eval_inputs[0],
                knowledge_base.inputs[evaluation_retrieval.indices[0]],
                evaluation_retrieval.distances[0],
                series_id=data.ids[0],
            )
            retrieval_figure.savefig(
                retrieval_plot_path,
                dpi=150,
                bbox_inches="tight",
            )
            writer.add_figure(
                f"retrieval/{data.ids[0]}",
                retrieval_figure,
                global_step=0,
                close=True,
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
        if isinstance(model, GatedFutureLSTMForecaster):
            learned_retrieval_gate = float(model.retrieval_gate().detach().cpu())
            writer.add_scalar(
                "retrieval/gate",
                learned_retrieval_gate,
                config.epochs,
            )

        normalized_predictions = predict(
            model,
            evaluation_model_inputs,
            batch_size=config.batch_size,
            device=selected_device,
            future_priors=evaluation_future_priors,
        )
        predictions = invert_standardization(
            normalized_predictions, locations, scales
        ).astype(np.float32)

        metric_values = {
            "smape": smape(evaluation_samples.targets, predictions),
            "mase": mase(
                evaluation_samples.targets,
                predictions,
                fit_series,
                seasonal_period=M4_SEASONAL_PERIODS[data.frequency],
            ),
        }
        retrieval_prior_metric_values: dict[str, float] | None = None
        if evaluation_future_priors is not None:
            retrieval_prior_predictions = invert_standardization(
                evaluation_future_priors,
                locations,
                scales,
            ).astype(np.float32)
            retrieval_prior_metric_values = {
                "smape": smape(
                    evaluation_samples.targets,
                    retrieval_prior_predictions,
                ),
                "mase": mase(
                    evaluation_samples.targets,
                    retrieval_prior_predictions,
                    fit_series,
                    seasonal_period=M4_SEASONAL_PERIODS[data.frequency],
                ),
            }
            writer.add_scalar(
                "baseline/retrieval_prior_smape",
                retrieval_prior_metric_values["smape"],
                config.epochs,
            )
            writer.add_scalar(
                "baseline/retrieval_prior_mase",
                retrieval_prior_metric_values["mase"],
                config.epochs,
            )
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
                fit_series,
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
                model_label=model_label,
                persistence_predictions=persistence_predictions[sample_index],
                retrieval_prior=(
                    None
                    if retrieval_prior_predictions is None
                    else retrieval_prior_predictions[sample_index]
                ),
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
    prediction_values = {
        "ids": np.asarray(data.ids),
        "targets": evaluation_samples.targets,
        "predictions": predictions,
        "persistence_predictions": persistence_predictions,
    }
    if evaluation_retrieval is not None:
        prediction_values["retrieved_indices"] = evaluation_retrieval.indices
        prediction_values["retrieval_distances"] = evaluation_retrieval.distances
    if retrieval_prior_predictions is not None:
        prediction_values["retrieval_prior_predictions"] = retrieval_prior_predictions
    np.savez_compressed(predictions_path, **prediction_values)

    artifact_values = {
        "checkpoint": str(checkpoint_path),
        "predictions": str(predictions_path),
        "prediction_plot": str(prediction_plot_path),
        "metrics": str(metrics_path),
        "tensorboard": str(tensorboard_dir),
    }
    retrieval_values: dict[str, object] | None = None
    if retrieval_cache_path is not None and knowledge_base is not None:
        artifact_values["retrieval_cache"] = str(retrieval_cache_path)
        artifact_values["retrieval_plot"] = str(retrieval_plot_path)
        retrieval_values = {
            "strategy": config.retrieval_strategy,
            "candidate_pool_size": config.candidate_pool_size,
            "knowledge_base_size": len(knowledge_base),
            "top_k": config.top_k,
            "candidate_horizon": candidate_horizon,
            "cache_loaded": cache_was_loaded,
            "elapsed_seconds": retrieval_seconds,
            "same_series_overlap_allowed": False,
            "same_series_policy": same_series_policy,
        }
        if learned_retrieval_gate is not None:
            retrieval_values["temperature"] = config.retrieval_temperature
            retrieval_values["learned_gate"] = learned_retrieval_gate

    baseline_values: dict[str, object] = {
        "persistence": persistence_metric_values,
    }
    if retrieval_prior_metric_values is not None:
        baseline_values["retrieval_prior"] = retrieval_prior_metric_values

    results: dict[str, object] = {
        "dataset": f"M4-{data.frequency}",
        "model": model_label,
        "device": selected_device.type,
        "evaluation_split": config.evaluation_split,
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
        "baselines": baseline_values,
        "artifacts": artifact_values,
    }
    if retrieval_values is not None:
        results["retrieval"] = retrieval_values
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
    if retrieval_prior_metric_values is not None:
        print(
            "Retrieval prior: "
            f"SMAPE={retrieval_prior_metric_values['smape']:.4f}, "
            f"MASE={retrieval_prior_metric_values['mase']:.4f}"
        )
    if learned_retrieval_gate is not None:
        print(f"Learned retrieval gate: {learned_retrieval_gate:.6f}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved prediction plot to {prediction_plot_path}")
    if retrieval_cache_path is not None:
        print(f"Saved retrieval plot to {retrieval_plot_path}")
    print(f"Saved TensorBoard logs to {tensorboard_dir}")
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate LSTM retrieval ablations on M4 data."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/m4"))
    parser.add_argument("--freq", dest="frequency", default="Weekly")
    parser.add_argument(
        "--model",
        choices=("lstm", "rag_lstm", "gated_rag_lstm"),
        default="lstm",
    )
    parser.add_argument(
        "--evaluation-split",
        choices=("official_test", "train_tail"),
        default="official_test",
    )
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
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--retrieval-strategy",
        choices=("exact", "euclidean_prefilter"),
        default="euclidean_prefilter",
    )
    parser.add_argument("--candidate-pool-size", type=int, default=512)
    parser.add_argument("--retrieval-query-batch-size", type=int, default=64)
    parser.add_argument("--retrieval-workers", type=int, default=1)
    parser.add_argument("--retrieval-temperature", type=float, default=0.25)
    parser.add_argument("--initial-retrieval-gate", type=float, default=0.1)
    parser.add_argument("--build-cache-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--retrieval-cache-dir", type=Path, default=Path("cache"))
    return parser


def parse_args(argv: Sequence[str] | None = None) -> ExperimentConfig:
    namespace = build_argument_parser().parse_args(argv)
    return ExperimentConfig(**vars(namespace))


def main(argv: Sequence[str] | None = None) -> None:
    run_experiment(parse_args(argv))


if __name__ == "__main__":
    main()
