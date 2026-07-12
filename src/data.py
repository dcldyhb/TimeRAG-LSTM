"""加载数据用的模块"""

from __future__ import annotations #延迟解析类型标注，避免报错，并且提高打开速度

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


M4_FREQUENCIES = {"Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"}


@dataclass(frozen=True) # 自动生成构造函数，frozen=True表示字段不会改变
class M4FrequencyData:

    frequency: str
    ids: tuple[str, ...] # 使用元组，使得数据加载完成后的序列集合、数量和顺序不被随意修改
    train: tuple[np.ndarray, ...]
    test: tuple[np.ndarray, ...]

    @property
    def horizon(self) -> int:
        horizons = {len(values) for values in self.test}
        if len(horizons) != 1:
            raise ValueError(f"Test series have inconsistent horizons: {sorted(horizons)}")
        return next(iter(horizons))


@dataclass(frozen=True)
class WindowedSamples:
    """Materialized forecasting samples and their source locations."""

    inputs: np.ndarray
    targets: np.ndarray
    series_indices: np.ndarray
    cutoffs: np.ndarray

    def __len__(self) -> int:
        return len(self.inputs)


def _normalise_frequency(frequency: str) -> str:
    value = frequency.strip().capitalize()
    if value not in M4_FREQUENCIES:
        choices = ", ".join(sorted(M4_FREQUENCIES))
        raise ValueError(f"Unknown M4 frequency {frequency!r}. Expected one of: {choices}")
    return value


def _read_m4_csv(path: Path) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    """Read an M4 wide CSV and remove only trailing empty cells."""
    if not path.is_file():
        raise FileNotFoundError(f"M4 file not found: {path}")

    ids: list[str] = []
    series: list[np.ndarray] = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration as exc:
            raise ValueError(f"M4 file is empty: {path}") from exc

        for row_number, row in enumerate(reader, start=2):
            if not row:
                continue

            series_id = row[0].strip()
            if not series_id:
                raise ValueError(f"Missing series id in {path} at row {row_number}")

            raw_values = [value.strip() for value in row[1:]]
            while raw_values and raw_values[-1] == "":
                raw_values.pop()

            if not raw_values:
                raise ValueError(f"Series {series_id!r} has no observations in {path}")
            if any(value == "" for value in raw_values):
                raise ValueError(
                    f"Series {series_id!r} has an interior missing value in {path} "
                    f"at row {row_number}"
                )

            try:
                values = np.asarray(raw_values, dtype=np.float32)
            except ValueError as exc:
                raise ValueError(
                    f"Series {series_id!r} contains a non-numeric value in {path} "
                    f"at row {row_number}"
                ) from exc

            ids.append(series_id)
            series.append(values)

    if not ids:
        raise ValueError(f"M4 file contains no series: {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate series ids found in {path}")

    return tuple(ids), tuple(series)


def load_m4_frequency(data_dir: str | Path, frequency: str = "Weekly") -> M4FrequencyData:
    """Load and align the official train/test files for one M4 frequency."""
    frequency = _normalise_frequency(frequency)
    data_dir = Path(data_dir)

    train_ids, train_values = _read_m4_csv(data_dir / f"{frequency}-train.csv")
    test_ids, test_values = _read_m4_csv(data_dir / f"{frequency}-test.csv")

    if set(train_ids) != set(test_ids):
        missing_test = sorted(set(train_ids) - set(test_ids))
        missing_train = sorted(set(test_ids) - set(train_ids))
        raise ValueError(
            "Train/test series ids do not match. "
            f"Missing from test: {missing_test[:5]}; missing from train: {missing_train[:5]}"
        )

    test_by_id = dict(zip(test_ids, test_values))
    aligned_test = tuple(test_by_id[series_id] for series_id in train_ids)

    data = M4FrequencyData(
        frequency=frequency,
        ids=train_ids,
        train=train_values,
        test=aligned_test,
    )
    _ = data.horizon
    return data


def build_training_windows(
    train_series: Sequence[np.ndarray],
    input_length: int,
    horizon: int,
    *,
    stride: int = 1,
    max_samples: int | None = None,
    seed: int = 42,
) -> WindowedSamples:
    """Build windows from training data only, with optional uniform debug sampling."""
    if input_length <= 0 or horizon <= 0:
        raise ValueError("input_length and horizon must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")

    counts = np.asarray(
        [max(0, 1 + (len(values) - input_length - horizon) // stride) for values in train_series],
        dtype=np.int64,
    )
    total_windows = int(counts.sum())
    if total_windows == 0:
        raise ValueError(
            "No valid training windows. Check series lengths, input_length, and horizon."
        )

    sample_count = total_windows if max_samples is None else min(max_samples, total_windows)
    if sample_count == total_windows:
        flat_indices = np.arange(total_windows, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        flat_indices = np.sort(
            rng.choice(total_windows, size=sample_count, replace=False).astype(np.int64)
        )

    cumulative_counts = np.cumsum(counts)
    series_indices = np.searchsorted(cumulative_counts, flat_indices, side="right")
    previous_counts = np.concatenate((np.asarray([0], dtype=np.int64), cumulative_counts[:-1]))
    local_window_indices = flat_indices - previous_counts[series_indices]
    starts = local_window_indices * stride
    cutoffs = starts + input_length

    inputs = np.empty((sample_count, input_length), dtype=np.float32)
    targets = np.empty((sample_count, horizon), dtype=np.float32)

    for output_index, (series_index, start) in enumerate(zip(series_indices, starts)):
        values = np.asarray(train_series[int(series_index)], dtype=np.float32)
        start = int(start)
        cutoff = start + input_length
        inputs[output_index] = values[start:cutoff]
        targets[output_index] = values[cutoff : cutoff + horizon]

    return WindowedSamples(
        inputs=inputs,
        targets=targets,
        series_indices=series_indices.astype(np.int32),
        cutoffs=cutoffs.astype(np.int32),
    )


def build_evaluation_windows(data: M4FrequencyData, input_length: int) -> WindowedSamples:
    """Use each training tail as input and the official test split as target."""
    if input_length <= 0:
        raise ValueError("input_length must be positive")

    too_short = [data.ids[i] for i, values in enumerate(data.train) if len(values) < input_length]
    if too_short:
        raise ValueError(
            f"{len(too_short)} training series are shorter than input_length={input_length}; "
            f"examples: {too_short[:5]}"
        )

    inputs = np.stack([values[-input_length:] for values in data.train]).astype(np.float32)
    targets = np.stack(data.test).astype(np.float32)
    series_indices = np.arange(len(data.ids), dtype=np.int32)
    cutoffs = np.asarray([len(values) for values in data.train], dtype=np.int32)
    return WindowedSamples(inputs, targets, series_indices, cutoffs)


def standardize_by_input(
    inputs: np.ndarray,
    targets: np.ndarray | None = None,
    *,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Standardize each sample using statistics from its input window only."""
    inputs = np.asarray(inputs, dtype=np.float32)
    if inputs.ndim != 2:
        raise ValueError(f"inputs must have shape [samples, time], got {inputs.shape}")

    locations = inputs.mean(axis=1, keepdims=True)
    scales = np.maximum(inputs.std(axis=1, keepdims=True), eps)
    normalized_inputs = (inputs - locations) / scales

    normalized_targets = None
    if targets is not None:
        targets = np.asarray(targets, dtype=np.float32)
        if targets.ndim != 2 or targets.shape[0] != inputs.shape[0]:
            raise ValueError(
                "targets must have shape [samples, horizon] and match the input sample count"
            )
        normalized_targets = (targets - locations) / scales

    return normalized_inputs, normalized_targets, locations, scales


def invert_standardization(
    normalized_values: np.ndarray,
    locations: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Map standardized forecasts back to each series' original scale."""
    return np.asarray(normalized_values) * np.asarray(scales) + np.asarray(locations)
