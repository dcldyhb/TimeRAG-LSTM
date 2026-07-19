"""加载数据用的模块"""

from __future__ import annotations #延迟解析类型标注，避免报错，并且提高打开速度

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


M4_FREQUENCIES = {"Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"}


# 数据类，含有频率、id、训练集和测试集等属性，带有检查训练集和测试集的内部的长度是否一致的成员函数 horizon
@dataclass(frozen=True) # 自动生成构造函数，frozen=True表示字段不会改变
class M4FrequencyData:

    frequency: str
    ids: tuple[str, ...] # 使用元组，使得数据加载完成后的序列集合、数量和顺序不被随意修改
    train: tuple[np.ndarray, ...]
    test: tuple[np.ndarray, ...]

    @property # 成员函数伪装成变量使用，比如可以直接调用 data.horizon 而不是 data.horizon()
    def horizon(self) -> int:
        horizons = {len(values) for values in self.test} # 测试集中的变量数组长度的集合，会删除重复值
        if len(horizons) != 1:
            raise ValueError(f"Test series have inconsistent horizons: {sorted(horizons)}") # 判断变量的长度是否一致
        return next(iter(horizons)) # 最终返回唯一的长度值


# 保存切好的训练集和测试集
@dataclass(frozen=True)
class WindowedSamples:
    """Materialized forecasting samples and their source locations."""

    inputs: np.ndarray # 输入窗口
    targets: np.ndarray # 正确答案
    series_indices: np.ndarray # 来自训练集中的第几条时间序列，可用于追溯
    cutoffs: np.ndarray # 输入窗口和预测目标的分界位置

    def __len__(self) -> int:
        return len(self.inputs)


# 规范化频率名称
def _normalise_frequency(frequency: str) -> str:
    value = frequency.strip().capitalize() # 去除字符串两侧空格，首字母大写
    if value not in M4_FREQUENCIES: # 检查是否为属于的六种频率
        choices = ", ".join(sorted(M4_FREQUENCIES))
        raise ValueError(f"Unknown M4 frequency {frequency!r}. Expected one of: {choices}")
    return value


# 阅读文件得到 ids 和 series 的数据
def _read_m4_csv(path: Path) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    if not path.is_file():
        raise FileNotFoundError(f"M4 file not found: {path}")

    ids: list[str] = []
    series: list[np.ndarray] = []

    # 打开文件，如果第一行都没有说明是空的，报错。
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration as exc:
            raise ValueError(f"M4 file is empty: {path}") from exc

        # 跳过第一行，如果某一行为空，即 not row = True，则跳过该行阅读下一行
        for row_number, row in enumerate(reader, start=2):
            if not row:
                continue

            # 每一行的第一个单元格是序列 ID，如果无 ID 则报错
            series_id = row[0].strip()
            if not series_id:
                raise ValueError(f"Missing series id in {path} at row {row_number}")

            # 从时间序列的 1 号元素开始，逐级加入到 raw_values 列表中，并且删除末尾的空元素
            raw_values = [value.strip() for value in row[1:]]
            while raw_values and raw_values[-1] == "":
                raw_values.pop()

            # 不允许空的时间序列，不允许时间序列中间有空值
            if not raw_values:
                raise ValueError(f"Series {series_id!r} has no observations in {path}")
            if any(value == "" for value in raw_values):
                raise ValueError(
                    f"Series {series_id!r} has an interior missing value in {path} "
                    f"at row {row_number}"
                )

            # CSV reader 得到的列表中的元素是字符串不是数字，转换为 NumPy 的浮点数数组
            try:
                values = np.asarray(raw_values, dtype=np.float32)
            except ValueError as exc:
                raise ValueError(
                    f"Series {series_id!r} contains a non-numeric value in {path} "
                    f"at row {row_number}"
                ) from exc

            ids.append(series_id)
            series.append(values)

    # 检查是否无 ID 以及 ID
    if not ids:
        raise ValueError(f"M4 file contains no series: {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate series ids found in {path}")

    return tuple(ids), tuple(series)


# 加载、对齐训练集和测试集，这里的 str | Path 表示 data_dir 可以使字符串或者 Path 对象，默认频率为 Weekly
def load_m4_frequency(data_dir: str | Path, frequency: str = "Weekly") -> M4FrequencyData:
    frequency = _normalise_frequency(frequency) # 48 行的函数，去除字符串两侧的空格和首字母大写用的，使得函数传入的参数可以不分大小写
    data_dir = Path(data_dir)

    train_ids, train_values = _read_m4_csv(data_dir / f"{frequency}-train.csv") # 这里直接输入路径拼接
    test_ids, test_values = _read_m4_csv(data_dir / f"{frequency}-test.csv")

    if set(train_ids) != set(test_ids):
        missing_test = sorted(set(train_ids) - set(test_ids))
        missing_train = sorted(set(test_ids) - set(train_ids))
        raise ValueError(
            "Train/test series ids do not match. "
            f"Missing from test: {missing_test[:5]}; missing from train: {missing_train[:5]}"
        )

    test_by_id = dict(zip(test_ids, test_values)) #建立以 ID 为索引的字典
    aligned_test = tuple(test_by_id[series_id] for series_id in train_ids) # 根据 ID 顺序整理数据，生成元组

    data = M4FrequencyData(
        frequency=frequency,
        ids=train_ids,
        train=train_values,
        test=aligned_test,
    )
    _ = data.horizon
    return data


# 构造训练窗口
def build_training_windows(
    train_series: Sequence[np.ndarray], # 所有的训练时间序列
    input_length: int, # 输入长度
    horizon: int, # 预测长度
    *,
    stride: int = 1, # 窗口每次移动步数
    max_samples: int | None = None, # 最多保留多少样本
    seed: int = 42, # 随机抽样种子
) -> WindowedSamples:
    if input_length <= 0 or horizon <= 0:
        raise ValueError("input_length and horizon must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")

    # 计算可以生成多少个滑动窗口
    counts = np.asarray(
        [max(0, 1 + (len(values) - input_length - horizon) // stride) for values in train_series],
        dtype=np.int64,
    )
    total_windows = int(counts.sum())
    if total_windows == 0:
        raise ValueError(
            "No valid training windows. Check series lengths, input_length, and horizon."
        )

    # 设置最大采样量，如果没有给出则设置为样本总数，如果有设置则为最大采样量和样本总数的较小值
    sample_count = total_windows if max_samples is None else min(max_samples, total_windows)

    # 设置所有滑动窗口的索引数组 flat_indices
    if sample_count == total_windows: # 如果全部样本都被使用则将所有样本的索引放入 flat_indices 中，从 0 到 len(total_window)
        flat_indices = np.arange(total_windows, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed) # 使用现代随机数生成器（PCG64）创建一个随机数生成器对象
        flat_indices = np.sort(
            rng.choice(total_windows, size=sample_count, replace=False).astype(np.int64)
        ) # 从 0 到 total_windows-1 无放回地随机抽取 sample_count 个整数，返回一个数组，其顺序随机

    # 将任意一个滑动窗口还原到在原来时间序列中的位置
    cumulative_counts = np.cumsum(counts) # 得到原有时间序列的结束位置的累积和，比如 counts = [3,4,2] 得到 cumulative_counts = [3,7,9]
    series_indices = np.searchsorted(cumulative_counts, flat_indices, side="right") # 对每个 flat_indices 在 cumulative_counts 中二分查找，返回第一个累计值大于 flat_index 的位置索引，np.searchsorted 是将被搜索序列插入到被搜索的序列中，得到某一条的位置，最终返回所有时间窗口在时间序列中的位置
    previous_counts = np.concatenate((np.asarray([0], dtype=np.int64), cumulative_counts[:-1]))
    # cumulative_counts[:-1] 去掉最后一个元素 -> ，然后在前面拼接 0,能够将 cumulative_counts 从表示序列的结束位置到起始位置
    local_window_indices = flat_indices - previous_counts[series_indices] # 得到位于序列内部的第几个窗口，从 0 开始计数
    # 接下来这两个是计数的起点和重点
    starts = local_window_indices * stride #具体时间点上的起始步骤
    cutoffs = starts + input_length #结束步骤

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


# 截取官方测试集的最后 input_length 个时间步
def build_evaluation_windows(data: M4FrequencyData, input_length: int) -> WindowedSamples:
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


# 将标准化后的输出值还原回去
def invert_standardization(
    normalized_values: np.ndarray,
    locations: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Map standardized forecasts back to each series' original scale."""
    return np.asarray(normalized_values) * np.asarray(scales) + np.asarray(locations)
