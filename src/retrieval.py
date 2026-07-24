"""Training-only DTW retrieval utilities for TimeRAG-LSTM."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable
from zipfile import BadZipFile

import numpy as np
import scipy
from scipy.spatial import cKDTree

from src.data import WindowedSamples, standardize_by_input


CACHE_SCHEMA_VERSION = "timerag-lstm-dtw-cache-v3"
CAUSAL_POLICY_VERSION = "same-series-complete-episode-before-query-v2"
DTW_VERSION = "unrestricted-squared-cost-v1"
RETRIEVAL_STRATEGIES = {"exact", "euclidean_prefilter"}


def _validated_candidate_horizon(candidate_horizon: int) -> int:
    if isinstance(candidate_horizon, (bool, np.bool_)) or not isinstance(
        candidate_horizon, (int, np.integer)
    ):
        raise ValueError("candidate_horizon must be a non-negative integer")
    if candidate_horizon < 0:
        raise ValueError("candidate_horizon must be non-negative")
    return int(candidate_horizon)


@dataclass(frozen=True)
class RetrievalKnowledgeBase:
    """Normalized training windows and their original source locations."""

    inputs: np.ndarray
    raw_inputs: np.ndarray
    series_indices: np.ndarray
    cutoffs: np.ndarray

    def __post_init__(self) -> None:
        if self.inputs.ndim != 2 or self.raw_inputs.shape != self.inputs.shape:
            raise ValueError("knowledge-base inputs must have shape [items, time]")
        if not np.all(np.isfinite(self.inputs)) or not np.all(
            np.isfinite(self.raw_inputs)
        ):
            raise ValueError("knowledge-base inputs must contain only finite values")
        item_count = self.inputs.shape[0]
        if self.series_indices.shape != (item_count,) or self.cutoffs.shape != (
            item_count,
        ):
            raise ValueError("knowledge-base metadata must have shape [items]")

    def __len__(self) -> int:
        return len(self.inputs)


@dataclass(frozen=True)
class RetrievalResult:
    """Top-k knowledge-base rows and their DTW distances per query."""

    indices: np.ndarray
    distances: np.ndarray

    def __post_init__(self) -> None:
        if self.indices.ndim != 2 or self.distances.shape != self.indices.shape:
            raise ValueError("retrieval results must have shape [queries, top_k]")


def build_knowledge_base(
    training_samples: WindowedSamples,
    *,
    relative_scale_floor: float = 1e-3,
) -> RetrievalKnowledgeBase:
    """Build a retrieval database exclusively from materialized train windows."""
    normalized_inputs, _, _, _ = standardize_by_input(
        training_samples.inputs,
        relative_scale_floor=relative_scale_floor,
    )
    return RetrievalKnowledgeBase(
        inputs=np.ascontiguousarray(normalized_inputs, dtype=np.float32),
        raw_inputs=np.ascontiguousarray(training_samples.inputs, dtype=np.float32),
        series_indices=np.asarray(training_samples.series_indices, dtype=np.int32),
        cutoffs=np.asarray(training_samples.cutoffs, dtype=np.int32),
    )


def build_outcome_bank(
    training_samples: WindowedSamples,
    *,
    relative_scale_floor: float = 1e-3,
) -> np.ndarray:
    """Normalize each training future using only its candidate input statistics."""
    _, normalized_targets, _, _ = standardize_by_input(
        training_samples.inputs,
        training_samples.targets,
        relative_scale_floor=relative_scale_floor,
    )
    if normalized_targets is None:
        raise RuntimeError("Training targets were not standardized")
    outcome_bank = np.ascontiguousarray(normalized_targets, dtype=np.float32)
    if not np.all(np.isfinite(outcome_bank)):
        raise ValueError("outcome-bank targets must contain only finite values")
    return outcome_bank


def dtw_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return the exact DTW distance using squared pointwise cost."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.ndim != 1 or second.ndim != 1 or not len(first) or not len(second):
        raise ValueError("DTW inputs must be non-empty one-dimensional arrays")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("DTW inputs must contain only finite values")

    previous = np.full(len(second) + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    for first_value in first:
        current = np.full(len(second) + 1, np.inf, dtype=np.float64)
        for second_index, second_value in enumerate(second, start=1):
            cost = (first_value - second_value) ** 2
            current[second_index] = cost + min(
                previous[second_index],
                current[second_index - 1],
                previous[second_index - 1],
            )
        previous = current
    return float(np.sqrt(previous[-1]))


def _dtw_distances(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Vectorize exact DTW over candidates while iterating through time."""
    query = np.asarray(query, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    candidate_count, candidate_length = candidates.shape
    previous = np.full((candidate_count, candidate_length + 1), np.inf)
    previous[:, 0] = 0.0

    for query_value in query:
        current = np.full_like(previous, np.inf)
        for candidate_step in range(candidate_length):
            costs = (query_value - candidates[:, candidate_step]) ** 2
            current[:, candidate_step + 1] = costs + np.minimum.reduce(
                (
                    previous[:, candidate_step + 1],
                    current[:, candidate_step],
                    previous[:, candidate_step],
                )
            )
        previous = current
    return np.sqrt(previous[:, -1])


def _batched_dtw_distances(
    queries: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """Compute DTW for aligned ``[queries, candidates, time]`` batches."""
    queries = np.asarray(queries, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    if queries.ndim != 2 or candidates.ndim != 3:
        raise ValueError("batched DTW expects [queries, time] and [queries, candidates, time]")
    if queries.shape[0] != candidates.shape[0] or queries.shape[1] != candidates.shape[2]:
        raise ValueError("batched DTW queries and candidates must align")

    query_count, input_length = queries.shape
    candidate_count = candidates.shape[1]
    previous = np.full(
        (query_count, candidate_count, input_length + 1),
        np.inf,
        dtype=np.float64,
    )
    previous[:, :, 0] = 0.0

    for query_step in range(input_length):
        current = np.full_like(previous, np.inf)
        for candidate_step in range(input_length):
            costs = (
                queries[:, None, query_step] - candidates[:, :, candidate_step]
            ) ** 2
            current[:, :, candidate_step + 1] = costs + np.minimum.reduce(
                (
                    previous[:, :, candidate_step + 1],
                    current[:, :, candidate_step],
                    previous[:, :, candidate_step],
                )
            )
        previous = current
    return np.sqrt(previous[:, :, -1])


def _valid_candidate_mask(
    candidate_indices: np.ndarray,
    knowledge_base: RetrievalKnowledgeBase,
    *,
    query_series_index: int,
    query_cutoff: int,
    input_length: int,
    exclude_index: int,
    candidate_horizon: int = 0,
) -> np.ndarray:
    candidate_horizon = _validated_candidate_horizon(candidate_horizon)
    candidate_series = knowledge_base.series_indices[candidate_indices]
    candidate_ends = (
        knowledge_base.cutoffs[candidate_indices].astype(np.int64)
        + candidate_horizon
    )
    query_start = query_cutoff - input_length
    same_series_not_in_prefix = (candidate_series == query_series_index) & (
        candidate_ends > query_start
    )
    valid = ~same_series_not_in_prefix
    if exclude_index >= 0:
        valid &= candidate_indices != exclude_index
    return valid


def _tree_neighbors(
    tree: cKDTree,
    query: np.ndarray,
    *,
    count: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    distances, indices = tree.query(query, k=count, workers=workers)
    distances = np.atleast_1d(np.asarray(distances, dtype=np.float64))
    indices = np.atleast_1d(np.asarray(indices, dtype=np.int64))
    order = np.lexsort((indices, distances))
    return distances[order], indices[order]


def _prefilter_candidates(
    queries: np.ndarray,
    knowledge_base: RetrievalKnowledgeBase,
    *,
    query_series_indices: np.ndarray,
    query_cutoffs: np.ndarray,
    excluded: np.ndarray,
    candidate_pool_size: int,
    query_batch_size: int,
    workers: int,
    candidate_horizon: int,
    progress_callback: Callable[[int, int], None] | None,
) -> RetrievalResult:
    query_count, input_length = queries.shape
    knowledge_size = len(knowledge_base)
    pool_size = min(candidate_pool_size, knowledge_size)
    initial_search_size = min(
        knowledge_size,
        max(pool_size + 32, pool_size * 2),
    )
    tree = cKDTree(knowledge_base.inputs)
    retrieved_indices = np.empty((query_count, pool_size), dtype=np.int64)
    retrieved_distances = np.empty((query_count, pool_size), dtype=np.float32)

    for batch_start in range(0, query_count, query_batch_size):
        batch_end = min(batch_start + query_batch_size, query_count)
        batch_queries = queries[batch_start:batch_end]
        initial_distances, initial_indices = tree.query(
            batch_queries,
            k=initial_search_size,
            workers=workers,
        )
        initial_distances = np.asarray(initial_distances, dtype=np.float64)
        initial_indices = np.asarray(initial_indices, dtype=np.int64)
        if initial_indices.ndim == 1:
            initial_distances = initial_distances[:, None]
            initial_indices = initial_indices[:, None]
        initial_order = np.lexsort((initial_indices, initial_distances), axis=1)
        initial_indices = np.take_along_axis(initial_indices, initial_order, axis=1)

        batch_size = batch_end - batch_start
        candidate_indices = np.empty((batch_size, pool_size), dtype=np.int64)
        candidate_valid = np.zeros((batch_size, pool_size), dtype=bool)
        for local_index in range(batch_size):
            query_index = batch_start + local_index
            row_indices = initial_indices[local_index]
            valid = _valid_candidate_mask(
                row_indices,
                knowledge_base,
                query_series_index=int(query_series_indices[query_index]),
                query_cutoff=int(query_cutoffs[query_index]),
                input_length=input_length,
                exclude_index=int(excluded[query_index]),
                candidate_horizon=candidate_horizon,
            )
            safe_indices = row_indices[valid]
            search_size = initial_search_size
            while len(safe_indices) < pool_size and search_size < knowledge_size:
                search_size = min(knowledge_size, search_size * 2)
                _, expanded_indices = _tree_neighbors(
                    tree,
                    queries[query_index],
                    count=search_size,
                    workers=workers,
                )
                expanded_valid = _valid_candidate_mask(
                    expanded_indices,
                    knowledge_base,
                    query_series_index=int(query_series_indices[query_index]),
                    query_cutoff=int(query_cutoffs[query_index]),
                    input_length=input_length,
                    exclude_index=int(excluded[query_index]),
                    candidate_horizon=candidate_horizon,
                )
                safe_indices = expanded_indices[expanded_valid]

            if not len(safe_indices):
                raise ValueError(f"Query {query_index} has no leakage-safe candidates")
            take_count = min(pool_size, len(safe_indices))
            candidate_indices[local_index, :take_count] = safe_indices[:take_count]
            candidate_valid[local_index, :take_count] = True
            if take_count < pool_size:
                candidate_indices[local_index, take_count:] = safe_indices[0]

        distances = _batched_dtw_distances(
            batch_queries,
            knowledge_base.inputs[candidate_indices],
        )
        distances[~candidate_valid] = np.inf
        order = np.lexsort((candidate_indices, distances), axis=1)
        sorted_indices = np.take_along_axis(candidate_indices, order, axis=1)
        sorted_distances = np.take_along_axis(distances, order, axis=1)
        retrieved_indices[batch_start:batch_end] = sorted_indices
        retrieved_distances[batch_start:batch_end] = sorted_distances
        if progress_callback is not None:
            progress_callback(batch_end, query_count)

    return RetrievalResult(retrieved_indices, retrieved_distances)


def retrieve_top_k(
    normalized_queries: np.ndarray,
    knowledge_base: RetrievalKnowledgeBase,
    *,
    top_k: int,
    query_series_indices: np.ndarray,
    query_cutoffs: np.ndarray,
    exclude_indices: np.ndarray | None = None,
    strategy: str = "exact",
    candidate_pool_size: int = 64,
    query_batch_size: int = 64,
    workers: int = 1,
    candidate_horizon: int = 0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> RetrievalResult:
    """Retrieve DTW neighbors under the same-series causal-prefix policy."""
    queries = np.asarray(normalized_queries, dtype=np.float32)
    if queries.ndim != 2 or queries.shape[1] != knowledge_base.inputs.shape[1]:
        raise ValueError("queries and knowledge-base items must share [items, time]")
    if not np.all(np.isfinite(queries)):
        raise ValueError("queries must contain only finite values")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if strategy not in RETRIEVAL_STRATEGIES:
        raise ValueError(f"Unknown retrieval strategy: {strategy}")
    if strategy == "euclidean_prefilter" and candidate_pool_size < top_k:
        raise ValueError("candidate_pool_size must be at least top_k")
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")
    if workers == 0 or workers < -1:
        raise ValueError("workers must be -1 or a positive integer")
    candidate_horizon = _validated_candidate_horizon(candidate_horizon)

    query_count, input_length = queries.shape
    series_indices = np.asarray(query_series_indices, dtype=np.int32)
    cutoffs = np.asarray(query_cutoffs, dtype=np.int32)
    if series_indices.shape != (query_count,) or cutoffs.shape != (query_count,):
        raise ValueError("query metadata must have shape [queries]")
    if exclude_indices is None:
        excluded = np.full(query_count, -1, dtype=np.int64)
    else:
        excluded = np.asarray(exclude_indices, dtype=np.int64)
        if excluded.shape != (query_count,):
            raise ValueError("exclude_indices must have shape [queries]")

    if strategy == "euclidean_prefilter":
        candidates = _prefilter_candidates(
            queries,
            knowledge_base,
            query_series_indices=series_indices,
            query_cutoffs=cutoffs,
            excluded=excluded,
            candidate_pool_size=candidate_pool_size,
            query_batch_size=query_batch_size,
            workers=workers,
            candidate_horizon=candidate_horizon,
            progress_callback=progress_callback,
        )
        if candidates.indices.shape[1] < top_k:
            raise ValueError(
                f"Only {candidates.indices.shape[1]} candidates available for top_k={top_k}"
            )
        selected_indices = candidates.indices[:, :top_k].copy()
        selected_distances = candidates.distances[:, :top_k].copy()
        if not np.all(np.isfinite(selected_distances)):
            raise ValueError("At least one query has fewer leakage-safe candidates than top_k")
        return RetrievalResult(selected_indices, selected_distances)

    retrieved_indices = np.empty((query_count, top_k), dtype=np.int64)
    retrieved_distances = np.empty((query_count, top_k), dtype=np.float32)
    all_indices = np.arange(len(knowledge_base), dtype=np.int64)

    for query_index, query in enumerate(queries):
        valid_mask = _valid_candidate_mask(
            all_indices,
            knowledge_base,
            query_series_index=int(series_indices[query_index]),
            query_cutoff=int(cutoffs[query_index]),
            input_length=input_length,
            exclude_index=int(excluded[query_index]),
            candidate_horizon=candidate_horizon,
        )
        candidate_indices = all_indices[valid_mask]
        if len(candidate_indices) < top_k:
            raise ValueError(
                f"Query {query_index} has only {len(candidate_indices)} leakage-safe "
                f"candidates, fewer than top_k={top_k}"
            )

        distances = _dtw_distances(query, knowledge_base.inputs[candidate_indices])
        order = np.lexsort((candidate_indices, distances))[:top_k]
        retrieved_indices[query_index] = candidate_indices[order]
        retrieved_distances[query_index] = distances[order]
        if progress_callback is not None and (
            query_index + 1 == query_count or (query_index + 1) % query_batch_size == 0
        ):
            progress_callback(query_index + 1, query_count)

    return RetrievalResult(retrieved_indices, retrieved_distances)


def assemble_rag_inputs(
    normalized_queries: np.ndarray,
    knowledge_base: RetrievalKnowledgeBase,
    retrieved_indices: np.ndarray,
) -> np.ndarray:
    """Stack query and retrieved windows as feature channels."""
    queries = np.asarray(normalized_queries, dtype=np.float32)
    indices = np.asarray(retrieved_indices)
    if queries.ndim != 2 or indices.ndim != 2 or indices.shape[0] != len(queries):
        raise ValueError("queries and retrieved indices must align by sample")
    if queries.shape[1] != knowledge_base.inputs.shape[1]:
        raise ValueError("queries and knowledge-base inputs must have equal length")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("retrieved indices must be integers")
    if np.any(indices < 0) or np.any(indices >= len(knowledge_base)):
        raise ValueError("retrieved index is outside the knowledge base")

    retrieved = knowledge_base.inputs[indices].transpose(0, 2, 1)
    return np.concatenate((queries[:, :, None], retrieved), axis=2).astype(
        np.float32,
        copy=False,
    )


def assemble_future_prior(
    outcome_bank: np.ndarray,
    retrieval: RetrievalResult,
    temperature: float,
) -> np.ndarray:
    """Aggregate retrieved training futures with stable DTW-distance weights."""
    try:
        temperature_value = float(temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature must be a positive finite number") from exc
    if not np.isfinite(temperature_value) or temperature_value <= 0:
        raise ValueError("temperature must be a positive finite number")

    outcomes = np.asarray(outcome_bank)
    indices = np.asarray(retrieval.indices)
    distances = np.asarray(retrieval.distances)
    if outcomes.ndim != 2 or outcomes.shape[1] == 0:
        raise ValueError("outcome_bank must have shape [candidates, horizon]")
    if indices.ndim != 2 or distances.shape != indices.shape or indices.shape[1] == 0:
        raise ValueError("retrieval results must have shape [queries, top_k]")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("retrieval indices must be integers")
    if np.any(indices < 0) or np.any(indices >= len(outcomes)):
        raise ValueError("retrieved index is outside the outcome bank")
    outcome_is_real = np.issubdtype(outcomes.dtype, np.integer) or np.issubdtype(
        outcomes.dtype, np.floating
    )
    if not outcome_is_real or not np.all(np.isfinite(outcomes)):
        raise ValueError("outcome_bank must contain only finite numeric values")
    distance_is_real = np.issubdtype(distances.dtype, np.integer) or np.issubdtype(
        distances.dtype, np.floating
    )
    if not distance_is_real or not np.all(np.isfinite(distances)):
        raise ValueError("retrieval distances must contain only finite numeric values")
    if np.any(distances < 0):
        raise ValueError("retrieval distances must be non-negative")

    distance_values = distances.astype(np.float64, copy=False)
    shifted_distances = distance_values - distance_values.min(axis=1, keepdims=True)
    with np.errstate(over="ignore"):
        weights = np.exp(-shifted_distances / temperature_value)
    weights /= weights.sum(axis=1, keepdims=True)

    retrieved_outcomes = np.asarray(outcomes[indices], dtype=np.float64)
    prior = np.einsum("qk,qkh->qh", weights, retrieved_outcomes, optimize=True)
    prior = np.asarray(prior, dtype=np.float32)
    if not np.all(np.isfinite(prior)):
        raise ValueError("future prior contains a non-finite value")
    return prior


def retrieval_cache_key(
    knowledge_base: RetrievalKnowledgeBase,
    *,
    train_queries: np.ndarray,
    train_series_indices: np.ndarray,
    train_cutoffs: np.ndarray,
    evaluation_queries: np.ndarray,
    evaluation_series_indices: np.ndarray,
    evaluation_cutoffs: np.ndarray,
    top_k: int,
    strategy: str = "exact",
    candidate_pool_size: int | None = None,
    candidate_horizon: int = 0,
) -> str:
    """Fingerprint all data and settings that determine retrieval results."""
    if strategy not in RETRIEVAL_STRATEGIES:
        raise ValueError(f"Unknown retrieval strategy: {strategy}")
    candidate_horizon = _validated_candidate_horizon(candidate_horizon)
    digest = sha256(CACHE_SCHEMA_VERSION.encode("ascii"))
    settings = (
        DTW_VERSION,
        CAUSAL_POLICY_VERSION,
        strategy,
        str(top_k),
        str(candidate_horizon),
        str(candidate_pool_size if strategy == "euclidean_prefilter" else None),
        scipy.__version__ if strategy == "euclidean_prefilter" else "not-used",
    )
    digest.update("|".join(settings).encode("ascii"))
    arrays = (
        knowledge_base.inputs,
        knowledge_base.series_indices,
        knowledge_base.cutoffs,
        train_queries,
        train_series_indices,
        train_cutoffs,
        evaluation_queries,
        evaluation_series_indices,
        evaluation_cutoffs,
    )
    for values in arrays:
        array = np.ascontiguousarray(values)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_retrieval_result(
    result: RetrievalResult,
    knowledge_base: RetrievalKnowledgeBase,
    *,
    query_series_indices: np.ndarray,
    query_cutoffs: np.ndarray,
    exclude_indices: np.ndarray | None = None,
    candidate_horizon: int = 0,
) -> None:
    """Validate cache structure and the causal-prefix retrieval contract."""
    candidate_horizon = _validated_candidate_horizon(candidate_horizon)
    indices = np.asarray(result.indices)
    distances = np.asarray(result.distances)
    query_count, top_k = indices.shape
    series_indices = np.asarray(query_series_indices, dtype=np.int32)
    cutoffs = np.asarray(query_cutoffs, dtype=np.int32)
    if series_indices.shape != (query_count,) or cutoffs.shape != (query_count,):
        raise ValueError("retrieval result and query metadata do not align")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("retrieval indices must be integers")
    if not np.issubdtype(distances.dtype, np.floating):
        raise ValueError("retrieval distances must be floating-point values")
    if np.any(indices < 0) or np.any(indices >= len(knowledge_base)):
        raise ValueError("retrieval cache contains an out-of-range index")
    if top_k > 1 and np.any(np.diff(np.sort(indices, axis=1), axis=1) == 0):
        raise ValueError("retrieval cache contains duplicate neighbors")
    if not np.all(np.isfinite(distances)) or np.any(distances < 0):
        raise ValueError("retrieval distances must be finite and non-negative")
    if top_k > 1 and np.any(np.diff(distances, axis=1) < -1e-6):
        raise ValueError("retrieval distances must be sorted")

    input_length = knowledge_base.inputs.shape[1]
    candidate_series = knowledge_base.series_indices[indices]
    candidate_ends = (
        knowledge_base.cutoffs[indices].astype(np.int64) + candidate_horizon
    )
    causal_violations = (candidate_series == series_indices[:, None]) & (
        candidate_ends > cutoffs[:, None].astype(np.int64) - input_length
    )
    if np.any(causal_violations):
        raise ValueError("retrieval cache violates the same-series causal-prefix policy")
    if exclude_indices is not None:
        excluded = np.asarray(exclude_indices, dtype=np.int64)
        if excluded.shape != (query_count,):
            raise ValueError("exclude_indices must have shape [queries]")
        if np.any(indices == excluded[:, None]):
            raise ValueError("retrieval cache contains an excluded query row")


def save_retrieval_cache(
    path: str | Path,
    *,
    cache_key: str,
    training: RetrievalResult,
    evaluation: RetrievalResult,
) -> None:
    """Persist training/evaluation retrieval results in one compressed file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(
                handle,
                cache_key=np.asarray(cache_key),
                train_indices=training.indices,
                train_distances=training.distances,
                evaluation_indices=evaluation.indices,
                evaluation_distances=evaluation.distances,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_retrieval_cache(
    path: str | Path,
    *,
    expected_key: str,
    expected_training_shape: tuple[int, int] | None = None,
    expected_evaluation_shape: tuple[int, int] | None = None,
    knowledge_base_size: int | None = None,
) -> tuple[RetrievalResult, RetrievalResult] | None:
    """Load a matching cache, returning ``None`` for a missing or stale file."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as saved:
            required = {
                "cache_key",
                "train_indices",
                "train_distances",
                "evaluation_indices",
                "evaluation_distances",
            }
            if not required.issubset(saved.files):
                return None
            if str(saved["cache_key"].item()) != expected_key:
                return None
            training = RetrievalResult(
                saved["train_indices"].copy(), saved["train_distances"].copy()
            )
            evaluation = RetrievalResult(
                saved["evaluation_indices"].copy(),
                saved["evaluation_distances"].copy(),
            )
    except (BadZipFile, EOFError, OSError, ValueError):
        return None

    if expected_training_shape is not None and training.indices.shape != expected_training_shape:
        return None
    if (
        expected_evaluation_shape is not None
        and evaluation.indices.shape != expected_evaluation_shape
    ):
        return None
    if knowledge_base_size is not None:
        for result in (training, evaluation):
            if not np.issubdtype(result.indices.dtype, np.integer):
                return None
            if not np.issubdtype(result.distances.dtype, np.floating):
                return None
            if np.any(result.indices < 0) or np.any(result.indices >= knowledge_base_size):
                return None
            if not np.all(np.isfinite(result.distances)) or np.any(result.distances < 0):
                return None
    return training, evaluation
