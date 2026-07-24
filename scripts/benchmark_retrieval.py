"""Compare bounded DTW retrieval with exact DTW on a Weekly subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.data import (
    build_evaluation_windows,
    build_training_windows,
    load_m4_frequency,
    standardize_by_input,
)
from src.retrieval import (
    RetrievalResult,
    build_knowledge_base,
    retrieve_top_k,
    validate_retrieval_result,
)


def _finite_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _quality_metrics(
    exact: RetrievalResult,
    approximate: RetrievalResult,
) -> dict[str, float | None]:
    recalls = [
        len(set(expected).intersection(actual)) / len(expected)
        for expected, actual in zip(exact.indices, approximate.indices)
    ]
    exact_top = exact.distances[:, 0].astype(np.float64)
    approximate_top = approximate.distances[:, 0].astype(np.float64)
    inflation = np.divide(
        approximate_top,
        exact_top,
        out=np.ones_like(approximate_top),
        where=exact_top > 1e-12,
    )
    inflation[(exact_top <= 1e-12) & (approximate_top > 1e-12)] = np.inf
    return {
        "top_k_recall": float(np.mean(recalls)),
        "top_1_agreement": float(
            np.mean(exact.indices[:, 0] == approximate.indices[:, 0])
        ),
        "top_1_distance_inflation_median": _finite_float(np.median(inflation)),
        "top_1_distance_inflation_p95": _finite_float(
            np.quantile(inflation, 0.95)
        ),
    }


def _timed_retrieval(*args, **kwargs) -> tuple[RetrievalResult, float]:
    started = perf_counter()
    result = retrieve_top_k(*args, **kwargs)
    return result, perf_counter() - started


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    data = load_m4_frequency(args.data_dir, args.frequency)
    training_samples = build_training_windows(
        data.train,
        args.input_length,
        data.horizon,
        max_samples=args.knowledge_base_size,
        seed=args.seed,
    )
    evaluation_samples = build_evaluation_windows(data, args.input_length)
    normalized_training, _, _, _ = standardize_by_input(training_samples.inputs)
    normalized_evaluation, _, _, _ = standardize_by_input(evaluation_samples.inputs)
    knowledge_base = build_knowledge_base(training_samples)

    rng = np.random.default_rng(args.seed)
    train_query_count = min(args.train_queries, len(training_samples))
    train_query_rows = np.sort(
        rng.choice(len(training_samples), size=train_query_count, replace=False)
    )
    evaluation_query_count = min(args.evaluation_queries, len(evaluation_samples))
    evaluation_query_rows = np.arange(evaluation_query_count)

    train_queries = normalized_training[train_query_rows]
    train_series = training_samples.series_indices[train_query_rows]
    train_cutoffs = training_samples.cutoffs[train_query_rows]
    evaluation_queries = normalized_evaluation[evaluation_query_rows]
    evaluation_series = evaluation_samples.series_indices[evaluation_query_rows]
    evaluation_cutoffs = evaluation_samples.cutoffs[evaluation_query_rows]

    exact_train, exact_train_seconds = _timed_retrieval(
        train_queries,
        knowledge_base,
        top_k=args.top_k,
        query_series_indices=train_series,
        query_cutoffs=train_cutoffs,
        exclude_indices=train_query_rows,
        strategy="exact",
    )
    exact_evaluation, exact_evaluation_seconds = _timed_retrieval(
        evaluation_queries,
        knowledge_base,
        top_k=args.top_k,
        query_series_indices=evaluation_series,
        query_cutoffs=evaluation_cutoffs,
        strategy="exact",
    )
    validate_retrieval_result(
        exact_train,
        knowledge_base,
        query_series_indices=train_series,
        query_cutoffs=train_cutoffs,
        exclude_indices=train_query_rows,
    )
    validate_retrieval_result(
        exact_evaluation,
        knowledge_base,
        query_series_indices=evaluation_series,
        query_cutoffs=evaluation_cutoffs,
    )

    candidate_results: dict[str, object] = {}
    for candidate_pool_size in args.candidate_pool_sizes:
        approximate_train, train_seconds = _timed_retrieval(
            train_queries,
            knowledge_base,
            top_k=args.top_k,
            query_series_indices=train_series,
            query_cutoffs=train_cutoffs,
            exclude_indices=train_query_rows,
            strategy="euclidean_prefilter",
            candidate_pool_size=candidate_pool_size,
            query_batch_size=args.query_batch_size,
            workers=args.workers,
        )
        approximate_evaluation, evaluation_seconds = _timed_retrieval(
            evaluation_queries,
            knowledge_base,
            top_k=args.top_k,
            query_series_indices=evaluation_series,
            query_cutoffs=evaluation_cutoffs,
            strategy="euclidean_prefilter",
            candidate_pool_size=candidate_pool_size,
            query_batch_size=args.query_batch_size,
            workers=args.workers,
        )
        validate_retrieval_result(
            approximate_train,
            knowledge_base,
            query_series_indices=train_series,
            query_cutoffs=train_cutoffs,
            exclude_indices=train_query_rows,
        )
        validate_retrieval_result(
            approximate_evaluation,
            knowledge_base,
            query_series_indices=evaluation_series,
            query_cutoffs=evaluation_cutoffs,
        )
        candidate_results[str(candidate_pool_size)] = {
            "train": {
                "seconds": train_seconds,
                **_quality_metrics(exact_train, approximate_train),
            },
            "evaluation": {
                "seconds": evaluation_seconds,
                **_quality_metrics(exact_evaluation, approximate_evaluation),
            },
        }

    results: dict[str, object] = {
        "dataset": f"M4-{data.frequency}",
        "knowledge_base_size": len(knowledge_base),
        "train_queries": train_query_count,
        "evaluation_queries": evaluation_query_count,
        "top_k": args.top_k,
        "exact_seconds": {
            "train": exact_train_seconds,
            "evaluation": exact_evaluation_seconds,
        },
        "candidate_pools": candidate_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/m4"))
    parser.add_argument("--frequency", default="Weekly")
    parser.add_argument("--input-length", type=int, default=26)
    parser.add_argument("--knowledge-base-size", type=int, default=10_000)
    parser.add_argument("--train-queries", type=int, default=200)
    parser.add_argument("--evaluation-queries", type=int, default=359)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--candidate-pool-sizes",
        type=lambda value: [int(item) for item in value.split(",")],
        default=[256, 512, 1024],
    )
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/retrieval_benchmark_weekly.json"),
    )
    return parser


if __name__ == "__main__":
    result = run_benchmark(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
