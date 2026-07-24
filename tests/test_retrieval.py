import tempfile
import unittest
from pathlib import Path

import numpy as np

from src import retrieval as retrieval_module
from src.data import WindowedSamples
from src.retrieval import (
    RetrievalKnowledgeBase,
    RetrievalResult,
    assemble_future_prior,
    assemble_rag_inputs,
    build_knowledge_base,
    build_outcome_bank,
    dtw_distance,
    load_retrieval_cache,
    retrieval_cache_key,
    retrieve_top_k,
    save_retrieval_cache,
    validate_retrieval_result,
)


class RetrievalTests(unittest.TestCase):
    def test_dtw_distance_is_zero_for_identical_sequences_and_symmetric(self):
        first = np.asarray([0.0, 1.0, 2.0, 1.0], dtype=np.float32)
        second = np.asarray([0.0, 0.5, 2.0, 1.0], dtype=np.float32)

        self.assertEqual(dtw_distance(first, first), 0.0)
        self.assertAlmostEqual(dtw_distance(first, second), dtw_distance(second, first))

    def test_batched_dtw_matches_scalar_dtw(self):
        queries = np.asarray([[0, 1, 2], [2, 1, 0]], dtype=np.float32)
        candidates = np.asarray(
            [
                [[0, 1, 2], [0, 2, 2]],
                [[2, 1, 0], [2, 0, 0]],
            ],
            dtype=np.float32,
        )

        batched = retrieval_module._batched_dtw_distances(queries, candidates)
        expected = np.asarray(
            [
                [dtw_distance(query, candidate) for candidate in row]
                for query, row in zip(queries, candidates)
            ]
        )

        np.testing.assert_allclose(batched, expected)

    def test_knowledge_base_contains_inputs_but_not_training_targets(self):
        samples = WindowedSamples(
            inputs=np.asarray([[0, 1, 2], [10, 11, 12]], dtype=np.float32),
            targets=np.asarray([[999, 999], [888, 888]], dtype=np.float32),
            series_indices=np.asarray([0, 1], dtype=np.int32),
            cutoffs=np.asarray([3, 3], dtype=np.int32),
        )

        knowledge_base = build_knowledge_base(samples)

        np.testing.assert_array_equal(knowledge_base.raw_inputs, samples.inputs)
        self.assertFalse(np.any(knowledge_base.raw_inputs == 999))
        self.assertFalse(np.any(knowledge_base.raw_inputs == 888))

    def test_outcome_bank_uses_each_candidate_input_statistics(self):
        samples = WindowedSamples(
            inputs=np.asarray([[0, 2], [10, 14]], dtype=np.float32),
            targets=np.asarray([[3, 5], [16, 20]], dtype=np.float32),
            series_indices=np.asarray([0, 1], dtype=np.int32),
            cutoffs=np.asarray([2, 2], dtype=np.int32),
        )

        outcomes = build_outcome_bank(samples, relative_scale_floor=0.0)

        np.testing.assert_allclose(outcomes, [[2, 4], [2, 4]])
        self.assertEqual(outcomes.dtype, np.float32)

    def test_retrieval_excludes_same_series_overlapping_windows(self):
        inputs = np.asarray(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 2.5],
                [0.0, 1.0, 2.0, 3.0],
            ],
            dtype=np.float32,
        )
        knowledge_base = RetrievalKnowledgeBase(
            inputs=inputs,
            raw_inputs=inputs.copy(),
            series_indices=np.asarray([0, 0, 1, 0], dtype=np.int32),
            cutoffs=np.asarray([4, 6, 6, 12], dtype=np.int32),
        )

        for strategy in ("exact", "euclidean_prefilter"):
            with self.subTest(strategy=strategy):
                result = retrieve_top_k(
                    inputs[:1],
                    knowledge_base,
                    top_k=2,
                    query_series_indices=np.asarray([0]),
                    query_cutoffs=np.asarray([8]),
                    strategy=strategy,
                    candidate_pool_size=len(knowledge_base),
                    query_batch_size=1,
                )

                np.testing.assert_array_equal(result.indices, [[0, 2]])
                self.assertNotIn(1, result.indices[0])
                self.assertNotIn(3, result.indices[0])

    def test_candidate_horizon_requires_complete_episode_before_query(self):
        inputs = np.asarray(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 2.5],
            ],
            dtype=np.float32,
        )
        knowledge_base = RetrievalKnowledgeBase(
            inputs=inputs,
            raw_inputs=inputs.copy(),
            series_indices=np.asarray([0, 0, 1], dtype=np.int32),
            cutoffs=np.asarray([4, 5, 10], dtype=np.int32),
        )

        for strategy in ("exact", "euclidean_prefilter"):
            with self.subTest(strategy=strategy):
                result = retrieve_top_k(
                    inputs[:1],
                    knowledge_base,
                    top_k=2,
                    query_series_indices=np.asarray([0]),
                    query_cutoffs=np.asarray([10]),
                    strategy=strategy,
                    candidate_pool_size=len(knowledge_base),
                    query_batch_size=1,
                    candidate_horizon=2,
                )
                np.testing.assert_array_equal(result.indices, [[0, 2]])

        validate_retrieval_result(
            RetrievalResult(
                np.asarray([[0]], dtype=np.int64),
                np.asarray([[0.0]], dtype=np.float32),
            ),
            knowledge_base,
            query_series_indices=np.asarray([0]),
            query_cutoffs=np.asarray([10]),
            candidate_horizon=2,
        )
        with self.assertRaisesRegex(ValueError, "causal-prefix"):
            validate_retrieval_result(
                RetrievalResult(
                    np.asarray([[1]], dtype=np.int64),
                    np.asarray([[0.0]], dtype=np.float32),
                ),
                knowledge_base,
                query_series_indices=np.asarray([0]),
                query_cutoffs=np.asarray([10]),
                candidate_horizon=2,
            )

    def test_prefilter_expands_until_it_finds_safe_candidates(self):
        invalid_count = 40
        inputs = np.vstack(
            (
                np.zeros((invalid_count, 4), dtype=np.float32),
                np.full((1, 4), 10.0, dtype=np.float32),
                np.full((1, 4), 11.0, dtype=np.float32),
                np.full((1, 4), 12.0, dtype=np.float32),
            )
        )
        knowledge_base = RetrievalKnowledgeBase(
            inputs=inputs,
            raw_inputs=inputs.copy(),
            series_indices=np.asarray(
                [0] * invalid_count + [1, 2, 3], dtype=np.int32
            ),
            cutoffs=np.asarray([8] * invalid_count + [8, 8, 8], dtype=np.int32),
        )

        result = retrieve_top_k(
            np.zeros((1, 4), dtype=np.float32),
            knowledge_base,
            top_k=2,
            query_series_indices=np.asarray([0]),
            query_cutoffs=np.asarray([8]),
            strategy="euclidean_prefilter",
            candidate_pool_size=2,
            query_batch_size=1,
        )

        np.testing.assert_array_equal(result.indices, [[40, 41]])
        self.assertTrue(np.all(np.isfinite(result.distances)))

    def test_prefilter_matches_exact_when_pool_covers_knowledge_base(self):
        inputs = np.asarray(
            [
                [0.0, 1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0, 0.0],
                [0.0, 0.5, 2.0, 3.0],
                [2.0, 1.0, 1.0, 2.0],
                [0.0, 1.0, 1.5, 3.0],
            ],
            dtype=np.float32,
        )
        knowledge_base = RetrievalKnowledgeBase(
            inputs=inputs,
            raw_inputs=inputs.copy(),
            series_indices=np.asarray([0, 0, 1, 2, 3], dtype=np.int32),
            cutoffs=np.asarray([4, 6, 6, 6, 6], dtype=np.int32),
        )
        kwargs = {
            "top_k": 3,
            "query_series_indices": np.asarray([0]),
            "query_cutoffs": np.asarray([8]),
        }

        exact = retrieve_top_k(inputs[2:3], knowledge_base, strategy="exact", **kwargs)
        prefiltered = retrieve_top_k(
            inputs[2:3],
            knowledge_base,
            strategy="euclidean_prefilter",
            candidate_pool_size=len(knowledge_base),
            query_batch_size=1,
            **kwargs,
        )

        np.testing.assert_array_equal(prefiltered.indices, exact.indices)
        np.testing.assert_allclose(prefiltered.distances, exact.distances)

    def test_rag_inputs_place_query_before_retrieved_channels(self):
        inputs = np.asarray(
            [[0, 1, 2], [10, 11, 12], [20, 21, 22]], dtype=np.float32
        )
        knowledge_base = RetrievalKnowledgeBase(
            inputs=inputs,
            raw_inputs=inputs.copy(),
            series_indices=np.asarray([0, 1, 2], dtype=np.int32),
            cutoffs=np.asarray([3, 3, 3], dtype=np.int32),
        )

        rag_inputs = assemble_rag_inputs(
            np.asarray([[100, 101, 102]], dtype=np.float32),
            knowledge_base,
            np.asarray([[2, 0]], dtype=np.int64),
        )

        self.assertEqual(rag_inputs.shape, (1, 3, 3))
        np.testing.assert_array_equal(rag_inputs[0, :, 0], [100, 101, 102])
        np.testing.assert_array_equal(rag_inputs[0, :, 1], inputs[2])
        np.testing.assert_array_equal(rag_inputs[0, :, 2], inputs[0])

    def test_future_prior_uses_distance_weights_and_is_permutation_invariant(self):
        outcomes = np.asarray([[10, 0], [0, 10], [5, 5]], dtype=np.float32)
        retrieval = RetrievalResult(
            np.asarray([[0, 1]], dtype=np.int64),
            np.asarray([[0.0, 1.0]], dtype=np.float32),
        )
        permuted = RetrievalResult(
            np.asarray([[1, 0]], dtype=np.int64),
            np.asarray([[1.0, 0.0]], dtype=np.float32),
        )
        expected_weight = np.exp(-1.0)
        expected = np.asarray(
            [[10.0 / (1.0 + expected_weight), 10.0 * expected_weight / (1.0 + expected_weight)]],
            dtype=np.float32,
        )

        prior = assemble_future_prior(outcomes, retrieval, temperature=1.0)
        permuted_prior = assemble_future_prior(outcomes, permuted, temperature=1.0)

        np.testing.assert_allclose(prior, expected, rtol=1e-6)
        np.testing.assert_allclose(permuted_prior, prior, rtol=1e-6)

    def test_future_prior_handles_zero_distances_and_rejects_invalid_inputs(self):
        outcomes = np.asarray([[10, 0], [0, 10]], dtype=np.float32)
        zero_distance = RetrievalResult(
            np.asarray([[0, 1]], dtype=np.int64),
            np.asarray([[0.0, 0.0]], dtype=np.float32),
        )

        prior = assemble_future_prior(outcomes, zero_distance, temperature=0.5)

        np.testing.assert_allclose(prior, [[5, 5]])
        self.assertTrue(np.all(np.isfinite(prior)))
        with self.assertRaisesRegex(ValueError, "temperature"):
            assemble_future_prior(outcomes, zero_distance, temperature=0.0)
        with self.assertRaisesRegex(ValueError, "outside"):
            assemble_future_prior(
                outcomes,
                RetrievalResult(
                    np.asarray([[2]], dtype=np.int64),
                    np.asarray([[0.0]], dtype=np.float32),
                ),
                temperature=1.0,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            assemble_future_prior(
                np.asarray([[np.nan, 0], [0, 1]], dtype=np.float32),
                zero_distance,
                temperature=1.0,
            )

    def test_cache_round_trip_and_stale_key_rejection(self):
        inputs = np.asarray([[0, 1], [1, 0]], dtype=np.float32)
        knowledge_base = RetrievalKnowledgeBase(
            inputs=inputs,
            raw_inputs=inputs.copy(),
            series_indices=np.asarray([0, 1], dtype=np.int32),
            cutoffs=np.asarray([2, 2], dtype=np.int32),
        )
        key = retrieval_cache_key(
            knowledge_base,
            train_queries=inputs,
            train_series_indices=knowledge_base.series_indices,
            train_cutoffs=knowledge_base.cutoffs,
            evaluation_queries=inputs[:1],
            evaluation_series_indices=np.asarray([0], dtype=np.int32),
            evaluation_cutoffs=np.asarray([4], dtype=np.int32),
            top_k=1,
        )
        training = RetrievalResult(
            np.asarray([[1], [0]], dtype=np.int64),
            np.asarray([[1.0], [1.0]], dtype=np.float32),
        )
        evaluation = RetrievalResult(
            np.asarray([[1]], dtype=np.int64),
            np.asarray([[1.0]], dtype=np.float32),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "retrieval.npz"
            save_retrieval_cache(
                path,
                cache_key=key,
                training=training,
                evaluation=evaluation,
            )

            loaded = load_retrieval_cache(path, expected_key=key)
            self.assertIsNotNone(loaded)
            loaded_training, loaded_evaluation = loaded
            np.testing.assert_array_equal(loaded_training.indices, training.indices)
            np.testing.assert_array_equal(loaded_evaluation.indices, evaluation.indices)
            self.assertIsNone(load_retrieval_cache(path, expected_key="stale"))

    def test_cache_key_changes_with_retrieval_strategy_pool_and_horizon(self):
        inputs = np.asarray([[0, 1], [1, 0]], dtype=np.float32)
        knowledge_base = RetrievalKnowledgeBase(
            inputs=inputs,
            raw_inputs=inputs.copy(),
            series_indices=np.asarray([0, 1], dtype=np.int32),
            cutoffs=np.asarray([2, 2], dtype=np.int32),
        )
        common = {
            "train_queries": inputs,
            "train_series_indices": knowledge_base.series_indices,
            "train_cutoffs": knowledge_base.cutoffs,
            "evaluation_queries": inputs[:1],
            "evaluation_series_indices": np.asarray([0], dtype=np.int32),
            "evaluation_cutoffs": np.asarray([4], dtype=np.int32),
            "top_k": 1,
        }

        exact = retrieval_cache_key(knowledge_base, strategy="exact", **common)
        pool_8 = retrieval_cache_key(
            knowledge_base,
            strategy="euclidean_prefilter",
            candidate_pool_size=8,
            **common,
        )
        pool_16 = retrieval_cache_key(
            knowledge_base,
            strategy="euclidean_prefilter",
            candidate_pool_size=16,
            **common,
        )
        horizon_2 = retrieval_cache_key(
            knowledge_base,
            strategy="exact",
            candidate_horizon=2,
            **common,
        )

        self.assertEqual(len({exact, pool_8, pool_16, horizon_2}), 4)

    def test_corrupt_and_out_of_range_caches_are_rejected(self):
        result = RetrievalResult(
            np.asarray([[3]], dtype=np.int64),
            np.asarray([[1.0]], dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "retrieval.npz"
            path.write_bytes(b"incomplete")
            self.assertIsNone(load_retrieval_cache(path, expected_key="key"))

            save_retrieval_cache(
                path,
                cache_key="key",
                training=result,
                evaluation=result,
            )
            self.assertIsNone(
                load_retrieval_cache(
                    path,
                    expected_key="key",
                    expected_training_shape=(1, 1),
                    expected_evaluation_shape=(1, 1),
                    knowledge_base_size=2,
                )
            )

            non_numeric = RetrievalResult(
                np.asarray([[0]], dtype=np.int64),
                np.asarray([["not-a-distance"]]),
            )
            save_retrieval_cache(
                path,
                cache_key="key",
                training=non_numeric,
                evaluation=non_numeric,
            )
            self.assertIsNone(
                load_retrieval_cache(
                    path,
                    expected_key="key",
                    expected_training_shape=(1, 1),
                    expected_evaluation_shape=(1, 1),
                    knowledge_base_size=2,
                )
            )

    def test_validation_rejects_duplicate_or_unsorted_results(self):
        inputs = np.asarray([[0, 1], [1, 0], [2, 0]], dtype=np.float32)
        knowledge_base = RetrievalKnowledgeBase(
            inputs=inputs,
            raw_inputs=inputs.copy(),
            series_indices=np.asarray([0, 1, 2], dtype=np.int32),
            cutoffs=np.asarray([2, 2, 2], dtype=np.int32),
        )
        metadata = {
            "query_series_indices": np.asarray([3]),
            "query_cutoffs": np.asarray([4]),
        }

        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_retrieval_result(
                RetrievalResult(
                    np.asarray([[0, 0]]), np.asarray([[0.0, 1.0]])
                ),
                knowledge_base,
                **metadata,
            )
        with self.assertRaisesRegex(ValueError, "sorted"):
            validate_retrieval_result(
                RetrievalResult(
                    np.asarray([[0, 1]]), np.asarray([[2.0, 1.0]])
                ),
                knowledge_base,
                **metadata,
            )


if __name__ == "__main__":
    unittest.main()
