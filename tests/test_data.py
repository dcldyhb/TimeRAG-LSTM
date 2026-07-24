import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data import (
    M4FrequencyData,
    build_evaluation_windows,
    build_temporal_validation_split,
    build_training_windows,
    invert_standardization,
    load_m4_frequency,
    standardize_by_input,
)


class M4DataTests(unittest.TestCase):
    def _write_csv(self, path: Path, rows):
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows(rows)

    def test_load_aligns_ids_and_trims_trailing_cells(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_csv(
                root / "Weekly-train.csv",
                [["V1", "V2", "V3", "V4", "V5"], ["W1", 1, 2, 3, ""], ["W2", 4, 5, 6, 7]],
            )
            self._write_csv(
                root / "Weekly-test.csv",
                [["V1", "V2", "V3"], ["W2", 10, 11], ["W1", 8, 9]],
            )

            data = load_m4_frequency(root, "weekly")
            self.assertEqual(data.ids, ("W1", "W2"))
            self.assertEqual(data.horizon, 2)
            np.testing.assert_array_equal(data.train[0], [1, 2, 3])
            np.testing.assert_array_equal(data.test[0], [8, 9])

    def test_training_windows_have_expected_cutoffs(self):
        samples = build_training_windows(
            [np.arange(8, dtype=np.float32)], input_length=3, horizon=2
        )
        np.testing.assert_array_equal(samples.inputs[0], [0, 1, 2])
        np.testing.assert_array_equal(samples.targets[0], [3, 4])
        np.testing.assert_array_equal(samples.inputs[-1], [3, 4, 5])
        np.testing.assert_array_equal(samples.targets[-1], [6, 7])
        np.testing.assert_array_equal(samples.cutoffs, [3, 4, 5, 6])

    def test_debug_sampling_is_reproducible_and_spans_series(self):
        series = [np.arange(20, dtype=np.float32) + offset for offset in (0, 100, 200)]
        first = build_training_windows(series, 4, 2, max_samples=8, seed=7)
        second = build_training_windows(series, 4, 2, max_samples=8, seed=7)
        np.testing.assert_array_equal(first.inputs, second.inputs)
        self.assertGreater(len(np.unique(first.series_indices)), 1)

    def test_evaluation_uses_train_tail_and_official_test(self):
        data = M4FrequencyData(
            frequency="Weekly",
            ids=("W1", "W2"),
            train=(np.arange(6, dtype=np.float32), np.arange(10, 16, dtype=np.float32)),
            test=(np.asarray([6, 7], dtype=np.float32), np.asarray([16, 17], dtype=np.float32)),
        )
        samples = build_evaluation_windows(data, input_length=3)
        np.testing.assert_array_equal(samples.inputs, [[3, 4, 5], [13, 14, 15]])
        np.testing.assert_array_equal(samples.targets, [[6, 7], [16, 17]])

    def test_temporal_validation_split_holds_out_each_training_tail(self):
        split = build_temporal_validation_split(
            (
                np.arange(10, dtype=np.float32),
                np.arange(100, 112, dtype=np.float32),
            ),
            input_length=3,
            horizon=2,
        )

        np.testing.assert_array_equal(split.fit_series[0], np.arange(8))
        np.testing.assert_array_equal(split.fit_series[1], np.arange(100, 110))
        np.testing.assert_array_equal(
            split.validation_samples.inputs,
            [[5, 6, 7], [107, 108, 109]],
        )
        np.testing.assert_array_equal(
            split.validation_samples.targets,
            [[8, 9], [110, 111]],
        )
        np.testing.assert_array_equal(
            split.validation_samples.series_indices,
            [0, 1],
        )
        np.testing.assert_array_equal(split.validation_samples.cutoffs, [8, 10])

    def test_temporal_validation_split_never_uses_official_test_targets(self):
        data = M4FrequencyData(
            frequency="Weekly",
            ids=("W1",),
            train=(np.arange(8, dtype=np.float32),),
            test=(np.asarray([999, 1000], dtype=np.float32),),
        )

        split = build_temporal_validation_split(
            data.train,
            input_length=3,
            horizon=2,
        )

        np.testing.assert_array_equal(split.validation_samples.inputs, [[3, 4, 5]])
        np.testing.assert_array_equal(split.validation_samples.targets, [[6, 7]])
        self.assertFalse(
            np.any(np.isin(split.validation_samples.targets, data.test[0]))
        )

    def test_temporal_validation_split_rejects_short_training_series(self):
        with self.assertRaisesRegex(
            ValueError, "shorter than input_length \+ horizon=5"
        ):
            build_temporal_validation_split(
                (np.arange(4, dtype=np.float32),),
                input_length=3,
                horizon=2,
            )

    def test_standardization_round_trip(self):
        inputs = np.asarray([[1, 2, 3], [10, 10, 10]], dtype=np.float32)
        targets = np.asarray([[4, 5], [11, 12]], dtype=np.float32)
        _, normalized_targets, locations, scales = standardize_by_input(inputs, targets)
        restored = invert_standardization(normalized_targets, locations, scales)
        np.testing.assert_allclose(restored, targets, rtol=1e-5, atol=1e-5)

    def test_standardization_uses_input_relative_floor_for_near_constant_window(self):
        inputs = np.asarray([[1199.99] + [1200.0] * 25], dtype=np.float32)
        first_targets = np.asarray([[1200.0, 1300.0, 1500.0]], dtype=np.float32)
        second_targets = np.asarray([[900.0, 1000.0, 1100.0]], dtype=np.float32)

        _, normalized_targets, locations, scales = standardize_by_input(
            inputs, first_targets
        )
        _, _, second_locations, second_scales = standardize_by_input(
            inputs, second_targets
        )

        expected_floor = 1e-3 * np.mean(np.abs(inputs), axis=1, keepdims=True)
        np.testing.assert_allclose(scales, expected_floor, rtol=1e-5)
        np.testing.assert_array_equal(locations, second_locations)
        np.testing.assert_array_equal(scales, second_scales)
        self.assertLess(float(np.max(np.abs(normalized_targets))), 300.0)

        restored = invert_standardization(normalized_targets, locations, scales)
        np.testing.assert_allclose(restored, first_targets, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
