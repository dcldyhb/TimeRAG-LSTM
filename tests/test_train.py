import csv
from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None
else:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from train import ExperimentConfig, run_experiment


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TrainingPipelineTests(unittest.TestCase):
    def _write_csv(self, path: Path, rows) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows(rows)

    def test_retrieval_temperature_must_be_positive_and_finite(self):
        for temperature in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(
                    ValueError,
                    "retrieval_temperature must be positive and finite",
                ):
                    ExperimentConfig(
                        model="gated_rag_lstm",
                        retrieval_temperature=temperature,
                    ).validate()

    def test_weekly_smoke_run_saves_metrics_predictions_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            output_dir = root / "results"
            checkpoint_dir = root / "checkpoints"
            log_dir = root / "runs"
            data_dir.mkdir()
            self._write_csv(
                data_dir / "Weekly-train.csv",
                [
                    ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9"],
                    ["W1", 1, 2, 3, 4, 5, 6, 7, 8],
                    ["W2", 2, 4, 6, 8, 10, 12, 14, 16],
                ],
            )
            self._write_csv(
                data_dir / "Weekly-test.csv",
                [["V1", "V2", "V3"], ["W1", 9, 10], ["W2", 18, 20]],
            )

            results = run_experiment(
                ExperimentConfig(
                    data_dir=data_dir,
                    input_length=3,
                    max_samples=4,
                    hidden_size=4,
                    batch_size=2,
                    epochs=1,
                    output_dir=output_dir,
                    checkpoint_dir=checkpoint_dir,
                    log_dir=log_dir,
                ),
                device=torch.device("cpu"),
            )

            metrics_path = output_dir / "weekly_lstm_metrics.json"
            predictions_path = output_dir / "weekly_lstm_predictions.npz"
            prediction_plot_path = output_dir / "weekly_prediction_plot.png"
            checkpoint_path = checkpoint_dir / "weekly_lstm.pt"
            self.assertTrue(metrics_path.is_file())
            self.assertTrue(predictions_path.is_file())
            self.assertGreater(prediction_plot_path.stat().st_size, 0)
            self.assertTrue(checkpoint_path.is_file())

            with metrics_path.open("r", encoding="utf-8") as handle:
                saved_results = json.load(handle)
            self.assertEqual(saved_results["data"]["prediction_shape"], [2, 2])
            self.assertEqual(len(saved_results["training"]["epoch_losses"]), 1)
            self.assertEqual(saved_results["training"]["loss"], "smooth_l1")
            self.assertTrue(np.isfinite(saved_results["metrics"]["smape"]))
            self.assertTrue(np.isfinite(saved_results["metrics"]["mase"]))
            self.assertTrue(
                np.isfinite(saved_results["baselines"]["persistence"]["smape"])
            )
            self.assertTrue(
                np.isfinite(saved_results["baselines"]["persistence"]["mase"])
            )

            with np.load(predictions_path) as saved_predictions:
                self.assertEqual(saved_predictions["predictions"].shape, (2, 2))
                self.assertEqual(saved_predictions["targets"].shape, (2, 2))
                self.assertEqual(
                    saved_predictions["persistence_predictions"].shape,
                    (2, 2),
                )

            tensorboard_dir = Path(results["artifacts"]["tensorboard"])
            self.assertTrue(list(tensorboard_dir.glob("events.out.tfevents.*")))
            event_data = EventAccumulator(str(tensorboard_dir)).Reload()
            scalar_tags = set(event_data.Tags()["scalars"])
            self.assertIn("train/loss", scalar_tags)
            self.assertIn("eval/smape", scalar_tags)
            self.assertIn("eval/mase", scalar_tags)
            self.assertIn("baseline/persistence_smape", scalar_tags)
            self.assertIn("baseline/persistence_mase", scalar_tags)
            self.assertIn("predictions/W1", event_data.Tags()["images"])
            self.assertEqual(results["device"], "cpu")

    def test_weekly_rag_smoke_run_saves_retrieval_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            output_dir = root / "results"
            checkpoint_dir = root / "checkpoints"
            log_dir = root / "runs"
            cache_dir = root / "cache"
            data_dir.mkdir()
            self._write_csv(
                data_dir / "Weekly-train.csv",
                [
                    ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9"],
                    ["W1", 1, 2, 3, 4, 5, 6, 7, 8],
                    ["W2", 2, 4, 6, 8, 10, 12, 14, 16],
                ],
            )
            self._write_csv(
                data_dir / "Weekly-test.csv",
                [["V1", "V2", "V3"], ["W1", 9, 10], ["W2", 18, 20]],
            )

            config = ExperimentConfig(
                data_dir=data_dir,
                model="rag_lstm",
                input_length=3,
                max_samples=4,
                hidden_size=4,
                batch_size=2,
                epochs=1,
                top_k=1,
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
                log_dir=log_dir,
                retrieval_cache_dir=cache_dir,
            )
            cache_results = run_experiment(
                replace(config, build_cache_only=True),
                device=torch.device("cpu"),
            )
            results = run_experiment(
                config,
                device=torch.device("cpu"),
            )

            metrics_path = output_dir / "weekly_rag_lstm_metrics.json"
            predictions_path = output_dir / "weekly_rag_lstm_predictions.npz"
            retrieval_plot_path = output_dir / "weekly_retrieval_example.png"
            checkpoint_path = checkpoint_dir / "weekly_rag_lstm.pt"
            self.assertTrue(metrics_path.is_file())
            self.assertTrue(checkpoint_path.is_file())
            self.assertGreater(retrieval_plot_path.stat().st_size, 0)
            self.assertEqual(cache_results["status"], "cache_only")
            self.assertTrue(Path(cache_results["artifacts"]["cache_summary"]).is_file())
            self.assertTrue(Path(results["artifacts"]["retrieval_cache"]).is_file())
            self.assertEqual(results["model"], "TimeRAG-LSTM")
            self.assertEqual(results["retrieval"]["knowledge_base_size"], 4)
            self.assertTrue(results["retrieval"]["cache_loaded"])
            self.assertFalse(results["retrieval"]["same_series_overlap_allowed"])

            with np.load(predictions_path) as saved_predictions:
                self.assertEqual(saved_predictions["retrieved_indices"].shape, (2, 1))
                self.assertEqual(saved_predictions["retrieval_distances"].shape, (2, 1))

            tensorboard_dir = Path(results["artifacts"]["tensorboard"])
            event_data = EventAccumulator(str(tensorboard_dir)).Reload()
            self.assertIn("retrieval/W1", event_data.Tags()["images"])

    def test_gated_rag_train_tail_smoke_uses_training_only_future_prior(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            output_dir = root / "results"
            checkpoint_dir = root / "checkpoints"
            log_dir = root / "runs"
            cache_dir = root / "cache"
            data_dir.mkdir()
            self._write_csv(
                data_dir / "Weekly-train.csv",
                [
                    [f"V{i}" for i in range(1, 12)],
                    ["W1", *range(1, 11)],
                    ["W2", *range(10, 110, 10)],
                ],
            )
            self._write_csv(
                data_dir / "Weekly-test.csv",
                [
                    ["V1", "V2", "V3"],
                    ["W1", 999, 1000],
                    ["W2", 1999, 2000],
                ],
            )

            config = ExperimentConfig(
                data_dir=data_dir,
                model="gated_rag_lstm",
                evaluation_split="train_tail",
                input_length=3,
                max_samples=6,
                hidden_size=4,
                batch_size=2,
                epochs=1,
                top_k=1,
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
                log_dir=log_dir,
                retrieval_cache_dir=cache_dir,
            )
            cache_results = run_experiment(
                replace(config, build_cache_only=True),
                device=torch.device("cpu"),
            )
            results = run_experiment(config, device=torch.device("cpu"))

            predictions_path = (
                output_dir
                / "weekly_gated_rag_lstm_train_tail_predictions.npz"
            )
            self.assertEqual(cache_results["status"], "cache_only")
            self.assertEqual(cache_results["evaluation_split"], "train_tail")
            self.assertEqual(results["evaluation_split"], "train_tail")
            self.assertEqual(results["model"], "TimeRAG-LSTM (gated future)")
            self.assertEqual(results["retrieval"]["candidate_horizon"], 2)
            self.assertEqual(
                results["retrieval"]["same_series_policy"],
                "complete_candidate_episode_before_query_input",
            )
            self.assertTrue(results["retrieval"]["cache_loaded"])
            self.assertGreater(results["retrieval"]["learned_gate"], 0.0)
            self.assertLess(results["retrieval"]["learned_gate"], 1.0)
            self.assertIn("retrieval_prior", results["baselines"])

            with np.load(predictions_path) as saved_predictions:
                np.testing.assert_array_equal(
                    saved_predictions["targets"],
                    [[9, 10], [90, 100]],
                )
                self.assertEqual(
                    saved_predictions["retrieval_prior_predictions"].shape,
                    (2, 2),
                )
                self.assertFalse(np.any(saved_predictions["targets"] >= 999))

            tensorboard_dir = Path(results["artifacts"]["tensorboard"])
            event_data = EventAccumulator(str(tensorboard_dir)).Reload()
            scalar_tags = set(event_data.Tags()["scalars"])
            self.assertIn("retrieval/gate", scalar_tags)
            self.assertIn("baseline/retrieval_prior_smape", scalar_tags)
            self.assertIn("baseline/retrieval_prior_mase", scalar_tags)


if __name__ == "__main__":
    unittest.main()
