import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None
else:
    from src.models import (
        GatedFutureLSTMForecaster,
        LSTMForecaster,
        RAGLSTMForecaster,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class LSTMForecasterTests(unittest.TestCase):
    def test_two_dimensional_input_produces_horizon_forecast(self):
        model = LSTMForecaster(hidden_size=16, horizon=13)
        inputs = torch.randn(8, 26)

        predictions = model(inputs)

        self.assertEqual(tuple(predictions.shape), (8, 13))

    def test_three_dimensional_input_is_supported(self):
        model = LSTMForecaster(input_size=2, hidden_size=16, horizon=5)
        inputs = torch.randn(4, 10, 2)

        predictions = model(inputs)

        self.assertEqual(tuple(predictions.shape), (4, 5))

    def test_incorrect_feature_count_is_rejected(self):
        model = LSTMForecaster(input_size=1)

        with self.assertRaisesRegex(ValueError, "Expected input_size=1"):
            model(torch.randn(4, 26, 2))

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "horizon must be positive"):
            LSTMForecaster(horizon=0)

    def test_rag_lstm_uses_query_and_retrieved_channels(self):
        model = RAGLSTMForecaster(top_k=5, hidden_size=16, horizon=13)
        inputs = torch.randn(8, 26, 6)

        predictions = model(inputs)

        self.assertEqual(model.input_size, 6)
        self.assertEqual(tuple(predictions.shape), (8, 13))

    def test_rag_lstm_rejects_missing_retrieval_channels(self):
        model = RAGLSTMForecaster(top_k=2)

        with self.assertRaisesRegex(ValueError, "Expected input_size=3"):
            model(torch.randn(4, 26, 2))

    def test_gated_future_lstm_produces_horizon_forecast(self):
        model = GatedFutureLSTMForecaster(hidden_size=8, horizon=5)

        predictions = model(torch.randn(4, 10), torch.randn(4, 5))

        self.assertEqual(tuple(predictions.shape), (4, 5))

    def test_gated_future_lstm_uses_convex_gate(self):
        model = GatedFutureLSTMForecaster(
            hidden_size=8,
            horizon=3,
            initial_gate=0.25,
        )
        inputs = torch.randn(2, 6)
        future_prior = torch.randn(2, 3)

        base = model.base_forecast(inputs)
        predictions = model(inputs, future_prior)
        gate = model.retrieval_gate()

        torch.testing.assert_close(gate, torch.tensor(0.25))
        torch.testing.assert_close(
            predictions,
            (1.0 - gate) * base + gate * future_prior,
        )
        self.assertGreater(float(gate.detach()), 0.0)
        self.assertLess(float(gate.detach()), 1.0)

    def test_gated_future_lstm_rejects_invalid_shapes(self):
        model = GatedFutureLSTMForecaster(hidden_size=8, horizon=3)

        with self.assertRaisesRegex(ValueError, "Expected input_size=1"):
            model(torch.randn(2, 6, 2), torch.randn(2, 3))
        with self.assertRaisesRegex(ValueError, "future_prior must have shape"):
            model(torch.randn(2, 6), torch.randn(2, 4))

    def test_gated_future_lstm_rejects_invalid_configuration(self):
        for initial_gate in (0.0, 1.0, -0.1, 1.1):
            with self.subTest(initial_gate=initial_gate):
                with self.assertRaisesRegex(
                    ValueError,
                    "initial_gate must be strictly between 0 and 1",
                ):
                    GatedFutureLSTMForecaster(initial_gate=initial_gate)

        with self.assertRaisesRegex(ValueError, "hidden_size must be positive"):
            GatedFutureLSTMForecaster(hidden_size=0)

    def test_gated_future_lstm_backpropagates_to_base_and_gate(self):
        torch.manual_seed(7)
        model = GatedFutureLSTMForecaster(
            hidden_size=8,
            horizon=3,
            initial_gate=0.25,
        )
        inputs = torch.randn(4, 6)
        future_prior = torch.full((4, 3), 10.0)

        model(inputs, future_prior).sum().backward()

        self.assertIsNotNone(model.lstm.weight_ih_l0.grad)
        self.assertGreater(float(model.lstm.weight_ih_l0.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.output_layer.weight.grad)
        self.assertGreater(float(model.output_layer.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.retrieval_gate_logit.grad)
        self.assertGreater(float(model.retrieval_gate_logit.grad.abs()), 0.0)


if __name__ == "__main__":
    unittest.main()
