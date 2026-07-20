"""Forecasting models used by the TimeRAG-LSTM experiments."""

from __future__ import annotations

from torch import Tensor, nn


# 直接预测未来一段时间的值，不含辅助的 Plain LSTM 模型
class LSTMForecaster(nn.Module): # 继承 nn.Module 类而不是实参

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        horizon: int = 13,
        dropout: float = 0.0,
    ) -> None:
        super().__init__() #调用父类的构造函数

        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        # 注册为模块的属性，以便调用
        self.input_size = input_size
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0, # 单层 LSTM 不易发生过拟合，不需要 drop out
        )
        self.output_layer = nn.Linear(hidden_size, horizon)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forecast from ``[batch, time]`` or ``[batch, time, features]``."""
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(dim=-1)

        if inputs.ndim != 3:
            raise ValueError(
                "inputs must have shape [batch, time] or [batch, time, features]"
            )
        if inputs.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected input_size={self.input_size}, got {inputs.shape[-1]}"
            )

        sequence_outputs, _ = self.lstm(inputs)
        final_output = sequence_outputs[:, -1, :]
        return self.output_layer(final_output)
