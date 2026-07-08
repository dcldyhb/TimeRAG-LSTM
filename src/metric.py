# 评价指标参数

import numpy as np

# smape 是时间序列预测误差指标，判断预测值 y_pred 和真实值 y_true 相差多少百分比。使用误差 / 平均规模 * 100 来计算
def smape(y_true, y_pred, eps=1e-8):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_pred)+ np.abs(y_true)) / 2.0
    return float(np.mean(numerator / np.maximum(denominator, eps)) * 100.0)

def mase(y_true, y_pred, insample, seasonal_period=1):
    errors = []
    scales = []

    for i, train_values in enumerate(insample):
        train_values = np.asarray(train_values, dtype=np.float32)

        if len(train_values) <= seasonal_period:
            scale = np.mean(np.abs(np.diff(train_values))) if len(train_values) > 1 else 1.0
        else:
            scale = np.mean(np.abs(train_values[seasonal_period:] - train_values[:-seasonal_period]))

        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0

        errors.append(np.abs(y_true[i] - y_pred[i]))
        scales.append(scale)

    return float(np.mean(np.stack(errors) / np.asarray(scales)[:, None]))