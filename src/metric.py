# 评价指标参数
import numpy as np

# smape 是时间序列预测误差指标，判断预测值 y_pred 和真实值 y_true 相差多少百分比。使用误差 / 平均规模 * 100 来计算
def smape(y_true, y_pred, eps=1e-8):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_pred) + np.abs(y_true)) / 2.0
    return float(np.mean(numerator / np.maximum(denominator, eps)) * 100.0)


# mase 是平均绝对缩放误差，将模型预测误差和一个简单基准误差做比较，insample 是训练集数据，seasonal_period 是季节性周期
def mase(y_true, y_pred, insample, seasonal_period=1):
    errors = []
    scales = []

    # 计算训练集的缩放因子
    # 遍历每一个样本，enumerate 函数会遍历列表，并且返回 i: 遍历到第几条数据和 train_values: 当前序列的具体数据
    for i, train_values in enumerate(insample):
        train_values = np.asarray(train_values, dtype=np.float32)

        # 计算缩放因子，即基准模型在训练集上的平均绝对误差
        # 当训练集长度小于季节性周期时退化为计算相邻差分
        if len(train_values) <= seasonal_period:
            scale = (
                np.mean(np.abs(np.diff(train_values))) if len(train_values) > 1 else 1.0 #np.diff 能够将该位元素减去前一位元素，得到一个新的数组，长度比原数组少 1，从 X_1-x_0 到 x_{n-1}-x_{n-2}，长度小于 1 时无法计算差分，直接设置为 1.0
            )
        # 大于季节性周期时
        else:
            scale = np.mean(
                np.abs(train_values[seasonal_period:] - train_values[:-seasonal_period])
            )
        # 当缩放因子不是有限数，包括 NaN
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0  

        errors.append(np.abs(y_true[i] - y_pred[i]))
        scales.append(scale)

    return float(np.mean(np.stack(errors) / np.asarray(scales)[:, None])) #[:,none] 将一维的数组设置为一个 len(errors) 行 1 列的二维数组，方便后续计算，并且会通过 numpy 的广播机制进行计算
