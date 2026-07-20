# 流程记录

## 数据集收集

使用 M4 训练集，路径为 /data/m4 和 /data/m4-unused

## 本地 M4 Weekly LSTM baseline 闭环

### 评价指标

[mertic.py](./src/metric.py)

分别使用平均绝对百分比误差和平均绝对标度误差进行分析

#### 平均绝对百分比误差

平均绝对百分比误差 MAPE 是衡量预测值与真实值之间百分比误差的平均值

计算公式如下

$$
MAPE = \frac{1}{n}\sum \frac{\vert \text{预测值}-\text{真实值}\vert}{\vert\text{真实值}\vert}*100
$$

如果使用 MAPE 作为评价指标，则如果真实值很小，则可能会将误差放大

在本项目中使用**对称平均绝对百分比误差（Symmetric Mean Absolute Percentage Eror, SMAPE）**

主要相差在分母上使用了真实值和预测值的平均大小

$$
SMAPE = \frac{1}{n}\sum \frac{\vert \text{预测值}-\text{真实值}\vert}{(\vert\text{真实值}\vert+\vert\text{预测值}\vert)/2}*100
$$

#### 平均绝对缩放误差

将当前模型的预测误差与一个简单的基准模型在训练集上的误差进行对比

对于一条训练序列：

$$
x_1,x_2,\dots,x_T
$$

假设季节周期为 $m$，缩放因子为：

$$
 Q = \frac{1}{T-m} \sum_{t=m+1}^{T} |x_t-x_{t-m}|
$$

这个 $Q$ 是季节性朴素预测模型在训练集上的平均绝对误差。

模型在测试集上的绝对预测误差是：

$$
|y_h-\hat{y}\_h|
$$

缩放后的误差是：

$$
q_h = \frac{|y_h-\hat{y}\_h|}{Q}
$$

如果预测区间有 $H$ 个点，则该序列的 MASE 是：

$$
\operatorname{MASE} = \frac{1}{H} \sum\_{h=1}^{H} \frac{|y_h-\hat{y}\_h|}{Q}
$$

### 数据读取

[data.py](./src/data.py)

这个模块实现了数据的载入和预处理，主要实现了

1. 读取数据
2. 构建训练使用的滑动窗口样本
3. 构建了评估用的样本
   - 使用每个序列的最后 `input_length` 个点作为输入，官方的测试集为目标
4. 样本标准化

#### 样本标准化

对于一批样本，包含输入 $X$ 和输出 $Y$

根据 $X$ 计算其统计量 $\mu$ 和 $\sigma$，然后分别对 $X$ 和 $Y$ 使用输入的统计量进行标准化

$$
X_{i,t}^{'} = \frac{X_{i,t}-\mu_i}{\sigma_i}
$$

$$
Y_{i..t}^{'} = \frac{Y_{i,t}-\mu_i}{\sigma_i}
$$

关键在于

1. 仅使用输入量来计算统计值，防止未来数据的均值被输入到模型中，防止向未来的大致水平透露给模型
2. 仅使用输入量来标准化 $Y$

### 模型

使用 [LSTM](./src/model.py) 作为 baseline 模型

#### LSTM 模型

即 Long Short Term Memory，长短期记忆模型，和传统 RNN 相比，LSTM 加入了门控机制，能够更好的处理长时间的序列数据，记录很久以前的重要信息同时忘记不重要的信息，防止出现梯度消失和梯度爆炸的问题

其核心为

- 两条并行的记忆线
  1. 细胞状态 $C_t$ 长期记忆
  2. 隐藏状态 $h_t$ 短期记忆
- 三个门
  1. 遗忘门
     查看当前输入和上一个隐藏状态，从而输出 $0$ 到 $1$ 之间的数值，表示当前细胞状态中哪些信息需要被遗忘，哪些信息需要被保留
  2. 输入门
     通过输入门层和后续值层筛选哪些信息需要被输入到长期记忆中，需要输入多少
  3. 输出门
     基于当前细胞状态决定需要输出多少内容

##### 核心计算公式为

###### 遗忘门

$$
f_t = \sigma(W_f\cdot [h_{t-1},x_t]+b_f)
$$

###### 输入门

$$
i_t = \sigma (W_i\cdot [h_{t-1},x_t]+b_i)
$$

###### 候选记忆

$$
\tilde{C_t}=\tanh(W_C\cdot [h_{t-1},x_t]+b_C)
$$

将上一刻的短期记忆和当前输入融合

###### 更新细胞状态

$$
C_t = f_t\odot C_{t-1}+i_t\odot \tilde{C_t}
$$

###### 输出门

$$
o_t = \sigma(W_o\cdot [h_{t-1},x_t]+b_o)
$$

###### 更新短期记忆

$$
h_t = o_t\odot \tanh(C_t)
$$

### 训练与评估入口

[`train.py`](./train.py) 已将 plain LSTM baseline 串成完整流程：

1. 加载指定频率的 M4 训练集和官方测试集。
2. 从训练集构造滑动窗口，并仅使用各输入窗口的统计量标准化输入和目标。
3. 使用 Smooth L1 损失和 Adam 优化器训练 `LSTMForecaster`。
4. 使用每条训练序列的末尾窗口预测官方测试集，随后反标准化。
5. 计算 SMAPE 和 MASE，并保存配置、逐轮 loss、预测数组和模型参数。

2026-07-19 的本地 Weekly CPU 调试命令：

```bash
conda run --no-capture-output -n TimeRAG-LSTM python train.py \
  --freq Weekly --max-samples 100 --epochs 2
```

已观察到：

- 训练样本数：100。
- epoch loss：`6.682118 -> 6.647610`。
- 官方评估预测形状：`(359, 13)`。
- SMAPE：`12.1946`。
- MASE：`4.9444`。
- 指标：`results/weekly_lstm_metrics.json`。
- 预测：`results/weekly_lstm_predictions.npz`。
- 预测图：`results/weekly_prediction_plot.png`。
- checkpoint：`checkpoints/weekly_lstm.pt`。
- TensorBoard 事件日志：`runs/weekly_lstm_<timestamp>/`。

这只是用于验证闭环的两轮小样本结果，不是正式实验精度。Codex 的非交互 Conda 进程使用 CPU；用户的交互式 Conda 终端已经验证 MPS 可用并被 `src.config.DEVICE` 选中。

### TensorBoard 可视化

`train.py` 会为每次运行创建带时间戳的独立事件目录，并记录：

- `train/loss`：每个 epoch 的平均 MSE loss。
- `eval/smape` 和 `eval/mase`：官方测试集上的最终指标。
- `baseline/persistence_smape` 和 `baseline/persistence_mase`：最后值外推基线。
- `run/configuration`：本次运行的完整配置。
- `predictions/<series_id>`：前三条评估序列的输入、真实目标和 LSTM 预测曲线。

启动 TensorBoard：

```bash
tensorboard --logdir runs --port 6006
```

然后访问 `http://localhost:6006`。JSON、NPZ、PNG 和 checkpoint 仍是独立保存的实验产物，TensorBoard 只负责交互式查看和多次运行对比。

### 近常数窗口稳定化

10,000 个训练窗口中曾出现一条 W10 异常窗口：输入值约为 1200，但输入标准差只有 `0.01087`，后续目标跳到 1300 以上。仅使用绝对下限 `1e-6` 时，最大标准化目标达到 `27601.6`，该窗口贡献了 `99.9439%` 的训练 MSE。

修复包含两部分：

1. scale 除绝对下限外，还使用只由输入 X 计算的相对下限：输入绝对均值的 `0.1%`。
2. 使用 Smooth L1 代替 MSE，避免孤立的大幅跳变通过平方误差控制整个训练目标。

修复后相同 10k 样本、10 epoch CPU 对照实验结果：

- loss：`1.0213 -> 0.9021`。
- SMAPE：`8.5773`，优于 persistence 的 `9.1613`。
- MASE：`2.3726`，优于 persistence 的 `2.7773`。
- LSTM 在 359 条评估序列中的 227 条上优于 persistence。

相对 scale 下限仍只使用输入窗口，因此没有让目标或官方测试集参与统计量估计。
