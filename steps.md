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

- `train/loss`：每个 epoch 的平均 Smooth L1 loss。
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

## Stage 2：DTW 检索与 TimeRAG-LSTM 本地闭环

`src/retrieval.py` 已实现：

1. 只从训练滑窗输入构建知识库，不保存训练目标或官方测试目标。
2. 对每个窗口独立标准化后计算精确 DTW top-k。
3. 训练查询排除自身；同一序列排除所有与查询输入区间重叠的候选。
4. 使用输入与元数据内容指纹校验 NPZ 缓存，防止参数或样本变化后误用旧结果。
5. 将 query 与 top-k 检索窗口拼为 `[batch, time, top_k + 1]` 通道输入。

`RAGLSTMForecaster` 复用 plain LSTM，只把输入特征数固定为
`top_k + 1`。`train.py --model rag_lstm --top-k 5` 会保存检索缓存、检索索引与
距离、标准化检索示例图、预测、指标、checkpoint 和 TensorBoard 日志。

2026-07-20 的真实 M4 Weekly CPU smoke 使用 100 个训练窗口、top-5、2 epoch：

- loss：`1.436290 -> 1.420194`。
- SMAPE：`12.1860`。
- MASE：`4.8831`。
- persistence：SMAPE `9.1613`，MASE `2.7773`。
- 第二次相同运行成功复用内容指纹一致的 DTW 缓存。
- 真实缓存审计通过：训练查询无自身命中，训练和评估查询均无同序列窗口重叠，距离均有限。
- 全部 25 个测试通过，包括 plain LSTM 回归和 RAG 端到端产物测试。

该结果只证明 RAG 闭环正确，小样本精度尚不优于 persistence。精确全对全 DTW
不适合直接扩展到 353,270 个 Weekly 窗口，正式实验前需要先测试规模并确定候选缩减策略。

### 有界候选检索优化

当前默认检索改为两阶段：

1. 使用 SciPy `cKDTree` 在标准化窗口上按欧氏距离预筛 512 个候选。
2. 按 query batch 对候选池执行向量化精确 DTW，最终保留 top-5。
3. `exact` 模式保留为小规模 oracle，不用于全量 Weekly。
4. 同序列候选必须在 query 输入窗口开始前结束；优化路径仍执行相同因果过滤。
5. 缓存 key 纳入策略、候选池、SciPy/DTW/因果策略版本和全部输入内容。
6. 缓存采用临时文件加原子替换，并在加载后复核形状、索引、距离、自身命中和因果约束。
7. `--build-cache-only` 支持先离线构建缓存，再从同一缓存启动训练。
8. 非数值距离缓存会作为损坏缓存拒绝；预筛选在因果过滤后候选不足时会扩展搜索范围。

10k KB oracle 基准（200 个训练 query + 359 个评估 query）：

- exact 总耗时：约 `19.20s`。
- 512 候选总耗时：约 `1.85s`，约快 `10.4x`。
- 512 候选评估 top-5 recall：`0.690`，top-1 一致率：`0.730`。
- 1024 候选评估 top-5 recall：`0.772`，但耗时约为 512 候选的 `3.4x`。
- 10k 完整 cache-only、batch 64：`37.76s`；batch 256 反而为 `81.28s`，因此默认 batch 为 64。
- 50k 完整 cache-only：`227.50s`，缓存约 `1.5MB`；二次加载约 `0.012s`。
- v2 路径最终本地回归为 31 个测试全部通过，当时的 50k 缓存也成功重新加载并通过因果校验；后续 gated-future 的 v3 完整 episode 规则会有意使该旧缓存失效。

匹配的 10k、10 epoch CPU 对照：

- plain LSTM：SMAPE `8.5773`，MASE `2.3726`。
- TimeRAG-LSTM（512 候选）：SMAPE `8.5530`，MASE `2.4054`。

RAG 的 SMAPE 仅改善 `0.0243`，但 MASE 变差 `0.0328`，属于混合结果，尚不能
宣称检索整体优于 plain LSTM。正式结论必须来自全量 Weekly 的匹配实验。

### Gated future prior 优化

逐序列分析显示，history-channel RAG 在 359 条序列中的 186 条上更好，中位误差也
略好，但少数大幅退化序列使整体 MASE 变差。DTW 距离和 exact recall 对相对收益的
解释力较弱，主要瓶颈是旧模型只看到相似历史，不知道这些历史随后发生了什么。

新路径实现了：

1. `train_tail` 训练期验证：每条 official train 的最后 `H` 点作为验证目标，前 `L`
   点作为验证输入，official test 不参与参数选择。
2. outcome bank：只保存训练窗口的 future，并使用对应 candidate input 的统计量标准化。
3. 完整 episode 规则：同序列候选必须满足
   `candidate_cutoff + H <= query_cutoff - L`。
4. future prior：对 top-k candidate future 使用稳定的
   `softmax(-distance / temperature)` 加权，结果与邻居排列无关。
5. gated residual：query-only LSTM 生成 base forecast，再用可学习凸 gate 与 future
   prior 融合。旧 `rag_lstm` 保留为 history-channel 消融。
6. 缓存 schema 升为 v3，并将 candidate horizon 纳入指纹。

10k、10 epoch 的训练尾部验证结果：

```text
Model                         SMAPE    MASE      Selection score
Plain LSTM                    9.0424   2.5607    0.9335
History-channel RAG           9.2236   2.5880    0.9481
Gated future, temperature .25 8.8750   2.5600    0.9243
Gated future, temperature .50 8.9259   2.5587    0.9268
Gated future, temperature 1.0 8.9384   2.5645    0.9285
```

选择分数预先固定为
`0.5 * (SMAPE / persistence_SMAPE + MASE / persistence_MASE)`，越低越好，因此
选择 temperature `0.25`。对应的 10k official-test 结果为：

- matched plain LSTM：SMAPE `8.5773`，MASE `2.3726`。
- gated-future RAG：SMAPE `8.4361`，MASE `2.3809`，learned gate `0.1327`。

新模型的 SMAPE 数值改善 `0.1412`（相对下降 `1.646%`），并将旧 RAG 的 MASE
退化从 `+0.0328` 缩小到 `+0.0083`。逐序列 bootstrap 的 SMAPE 差值 95% 区间为
`[-0.2843, 0.0130]`，因此不能声称统计显著；整体仍是混合结果，也不能宣称两项指标
都优于 plain LSTM。温度选择没有直接使用 official-test target，但更早的架构诊断曾查看
official-test 结果，所以该测试结果不应描述为完全未查看的最终盲测。当前完整回归为 45 个测试通过。
