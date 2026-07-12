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
