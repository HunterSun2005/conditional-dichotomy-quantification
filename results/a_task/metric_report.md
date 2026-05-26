# 指标报告

## DCF 衡量什么

Conditional Dichotomy Quantification 关注的是：在同一个上下文下，positive 和 negative 是否成为语义几何空间中最对立的两个端点。对每条样本，评测器会编码三条带上下文的文本：

- `context + positive`
- `context + negative`
- `context + neutral`

官方实现位于 `oppositescore/evaluation/evaluation.py`，它会计算这三组 embedding 两两之间的 cosine distance。

## 指标定义

对单条样本：

- `pos_neg`：`context + positive` 与 `context + negative` 的距离。
- `pos_neutral`：`context + positive` 与 `context + neutral` 的距离。
- `neg_neutral`：`context + negative` 与 `context + neutral` 的距离。

当 neutral 到两端的距离都小于 positive-negative 距离时，`DCF` 判定通过：

```text
pos_neutral < pos_neg and neg_neutral < pos_neg
```

`DCF-positive` 只检查 `pos_neutral < pos_neg`。

`DCF-negative` 只检查 `neg_neutral < pos_neg`。

`pos_neg_degree` 是 positive-negative 二分对立强度的平均值。官方代码中，它是 positive-negative pair 上 `1 - cos(angle)` 的均值；这里的 `angle` 实际上是 scikit-learn 返回的 cosine distance。

## 结果解读

DCF 越高，说明 embedding 几何关系越经常把 neutral 放在 positive 与 negative 两个端点之间。`pos_neg_degree` 越高，说明模型构造了更强的 positive-negative 分离；但它必须和 DCF 一起看，因为仅有较大的端点分离并不能保证 neutral 处在合理的中间位置。

## 实现注意事项

官方 evaluator 会把 `angles_data.csv` 写入当前工作目录。批量实验时应在每次运行后重定向或移动该文件，避免结果被覆盖。
