# 机器学习 - 个人作业 - 区分混合图像数据集

## 问题描述

本题以**无监督学习**为背景，你需要学习图像特征表示，将一批来**源于不同数据集并混合到一起**的图像区分开（无来源信息和语义标签，混合比例未知）。
该问题不区分训练集和测试集。为测试，你需要回答多个查询，每个询问图像A和图像B**是否来自同一个源数据集**，答案为是（1）或否（0）。

## 技术方案

### 特征提取

首先，通过深度神经网络模型，来提取图像的特征。由于是**无监督学习**，采用 **Autoencoder** + **Contrastive Learning** 的方法，用于学习图像特征表示。
由于数据集中图像大小仅有 $32×32$ ，所以默认采用三层卷积神经网络来提取图像特征，以及三层反卷积神经网络来重建图像。

### 特征聚类

在训练好深度神经网络模型之后，计算每张图像的特征向量。对这些特征向量，采用 **KMeans** 方法进行聚类，以区分两张图像是否来自同一个源数据集。

## 模型训练

执行以下命令训练网络模型：`python train.py`.
一些参数设置可以在`train.py`中开头部分修改，如`batch_size`, `epochs`, `data_dir`...

## 图像聚类

在训练好网络模型之后，通过以下命令对所有图像进行聚类：`python cluster.py`.

聚类结果将以 `json` 格式进行保存，例如：

```
{
    "0001579a.png": 0,
    "0025413a.png": 1,
    "image": label,
    ...
}
```

## 查询结果

执行以下命令来查询两张图像是否来自于同一个源数据集：`python query.py`.

查询结果将以 `csv` 格式进行保存，例如：

```csv
fileA        fileB        label
0001579a     0025413a     1
mnjh78a5     k9jju88c     0
...
```

## 可视化

执行以下命令可视化重建图像、图像特征：`python visual.py`.

以下是可视化结果示例：

<img src="./img/reconstructed%20image1.png" title="" alt="" width="395"><img src="./img/reconstructed%20image2.png" title="" alt="" width="395"><img src="./img/image%20features.png" title="" alt="" width="400">
