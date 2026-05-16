# Federated-Class-Continual-Learning

本项目是论文《Federated Continual Learning With Bounded Forgetting via Diffusion-Based Generative Replay in Edge Computing》的实验代码。

This project is the experimental code of the paper "Federated Continual Learning With Bounded Forgetting via Diffusion-Based Generative Replay in Edge Computing".

## Table of Contents

- [项目简介 Project Overview](#项目简介-project-overview)
- [项目目录 Project Directory](#项目目录-project-directory)
- [使用方法 Getting Started](#使用方法-getting-started)
  - [环境依赖 Environment](#环境依赖-environment)
  - [数据集 Datasets](#数据集-datasets)
  - [训练 Training](#训练-training)
- [项目声明 Project Statement](#项目声明-project-statement)

## 项目简介 Project Overview

本项目提出了一种基于扩散生成回放的联邦持续学习框架，旨在解决边缘计算场景下的持续学习问题。该方法通过生成式回放技术有效缓解了灾难性遗忘问题，同时实现了有界遗忘的理论保证。

This project proposes a federated continual learning framework based on diffusion-based generative replay, addressing the challenge of continual learning in edge computing scenarios. The method effectively mitigates catastrophic forgetting through generative replay technology while achieving bounded forgetting guarantees.

### 主要贡献 Main Contributions

- **联邦持续学习框架**: 支持多客户端协同训练，保护数据隐私
- **有界遗忘机制**: 理论保证学习新任务时对旧知识的遗忘程度
- **扩散生成回放**: 利用扩散模型生成高质量的伪样本进行回放
- **原型对比学习**: 采用原型对比损失增强类别可分性
- **边缘计算优化**: 适配边缘设备的资源受限场景

## 项目目录 Project Directory

```
Federated-Class-Continual-Learning/
├── convs/                           # 神经网络架构
│   ├── cifar_resnet.py             # CIFAR数据集专用的ResNet
│   ├── generator.py                # 生成器模型
│   ├── linears.py                  # 线性层分类器
│   ├── modified_represnet.py       # 改进的代表性网络
│   ├── resnet.py                   # 标准ResNet实现
│   ├── resnet_cbam.py              # 带CBAM注意力的ResNet
│   └── ucir_resnet.py              # UCIR专用ResNet
├── Loss/                           # 损失函数
│   └── losses.py                   # 包含SupConLoss等损失实现
├── methods/                        # 持续学习方法实现
│   ├── base.py                     # 基础学习器类
│   ├── ewc.py                      # EWC方法
│   ├── finetune.py                # 简单微调方法
│   ├── lwf.py                      # LwF方法
│   ├── PPCL.py                     # 本项目主要方法
│   └── target.py                   # 目标检测相关方法
├── utils/                         # 工具函数
│   ├── data.py                     # 数据集定义
│   ├── data_manager.py             # 数据管理及联邦学习数据划分
│   ├── empirical_feature_matrix.py  # 经验特征矩阵
│   ├── inc_net.py                  # 增量网络
│   └── toolkit.py                  # 通用工具函数
```

## 使用方法 Getting Started

### 环境依赖 Environment

本项目基于PyTorch实现，主要依赖包括：

```bash
torch >= 1.8.0
torchvision
numpy
scipy
Pillow
tqdm
wandb  # 可选，用于实验记录
kornia  # 用于数据增强
```

### 数据集 Datasets

项目支持以下数据集：

- **CIFAR-100**: 默认数据集
- **ImageNet100/ImageNet1000**
- **TinyImageNet200**

数据集路径需在代码中配置，默认使用CIFAR-100。

### 训练 Training

本项目实现了联邦学习训练流程，主要参数配置如下：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `dataset` | 数据集名称 | `cifar100` |
| `net` | 网络架构 | `resnet32` |
| `num_users` | 联邦学习客户端数量 | `5` |
| `frac` | 参与训练的客户端比例 | `0.4` |
| `local_bs` | 本地批次大小 | `128` |
| `com_round` | 通信轮数 | `200` |
| `lr` | 学习率 | `0.1` |
| `seed` | 随机种子 | `1993` |
| `init_cls` | 初始类别数 | `10` |
| `increment` | 每任务增量类别数 | `10` |

#### 运行示例 Example

```python
from methods.PPCL import PPCL
from utils.data_manager import DataManager

args = {
    "dataset": "cifar100",
    "net": "resnet32",
    "num_users": 5,
    "frac": 0.4,
    "local_bs": 128,
    "com_round": 200,
    "lr": 0.1,
    "seed": 1993,
    "init_cls": 10,
    "increment": 10,
    "wandb": 0,
    "save_dir": "./output"
}

data_manager = DataManager(
    dataset_name=args["dataset"],
    shuffle=True,
    seed=args["seed"],
    init_cls=args["init_cls"],
    increment=args["increment"]
)

learner = PPCL(args)
learner.train(data_manager)
```

## 项目声明 Project Statement

本项目的作者及单位：

The author and affiliation of this project:

- **项目名称 (Project Name)**: Federated-Class-Continual-Learning
- **作者 (Authors)**: Zaobo He, Yunkun Wang, Zhipeng Cai
- **作者单位 (Affiliation)**: 暨南大学网络空间安全学院 (College of Cyber Security, Jinan University)
- **论文 (Paper)**: Federated Continual Learning With Bounded Forgetting via Diffusion-Based Generative Replay in Edge Computing
- **期刊 (Journal)**: IEEE Transactions on Mobile Computing
- **卷期 (Volume/Issue)**: Volume 25, Issue 3, Pages 3001-3017
- **出版时间 (Date)**: March 2026
- **DOI**: 10.1109/TMC.2025.3618275

### 核心方法说明

本项目实现的方法包含以下核心组件：

1. **全局合成器 (GlobalSynthesizer)**: 基于扩散生成模型的图像合成器，用于合成高质量伪样本进行回放
2. **联邦训练 (Federated Training)**: 支持多客户端协同学习，采用FedAvg算法聚合模型权重
3. **本地微调 (Local Finetune)**: 客户端本地模型更新策略，结合知识蒸馏保持旧知识
4. **原型对比学习 (Prototype Contrastive Learning)**: 采用原型对比损失增强类别可分性，减少任务间的干扰
5. **有界遗忘 (Bounded Forgetting)**: 通过梯度重加权、间隔损失和原型对齐机制控制对旧知识的遗忘程度
6. **硬样本挖掘 (Hard Sample Mining)**: 关注决策边界附近的困难样本，提升模型判别能力

### 引用 Citation

如果您在研究中使用本项目代码，请引用我们的论文：

If you use this code in your research, please cite our paper:

```
@ARTICLE{2026TMCHe,
  author={He, Zaobo and Wang, Yunkun and Cai, Zhipeng},
  journal={IEEE Transactions on Mobile Computing},
  title={Federated Continual Learning With Bounded Forgetting via Diffusion-Based Generative Replay in Edge Computing},
  year={2026},
  volume={25},
  number={3},
  pages={3001--3017},
  doi={10.1109/TMC.2025.3618275}
}
```
