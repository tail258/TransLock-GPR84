<div align="center">

![TransLock-GPR84 工作流程](assets/workflow.svg)

# TransLock-GPR84

**面向低数据条件的 GPR84 可追溯分子筛选研究**

[English](README.md) · [简体中文](README.zh-CN.md) · [模型说明](models/Model_Card.md) · [研究方法](docs/methods.md)

</div>

TransLock-GPR84 是一个用于提出 GPR84 小分子优先级假设的计算研究项目。项目治理公开实验测量，微调 [CheMeleon](https://doi.org/10.1021/acs.jcim.6c01546) 分子图模型，进行整篇文献留出与单个专利系列的外部回顾评价，再用固定规则筛选公开分子目录。本仓库提供研究代码、小体积来源与开发数据表、精选结果，以及大型资源的获取说明。

> **当前证据范围：**十项提名分子的 GPR84 实际活性和拮抗作用均未知。外部评价是单个专利系列的回顾分析；短程分子动力学仅用于结构解释。

## 项目一览

| 阶段 | 已记录结果 |
|:--|:--|
| 数据治理 | ChEMBL 37 原始 2,070 条活性记录 → 285 条准入测量 → 277 个实验内单位 → 248 个编码图，来自 6 篇论文、17 个实验体系 |
| 内部开发 | 前20%预算的平均富集因子（EF）：微调 **1.112**、冻结编码器 **1.016**、指纹相似性 **1.083**。这些比较参与了方法选择。 |
| 目录应用 | 100,000 条原始目录记录 → **82,883** 个独立评分分子图 → **10** 项按骨架规则提名的候选假设 |
| 外部回顾评价 | [WO2024028364A1](https://patents.google.com/patent/WO2024028364A1/en) 中 61 项实测：固定主模型前13项命中 **12** 个强效分子，相似性参照命中 **10** 个；ROC-AUC 分别为 **0.646** 和 **0.401**。 |
| 短程 MD | 两项候选和 GLPG1205 对照各进行三条 5 ns 膜模拟：合计 **9 条轨迹 / 45 ns**。 |

外部集合中有53项强效、8项较弱，随机选13项的强效期望约为 **11.30**。61样品和60连接图的评价、三个预定种子及证据边界见[评价说明](docs/evaluation.md)。这组数据没有检验十项提名分子的活性。

## 仓库内容

```text
src/translock_screening/   数据治理、分组评价、训练、筛选与分析代码
scripts/                   研究脚本入口
configs/                   固定参数及协议配置
data/source/               小体积 ChEMBL 37 来源记录与出处
data/training/             准入测量、训练单位和留出成员
results/                   精选开发指标、外部汇总与候选清单
tests/                     相关行为检查
docs/                      方法、复现和资源说明
models/Model_Card.md       模型结构、用途与限制
```

原赛事提交件另行保留。此公开版只收入适合源码仓库的材料。大型权重、完整十万条目录、对接姿势、MD轨迹及再发布许可待核的专利转录不进入 Git 历史；见[资源说明](docs/resources.md)。

## 开始使用

原研究生产环境为 **WSL2 Ubuntu 22.04、Python 3.11 和 uv**。根锁文件指定主 CheMeleon 环境与 CUDA 版 PyTorch；新的公开检出尚未在所有设备上验收。

```bash
git clone https://github.com/tail258/TransLock-GPR84.git
cd TransLock-GPR84
uv sync --frozen
```

无需下载模型，即可阅读[训练单位表](data/training/training_units.csv)、[内部开发结果](results/internal_development/)、[外部汇总指标](results/external/metrics.csv)和[十项候选清单](results/screening/candidate_hypotheses.csv)。随仓数据不包含本团队开展的湿实验测量。

固定完整目录流程的入口为 `govern.py`、`train.py`、`predict.py`、`screen.py` 和 `main.py`，各自需要不同资源。[复现说明](docs/reproduction.md)列明输入、缺失提示和已验证程度。[`examples/`](examples/README.md)提供少量分子输入；从 [v0.1.0 模型发布页](https://github.com/tail258/TransLock-GPR84/releases/tag/v0.1.0)取得三份微调权重放入 `models/`，再从[作者固定记录](https://doi.org/10.5281/zenodo.15460715)取得预训练权重放入 `models/pretrained/`，即可使用正式模型逻辑推理。

## 证据边界

- 整篇文献留出比较属于内部开发评价，不能称独立测试。
- 外部排序只来自一个公开专利中的 GPR84 拮抗测量系列。评分前模型和规则已固定，但标签此前已知；不能据此推广为跨化学系列稳定优势。
- 模型输出为未经校准的**原始 logit 排序分数**，不是 IC50 或活性概率。
- 不同骨架、对接姿势及 5 ns MD 轨迹只支持候选假设和有限结构解释，不能证明实验活性、拮抗方向、结合自由能、选择性或安全性。

解读具体分数前请阅读[模型说明](models/Model_Card.md)、[方法](docs/methods.md)和[评价](docs/evaluation.md)。

## 来源与使用

数据来源及适用条款见[data/README.md](data/README.md)。第三方代码与 CheMeleon 权重来源见[THIRD_PARTY.md](THIRD_PARTY.md)，许可原文见[`licenses/`](licenses/)。本项目自有代码尚未声明对外许可；仓库公开可见不等于已授予再使用权。引用方法或数据时，请同时参考[文献清单](docs/references.md)中的原始论文和数据来源。

仓库默认英文文档；此页提供与英文首页一致的中文研究口径。
