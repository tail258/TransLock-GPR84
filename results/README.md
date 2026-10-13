# 正式清单与结果字段

> GitHub 公开版说明：这里只收录精选开发表、外部汇总指标与十项筛选清单。完整目录评分、专利逐项转录、姿势与大体积MD分析在原项目档案中；获取范围见 [`docs/resources.md`](../docs/resources.md)。下文“随包”指原正式代码包。

根目录`results.csv`是10项正式提名的UTF-8交付表。`main.py`默认核验归档并在指定新目录生成相同正式表；`--workflow full`执行正式训练、参数优化及完整目录筛选，生成本次重拟合模型的清单。`screen.py`默认用归档权重完整推理，也可用`--model-dir`读取本包新训练目录。`predict.py`输出完整目录中的全部准入图结果。

本次重拟合结果的`model_version`和`run_version`独立记录，不自动替换根目录正式表或历史外部指标。只有候选身份与已计算结构匹配时才关联既有PDB/SDF；未指定中心可对应其已归档条件性构型，已指定构型必须保持。新增身份的结构字段为空，筛选入口不自动计算对接或MD。

| 列 | 类型/单位/意义 |
|---|---|
| candidate_id | ChEMBL候选编号，字符串 |
| track | 赛道三-大规模虚拟筛选 |
| candidate_smiles | 标准化有立体SMILES；未指定立体继续未指定 |
| prediction_raw_logit | 固定主模型排序分数，无量纲；IC50需实验测量，含义见模型说明 |
| prediction_unit | dimensionless |
| primary_rank | 20261001名次；针对82883个评分图 |
| seed_20261002_rank / seed_20261003_rank | 两个预定敏感种子的名次 |
| nearest_training_similarity | 与训练图的最大r3含手性计数Morgan Tanimoto，0–1；描述结构相似度 |
| nomination_order | 固定骨架提名顺序；全部图推理表未提名时留空 |
| model_version / run_version | 正式模型与运行版本 |
| structure_pdb_files / ligand_sdf_files | 对应文件名，多个文件用分号分隔；路径按代码包根目录解析 |
| md_completed_replicates | CHEMBL120691、CHEMBL1086046各3；其余0表示未做该阶段，不表示阴性 |
| notes | 结果范围、候选活性未知及计算姿势假设等必要解释 |

浮点分数保存归档精度，不人为四舍五入后重新排名；CSV默认浮点尾差随环境可能有微小变化，权重小样回读允许绝对误差1e-5。缺失信息留空。清单没有填实验活性数值，因为10项候选尚未取得GPR84功能实验结果。

`screening/`保存完整评分、所有记录去向、10项提名及近邻测量；`internal_development/`、`internal_cross/`为历史开发证据；`external/`为61项公开专利外部回顾证据。两类评价按各自的数据来源和指标分别报告。

`structures/`为28份候选和2份对照复合物，PDB为Å、受体A/B、配体C/UNL；SDF保留化学身份。四个未指定手性候选各有两状态、每状态两个对接种子，其余每候选两份。源于中性登记互变形式的条件性姿势，不是实验复合物；不使用PDB推断未明确的键级或手性。

`md_interpretation/`的原代表SDF可能带历史含氢参数属性，当前干净展示导出以`md_representatives/`为准；其坐标未重新计算。`md_torsion_current/`为当前角度解释范围，原sp近线性角保留为历史图/表，不纳入当前解释。全部派生表来自已有轨迹后处理，候选实验状态见模型说明。
