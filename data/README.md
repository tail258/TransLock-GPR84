# 数据、来源与准入

> GitHub 公开版说明：本页沿用原正式代码包的数据治理记录。当前 Git 仓库只收录 `source/`、`training/`、`registry/`和预训练来源元数据；完整十万条目录、专利逐项转录及大型结果按 [`docs/resources.md`](../docs/resources.md)另行管理。本页出现的“随包”指原正式代码包。

获取日期使用原运行记录，均以文件中的UTC/+08:00时间为准；“保存mtime”与真实服务器查询时间分开。本文记录数据来源、处理方式及使用范围。

| 数据 | 版本/获取时间 | 用途与随包位置 | 来源与许可 |
|---|---|---|---|
| GPR84活动、实验、分子、文献和来源记录 | ChEMBL 37；2026-09-03 | `source/*.jsonl`：2070条活动、168实验、1033分子、20文献、3来源；训练前逐条审计 | [ChEMBL官方API](https://www.ebi.ac.uk/chembl/api/data/)，CC BY-SA 3.0；原请求范围见`snapshot_manifest.json` |
| 测量审计与准入 | 原审计2026-09-03；重构准入2026-10-01 | `source/activity_audit.parquet`、`training/`：285测量、277单位、248图 | ChEMBL派生记录；原ID/DOI及测量关系保留，遵循源数据CC BY-SA 3.0 |
| 目录筛选输入与已知靶点身份 | ChEMBL 37；2026-10-01 06:35:43 UTC起 | `catalogue/structures.csv`，100000条；当前GPR84身份951分子及2070活动，详见manifest | 同上；非随机目录子集 |
| 隔离身份与历史桥接 | 历史身份隔离2026-09-05；桥接2026-10-01 | `source/identity_quarantine.json`、`legacy_identity.csv`、`registry/`：不纳入50c/50d等冲突身份 | 本项目审计与ChEMBL派生；保留历史身份隔离范围 |
| CheMeleon预训练资源 | Zenodo固定记录15460715；2026-09-30 16:32:52 UTC起（北京时间10月1日） | `pretraining/`来源元数据；权重在`models/pretrained/` | [固定模型记录](https://doi.org/10.5281/zenodo.15460715)，MIT声明与原许可随包保留 |
| 预训练作者语料 | PubChem 100万个分子、1613项Mordred描述符 | 作者预训练背景；本项目没有复制或重跑此语料 | [CheMeleon正式论文](https://doi.org/10.1021/acs.jcim.6c01546)；本项目获取的是预训练权重，不是训练全集；语料逐项身份重叠未核清 |
| 外部功能实验 | WO2024028364A1 Table10；2026-10-03转录、核对及评分 | `external/`：61项转录、准入、SDF、历史排重；`results/external/`分数和指标 | [公开专利](https://patents.google.com/patent/WO2024028364A1/en)；仅保存本项目结构/实验事实转录与分析，不随包复制专利PDF；单独再发布许可尚未确认 |
| 骨架参照 | 2026-10-02已取得：训练277、ChEMBL951、BindingDB592、历史ChEMBL2070行 | `results/scaffold/`行级结构参照、范围比较与核验；各范围可能重叠 | ChEMBL遵循上述许可；BindingDB来源与原记录见参照表及下载记录，单独再发布许可尚未确认，不能统称实测拮抗剂 |
| 受体结构 | PDB 9WIJ，既有准备来源2026-09-03/09-08；候选姿势2026-10-02 | `results/structures/`与`structure_evidence/`，处理后的PDB和原配体身份SDF | [RCSB 9WIJ](https://www.rcsb.org/structure/9WIJ)；原始坐标来源与处理记录保留，不复制论文全文 |
| 既有MD与派生分析 | 2026-10-02–03；9×5ns，4500帧 | `results/md_*`：位移、接触、氢键、波动、聚类、当前角度范围、代表SDF、核验 | 本项目计算派生证据；不混同实测；原膜系统和轨迹在本地历史产物中，未纳入主入口依赖 |

## 准入和身份口径

准入：确认人源GPR84；功能实验、IC50、nM、明确拮抗语境，置信度≥9；标准化标志合格，非潜在重复，无有效性警告，排除分档样式实验；来源/assay/结构完整，隔离身份继续排除。点值和删失区间只有整个区间位于1 µM阈值同侧才赋类；缺失/未测不作阴性。全部排除理由在`training/exclusions.csv`，同实验图冲突在`training/conflicts.csv`。

同assay同标准化有立体图的同类测量合并成一个计权单位，保留全部测量ID；跨阈值矛盾不取中位标签。跨实验相同图按图重复次数降权。285是测量数、277是实验内单位、248是输入图，不能互换。

文献留出：六篇逐篇留出，排除训练侧与测试侧相同的无立体连接结构，包括连接图相同而已编码构型不同的记录。留出表完整保存三个角色。内部留出评价用于方法开发。

目录：按字符串ID升序，100000条公开记录，经已知/隔离ID和无立体连接图排除、结构范围、规范有立体图去重，实际82883图。该大小/元素范围来自训练图，不是已验证适用域。

外部：A≤0.3µM、0.3<B≤1µM记强效；1<C≤3µM记较弱，按区间标签分类。61项均在原范围；I-244/I-245为不同构型，主分析保留61样品，敏感性按编号固定合并连接图为60项。外部宽身份检查还去同位素并规范互变异构体，不能与训练连接图规则混写。相对277训练单位、2070已见ChEMBL、592 BindingDB及142历史作者结构补查均无宽身份匹配。主评分前固定既有模型与规则，标签已阅读，独立性范围见模型说明。

## 获取和重放

主流程已包含冻结输入，不需要联网再下载。原API/源码获取命令和URL随manifest保留；`scripts/fetch_screening_catalogue.py --help`与`fetch_chemeleon_weights.py --help`提供相同获取方式，默认须新目录。数据库更新后的下载属于新的数据版本；完整结果核验优先使用本包ChEMBL37快照。

原始ChEMBL活动获取API、目标ID和版本在`snapshot_manifest.json`；所有实体原JSONL已提供。`govern.py --from-raw`使用原审计配置和扩展隔离策略回放，并逐项对比正式训练单位。数据回放核验见`logs/data_replay_v1/summary.json`。

目录请求的原分页响应按`catalogue/manifest.json`中的`raw_file`命名；随包归档范围见`catalogue/raw_responses/`及整理清单。代码读取的是保持原内容的`structures.csv`，不以重新下载的当前版本替代。

专利文件页244–245为实验方法、249为分档、250–251为活动表；manual JSON和准入CSV保留样品编号、页码、SMILES与核对状态，可按公开专利原图再次核查。原PDF不随包复制；若需要全文，使用上述公开入口。

## 许可与发布范围

ChEMBL归属EBI及原文献作者，保留CC BY-SA 3.0：[许可全文](https://creativecommons.org/licenses/by-sa/3.0/legalcode)。原始数据库许可不等于全部代码或模型权重的统一授权。CheMeleon/Chemprop许可在`licenses/`；其他依赖按各自条款。

2026-10-03核对[BindingDB官方许可说明](https://www.bindingdb.org/rwd/bind/info.jsp)：其自行整理的数据标注CC BY 3.0，导入ChEMBL的数据保留CC BY-SA 3.0。本包592条结构参照保留的是网页行号与结构，尚未逐行确认所属来源类别，因此不统一改写其许可。专利手工转录派生材料的单独再发布许可状态保持原记录。

[RCSB官方政策](https://www.rcsb.org/pages/policies)将PDB档案数据列为CC0 1.0；本包9WIJ派生坐标保留PDB编号、结构来源和处理说明。CheMeleon使用[Zenodo固定v2记录15460715](https://zenodo.org/records/15460715)及随包原MIT许可。上述许可各自对应源资源，不扩展为全部代码、权重和数据的统一授权。本项目自有代码与文档采用[MIT许可](../LICENSE)；数据、权重和第三方材料仍分别遵守其适用条款。
