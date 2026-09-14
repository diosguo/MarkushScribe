# AGENTS.md

本文件为 AI 编码助手（opencode 等）在本仓库工作时的指导说明。修改代码前请先阅读本文件。

## 1. 项目概述

**MarkushScribe** 是一个 **Markush 结构解析（Markush structure recognition）** 工具：输入一张化学结构图（通常来自专利、论文），输出其可机读的 Markush 结构表示。

- 语言：**Python 3.10+**
- 思路参考：[MolScribe](https://github.com/thomas0809/MolScribe)（图像 → 分子图 → SMILES 的 image-to-graph 模型）
- 与普通 OCSR 的核心区别：Markush 结构包含**可变部分**（R 基团、变量原子/基团如 `X`、`Y`、`Ar`、`halogen`，可变取代基，环大小/连接可变等），不能简单用一条确定 SMILES 表示。

### 1.1 目标产物

给定结构图，输出：

1. 识别到的原子/节点（含 R 基团标签、缩写/超原子、变量节点）及其归一化坐标；
2. 识别到的化学键（含键型、楔形键、可变连接）；
3. 组装后的 Markush 图表示（见 §3.2），并可导出为：
   - RDKit `RWMol`（用 dummy atom / R-group label 表示可变点）；
   - Markush 扩展 SMILES / CXSMILES；
   - 结构化 JSON（节点、边、R 基团定义）。

### 1.2 参考实现

仓库内 `MolScribe/` 是只读的上游参考实现，**不要修改**（除非明确要求）。关键文件：

| 文件 | 作用 |
| --- | --- |
| `molscribe/interface.py` | 推理入口 `MolScribe.predict_image_file`，返回 `smiles/molfile/atoms/bonds/confidence` |
| `molscribe/model.py` | `Encoder`（timm Swin/ResNet/EfficientNet）、`Decoder`（TransformerAR + GraphPredictor 边预测） |
| `molscribe/tokenizer.py` | 字符/原子 tokenizer，含 `<sos>/<eos>/<pad>/<unk>/<mask>` |
| `molscribe/chemistry.py` | 图→SMILES、原子/坐标处理，含 R 基团相关工具 |
| `molscribe/constants.py` | `RGROUP_SYMBOLS`、`SUBSTITUTIONS`、`ABBREVIATIONS`、元素表 |
| `molscribe/dataset.py` | 数据加载与增强，基于 Indigo 动态渲染 |
| `molscribe/indigo/` | 内置修改版 Indigo，用于渲染结构图 |
| `train.py` / `evaluate.py` / `predict.py` / `web.py` | 训练、评测、预测、Web demo |

## 2. 规划目录结构

新代码不要放进 `MolScribe/`，与参考实现解耦。建议结构：

```
MarkushScribe/
├── AGENTS.md
├── README.md
├── pyproject.toml            # 依赖与工具配置（优先用 pyproject，而非 setup.py）
├── requirements.txt          # 可保留，与 pyproject 保持一致
├── MolScribe/                # 只读参考实现
├── markushscribe/            # 主 Python 包
│   ├── __init__.py
│   ├── interface.py          # 对外推理 API（对应 MolScribe 的 interface.py）
│   ├── model.py              # Encoder / Decoder / 图预测头
│   ├── tokenizer.py          # 含 R 基团与变量 token 的 tokenizer
│   ├── chemistry.py          # Markush 图构建、RWMol/CSMILES 导出
│   ├── constants.py          # 元素、R 基团符号、可变基团词表
│   ├── dataset.py            # 数据集与增强
│   ├── train.py              # 训练循环
│   ├── evaluate.py           # 评测
│   └── inference/            # 解码策略（greedy/beam）
├── scripts/                  # 训练 / 评测 shell 脚本
├── configs/                  # 实验配置（YAML）
├── data/                     # 数据集（不提交大文件）
├── ckpts/                    # 模型权重（不提交）
├── tests/                    # pytest 单元测试
└── assets/                   # 示例图片、文档图
```

## 3. 技术方案

### 3.1 模型

沿用 MolScribe 的 image-to-graph 范式，但扩展以支持可变结构：

- **Encoder**：`timm` 的 Swin Transformer / ViT / ConvNeXt，输入 384×384。保持 encoder 可替换。
- **Decoder**：Transformer decoder，多头输出：
  - 节点 token 序列（元素、R 标签、缩写、变量标记）+ 坐标；
  - 边预测头（全连接对之间 7 类：无键/单/双/三/芳香/实楔/虚楔）；
  - （可选）节点属性头：可变/普通、变量类型、R 基团编号。
- **Token 词表**：在 MolScribe 词表基础上加入 Markush 专用 token，例如
  `R`, `R1`…`R12`, `X`, `Y`, `Z`, `Ar`, `Hal`, `[Rg]`, 以及可变键标记。词表定义集中在 `constants.py`。
- 权重可从 MolScribe 预训练 checkpoint 初始化（encoder/decoder 部分加载，词表扩展处新增 embedding 随机初始化）。

### 3.2 Markush 图表示（内部约定）

统一用一套内部数据结构，序列化与导出都基于它：

```python
{
  "nodes": [
    {"id": 0, "kind": "atom"|"rgroup"|"variable"|"abbreviation",
     "symbol": "C"|"R1"|"X"|"CO2Et", "x": 0.57, "y": 0.95,
     "variable_def": {...},          # 可变节点的取值范围（可选）
     "confidence": 0.97}
  ],
  "edges": [
    {"source": 0, "target": 1,
     "bond_type": "single"|"double"|"triple"|"aromatic"|"solid wedge"|"dashed wedge",
     "variable": false, "confidence": 0.99}
  ],
  "rgroup_definitions": {"R1": ["C1-C6 alkyl", "halogen", "..."]}
}
```

导出约定：

- RDKit：可变节点用 dummy atom `[*:n]`，R 标签映射到 attachment point；
- CXSMILES：使用 `|$...$|` R 基团块；
- 保留归一化坐标（相对图像宽高，左上为原点，y 向下），与 MolScribe 一致。

## 4. 环境与依赖

- Python 3.10+，建议独立虚拟环境。
- 核心依赖：`torch`, `torchvision`, `timm`, `numpy`, `pandas`, `opencv-python`, `rdkit`, `albumentations`, `huggingface-hub`, `transformers`, `tensorboardX`, `pytest`, `ruff`。
- 参考 MolScribe `requirements.txt` 的版本约束，但**新项目优先使用 `pyproject.toml`** 管理依赖。
- 数据/权重目录（`data/`, `ckpts/`）不入库，加入 `.gitignore`。

常用命令（待项目脚手架落地后按实际更新）：

```bash
# 安装（开发模式）
pip install -e ".[dev]"

# 1) 从 ChEMBL .smi 生成 Markush IR 分片（需要 ../MarkushRender 供渲染）
python tools/fetch_chembl_sample.py --out data/chembl_sample.smi --count 2000
python -m markushscribe.markushgen.cli --smi data/chembl_sample.smi --out data/markush_ir \
    --count-per-mol 8 --seed 0 --shard-size 20000

# 2) 渲染 + 真值 + 像素/几何增强 -> tar shard（bundle 内自带 split）
python tools/build_render_dataset.py --ir data/markush_ir/shard_*.jsonl \
    --out data/rendered --preset-choices patent_bw acs chemdraw --shard-size 1000 --drop-overlaps

# 3) 校验 shard 内每个 markush_graph_v2 目标
python tools/validate_dataset.py data/rendered --strict

# 4) overlay 抽检（anchor/边/注释回画到图上）
python tools/inspect_dataset.py data/rendered --out data/inspect --split train --save-overlays

# 5) 独立切分后检查跨 split 泄漏
python tools/check_split_leakage.py --split train=a.jsonl --split test=b.jsonl --key scaffold

# 6) Stage 1 普通 OCSR 语料（RDKit -> IR -> 渲染）
python tools/build_plain_corpus.py --smi data/chembl_sample.smi --out data/plain_ir
python tools/build_render_dataset.py --ir data/plain_ir/shard_*.jsonl --out data/plain_rendered \
    --preset-choices patent_bw acs chemdraw --shard-size 1000 --drop-overlaps
python tools/validate_dataset.py data/plain_rendered --strict

# 训练 / 预测 / 评测（M1 NODE AR baseline，见 M1方案.md）
.venv/bin/python -m markushscribe.train --config configs/train_node_ar_debug.yaml --debug
.venv/bin/python -m markushscribe.train --config configs/train_node_ar_debug.yaml --overfit
.venv/bin/python -m markushscribe.train --config configs/train_node_ar_stage1.yaml --limit 32 --epochs 1
.venv/bin/python -m markushscribe.predict --ckpt runs/node_ar_debug/last.pt --image assets/example.png
.venv/bin/python -m markushscribe.evaluate --ckpt <ckpt> \
    --shard data/plain_rendered/valid/shard_*.tar --limit 20 --max-len 128 --out data/eval_stage1
python tools/inspect_predictions.py --ckpt runs/node_ar_debug/last.pt \
    --shard data/rendered/valid/shard_00000.tar --out data/pred_inspect --limit 8
python tools/verify_overfit.py --limit 10 --steps 800 --tolerance 1e-2 --eval

# MolScribe SMILES 基线（需 pyproject [baseline] 依赖）
python tools/run_molscribe_baseline.py --shard data/plain_rendered/valid/shard_*.tar --limit 20 --out data/baseline_plain
python tools/run_molscribe_baseline.py --shard data/rendered/valid/shard_*.tar --limit 20 --out data/baseline_markush

# 测试 / 静态检查
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format .
```

## 5. 数据

- 输入：结构图（PNG/JPG，RGB）。
- 标注：Markush 图（§3.2）或其序列化形式；至少需要节点（含 R/变量标签）与边。
- 数据来源建议：
  - 由 RDKit / Indigo 从已知 Markush 结构**动态渲染**合成训练数据（参考 `molscribe/dataset.py` 的增强与渲染流程）；
  - 真实专利结构图 + 人工/半自动标注作评测集。
- 划分：`train / valid / test`，评测集须与训练集来源分离。
- 大文件不入 git；仓库内只保留少量示例与下载脚本。

## 6. 评测

主指标：

- **Graph exact match**：节点（含 R 标签/变量）+ 边完全一致（关键指标）。
- **Node / Edge F1**：节点符号、边类型分别统计。
- **Canonical Markush match**：导出表示规范化后的精确匹配。
- **R-group ID 准确率**：R 标签与编号识别正确率（Markush 特有）。

评测脚本须支持按上述维度输出汇总，并保留逐样本结果便于分析。

## 7. 编码规范

- 遵循 PEP 8；统一用 `ruff` 做 lint/format，行宽 100。
- 新增公共函数/类写类型注解与简短 docstring（英文）。
- **不要添加无关注释**；仅在逻辑不直观处解释“为什么”。
- 张量统一用 `torch`，图像处理优先 `cv2`/`albumentations`，化学操作统一 `rdkit`。
- 与 MolScribe 对齐的命名保持一致性（如 `predict_image`, `return_atoms_bonds`, `FORMAT_INFO`）。
- 禁止硬编码绝对路径、密钥、token；路径通过参数或配置传入。
- 修改 `MolScribe/` 前必须确认，默认视为只读。

## 8. 测试要求

- 每个新模块至少覆盖：输入/输出形状、边界情况（空图、单节点、无 R 基团、多 R 基团）。
- 化学导出（RWMol / CXSMILES）需做 round-trip 测试：`图 → 导出 → 解析 → 结构等价`。
- 训练相关改动需保证 CPU 上能跑通一个 `--debug` 小 batch。
- 提交前运行 `pytest -q` 与 `ruff check .`，确保通过。

## 9. 里程碑（建议）

1. **脚手架**：包结构、依赖、配置、CI、示例脚本。
2. **数据管线**：Markush 结构渲染 + 标注格式 + DataLoader（M0 已完成，见 `docs/m0_report.md`）。
3. **基线模型**：在 MolScribe 架构上扩展词表与图头，跑通训练。
   - 实施蓝本见 **`M1方案.md`**（NODE AR baseline：tokenizer / 张量契约 / 模型 / 损失 / 训练 / 评测 / T1–T7）。
   - M1 已实现：`markushscribe/{constants,tokenizer,torch_data,losses,matching,evaluation,inference,training,model,predict,train,evaluate}.py`、`chemistry/graph_to_smiles.py`、`configs/*`、`tools/{build_plain_corpus,inspect_predictions,verify_overfit,run_molscribe_baseline}.py`；154 tests 通过。
   - Stage 1 数据已生成：`data/plain_ir`(2000) + `data/plain_rendered`(2000)；`configs/train_node_ar_stage1.yaml` 已加（正式 GPU 训练待做）。
   - 进度/待办交接见 **`docs/m1_status_and_next_steps.md`**（Stage 1 语料 + 评测 runner 已完成；剩余 P0：Stage 1 GPU 正式训练、SMILES 基线对比）。
4. **推理接口**：`predict_image_file` 输出 §3.2 结构并导出 RWMol/CXSMILES。
5. **评测**：合成 + 真实评测集，报告指标。
6. **文档与 demo**：README、示例图片、可选 Web demo。

## 10. 给 Agent 的注意事项

- 动手前先阅读本文件与 `MolScribe/` 中相关实现，理解约定后再改代码。
- 保持改动聚焦，不要顺手重构无关文件；不修改 `MolScribe/`。
- 不确定的 Markush 表示/导出格式决策，先问用户，不要自行臆造标准。
- 完成工作后运行 `pytest` 和 `ruff`；若命令不存在，先确认项目实际工具链。
- 不要提交（commit）任何改动，除非用户明确要求。
