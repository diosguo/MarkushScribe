# MarkushScribe

**Markush 结构识别**：输入一张化学结构图（多来自专利 / 论文），输出可机读的 Markush 结构表示。

与普通 OCSR（光学化学结构识别）不同，Markush 结构包含**可变部分**：R 基团、变量原子/基团（`X`、`Y`、`Ar`、`halogen`）、可变取代基、可变环大小/连接等，无法用一条确定 SMILES 表示。本项目的目标是识别这些可变点、组装成一张 Markush 图，并导出为可继续处理的表示。

> Image-to-graph Markush structure recognition. Input: a rendered chemical structure image. Output: a normalized, machine-readable Markush graph.

## 目标产物

1. 识别到的原子/节点（元素、R 基团标签、缩写/超原子、变量节点）及归一化坐标；
2. 识别到的化学键（键型、楔形键、可变连接）；
3. 组装后的 Markush 图，并可导出为：RDKit `RWMol`（dummy atom / atom-map 表示可变点）、Markush 扩展 SMILES / CXSMILES、结构化 JSON。

## 当前进度

- **M0（契约与数据闭环）已完成**：`markush_graph_v2` schema/validator、分组切分 + 泄漏检查、几何与像素增强、渲染 shard、overlay 抽检工具。
- **M1（显式 NODE AR baseline）已实现并验证**：
  - 真实渲染 shard 上小规模过拟合：10 条样本 loss `36.08 → 0.0100`，重建 `graph_exact = 10/10`；
  - `--debug` CPU 训练可复现；
  - Stage 1 普通 OCSR 语料已生成（2000 条 IR / 渲染），MolScribe SMILES 基线已跑通。
- 尚未完成：GPU 上的 Stage 1/Stage 2 正式训练、teacher forcing 课程、与 NODE AR 的正式对比。

详细进度与待办见 [`docs/m1_status_and_next_steps.md`](docs/m1_status_and_next_steps.md)。

## 目录结构

```
markushscribe/          # 主 Python 包
  schema.py             # markush_graph_v2 数据结构与校验
  markushgen/           # SMILES -> Markush IR 抽象
  chemistry/            # mol_to_ir、graph_to_smiles
  tokenizer.py          # 含 R 基团/变量 token 的 tokenizer
  torch_data.py         # 张量契约与 Dataset
  model/                # Encoder / NODE AR / 节点头 / label decoder / 边头
  losses/ matching/     # 掩码损失、匈牙利匹配
  evaluation/           # node / label / edge / graph_exact 指标
  inference/            # 贪心解码、图组装、置信度
  training/             # 配置、checkpoint、teacher forcing 课程、训练循环
  train.py predict.py evaluate.py
tools/                  # 数据构建、校验、overlay、评测、基线脚本
configs/                # 实验配置（YAML）
tests/                  # pytest 单元测试
docs/                   # 设计与进度文档
examples/               # 示例图与真值
方案.md                 # 总体设计
M1方案.md               # M1 实施蓝本 + 实施记录
AGENTS.md               # 仓库约定（供协作者/AI 助手）
```

`MolScribe/` 为上游只读参考实现，**不入库**，需要时单独 clone（见下文）。`data/`、`runs/`、`ckpts/` 等大文件同样不入库。

## 环境准备

需要 Python 3.10+，建议独立虚拟环境。渲染与真值来自兄弟项目 **MarkushRender**（`../MarkushRender`）。

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip

# torch：若 download.pytorch.org 不可达，可用镜像
.venv/bin/pip install --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cpu torch
.venv/bin/pip install --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cpu \
    --force-reinstall --no-deps torchvision

.venv/bin/pip install timm numpy pillow rdkit pyyaml tensorboardX pytest ruff
.venv/bin/pip install -e ../MarkushRender   # 渲染 + v2 真值
.venv/bin/pip install -e . --no-deps        # 本包（tools/*.py 可直接运行）
```

MolScribe 基线所需的旧依赖（可选）：

```bash
.venv/bin/pip install pandas matplotlib opencv-python-headless \
    "albumentations==1.1.0" "SmilesPE==0.0.3" "OpenNMT-py==2.2.0"

# 上游参考实现与预训练权重（不入库）
git clone https://github.com/thomas0809/MolScribe.git MolScribe
wget -P MolScribe/ckpts \
    https://huggingface.co/yujieq/MolScribe/resolve/main/swin_base_char_aux_1m680k.pth
```

## 快速开始

### 1. 数据管线

```bash
# 从 ChEMBL 抽样 SMILES（需要网络）
python tools/fetch_chembl_sample.py --out data/chembl_sample.smi --count 2000

# SMILES -> Markush IR -> 渲染 + 增强 -> tar shard
python -m markushscribe.markushgen.cli --smi data/chembl_sample.smi --out data/markush_ir \
    --count-per-mol 8 --seed 0 --shard-size 20000
python tools/build_render_dataset.py --ir data/markush_ir/shard_*.jsonl \
    --out data/rendered --preset-choices patent_bw acs chemdraw --shard-size 1000 --drop-overlaps

# 校验 shard 内的 markush_graph_v2 目标
python tools/validate_dataset.py data/rendered --strict
python tools/inspect_dataset.py data/rendered --out data/inspect --split train --save-overlays

# Stage 1：普通 OCSR 语料（RDKit -> IR -> 渲染）
python tools/build_plain_corpus.py --smi data/chembl_sample.smi --out data/plain_ir
python tools/build_render_dataset.py --ir data/plain_ir/shard_*.jsonl \
    --out data/plain_rendered --preset-choices patent_bw acs chemdraw --drop-overlaps
python tools/validate_dataset.py data/plain_rendered --strict
```

### 2. 训练 / 预测 / 评测

```bash
# 冒烟：单 batch 过拟合 + debug 训练（CPU）
.venv/bin/python -m markushscribe.train --config configs/train_node_ar_debug.yaml --debug
.venv/bin/python -m markushscribe.train --config configs/train_node_ar_debug.yaml --overfit --steps 250

# Stage 1（冻结 encoder）/ Stage 2（合成 Markush）
.venv/bin/python -m markushscribe.train --config configs/train_node_ar_stage1.yaml
.venv/bin/python -m markushscribe.train --config configs/train_node_ar.yaml

# 预测单图 / 评测一个 split
.venv/bin/python -m markushscribe.predict --ckpt runs/node_ar_debug/last.pt --image examples/markush_example_01.png
.venv/bin/python -m markushscribe.evaluate --ckpt <ckpt> \
    --shard data/plain_rendered/valid/shard_*.tar --limit 20 --max-len 128 --out data/eval_stage1

# 小规模过拟合/收敛验证 + 预测 overlay 抽检
python tools/verify_overfit.py --limit 10 --steps 800 --tolerance 1e-2 --eval
python tools/inspect_predictions.py --ckpt <ckpt> --shard data/rendered/valid/shard_00000.tar --out data/pred_inspect --limit 8
```

### 3. SMILES 基线（MolScribe）

```bash
python tools/run_molscribe_baseline.py --shard data/plain_rendered/valid/shard_*.tar \
    --limit 20 --out data/baseline_plain
python tools/run_molscribe_baseline.py --shard data/rendered/valid/shard_*.tar \
    --limit 20 --out data/baseline_markush
```

## 数据格式

内部统一使用 `markush_graph_v2`，核心字段：

```json
{
  "format": "markush_graph_v2",
  "coordinate_system": "normalized [0,1], origin top-left, y increases downward",
  "nodes": [
    {"id": 0, "node_class": "atom|rgroup|variable|abbreviation|ring_placeholder",
     "anchor": {"x": 0.36, "y": 0.02},
     "visual_label": {"visible": true, "plain_text": "N", "runs": [{"text": "N", "script": "base"}]},
     "semantic_label": {"element": "N", "h_count": null, "charge": 0, "isotope": null, "aromatic": false}}
  ],
  "edges": [
    {"source": 0, "target": 1,
     "normalized": {"order": 3, "aromatic": false, "kekule_order": null},
     "depicted": {"type": "triple", "visible_order": 3}}
  ]
}
```

坐标统一为归一化 `[0,1]`（左上原点，y 向下）；节点按阅读序、id 从 0 连续。

## 评测指标

- **Graph exact match**（主指标）：节点（含 R 标签/变量）+ 边完全一致；
- **Node / Edge F1**：节点符号、边类型分别统计；
- **Label 指标**：plain/script/family/index/multiplicity 等精确率；
- **R-group ID 准确率**（Markush 特有）；
- **SMILES 基线**：valid / exact / Tanimoto。

评测脚本输出 `summary.json`（聚合）与 `per_sample.jsonl`（逐样本）。

## 测试与静态检查

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

## 相关文档

- [`方案.md`](方案.md) — 总体设计（schema、模型、训练阶段、评测、里程碑）
- [`M1方案.md`](M1方案.md) — M1 NODE AR baseline 实施蓝本与实施记录
- [`docs/m1_status_and_next_steps.md`](docs/m1_status_and_next_steps.md) — 进度、技术债与后续任务清单
- [`docs/m0_report.md`](docs/m0_report.md) — M0 数据闭环报告
- [`AGENTS.md`](AGENTS.md) — 仓库约定与常用命令

## 上游参考

- [MolScribe](https://github.com/thomas0809/MolScribe) — image-to-graph OCSR 参考实现（只读，单独 clone）。

## License

Apache-2.0，见 [`LICENSE`](LICENSE)。
