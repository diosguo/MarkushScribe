# M1 进度与后续待办（交接文档）

> 用途：本文件面向「下一个会话 / 压缩上下文后的自己 / 协作者」。读完本文件即可知道
> **已经做到哪、证据是什么、接下来按什么顺序做什么、每步怎么验收**。
> 相关文档：`M1方案.md`（M1 实施蓝本 + §16 实施记录）、`方案.md`（总体设计 §13 训练阶段、§18 评测、§20 里程碑）、`AGENTS.md`（仓库约定）。
> 最后更新：2026-09（M1 已实现；Stage 1 数据 + 评测 runner + MolScribe 基线完成，154 tests 通过）。

---

## 0. TL;DR

- **M0（数据闭环）已完成**：schema/validator、split、几何+像素增强、渲染 shard、overlay、QA 工具。
- **M1（显式 NODE AR baseline）已完成并验证**：
  - 真实 shard 上单 batch 过拟合 `34.98 → 0.0013`，且重建图 **node_f1=1.00 / edge_f1=1.00 / graph_exact=1**。
  - `--debug` CPU 训练可复现（两次运行除耗时外逐项一致）。
  - `150 passed` + `ruff check` clean。
- **M1 唯一未完成的退出标准**：退出标准 3「oracle edge F1 明显优于 SMILES 直接扩词表基线」——需要先建 Stage 1 普通 OCSR 语料并在 GPU 上正式训练。
- 下一步按本文件 **§6 P0 清单**顺序推进即可。

---

## 1. 环境与运行方式

仓库根：`/home/xuyang/Documents/Projects/MarkushScribe`。独立虚拟环境 `.venv`（**不要**用 `../MarkushRender/.venv`）：

```bash
# 已创建；如需重建：
python3 -m venv .venv
.venv/bin/pip install -U pip
# 注意：download.pytorch.org 在本机不可达（SSL EOF），用 SJTU CPU 镜像：
.venv/bin/pip install --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cpu torch
.venv/bin/pip install --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cpu --force-reinstall --no-deps torchvision
.venv/bin/pip install timm numpy pillow rdkit pyyaml tensorboardX pytest ruff
.venv/bin/pip install -e ../MarkushRender
.venv/bin/pip install -e . --no-deps     # 让 tools/*.py 与 python -m 能直接 import markushscribe
# MolScribe 基线所需（旧依赖，见 pyproject [baseline]）：
.venv/bin/pip install pandas matplotlib opencv-python-headless "albumentations==1.1.0" \
    "SmilesPE==0.0.3" "OpenNMT-py==2.2.0"
```

`pyproject.toml` 的 `ruff extend-exclude` 已排除 `MolScribe/data/ckpts/runs/*.egg-info/*.md`
（否则 `ruff check .` 会扫 `.venv` 与文档代码块；`*.md` 是因新版 ruff 会格式化 Markdown 代码块）。

已装版本：`torch==2.14.0+cpu`、`torchvision==0.29.0+cpu`、`timm==1.0.29`、`rdkit`、`pillow`、`markushrender==0.1.0`（editable）。

常用命令（统一 `.venv`）：

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/python -m pytest -q                 # 全量（含 slow overfit，约 80s）
.venv/bin/python -m pytest -q -m "not slow"   # 跳过训练慢测
```

`pyproject.toml` 已加 `ml` 可选依赖组与 pytest `slow` marker（行宽 100，ruff ignore E501）。

---

## 2. 已实现模块清单

### 2.1 M0（此前完成）

| 模块 | 作用 |
| --- | --- |
| `markushscribe/schema.py` | `markush_graph_v2` 冻结 dataclass + `parse_target`/`validate_target`/`to_dict` |
| `markushscribe/markushgen/` | SMILES → Markush IR（`abstract_molecule`、CLI） |
| `markushscribe/split.py` | canonical/scaffold 分组切分 + 泄漏检查 |
| `markushscribe/transforms.py` | 像素退化 + 几何 warp（`warp_target`）+ `ink_fraction` |
| `markushscribe/dataset.py` | `render_sample`/`build_dataset`/`validate_dataset`/`read_shard`/`RenderConfig` |
| `markushscribe/visualization/overlay.py` | `draw_overlay`/`contact_sheet`/`inspect_dataset` |
| `tools/{fetch_chembl_sample,build_markush_corpus,build_render_dataset,validate_dataset,inspect_dataset,check_split_leakage}.py` | 数据管线 CLI |

### 2.2 M1（本次完成）

| 模块 | 作用 |
| --- | --- |
| `markushscribe/constants.py` | 词表分组、kind/subtype 枚举、坐标 bin、闭集属性表、bucket 函数 |
| `markushscribe/tokenizer.py` | `Vocab`、`build_vocab`、`encode_target`、`decode_tokens`（抗截断） |
| `markushscribe/chemistry/mol_to_ir.py` | RDKit Mol → Markush IR（Stage 1 用） |
| `markushscribe/chemistry/graph_to_smiles.py` | v2 图 → RDKit → canonical SMILES（基线比对/后续 canonical 匹配用） |
| `markushscribe/torch_data.py` | `TargetDataset`、`build_sample`、`collate_fn`、`normalize_image` |
| `markushscribe/model/` | `encoder`（timm 多尺度）、`node_ar`、`node_heads`、`label_decoder`（ROI crop）、`edge_heads`（因子化）、`model`、`init`（权重加载） |
| `markushscribe/losses/` | `common`（掩码 loss）、`node`、`label`、`edge` + `compute_losses` 加权汇总 |
| `markushscribe/matching/` | `hungarian`（无 SciPy 依赖）、`costs`（script-aware 编辑距离） |
| `markushscribe/evaluation/` | `labels`、`edges`（oracle & e2e）、`metrics`（node/label/edge/graph_exact、`evaluate_dataset`） |
| `markushscribe/inference/` | `decode`（贪心）、`assemble`（组装 + 语义解析）、`confidence` |
| `markushscribe/training/` | `config`（YAML）、`checkpoint`、`curriculum`（teacher forcing）、`trainer` |
| `markushscribe/train.py` / `predict.py` / `evaluate.py` | 训练 / 预测 / 评测 CLI |
| `tools/build_plain_corpus.py` / `tools/inspect_predictions.py` | Stage 1 语料 / 预测 overlay |
| `tools/verify_overfit.py` | 真实 shard 上小规模过拟合收敛 + 重建评测（`--limit/--steps/--eval`） |
| `tools/run_molscribe_baseline.py` | MolScribe 预训练模型作 SMILES 基线（valid/exact/tanimoto，写 summary + per-sample） |
| `configs/{model_node_ar,train_node_ar,train_node_ar_debug,train_node_ar_stage1}.yaml` | 配置 |
| `tests/*`（12 个新文件 + `conftest.py`） | 154 tests |

---

## 3. 数据资产现状

| 路径 | 内容 | 规模 |
| --- | --- | --- |
| `data/chembl_sample.smi` | ChEMBL 抽样 SMILES | 2000 |
| `data/markush_ir/` | `markushgen` 生成的 Markush IR JSONL | 8000 条 |
| `data/rendered/{train,valid,test}/shard_*.tar` | 渲染 + v2 真值 + 增强的 shard（PNG+JSON） | 296 样本（raw split 180/60/60，QA 淘汰 4） |
| `data/rendered/splits.json` | group→split 映射、ratios、counts | — |
| `data/inspect/` | overlay 抽检图 + contact sheet | 296 overlay + 4 sheet |
| `data/plain_ir/` | Stage 1 普通 OCSR IR（`tools/build_plain_corpus.py`） | 2000 条（1 shard，0 失败） |
| `data/plain_rendered/{train,valid,test}/shard_*.tar` | Stage 1 渲染 shard（3 样式 + 增强） | 2000（train 1960 / valid 20 / test 20，0 失败/空白/重叠） |
| `data/baseline_{plain,markush}/` | MolScribe 基线 `summary.json` + `per_sample.jsonl` | valid 各 20 条 |

Shard 格式：tar 内成对 `<uid>.png` + `<uid>.json`（`markush_graph_v2` target）。目标节点按阅读序、id 从 0 连续。

---

## 4. 已完成里程碑与验证证据

### M0 验收（摘要）
- `validate_dataset --strict`：valid 296，invalid/unpaired/blank 全 0；跨 split group 泄漏 0。
- overlay 抽检：anchor 中位距 0px；近空白图 3 张、重叠 1 张已在数据侧淘汰。

### M1 验收（含本轮加强）
- **退出标准 1（单 batch 过拟合 + 重建一致）**：
  - 小样本（2 条，各 31 节点，真实 `data/rendered/train` 首个 shard）：800 步 loss `34.98 → 0.0013`；
  - 略大规模（**10 条**，18–42 节点）：343 步 loss `36.08 → 0.0100`（early stop），逐条
    `node_f1=edge_f1=order=aromatic=depiction=1.00`，**`graph_exact = 10/10`**；
  - 复现：`python tools/verify_overfit.py --limit 10 --steps 800 --tolerance 1e-2 --eval`（CPU 约 8 分钟）。
- **退出标准 2（debug 可复现）**：两次 `--debug` 训练 `history.json` 除 `seconds` 外逐项一致。
- **测试**：`154 passed`；`ruff check .` / `ruff format --check .` clean。
- 本轮修复：`assemble_graph` 邻接形状处理（`[N,N,2]` logits / `[N,N]` 概率）+ 接入 aromatic 头；
  回归测试 `tests/test_inference.py::test_assemble_graph_accepts_edge_logits_and_aromatic`。
- **退出标准 3（优于 SMILES 扩词表基线）**：基线 harness 已完成并出数（MolScribe 预训练权重，离线运行）：
  | 数据（valid，各 20 条） | valid_rate | exact_match | mean_tanimoto |
  | --- | --- | --- | --- |
  | 普通 OCSR `data/plain_rendered` | 0.55 | 0.50 | 0.55 |
  | Markush `data/rendered` | 0.25 | **0.00** | 0.09 |
  即基线在普通 OCSR 上尚可，但在含 R/变量/占位的 Markush 上几乎完全失败。**NODE AR 侧的对比数字仍待
  正式训练后的 checkpoint**（当前只有 Stage 1 冒烟权重）。复现见 §9；结果在 `data/baseline_{plain,markush}/`。
- **Stage 1 全链路冒烟**：`data/plain_ir → data/plain_rendered → train（冻结 encoder）→ evaluate`
  已跑通；1 epoch/32 样本 train total `28.98`、valid `24.01`（45s），checkpoint `runs/node_ar_stage1_smoke/last.pt`。
  未充分训练时贪心解码会立刻输出 `<eos>`（`pred_nodes=0`），属训练不足而非流程问题（runner 已在单测中用随机模型验证能产出指标）。

复现命令：

```bash
# 单 batch 过拟合（CLI）
.venv/bin/python -m markushscribe.train --config configs/train_node_ar_debug.yaml --debug --overfit --steps 250
# debug 训练
.venv/bin/python -m markushscribe.train --config configs/train_node_ar_debug.yaml --debug
# 预测 + overlay 抽检
.venv/bin/python -m markushscribe.predict --ckpt runs/node_ar_debug/last.pt --image <png>
python tools/inspect_predictions.py --ckpt runs/node_ar_debug/last.pt \
    --shard data/rendered/valid/shard_00000.tar --out data/pred_inspect --limit 8
```

---

## 5. 已知缺口 / 技术债（G 编号）

| 编号 | 缺口 | 影响 |
| --- | --- | --- |
| G1 | ~~Stage 1 普通 OCSR 语料未生成~~ **已解决**：`data/plain_ir`(2000) + `data/plain_rendered`(2000) 已生成校验 | — |
| G2 | Stage 1 配置已加（`configs/train_node_ar_stage1.yaml`），但**尚未做正式（GPU）训练** | encoder 仍未真正适配 |
| G3 | ~~无「SMILES 直接扩词表」基线~~ **部分解决**：MolScribe harness + 数字已出；NODE AR 侧对比待训练 | 退出标准 3 差最后一列数字 |
| G4 | ~~无评测 runner CLI~~ **已解决**：`markushscribe/evaluate.py`（`--shard/--limit/--out/--strict-exact`） | — |
| G5 | teacher forcing curriculum 已接好但未在训练中启用/验证 | e2e 与 oracle 差距未知 |
| G6 | 边头缺沿线 `K=16` 视觉采样与法向窄带特征 | edge 语义上限受限（当前仅特征+几何） |
| G7 | label decoder 用双线性 ROI crop，非 ROIAlign；无对 memory 的逐节点交叉注意力 | 缩写/上下标识别的视觉信息有限 |
| G8 | MolScribe 预训练权重未真正下载/验证（`model/init.py` 为通用形状匹配加载器） | encoder 初始化收益未知 |
| G9 | 预测图：`parse_semantic` 为 best-effort；输出不含置信度；未保存 raw prediction；未做 schema 回环校验 | 生产可审计性不足 |
| G10 | 无 `scripts/`、无 `ckpts/` 管理、无 Stage1 配置 | 工程化不足 |
| G11 | 评测仅简单平均；无阈值扫描、无按 split/类别分层 | 达不到 §18 报告要求 |
| G12 | `graph_exact` semantic 模式忽略 depicted；kekule_order/ring_mode 未导出 | 评测口径待补齐 |
| G13 | `assemble_graph` 未写 `kekule_order`/`ring_mode` | aromatic 双目标导出不完整 |

---

## 6. 下一步任务清单

### P0 —— 收尾 M1（按顺序做）

- [x] **P0.1 构建 Stage 1 普通 OCSR 语料（G1）** —— 已完成（2000 条 IR，渲染 train 1960/valid 20/test 20，0 失败/空白/重叠；`validate_dataset --strict` 全绿）。
- [x] **P0.2 Stage 1 训练配置（G2）** —— `configs/train_node_ar_stage1.yaml` 已加（冻结 encoder、resnet18/256）；
  已跑 1 epoch/32 样本冒烟（train 28.98 / valid 24.01，45s）验证流程。**正式 GPU 训练仍待做**：
  `encoder_name` 换 Swin + `pretrained: true`、`image_size: 384`、多 epoch。

- [x] **P0.3 SMILES 直接扩词表基线（G3）** —— 采用「复用 MolScribe 预训练」方案：
  `tools/run_molscribe_baseline.py`（离线跑 `MolScribe/ckpts/swin_base_char_aux_1m680k.pth`，把预测 SMILES 与
  `graph_to_smiles(gold)` 做 valid/exact/tanimoto 比对）。已出数：普通 OCSR exact 0.50，Markush exact 0.00。
  **剩余**：等 NODE AR 正式训练后，用 `markushscribe.evaluate` 在同样 shard 上出 node/edge/graph 指标，
  与上表并列成最终对比（脚本与指标均已就绪）。

- [x] **P0.4 评测 runner CLI（G4）** —— `markushscribe/evaluate.py` 已实现：加载 ckpt + shard → 逐样本
  `predict_graph`/`evaluate_graph` → `evaluate_dataset` 聚合，输出 `summary.json` + `per_sample.jsonl`；
  `--max-len` 控制贪心解码上限（默认 `2+8*q_capacity`）；测试 `tests/test_evaluate_runner.py`。

- [ ] **P0.5 Stage 2 合成训练 + teacher forcing 课程（G5）**
  - 用 `configs/train_node_ar.yaml`（`data/rendered`）先跑一版；训练稳定后设 `teacher_forcing_epochs>0`，
    观察 validation node recall / anchor error 门控下 e2e 与 oracle 差距收窄。
  - 验收：报告 oracle vs e2e edge F1 差距；`node_recall` 不退化。
  - 备注：`training/curriculum.py` 已实现按 epoch 线性提升 predicted-node 概率。
  - **前置（建议先做）**：给 `Trainer` 加 AMP（autocast + GradScaler）与梯度累积，使 8–12 GB 卡可跑；
    显存需求见 §10。当前 trainer 为纯 fp32、无 AMP/累积/梯度检查点。

### P1 —— M1 加固 / M2 准备

- [ ] **P1.1 边头沿线视觉特征（G6）**：在 `model/edge_heads.py` 增加沿两 anchor 连线采样 `K=16` 的 pyramid
  特征（主线 + 法向窄带），`use_line_features` 开关 + 形状/性能测试。
- [ ] **P1.2 label decoder 强化（G7）**：ROIAlign（或可学习采样）替代固定双线性 crop；考虑对 memory 的
  逐节点交叉注意力；缩写/上下标专项测试。
- [ ] **P1.3 MolScribe 初始化实测（G8）**：下载 MolScribe checkpoint，核对 `state_dict` 命名并在
  `model/init.py` 做逐键映射，打印 loaded/missing/unexpected + 比例；测试阈值断言（防假加载）。
- [ ] **P1.4 推理输出加固（G9/G13）**：`assemble_graph` 补 `kekule_order`/`ring_mode`；`predict` 输出加
  节点/边置信度、保存 raw prediction；增加「预测图 → `schema.to_dict` → `parse_target`」回环测试
  （如不稳定，明确标注哪些字段可缺省）。
- [ ] **P1.5 工程化（G10）**：新增 `scripts/{train_stage1.sh,train_stage2.sh,evaluate.sh}`；`ckpts/`、`runs/`
  加入 `.gitignore`（若无 `.gitignore` 则创建）。
- [ ] **P1.6 评测分层（G11/G12）**：阈值扫描、按 split / 按样本类别（rich-label、placeholder、aromatic、
  variable/wedge）分层报告；补 `graph_exact` 的 depicted 口径。

### P2 —— 后续里程碑（详见 `方案.md §20`）

- **M2 Set-based 主模型**：learned queries + Hungarian matching 训练；node/anchor/label 多头；相对坐标 +
  沿线视觉特征边头；与 NODE AR 公平消融（同数据/同 encoder）。
  - 退出：相同数据下更高 Graph Exact，或同精度下长结构失败率更低。
- **M3 课程训练与完整合成域**：GT query→predicted query 课程、芳香双目标、variable、wedge direction、
  全样式/布局增强、普通 OCSR replay。
  - 退出：e2e edge F1 与 oracle edge F1 差距显著收窄；普通 OCSR graph exact 无可接受退化。
- **M4 真实专利微调**：family-disjoint 真实集、来源均衡微调、错误分析与置信度校准、可选 CXSMILES。
  - 退出：真实集达到本文验收门槛，合成→真实差距有分层解释。
- **M5 生产化**：稳定批量 API、模型卡、可复现实验配置、性能/显存/延迟基准、数据与 ckpt 版本管理、
  为未来 OCR/VTL 定义不破坏 v2 主图的扩展接口。

---

## 7. 关键约定与决策（勿遗忘）

- **坐标系**：归一化 `[0,1]`，左上原点，y 向下；各向异性 resize 到 384 不改归一化坐标。
- **节点顺序**：target 按阅读序、id 连续；dataset 张量下标 == target id == graph.nodes 顺序。
- **节点 embedding gather**：当前在 `<node>` 起始位置 gather（使 kind/subtype/坐标头无泄漏）；
  节点内容经 label decoder 的 `label_summary` 汇入 `node_features` 供边头使用。
- **容量**：`q_capacity/t_capacity/l_capacity` 超限标记 `capacity_exceeded`，训练器过滤而非静默截断。
- **词表**：`BASE_TOKENS` = special + script + kind + subtype + x/y bin；字符 token 从数据统计附加；
  训练/推理必须用同一 `vocab.json`（checkpoint 内已存 `vocab.tokens`）。
- **只读边界**：`MolScribe/` 只读参考，不修改；`../MarkushRender` 可改但改动需同步其 docs+tests（本 M1 周期内未改）。
- **提交**：除非明确要求，否则不 `git commit`。当前仓库非 git 仓库（无 `.git`）。

---

## 8. 风险 / 注意事项

- **网络**：`download.pytorch.org` 不可达，装 torch 必须用 SJTU/aliyun CPU 镜像（见 §1）。
- **markushrender 以 editable 安装**：修改 `../MarkushRender` 会即时影响本仓库测试。
- **CPU 限制**：所有模型能在 CPU 前向；正式训练建议 GPU（Swin + 384 + batch 2–8）。
- **长序列/大图**：NODE AR 依赖「何时发节点」的学习，隐式 C 仍是最大不确定点（见 `方案.md §21` 风险表）；
  必要时加节点中心 heatmap 辅助头。
- **预测图 schema 合法性**：`assemble_graph` 目前直接构造 dataclass；`parse_semantic` 为 best-effort，
  在放开「预测图必须通过 `schema.parse_target`」前，先做 P1.4 的回环测试。
- **评测口径**：oracle 与 e2e 必须分开报告；不要用化学后处理掩盖 raw graph 错误（`方案.md §21`）。

---

## 9. 一页命令速查

```bash
# 测试 / 静态检查
.venv/bin/python -m pytest -q
.venv/bin/ruff check . && .venv/bin/ruff format --check .

# 数据（M0 已有）
.venv/bin/python tools/validate_dataset.py data/rendered --strict
python tools/inspect_dataset.py data/rendered --out data/inspect --split train --save-overlays

# Stage 1 数据（P0.1）
.venv/bin/python tools/build_plain_corpus.py --smi data/chembl_sample.smi --out data/plain_ir
.venv/bin/python tools/build_render_dataset.py --ir data/plain_ir/shard_*.jsonl \
    --out data/plain_rendered --preset-choices patent_bw acs chemdraw --drop-overlaps
.venv/bin/python tools/validate_dataset.py data/plain_rendered --strict

# 训练 / 预测 / 评测
.venv/bin/python -m markushscribe.train --config configs/train_node_ar_debug.yaml --debug
.venv/bin/python -m markushscribe.train --config configs/train_node_ar_stage1.yaml --limit 32 --epochs 1 --batch-size 4
.venv/bin/python -m markushscribe.predict --ckpt runs/node_ar_debug/last.pt --image <png>
.venv/bin/python -m markushscribe.evaluate --ckpt <ckpt> \
    --shard data/plain_rendered/valid/shard_*.tar --limit 20 --max-len 128 --out data/eval_stage1
python tools/inspect_predictions.py --ckpt <ckpt> --shard <shard.tar> --out data/pred_inspect --limit 8
python tools/verify_overfit.py --limit 10 --steps 800 --tolerance 1e-2 --eval

# MolScribe SMILES 基线（普通 OCSR / Markush）
python tools/run_molscribe_baseline.py --shard data/plain_rendered/valid/shard_*.tar \
    --limit 20 --out data/baseline_plain
python tools/run_molscribe_baseline.py --shard data/rendered/valid/shard_*.tar \
    --limit 20 --out data/baseline_markush
```

---

## 10. GPU 训练：显存需求与准备清单

> 估算口径：当前 `Trainer` 为 **fp32 + AdamW，无 AMP / 无梯度累积 / 无梯度检查点**。下面数字为估计值
> （参数已实测，激活为经验估计），正式采购/租用前应在目标卡上实测一次峰值。

### 10.1 参数量（实测）

| 配置 | 总参数 | 可训练 | 其中 encoder |
| --- | --- | --- | --- |
| Stage1 resnet18 / dim128 / 256px / 冻结 encoder | 14.3M | 3.1M | 11.3M（可训 0.1M） |
| Stage2 resnet18 / dim256 / 384px | 22.5M | 22.5M | 11.4M |
| Swin-T / dim256 / 384px | 38.9M | 38.9M | 27.9M |
| Swin-B / dim256 / 384px | 98.3M | 98.3M | 87.2M |

### 10.2 显存需求（估计，含参数+梯度+优化器状态+激活）

| 训练阶段 | batch | 预估峰值 |
| --- | --- | --- |
| Stage1（冻结 resnet18，256px，dim128） | 8 | ~4–6 GB |
| Stage2（resnet18，384px，dim256） | 4–8 | ~8–12 GB |
| Stage2（Swin-T，384px，dim256） | 8 | ~10–16 GB |
| Stage2（Swin-B，384px，dim256） | 8 | ~20–28 GB |

- **推荐**：24 GB（3090 / 4090 / A5000）最省心；**够用**：12–16 GB（3060 12G / 4070 / 4080）跑 Swin-T；
  **8 GB** 仅 resnet18 + 小 batch + 梯度累积。
- 激活是主要开销，随 `image_size²` 与 `batch` 增长；`l_capacity=2048 / max_len=4096` 是长图最坏情形。

### 10.3 降显存手段

1. 冻结 encoder（Stage1 已用，省最多）。
2. **AMP（autocast + GradScaler）**——当前未实现，建议 GPU 训练前先加。
3. **梯度累积**（等效大 batch）——当前未实现。
4. 降低 `image_size`（384→256）或对大图分块。
5. encoder 梯度检查点（timm `features_only`，需改造 `Encoder`）。
6. 降低 `q_capacity / l_capacity`，或对超大图裁剪。

### 10.4 GPU 训练前 checklist

- [ ] 给 `Trainer.fit` 加 AMP + 梯度累积（`--amp`/`--accum` 或 config 字段），CPU 上保持 fp32。
- [ ] 在目标卡上跑 1 个 epoch 记录峰值显存，据此定 batch/AMP。
- [ ] Stage1：`configs/train_node_ar_stage1.yaml` 换 Swin-T、`pretrained: true`、`image_size: 384`、解冻顶层。
- [ ] Stage2：`configs/train_node_ar.yaml` 正式训练；稳定后开 `teacher_forcing_epochs>0`。
- [ ] 用 `markushscribe.evaluate` 在 `data/rendered/valid` 出 NODE AR 指标，与 §4 的 MolScribe 基线并列。
