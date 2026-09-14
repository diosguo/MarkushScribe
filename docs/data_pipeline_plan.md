# MarkushScribe 数据构造与增强优化方案

> 状态：实施蓝本（v1）
> 依赖：`../MarkushRender`（渲染与 `markush_graph_v2` 真值）
> 关联：`方案.md` §6/§7/§8/§14/§18/§19
> 本文回答三件事：① 如何从 ChEMBL `.smi` 批量构造 Markush 原始结构；② 图像级退化增强放在哪；③ 样式多样性 + 目标字段缺失的完整补齐方案。

---

## 0. 结论速览

| 议题 | 结论 |
| --- | --- |
| 结构源构造 | 已实现 `markushscribe/markushgen/`：`SMILES → 随机骨架保留 + 分支/R/M/L/A-B/重复基团抽象 → MarkushRender IR JSONL`。已在 `pytest` 中与 MarkushRender 打通（渲染 + `prediction_target` + round-trip）。 |
| 图像级像素退化 | **放 MarkushScribe**（本项目 dataset pipeline，epoch 级在线增强）。MarkushRender 只保留"改几何"的增强（旋转/镜像/抖动/折链等），因为它拥有 `Transform` 与逐像素对齐保证，且要保持标定图逐字节一致。 |
| 图像级几何增强（透视/扫描倾斜） | **放 MarkushScribe**，用一个与 anchor/bin 同步的 warp 模块；也可后续下沉到 Render。 |
| 样式多样性 | Render 的 `StyleSampler` 只在 `patent_bw` 族内采样（`styles.py:342`）。方案：改为"预设族 + 族内参数"双层采样；过渡期在 MarkushScribe 侧用 `StylePool` 先兜住。 |
| 字段缺失 | `kekule_order`、`placeholder_shape`、CXSMILES、`graph_annotations` 按"能在 Render 端确定画法的放 Render，纯化学导出的放 MarkushScribe"分工补齐。 |
| 切分 | 先按 canonical/scaffold 切分，再在 split 内生成变体（`方案.md` §14.2），否则同结构跨 split 泄漏。 |

---

## 1. 数据源构造（已落地）

### 1.1 目标与输入

- 输入：ChEMBL 导出的 `.smi`（每行 `SMILES [id]`，200 万+ 分子）。
- 输出：MarkushRender 可直接消费的 **MarkushRender IR JSON**（`_meta + nodes + edges`），并带来源审计字段。
- 目标规模：单分子生成 `K` 个抽象变体（默认 4–16），2M 分子 → 千万级 Markush 结构。见 §1.6 的规模策略。

### 1.2 模块结构

```text
markushscribe/
└── markushgen/
    ├── __init__.py      # 对外导出 MarkushConfig / abstract_molecule / MarkushIR
    ├── ir.py            # IR 数据类 + JSON 序列化 + validate（与 markushrender.ir 对齐）
    ├── config.py        # MarkushConfig：所有随机化的概率与上限
    ├── molecule.py      # RDKit：解析/最大片段/Murcko/环系/连通分量/核心选择
    ├── abstract.py      # 核心抽象算法
    └── cli.py           # .smi → 分片 JSONL（+ 可选单文件 IR）
tools/build_markush_corpus.py   # CLI 薄封装
tests/test_markushgen.py        # 12 个用例，含 Render 集成
```

### 1.3 抽象算法（`abstract.py::abstract_molecule`）

对每个分子：

1. **核心选择** `select_core`：`random_bfs`（默认）从随机环原子 BFS，保留 `core_retention=0.6` 的连通核心；`murcko` 保留环 + 环间 linker。核心必须连通且有 `min_core_atoms`。
2. **分支分解**：`removed = all - core`，按连通分量分解。每个只与核心有 **1 条**连接的分量是一个分支；连接数 ≠1 的分量整体保留（避免破坏拓扑）。
3. **裁剪深度**：对分支有二选一：
   - `whole_branch`（整枝 → 一个 R，默认概率 0.5；含环分支强制整枝，避免把环切断）；
   - `keep_length`（保留从连接点起 `k∈{0,0,1,2}` 个原子，其余末端 → R）。这同时覆盖了"末端取代基→R"和"整枝→R"两种诉求。
4. **R 基团**：每个裁剪前沿边生成一个 `rgroup` 节点（`R1..R12`），`p_multiplicity` 概率带 `multiplicity ∈ {m,n,x,y,p,q}` → 渲染为 `(R7)m`。
5. **重复基团**：整枝且移除片段是"同元素单链"时，`p_repeat_chain` 概率折叠成一个带多重度的变量/`rgroup`，即 `(L)n` / `(R)n`。
6. **M 变量**：环上保留原子以 `p_ring_atom_m` 替换为 `variable + variable_type=ring_atom`（`M1..M2`），保留全部键级。
7. **L 变量**：非环、度为 2 的 linker 原子以 `p_linker_l` 替换为 `variable + variable_type=linker`（`L1..L2`）。
8. **A/B 占位环**：对"全部原子都在核心、3–8 元、至少一条外向键"的环系，`p_ring_placeholder` 替换为 `variable + variable_type=ring`（`A/B/C…`），其外向键与 R 全部接到占位节点。
9. **可变键**：附着键以 `p_variable_bond` 画成 `wavy + variable=true`。
10. **H 数**：`new_h = max(0, rdkit_total_h + Σ(order-1))`，保证抽象后杂原子仍能画出 `NH/OH` 等。
11. **审计元数据**：`_meta` 写入 `source_smiles / canonical_smiles / scaffold_smiles / seed / core_policy / node_kind_counts / abstraction_ratio / rgroup_fragments`。`rgroup_fragments` 是每个 R 对应的原始片段 SMILES，**为后续 `variable_def` / 图到图枚举保留真值**。

坐标一律不写，交给 MarkushRender 的 CoordGen 布局（`engine=auto` 时缺坐标即重算）。

### 1.4 CLI 用法

```bash
python -m markushscribe.markushgen.cli \
    --smi data/chembl_2m.smi \
    --out data/markush_ir \
    --count-per-mol 8 \
    --core-policy random_bfs --core-retention 0.6 \
    --seed 0 --shard-size 20000 \
    --emit-single
```

产物：
- `data/markush_ir/shard_00000.jsonl`：每行一个 IR（`_meta + nodes + edges`）；
- `data/markush_ir/ir/<id>_<seed>.json`：可选单文件，便于直接喂 MarkushRender；
- `--start/--limit` 支持分片续跑。

### 1.5 输出契约示例

```json
{
  "_meta": {
    "generator": "markushscribe.markushgen",
    "source_smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "scaffold_smiles": "c1ccccc1",
    "seed": 1000003,
    "core_policy": "random_bfs", "core_retention": 0.6,
    "abstraction_ratio": 0.3846,
    "rgroup_fragments": {"R1": ["CC=O"], "R2": ["O"]},
    "node_kind_counts": {"atom": 8, "variable": 1, "rgroup": 3},
    "ir_version": 1
  },
  "nodes": [
    {"id": 0, "kind": "atom", "symbol": "O", "element": "O", "h_count": 1},
    {"id": 1, "kind": "variable", "symbol": "M1", "variable_type": "ring_atom"},
    {"id": 2, "kind": "rgroup", "symbol": "R1", "multiplicity": "m"}
  ],
  "edges": [
    {"source": 0, "target": 1, "bond_type": "aromatic"},
    {"source": 1, "target": 2, "bond_type": "single"}
  ],
  "rgroup_definitions": {"R1": ["CC=O"]}
}
```

注意：`element/h_count/charge/isotope` 显式给出，同时 `symbol` 也构造成可被 MarkushRender 重新解析的形式（`NH`、`CH3`、`13C`、`N+`），避免 `element` 字段被丢弃后把缩写误判成元素。

### 1.6 规模策略

- **去重与过滤**：先 `canonical_smiles` 去重；过滤金属/自由基/过小分子（`< min_core_atoms`）/超大分子（> 120 heavy atoms）。
- **K 变体**：每分子 `K` 个 seed；`K` 与 `core_retention` 组合即可覆盖"抽象度"谱系。建议 `K=8–16`，2M 分子 → 1600 万–3200 万个 Markush IR。
- **多进程**：CLI 目前单进程；建议按 `--start/--limit` 切片后 `xargs -P` 并行（rdkit 与 random 都是进程内独立）。
- **组合枚举（第二阶段）**：用 `rgroup_fragments` 把不同分子的骨架/取代基交叉重组（scaffold A + 分子 B 的片段 → 新 Markush），或用 `MarkushRender` 的变量枚举能力，突破"每分子只抽象自己"的上限。

### 1.7 已知限制

- 未处理立体化学（楔形键）。ChEMBL 的 `@/@@` 在抽象时被丢弃；若要监督楔形键，需要在抽象时用 RDKit 计算 2D 坐标并 wedging，且渲染必须用 `engine=given` 保证构型不变。见 §4.6。
- 抽象后少数"芳香环上换成 M/A"的结构会让 RDKit 布局报 `Can't kekulize mol`（仅 warning，渲染与标签正常，已实测 32/32 通过）。
- `variable_def` / `rgroup_definitions` 不参与模型目标；`rgroup_fragments` 只作为审计与后续扩展。

---

## 2. 图像级退化增强的归属决策

### 2.1 建议：像素退化放 MarkushScribe，几何增强留 MarkushRender

| 增强类型 | 举例 | 归属 | 理由 |
| --- | --- | --- | --- |
| 构图/几何（合法） | 旋转、镜像、坐标抖动、折链、布局根、E/Z 楔形补偿 | **Render** | 只有 Render 拥有最终 `Transform` 与"标签-像素严格对齐"保证；且已实现立体安全镜像/翻转（`augment.py:49`）。 |
| 栅格质量 | 低分辨率、关抗锯齿、最近邻放大 | **Render**（已有 `RasterStyle`） | 属于"怎么画"，且标定图要求默认逐字节不变。 |
| 纯像素退化 | 噪点、模糊、JPEG、阈值化、断墨、光照不均、扫描底色 | **MarkushScribe** | 不改任何几何/标签，纯训练期随机性；放训练管线可 **每个 epoch 重新采样**，样本多样性远高于预烘焙。 |
| 几何退化（像素空间） | 轻微透视、扫描倾斜、仿射/裁剪 | **MarkushScribe**（需同步目标） | 属于训练期增强；可通过 `warp_target()` 精确变换 anchor 与重算 `coord_target`，无需改 Render。 |

**为什么不把像素退化塞进 Render：**

1. Render 的黄金测试与"标定图逐字节一致"契约会被训练专用随机性污染；
2. 像素退化不需要 Render 的几何信息，放 Render 会让"渲染真源"承担训练框架职责；
3. 在线采样（按 epoch）比一次性生成固定退化图更省存储、多样性更高；
4. Render 作为独立仓库/工具，保持"确定性渲染器"边界更清晰（其 `DESIGN.md` §6.7 已定义"Render 管怎么画，MarkushScribe 管画什么"）。

### 2.2 MarkushScribe 侧增强模块设计

新增 `markushscribe/transforms.py`（图像 + 目标同步）：

```python
from dataclasses import dataclass

@dataclass
class ImageDegradeConfig:
    blur_prob: float = 0.3          # GaussianBlur / MotionBlur
    noise_prob: float = 0.4         # Gaussian / ISO / salt-pepper
    jpeg_prob: float = 0.4          # quality 30..90
    binarize_prob: float = 0.2      # adaptive threshold
    illumination_prob: float = 0.2  # 乘性光照梯度
    ink_dropout_prob: float = 0.15  # 细线腐蚀/随机抹除
    invert_prob: float = 0.0        # 黑底白线（少见）
    perspective_prob: float = 0.2   # 透视/倾斜
    affine_prob: float = 0.3        # 小角度旋转 + 平移 + 缩放
```

- **像素级**（blur/noise/jpeg/binarize/illumination/ink-drop）只改 PNG，不动 `prediction_target`。
- **几何级**（perspective/affine）必须调用同步函数：

```python
def warp_target(target: dict, matrix, image_size) -> dict:
    """用同一矩阵变换 anchor/coord_target 并重新量化到 64 bin；
    若变换后 anchor 出界，按 partial-object 规则删除该节点及其边或标记。"""
```

- 增强参数与 `(source_id, render_seed, epoch)` 绑定，保证可复现；不使用全局随机状态。
- 所有增强在 DataLoader worker 内执行（CPU），避免占用训练主进程。

### 2.3 与 Render 现有增强的衔接

- Render 已提供构图/样式多样性，MarkushScribe 不再重复做旋转/镜像。
- 调用顺序固定为：

```text
IR -> render_variant(ir, seed)     # Render：构图 + 样式 + 栅格质量
   -> prediction_target(result)    # Render：v2 标签
   -> pixel degrade (transforms)   # 本项目：像素退化
   -> optional warp + warp_target  # 本项目：几何退化（同步标签）
   -> schema/QA -> shard
```

---

## 3. 样式多样性优化

### 3.1 现状

`styles.py:333 StyleSampler.sample()` 在 `base != "patent_bw"` 时**直接返回固定预设**（`styles.py:342`），且 `AugmentConfig.sample_style` 只在 `base=="patent_bw"` 时采样（`augment.py:267`）。结果：合成的样式只覆盖黑白专利族，缺少 `acs/chemdraw/indigo/hand_drawn` 与元素配色。

### 3.2 推荐方案：双层采样（预设族 × 族内参数）

在 MarkushRender 的 `StyleSampler` 增加"预设族"维度：

```python
@dataclass
class StyleSampler:
    seed: int = 0
    base: str = "patent_bw"
    preset_choices: tuple[str, ...] = ("patent_bw", "acs", "chemdraw", "hand_drawn")
    def sample(self) -> Style:
        rng = np.random.default_rng(self.seed)
        base = str(rng.choice(self.preset_choices)) if self.base == "auto" else self.base
        style = get_preset(base)
        # 族内参数采样：字号/线宽/芳香画法/show_h/清晰度（现有逻辑）
        # acs 额外采样 color_mode=element + element_colors
        ...
```

- `AugmentConfig` 增加 `preset_choices` 透传；`render_variant` 不再限定 `patent_bw`。
- 元素配色只在 `acs` 族开启，键仍是单色（Render 不支持半键着色，见其 §9.1）。
- 用 `tools/augment_mix_grid.py` 做跨族联络表验收。

### 3.3 过渡方案（不改 Render）

在 MarkushScribe 侧建 `StylePool`：循环 base ∈ presets，对每个 base 用 `StyleSampler(seed, base=base).sample()`，再叠加 `--set` 级随机（字号、线宽、`show_all_carbon`、`show_h`、`margin`、`aromatic`）。缺点是 `acs/chemdraw` 的族内参数不随 seed 变化，多样性弱于 §3.2。

### 3.4 额外建议的样式轴

- `show_all_carbon` 有/无（影响是否出现显式 C 标签）；
- `show_h ∈ {none, terminal, hetero}`（现有已采样，建议加大比例跨度）；
- 画布长宽比/长边（`layout.image_size`、`target_long_side`）+ `margin`；
- 字体族只有 Liberation，可接受；如要 Arial/Helvetica 形态，Render 已将其映射到同一内置字体。

---

## 4. 目标字段缺失补齐

对照 `方案.md` §6/§7/§8，当前 `prediction_target` (`target.py:247`) 已覆盖绝大部分；以下 4 项需补。

### 4.1 `normalized.kekule_order`（§8.2）

现状：`_normalized_edge` 恒为 `kekule_order: None`（`target.py:224`），芳香双目标只完成一半。
方案：在 Render 的 `target.py` 为芳香边计算确定 Kekulé 分配：

```python
def _kekule_orders(ir, result) -> dict[frozenset[int], int]:
    # 用 RDKit 从 IR 重建 mol：atom->真实元素；rgroup/variable/placeholder->dummy；
    # 非芳香边按 bond_type；芳香边保留 aromatic。
    # Chem.Kekulize(mol, clearAromaticFlags=True)；逐键读 GetBondTypeAsDouble()。
    # 失败（变量环无法 kekulize）则整体返回 {}。
```

- 能确定时填 `kekule_order ∈ {1,2}`，否则保持 `null` 并 mask；`aromatic` 与 `depicted` 不变（严禁由 normalized 反推 depicted，§8.3）。
- 该字段进入 `normalized` 后，`target_to_ir` 也要能读回（有值则作为 `bond_type=double/single` 的可选 Kekulé 提示，默认仍用 aromatic）。
- 若暂不改 Render：MarkushScribe 侧可在拿到 `prediction_target` 后用**同一分子源**补，但节点到源原子的映射已丢失，因此**推荐在 Render 端做**。

### 4.2 `placeholder` shape / radius（§6.4）

现状：`prediction_target` 节点只有 `semantic_label.placeholder_id`，placeholder 的形状/半径只存在于 `result.graph`（`placeholder_radius`）。
方案：在 `target.py` 节点字典增加：

```python
if node.is_ring_placeholder:
    entry["placeholder"] = {
        "shape": result.style.ring.shape,   # circle | polygon | bracket
        "radius": result.style.ring.radius,
    }
```

- 主图契约仍只要求中心 anchor 与节点身份（§6.4），shape/radius 仅作辅助头。
- `_node_from_target` 的 round-trip 需容忍该字段（作为非预测信息，可忽略）。

### 4.3 CXSMILES 辅助导出（§17）

现状：`masks: {"cxsmiles": False}`（`target.py:359`），没有辅助信号。
方案：放 **MarkushScribe** `markushscribe/chemistry/cxsmiles.py`：

```python
def target_to_cxsmiles(target: dict) -> tuple[str | None, bool]:
    # 用 RDKit 构建 RWMol：
    #   atom -> 真实元素 + H/charge/isotope；
    #   rgroup -> dummy [*:n]，按 symbol index 设 isotope/atom map 作为 attachment point；
    #   variable -> 通用 dummy [*]（或按 family 用 map），ring_placeholder -> 单独的假原子/环；
    # 边按 normalized（芳香优先，否则 order）；
    # 输出 CXSMILES 的 R 基团块 |$...$|，失败返回 (None, False)。
```

- 成功的样本把 `masks.cxsmiles` 置 True 并写入 `auxiliary.cxsmiles`；失败不影响主图，也不掩盖（`warnings` 记录）。
- 评测按 §18.7 单独报告，不作为主指标。

### 4.4 `graph_annotations` 与环级芳香画法（§8.2）

现状：恒为 `[]`（`target.py:358`）。
方案：Render `target.py` 对"圆式/伴线式芳香环"生成环级注释：

```python
entry = {
  "kind": "aromatic_ring",
  "nodes": [...],                  # 环成员（target 内 id）
  "depiction": "circle" | "companion",
}
```

成员边仍标 `aromatic_circle_member`/`companion`；评测时环级与边级分开报告。

### 4.5 其他小项

| 项 | 现状 | 处理 |
| --- | --- | --- |
| `variable_subtype` 集合 | Render 含 `bond`（`target.py:23`），`方案.md` §6.1 列 `ring_atom/linker/generic` | 兼容超集，无需改；若生成器产生 `bond` 变量再补文档 |
| `loss_mask` | 仅 `h_count` | 逐步加入 `aromatic_atom`（像素不可判时）、`charge` 等；不可判即 `false` 并置 `semantic` 为 `null` |
| 节点 `aromatic` | 未被 Render 使用（`USER_GUIDE` §3.2） | 生成器已写；作为 `semantic_label.aromatic` 的辅助监督，不影响渲染 |

---

## 5. 端到端数据流水线

### 5.1 阶段

```text
[1] 源结构
    chembl.smi
      -> markushgen：canonical 去重/过滤 -> K 个抽象变体 -> IR JSONL
[2] 切分（必须在渲染前，§14.2）
    canonical_structure_id / scaffold_id / patent_family_id 分组 -> train/valid/test
[3] 渲染 + 真值
    每个 IR -> markushrender.render_variant(seed) -> PNG + prediction_target
[4] 图像增强
    像素退化（transforms）+ 可选几何 warp（同步 target）
[5] QA
    schema / 容量 / anchor 在画布内 / visual label 一致 / 无 dangling /
    楔向保持 / 芳香双标签一致 / 结构 ID 与图像 hash 不跨 split / overlay 抽检
[6] 分片存储
    webdataset 或 tar shard（image + target json）
```

### 5.2 切分与防泄漏

- **切分键优先级**：`patent_family_id > canonical_structure_id > scaffold_id > source_record_id`。
- 同一 canonical 的 K 个抽象变体、同一 scaffold 的不同分子必须同 split。
- 评测至少给 scaffold-disjoint 与（有专利时）family-disjoint 两套结果。
- `markushgen` 已写 `canonical_smiles/scaffold_smiles`，`split.py` 直接据此分组。

### 5.3 QA 闸门（必须自动化）

- `variant_report(...)["overlaps"] == 0`、`bond_scale_ok is True`、`chirality` 在同结构变体间一致；
- `prediction_target` 通过 schema 校验；anchor ∈ [0,1]；bin/offset 一致；
- 对生成 IR 抽样渲染 overlay（anchor 十字 + 节点 ID + depicted 边 + 楔向）；
- 记录 `(source_id, render_seed, degrade_seed)` 以便复现。

### 5.4 性能

- Render 无缓存（其 M4 未做）；建议对 `(canonical_ir_hash, style_params)` 建多层缓存，避免同结构重复布局；
- 渲染与像素退化为 CPU 密集，使用多进程 worker（每 worker 独立 rdkit/resvg）；
- target 与图像写 shard，避免百万小文件。

---

## 6. 立体化学（后续，非本期必需）

若要把楔形键纳入监督：

1. 生成抽象 IR 时保留 `solid wedge/dashed wedge` 与 `source=顶点` 约定；
2. 调用 Render 时必须 `engine=given` 并提供 2D 坐标，否则 CoordGen 重排会改变构型；
3. 用 `markushrender.depicted_chirality` 做跨变体构型一致性断言（Render 已提供）；
4. 镜像/键翻转只允许走 Render 的立体安全通道。

---

## 7. 实施优先级

| 优先级 | 项 | 归属 | 交付 |
| --- | --- | --- | --- |
| P0 | `markushgen` 结构构造 | MarkushScribe | ✅ 已实现，含 CLI + 测试 |
| P0 | `schema.py`（v2 契约） | MarkushScribe | ✅ 已实现 dataclass + validator + 测试 |
| P0 | `split.py`（先切分） | MarkushScribe | ✅ 已实现分组切分 + 泄漏检查 |
| P0 | 像素退化 `transforms.py` | MarkushScribe | ✅ 已实现像素退化 + 几何 warp 同步 |
| P0 | 数据组装 `dataset.py` + tools | MarkushScribe | ✅ 端到端 shard 构建 + 校验 CLI |
| P0 | overlay 查看器 + QA 闸门 | MarkushScribe | ✅ `visualization/overlay.py` + blank/overlap 过滤 |
| P0 | 真实小批量跑批 + 抽检 | MarkushScribe | ✅ 296 样本 0 失败，报告见 `docs/m0_report.md` |
| P1 | 样式双层采样 §3.2 | MarkushRender | ✅ 已实现（`StyleSampler.preset_choices` + `AugmentConfig` 透传） |
| P1 | `kekule_order` §4.1 | MarkushRender | ✅ 已实现（`kekule_aromatic_orders` + `normalized.kekule_order`） |
| P1 | `placeholder.shape` §4.2 | MarkushRender | ✅ 已实现（节点 `placeholder {shape, radius}`） |
| P2 | CXSMILES §4.3 | MarkushScribe | 可选导出 + `masks.cxsmiles` |
| P2 | `graph_annotations` §4.4 | MarkushRender | ✅ 已实现（环级 `aromatic_ring` 注释） |
| P2 | 几何 warp + `warp_target` | MarkushScribe | 透视/倾斜同步 |
| P3 | 立体化学 §6 | 两者 | 楔形键训练信号 |

---

## 8. 验收清单

- [ ] `pytest -q` + `ruff check .` 全绿（MarkushScribe）。
- [ ] 2M ChEMBL 抽样生成 1 万 IR，Render + target 成功率 ≥ 99%，`overlaps==0`。
- [ ] 生成的 `node_kind` 覆盖 atom/rgroup/variable(ring_atom,linker,ring)；`multiplicity` 出现。
- [ ] 切分后同一 canonical 不跨 split（有测试）。
- [ ] `prediction_target` schema 校验 100% 通过；anchor 全在画布内。
- [ ] 像素退化后标签不变；几何 warp 后 anchor 经 `warp_target` 与图像一致（overlay 验证）。
- [ ] 补齐字段后：`kekule_order` 在可确定芳香环上非空、`placeholder.shape` 非空、CXSMILES 成功率可统计。

---

## 9. 命令备忘

```bash
# 结构构造
python tools/build_markush_corpus.py --smi data/chembl.smi --out data/markush_ir \
    --count-per-mol 8 --core-retention 0.6 --seed 0 --shard-size 20000

# Render + 真值（示意）
python - <<'PY'
import json
from markushrender import render_variant, prediction_target
for line in open("data/markush_ir/shard_00000.jsonl"):
    ir = json.loads(line)
    r = render_variant(ir, seed=ir["_meta"]["seed"])
    target = prediction_target(r)
    r.write("data/rendered", f'{ir["_meta"]["source_id"]}_{ir["_meta"]["seed"]}')
    json.dump(target, open(f'data/targets/{ir["_meta"]["seed"]}.json', "w"))
PY

# 回归
pytest -q && ruff check .
```

---

## 10. 实施记录：MarkushRender 侧补丁（已完成）

> 位置：`../MarkushRender`；全部改动通过 `pytest -q`（222 passed）与 `ruff check .`，
> 并用 `tools/compare.py` 确认 `patent_bw` 标定样例比值未漂移（stroke/bond 0.0619）。

### 10.1 样式双层采样

- `markushrender/styles.py`：`StyleSampler` 新增 `preset_choices`；非空时先跨族抽预设，
  再在族内对 `size/width/radius` 做 ±10%/±15%/±8% 抖动并采样清晰度降质；空时保持原行为
  （`patent_bw` 走标定采样，其它 base 原样返回）。
- `markushrender/augment.py`：`AugmentConfig.preset_choices` 透传。
- 测试：`tests/test_styles.py`（跨族确定性、族内抖动范围、非 patent base 兼容、AugmentConfig 透传）。

### 10.2 `normalized.kekule_order`

- `markushrender/draw/bonds.py`：新增公开函数 `kekule_aromatic_orders(ir) -> {bond: 1|2}`，
  复用既有 `_kekule_aromatic_modes` 的 RDKit Kekulé 匹配；不可 Kekulé 化时返回 `{}`。
- `markushrender/target.py`：`_normalized_edge(edge, kekule_orders)` 为芳香边填
  `kekule_order`（含变量环保持 `null`），与 `depicted` 完全解耦。
- 测试：`tests/test_target.py::test_normalized_kekule_order_*`（真实苯环填 1/2，含 M 变量环全 `null`）。

### 10.3 `placeholder` 辅助几何

- `markushrender/target.py`：`ring_placeholder` 节点新增
  `placeholder = {shape, radius}`（取自实际 `result.style.ring`）；其它节点不写该键。
- 测试：`tests/test_target.py::test_placeholder_shape_and_radius_are_serialized`。

### 10.4 环级芳香注释 `graph_annotations`

- `markushrender/renderer.py`：`RenderResult` 新增 `rings`，由最终布局的 `layout.rings` 填充。
- `markushrender/target.py`：新增 `_aromatic_ring_annotations(...)`，对"成员边全为 aromatic、
  且画法一致为 circle / companion"的环输出
  `{kind: "aromatic_ring", nodes: [target id...], depiction: "circle"|"companion"}`；
  Kekulé 环不注释。
- 测试：`tests/test_target.py::test_graph_annotations_mark_circle_and_companion_rings`。

### 10.5 文档同步

- `../MarkushRender/DESIGN.md` §5.8（target 新字段）与 §6.3（采样器）。
- `../MarkushRender/USER_GUIDE.md` §6.3、§7.1、§11.2。

### 10.6 尚未做（MarkushScribe 侧或后续）

- CXSMILES 辅助导出（§4.3）、`split.py`、像素退化 `transforms.py`、几何 `warp_target`、立体化学。

---

## 11. 实施记录：MarkushScribe 数据闭环（已完成）

> 新增 `markushscribe/{schema,split,transforms,dataset}.py` 与 `tools/{build_render_dataset,
> validate_dataset,check_split_leakage}.py`；`pytest -q` 85 passed，`ruff check` + `ruff format --check`
> 全绿。已用 9 条 IR 跑通 `IR -> split -> render -> target -> degrade -> shard -> validate` 全链路。

### 11.1 `schema.py`：`markush_graph_v2` 契约

- 冻结 dataclass：`MarkushGraph / TargetNode / TargetEdge / Anchor / CoordTarget /
  VisualLabel / Placeholder / NormalizedEdge / DepictedEdge / GraphAnnotation`。
- `parse_target(payload)` / `validate_target(payload)`：校验 format、坐标系、`coord_bins`、
  `image_size`、节点类与 `variable_subtype`、anchor↔`coord_target` 一致性、视觉标签 runs、
  每类 semantic_label 必填字段、`loss_mask` 布尔、placeholder 形状/半径、边端点/自环/重复边、
  aromatic↔depicted 兼容、`kekule_order`、wedge 方向、`ring_mode` 仅限占位环、环级注释与 masks。
- `to_dict(graph)` 与 Render 输出逐字段一致；`tests/test_schema.py` 覆盖 30+ 非法用例与真实
  Render target 往返。

### 11.2 `split.py`：先切分、防泄漏

- `group_key(record, key)`：`patent_family > canonical > scaffold > source > fallback`；
  `auto` 按该优先级选第一个可用字段。
- `assign_splits`：按组大小降序 + 最大缺口贪心填充，seed 决定顺序，保证组不跨 split。
- `split_records` 返回 `SplitResult`（`group_to_split` / `counts` / `split_for` / `require`）。
- `leakage_report` / `assert_no_leakage` 接受 `SplitResult`、`group->split` 映射或
  `record->split` 可调用，用于检出外部赋值造成的跨 split。
- 测试：同分子多变体同 split、比例近似、seed 确定性、非法比例、泄漏检出。

### 11.3 `transforms.py`：像素退化 + 目标同步

- `ImageDegradeConfig`：模糊/噪声/JPEG/二值化/光照/断墨/反色概率与区间；`degrade_image`
  仅动像素，同 seed 可复现、全 0 概率严格不变。
- `GeometryConfig` + `sample_geometry_matrix`：仿射（旋转/缩放/平移）或轻微透视，返回
  forward 像素单应；`warp_image` 用其逆矩阵经 `Image.transform` 重采样。
- `warp_target`：用同一单应变换 anchor 与 `coord_target`，出界节点连同其边/环注释一起删除，
  按阅读序重新编号并排序边，更新 `image_size`。
- `degrade_sample`：像素退化 -> 可选几何 warp（含 target），test 断言 warp 后 schema 仍通过。
- 仅依赖 numpy + Pillow（当前环境无 cv2）。

### 11.4 `dataset.py` + tools：端到端组装与校验

- `RenderConfig` 透传 `base / preset_choices / style / lowres / rotate / mirror` 到
  MarkushRender `AugmentConfig`；`render_record` 渲染并 `prediction_target`。
- `build_dataset`：读 IR JSONL -> `split_records`（渲染前）-> 每个 split 渲染+退化 ->
  写 `{split}/shard_*.tar`（`<uid>.png` + `<uid>.json`）-> 写 `splits.json`（含组映射）。
- `validate_dataset`：校验每个 target、统计 valid/invalid/images/nodes/edges/unpaired 与
  分 split 计数，`--strict` 时失败退出码 1。
- `tools/build_render_dataset.py` / `tools/validate_dataset.py` / `tools/check_split_leakage.py`：
  薄 CLI，沿用 `sys.path` 引导（与 `build_markush_corpus.py` 一致）。

### 11.5 尚未做（下一步）

- `chemistry/cxsmiles.py`（§4.3）、真实 ChEMBL 抽样跑批与 ≥100 样本人工抽检、模型（tokenizer / NODE AR baseline）。

---

## 12. 实施记录：M0 收尾（已完成）

> 详见 `docs/m0_report.md`。本轮新增 overlay 查看器与数据 QA 闸门，并用 2000 个 ChEMBL
> 分子跑通 300 样本端到端（0 失败 / 0 invalid / 0 unpaired / 0 blank，split 179/57/60）。

### 12.1 overlay 查看器

- `markushscribe/visualization/overlay.py`：`draw_overlay` 标注 anchor 十字、节点
  `#id 文本 [class]`、depicted 边与环级注释；`inspect_shard` / `inspect_dataset` 生成
  contact sheet，`--save-overlays` 输出全尺寸单图。
- `tools/inspect_dataset.py`：CLI，本轮产出 296 张 overlay + 4 张 sheet（`data/inspect/`）。

### 12.2 QA 闸门

- `transforms.ink_fraction` + `build_split_shards(min_ink_fraction=0.001)`：丢弃近空白渲染，
  记录 `blank` / `blank_ids`；`validate_dataset` 报告并可在 `--strict` 下拒绝。
- `build_split_shards(qa=..., drop_overlaps=...)`：基于 `variant_report()["overlaps"]` 记录
  `overlaps` / `overlap_ids`，可选丢弃。
- CLI 新增 `--no-qa`、`--drop-overlaps`、`--min-ink`（build）与 `--min-ink` / `--no-ink`（validate）。

### 12.3 关键发现

- Render 在「大环分子 × 跨族采样样式/布局」下会产出近空白图（3/300），已由 ink 闸门兜住，
  需上游修复 auto-fit/密度控制。
- 「anchor 离墨」核验中位数 0 px；表面异常均为字母 `O` 中心空洞，非坐标 bug。

### 12.4 仍待做

- 2M 全量跑批（工具就绪）、专利 family-disjoint 切分、`chemistry/cxsmiles.py`。
- 模型：`constants.py` + `tokenizer.py` + torch Dataset/collate + NODE AR baseline。
