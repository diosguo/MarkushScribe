# M0 数据闭环验收报告

> 日期：2026-09-12
> 范围：`markush_graph_v2` 契约 + 数据构造/增强/切分/组装/QA 全链路
> 数据：ChEMBL 在线抽样 2000 分子（`tools/fetch_chembl_sample.py`）

## 1. 链路与产物

```text
ChEMBL 2000 ──markushgen K=4──> 8000 IR ──build_render_dataset──> 296 samples
   (data/chembl_sample.smi)      (data/markush_ir)                (data/rendered)
```

命令：

```bash
python tools/fetch_chembl_sample.py --out data/chembl_sample.smi --count 2000
python -m markushscribe.markushgen.cli --smi data/chembl_sample.smi \
    --out data/markush_ir --count-per-mol 4 --seed 0 --shard-size 20000
python tools/build_render_dataset.py --ir data/markush_ir/shard_00000.jsonl \
    --out data/rendered --preset-choices patent_bw acs chemdraw \
    --shard-size 100 --limit 300 --train 0.6 --valid 0.2 --test 0.2 --drop-overlaps
python tools/validate_dataset.py data/rendered --strict
python tools/inspect_dataset.py data/rendered --out data/inspect --split train \
    --per-shard 1000 --cols 3 --save-overlays
```

## 2. 结果汇总（296 samples）

| 指标 | 值 |
| --- | --- |
| 目标样本 | 300 |
| 成功渲染 | 300（failed=0） |
| 写入 shard | 296 |
| schema 通过 | 296 / 296 |
| invalid / unpaired / blank | 0 / 0 / 0 |
| 节点 / 边 | 7273 / 7921 |
| 切分（canonical 分组，seed 0） | train 179 / valid 57 / test 60 |
| 分组数 / 跨 split 泄漏 | 75 / 0 |
| 平均 PNG 体积 | ≈ 0.30 MB |

被 QA 闸门淘汰：**3 个近空白渲染** + **1 个标签重叠**（均为大环分子，见 §3）。

## 3. 发现与处理

1. **近空白渲染（3/300）**：MarkushRender 在「大环分子 × 跨族采样样式/布局」组合下会产出几乎全白图
   （`min >= 130`、ink fraction ≈ 0）。已定位为 Render 侧布局/auto-fit 边界问题，**未在本仓修复**；
   数据侧新增 `transforms.ink_fraction` 闸门（默认 `min_ink_fraction=0.001`）在写入前丢弃，并在
   `validate_dataset` 报告 `blank` 计数。受影响样本：`CHEMBL6266_2323`、`CHEMBL258254_2830`、`CHEMBL446445_3031`。
2. **标签重叠（1/300）**：同上大环分子 `CHEMBL258254_2829`。`build_render_dataset.py --drop-overlaps`
   可基于 `variant_report()["overlaps"]` 丢弃，本轮已启用。
3. **anchor 对齐核验**：对未退化图做「anchor 到最近墨迹距离」统计，中位数 0 px；表面上的
   「10–27 px 离墨」全部是字母 `O` 的中心空洞（anchor 正确落在 O 中心）。**无坐标/同步 bug**。
4. **大环布局**：Macrocycle 被 CoordGen 画成横跨画布的大圆，信息密度低但标签仍自洽；
   后续可考虑对超大环的 `image_size`/缩放做上限控制。
5. **存储**：PNG ≈ 0.3 MB/张，2M 分子 × K 会达到 TB 级；规模化前需评估 JPEG 或降低长边。

## 4. M0 退出检查

- [x] `markush_graph_v2` schema + validator（`markushscribe/schema.py`，含 30+ 非法用例）。
- [x] 切分工具 + 泄漏检查（`split.py`、`tools/check_split_leakage.py`）。
- [x] 增强后 anchor/edge 同步（`transforms.warp_target` + overlay 抽检 + 距离核验）。
- [x] overlay 查看器 + 296 张个体 overlay / 4 张 contact sheet（`data/inspect/`）。
- [x] 真实数据小批量跑批：0 失败、0 invalid、0 unpaired、0 blank。
- [ ] 2M 全量跑批（本轮为 300 样本 smoke；工具就绪）。
- [ ] 专利 family-disjoint 切分（需真实专利来源）。
- [ ] CXSMILES 辅助导出（P2，非 M0 必需）。

## 5. 已知开放项

- Render 侧近空白渲染（§3.1）需在上游修 `auto_fit`/密度控制，或对 `atom.size` 设下限。
- `O` 空洞类 anchor 需要更合理的对齐指标（bbox 内判定，而非邻域墨迹比例）。
- `validate_dataset` 目前逐图解码以测空白，规模化时可改为抽样或关闭（`--no-ink`）。
