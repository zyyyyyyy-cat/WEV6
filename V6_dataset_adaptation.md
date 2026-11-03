# V6 数据集适配检查：1000 / 2337 / 1000-GD

> 版本：2026-11-03
> 目的：确认 V6 的 loss/代码不仅适用于 1000，也能支持 2337；1000-GD 只做评测。
> 当前状态：代码已加 2337 config、preflight、SKIP_INCOMPLETE_IDS；未训练/未 test。

---

## 1. 数据集事实

| 项目 | 1000 | 2337 | 1000-GD |
|---|---|---|---|
| 角色 | 主实验 / Table 6 | 独立数据规模/泛化 | 1000 权重 GD 评测 |
| 总 IDs | 1,000（500 train + 500 test） | 2,337（1,000 train + 1,337 test） | 1000 test 子集 |
| train IDs / classes | 500 | **1,000** | 不训练 |
| train images/模态 | 30,711 | 约 75k（296 iters × 256） | — |
| iters/epoch | 117 | **296** | — |
| query/gallery | 2,135 / 31,203 per modality | 由 2337 dataset 动态读取 | FILTER_GD 子集；984 valid overall query |
| 相机 | 7（0–4 ground，5–6 aerial） | 需 preflight 确认 | 同 1000 |
| 缺失模态 | 预期 0 | **可能有；必须 preflight** | 不训练 |
| V6 训练 | 是 | 是（三模态齐全 IDs） | 否 |
| V6 loss | 直接适用 | 先 preflight；缺模态 skip/过滤 | 不计算 |
| Gate | V4 1000 +0.20 | 单独 baseline/gate | V4 GD 护栏 |
| 混表 | 基准 | **禁止** | 禁止 |

2337 训练日志证据：`identities: 1000`，`Iteration[50/296]`；1000 为 117 iters/epoch。两者 class 数和数据规模不同。

---

## 2. V6 的硬假设

1. 每个训练 tuple 必须三模态齐全：`(RGB idx, NIR idx, TIR idx)`；
2. `L_corr` 的 `M=3`，三个 unordered pair；
3. 每个 ID/模态至少 1 张图；cross-camera positive 可空（loss=0），但不能缺整个模态；
4. `num_classes` 由 dataset 动态决定；
5. 1000-GD 只评测，不训练；`L_corr` 不参与。

---

## 3. 2337 missing modality：已知风险，尚未本地确认

历史记录/README 明确提示：

- 2337 某些 `(pid, modality)` 组稀疏，TIR 更少；
- `PKMSampler_DDP.sample_list` 对 `cur=0` 会 `divmod(need,0)`；
- GPD warm start 需要 valid mask/身份回退；
- 当前 `ImageDataset`/`corr_loss` 假设 M=3 且 tuple 三模态齐全。

**本地没有 2337 数据**（只有 1000/V4 缓存），所以不能给出精确的 missing-modality ID 数。  
**必须服务器 preflight 确认；不能假设为 0。**

---

## 4. 已补的 2337 支持

### 4.1 Config

- `configs/V6_CORR_2337.yml`
  - `DATASETS.DATASET_DIR: 'WHU-MARS-2337'`（与历史 2337 log 一致；若实际为 `WHU-MARS_2337`，命令行覆盖）
  - `OUTPUT_DIR: './logs/V6_CORR_2337_DDP'`
  - `SKIP_INCOMPLETE_IDS: True`
  - 其余 V4 recipe + CORR_LOSS 不变
  - 从 ImageNet 从零训练；不加载 1000 checkpoint；不加 FILTER_GD

### 4.2 Sampler skip

- `config/defaults.py`: `DATALOADER.SKIP_INCOMPLETE_IDS=False` 默认；
- V6 两个 config 设 True；
- `PKMSampler` / `PKMSampler_DDP`：构造时过滤任一模态为 0 的 train ID；
- 1000 三模态齐全时：pids/length/RNG 不变，no-op；
- 2337 有缺模态时：只训练三模态齐全的 IDs；rank0 打印 skipped 数量；
- class count 仍按 dataset 全部 train IDs；GPD/AUX 形状不变；被 skip 的类不采样；warm start 对缺失模态用 id_mean 填充，GAC head 不全零；
- 若缺模态 ID 很多：不能静默接受；preflight 后决定是否过滤/修数据/换方案。

### 4.3 Preflight

`tools/v6_data_preflight.py` 输出：

- train/query/gallery 每模态 IDs/图数；
- `missing_modality_pids` / `zero_image_groups`；
- 每 ID/模态相机数、camera IDs；
- 模拟 PKM 的 per-pair cross-camera coverage；
- AG ground/aerial 数量；
- train/query/gallery raw ID overlap。

命令：

```bash
python tools/v6_data_preflight.py --root /project/code/data \
  --dataset_dir WHU-MARS --modalities RGB,IR,Thermal \
  --out ./logs/v6_preflight_1000.json

python tools/v6_data_preflight.py --root /project/code/data \
  --dataset_dir WHU-MARS-2337 --modalities RGB,IR,Thermal \
  --out ./logs/v6_preflight_2337.json

python tools/v6_data_preflight.py --root /project/code/data \
  --dataset_dir WHU-MARS --modalities RGB,IR,Thermal --filter_gd \
  --out ./logs/v6_preflight_gd.json
```

若 2337 目录是 `WHU-MARS_2337`，把 `--dataset_dir` 换成实际目录。

---

## 5. 判定规则

### 1000

- 预期 `missing_modality_pids=0`；
- per-pair coverage 约 89.5–90.2%；
- cameras 0–6；
- 若不符合：停，不能假设历史统计仍成立。

### 2337

- `missing_modality_pids == 0`：V6 2337 可直接用 `V6_CORR_2337.yml` 训练；仍确认 coverage/相机；
- `missing_modality_pids > 0`：
  1. 若数量少：skip 后训练，记录 skipped IDs；class count 不变；
  2. 若数量多：不能静默丢 ID；需要修数据/修 sampler/换 modality-missing 方案；
  3. 禁止用零向量补缺失模态进 `corr_loss`。
- 2337 的 gate/baseline 必须单独：UAD 2337 或新训 V4 2337；不能套 1000 +0.20。

### 1000-GD

- 不训练；用 1000 训练的 V6 final checkpoint；
- `DATASETS.FILTER_GD True` 评测；
- 只作护栏，不作 GO endpoint；
- 不训练 V6-GD，不用 GD 过滤训练集。

---

## 6. 当前代码状态

- `WEV6/WHU` 已复制 V4 recipe；
- 新增 `corr_loss`、`V6_CORR.yml`、`V6_CORR_2337.yml`、`v6_data_preflight.py`、`v6_eval.py`、`v6_delta_ci.py`；
- 修改 `defaults.py`、`processor.py`、`sampler.py`、`sampler_ddp.py`、`make_dataloader.py`；
- `model/`、`test.py`、`train.py` 未改；
- 未训练/未 test。

---

## 7. 一句话

**V6 的 loss 对 1000/2337 通用；2337 的 class 数、iters、query/gallery、缺模态风险已隔离到独立 config + preflight + SKIP_INCOMPLETE_IDS。**
**2337 是否能直接训练，取决于服务器 preflight 的 `missing_modality_pids`；在结果回来之前不启动 2337 训练。**

---

## 8. Preflight 实测结果（2026-11-03 下载）

文件：

- `WEV6/logs/v6_preflight_1000.json`
- `WEV6/logs/v6_preflight_2337.json`
- `WEV6/logs/v6_preflight_gd.json`

### 8.1 关键结论

| 数据集 | train IDs | missing_modality_pids | zero_image_groups | cameras/ID max mean | query / gallery |
|---|---:|---:|---:|---:|---:|
| 1000 | 500 | **0** | 0 | 2.392 | 2,135 / 31,203 per modality |
| 2337 | **1,000** | **0** | 0 | **2.444** | RGB 5,254/78,639；NIR 5,275/78,680；TIR 4,491/59,797 |
| 1000-GD | 500（train 不变） | 0 | 0 | 2.392 | 399 / 6,836 per modality；112 IDs |

**2337 训练集没有 missing modality**：
- 三个模态 train IDs 都是 **1,000**；
- `missing_modality_pids=0`，`zero_image_groups=0`；
- 因此 V6 的 M=3 tuple、`corr_loss`、`divmod(need,cur)` 在 2337 **训练集** 上没有缺模态问题；
- `SKIP_INCOMPLETE_IDS=True` 在 2337 训练时是 **no-op**，不会丢 ID。

**2337 test 侧有 TIR 缺失**：
- RGB/NIR query IDs 1,337；TIR query IDs **1,329**；
- RGB/NIR gallery IDs 1,337；TIR gallery IDs **1,329**；
- 即 **8 个 test IDs 没有 TIR**；这是 query/gallery 侧，不影响训练 loss；`v6_eval` 按模态动态读取，TIR pair 的 query/gallery 数会少 8 IDs，九对/TIR cross 的 n 不同，不能假设 2,135/31,203。
- 2337 TIR train 63,276 图 vs RGB 77,253 / NIR 76,975；TIR 少但 **1,000 IDs 齐全**。

**1000-GD**：
- query/gallery 只有 112 IDs、399/6,836 per modality；
- cameras 3–4，**没有 aerial**；
- 因此 AG A→G/G→A 在 GD 上不可用/会跳过；GD 只作 overall/九对护栏；
- train 仍是完整 500 IDs；GD 只评测，不训练。

### 8.2 覆盖率的读数

当前 JSON 的 `pkm_coverage.per_pair.valid_ratio` 是 **tuple-level cross-camera 比例**（约 0.49–0.52），不是 `corr_loss` 的 anchor-level 覆盖率。  
`v6_data_preflight.py` 已改为输出：
- `anchor_valid_ratio`（2B anchors 中 ≥1 cross-camera 正对的比例）；
- `positives_per_anchor`（anchor 的跨相机正对数）；
- `tuple_cross_ratio`（tuple 级比例）；
- `coverage_level` 标注。

**请用更新后的 preflight 重跑一次**，才能直接和设计中的 89.5–90.2% anchor-level 对齐；当前 JSON 的 0.50 不是覆盖率下降，而是统计口径不同。

### 8.3 2337 训练判定

- `missing_modality_pids=0` → **2337 可以按 `V6_CORR_2337.yml` 直接训练**；
- `SKIP_INCOMPLETE_IDS=True` 不会生效（no-op）；
- 仍需用更新后的 preflight 确认 anchor-level coverage；
- 2337 的 baseline/gate 仍必须单独（UAD 2337 / V4 2337），不能套 1000 的 +0.20。

### 8.4 1000 判定

- `missing_modality_pids=0`；train/query/gallery 均 500 IDs；
- cameras 0–6；2–3 cameras/ID；
- 1000 的 V6 训练路径不受 `SKIP_INCOMPLETE_IDS` 影响（no-op）；
- 用更新后的 preflight 确认 anchor-level coverage 约 89.5–90.2%。

### 8.5 1000-GD 判定

- 只评测 1000 训练模型；
- `FILTER_GD=True` 只过滤 query/gallery，不改 train；
- 无 aerial → `v6_eval` 的 AG 会跳过；不要把 AG 当 GD 指标；
- GD 只作 overall/九对护栏。

---

## 9. 当前结论

> **2337 训练集 missing modality = 0（本次服务器数据）；V6 的 2337 训练路径成立。**
> **2337 test 有 8 个 IDs 缺 TIR；`v6_eval` 动态处理，但 TIR pair 的 n 会比 1000 小。**
> **1000-GD 无 aerial；AG 不适用；GD 仅评测。**
> **`SKIP_INCOMPLETE_IDS` 在 1000/2337 训练中均为 no-op；保留它是为了未来数据版本/其他 2337 副本的安全。**
