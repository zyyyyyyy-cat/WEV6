# TIR calibration probe（WEV5/WHU）

`tir_calib_probe.py` 是 **frozen-feature、identity-disjoint** 的 calibration probe。它不训练 backbone，不修改 `model/processor/datasets/loss/train/test`；只缓存 V2/V4 的 pre-BN CLS，训练 TIR adapter，评估 TIR cross，决定是否值得进入 120ep V5。

> **2026-11-02 S64 重要：** `CC/WEV5/logs/tir_calib/v2/`、`v4/` 的 adapter 结果是在 **label-row 对齐修复之前**生成的；CE/GPD 的 batch label 与 sorted classifier/prototype 行错位，adapter 实际拟合错误身份。**这些输出不能作为 NO_GO/GO 证据**。当前工具已修复；必须在新输出目录重跑 `val`/`test`。详见 `CC/WEV5/analysis_tir_probe_results.md`。
>
> 修正后重跑建议：用新目录（如 `./logs/tir_calib/v2_fixed`）；可直接复用旧的 `features.npz` / `split.json`，或重新 `cache` 校验 checkpoint hash；`val` 全 arms，`test` primary 只评一次。val 是 seen-ID 天花板，只作 sanity，不作 go/no-go。
>
> **2026-11-02 S67 结果：** `logs/tir_calib/v2_fixed` 的 corrected probe 已完成，**有效 NO_GO**（TIR cross mAP −0.104pp、R5 −0.409pp、R10 −0.351pp；TIR same mAP −1.059pp；mixed mAP −0.265pp；cos +0.0143 但不足以救 retrieval）。不要再重跑同一个 primary test，不要跑 120ep。详见 `CC/WEV5/analysis_tir_probe_corrected.md`。

## 运行顺序与并行性

- **严格顺序**：`cache → split → val → test`。
- `cache` 和 `split` **不能并行**：`split` 读 `features.npz`。
- `val` 和 `test` **不能并行**：`test` 要求 `val_metrics.csv` 已存在，且 test 只评一次。
- `cache`/`split` 内部也不能和下游并行；先完成当前 stage 再下一步。
- V2 与 V4 的 cache 只有在你确认 GPU/内存足够且输出目录互不冲突时才可并行；默认**先跑完 V2 再跑 V4**。
- `selftest` 不依赖数据/checkpoint，可单独随时跑。
- 重跑 corrected probe 用**新目录**；旧的 `v2/decision.json` 不要覆盖，也不要当 NO_GO。

### 修正后 V2 重跑（复用已验过的 cache/split）

```bash
cd /project/code/WHU_base/WEV5/WHU
OLD=./logs/tir_calib/v2
NEW=./logs/tir_calib/v2_fixed
mkdir -p "$NEW"
cp "$OLD/features.npz" "$OLD/split.json" "$OLD/split_audit.csv" "$NEW"/

python tools/tir_calib_probe.py --stage val \
  --tag v2 --output_dir "$NEW" --device cuda \
  --seeds 1234,5678,9012 --epochs 120 --p 16 --k 4 \
  --rank 64 --gamma 0.1 \
  --arms primary,affine,linear,r32,shared,ce_only,ce_gpd --bootstrap 10000

python tools/tir_calib_probe.py --stage test \
  --tag v2 --output_dir "$NEW" --device cuda \
  --seeds 1234,5678,9012 --epochs 120 --p 16 --k 4 \
  --rank 64 --gamma 0.1 --bootstrap 10000
```

`val` 与 `test` 必须串行；`test` 只跑一次。若要干净重算 cache，用 `--stage cache` + `--stage split` 到 `$NEW`，再执行上面两步。

## 0. 关键约定

- **V2 是 primary**；V4 只做 val-only robustness，不跑 test、不进 gate。
- 1000 数据：**500 train IDs + 500 test IDs**；test query 2,135/模态，TIR cross 8,540 个 query-gallery 对（6,405 唯一 query）。
- **5-fold**：400 train / 100 val；val 每 ID/模态 4 张 query，27% ID 加 1 张（5 张）；pooled 2,135 query/模态。
- 只训练 **TIR adapter**；冻结 V2 BN/classifier/prototypes；CE 用 V2 BN + 该折 400 行 classifier；GPD/WRT/ProCA 按 V2 recipe。
- Primary 固定 **r=64、γ=0.1、全 loss**；不在 val 上选 γ/rank/loss；controls 只作解释。
- **test 只评一次**；`--force` 只用于预注册的 rerun。
- **统计口径**：final adapter state（epoch 120）的 test 为 primary；last-3（100/110/120）均值只作辅助，不是 UAD 官方口径，也不是独立验证集。
- 256×128；不再做 288×144。

## 1. 命令顺序（必须按这个顺序）

### 1.1 V2 cache

```bash
cd /project/code/WHU_base/WEV5/WHU

python tools/tir_calib_probe.py --stage cache \
  --config_file configs/HPD_UAD.yml \
  --checkpoint /project/code/WHU_base/WEV2/logs/HPD_1000_DDP/transformer_120.pth \
  --tag v2 \
  --output_dir ./logs/tir_calib/v2 \
  --dataset_root /project/code/data \
  --dataset_dir WHU-MARS \
  --device cuda --batch_size 512 --num_workers 8
```

检查：

- `logs/tir_calib/v2/features.npz`、`cache_meta.json` 存在；
- `train_counts` 约 30,711/模态；`query_counts` 2,135/模态；`gallery_counts` 31,203/模态；
- 若 counts 明显不对，先停，查 config/dataset。

### 1.2 V2 split

```bash
python tools/tir_calib_probe.py --stage split \
  --tag v2 --output_dir ./logs/tir_calib/v2 \
  --folds 5 --seed 1234
```

检查：

- `split.json`：5 folds，每折 400 train/100 val；
- `split_audit.csv`：每折每模态 query/gallery 数；无空 query/gallery；
- 每个 query 在 same-camera 排除后对 TIR cross 4 对至少 1 个 positive；不合格 query 会被 prune 并记录。

### 1.3 V2 val（primary + controls）

```bash
python tools/tir_calib_probe.py --stage val \
  --tag v2 --output_dir ./logs/tir_calib/v2 \
  --device cuda --seeds 1234,5678,9012 \
  --epochs 120 --p 16 --k 4 --rank 64 --gamma 0.1 \
  --arms primary,affine,linear,r32,shared,ce_only,ce_gpd
```

说明：

- `primary`：3 seeds ×5 folds；controls：1 seed ×5 folds。
- 输出 `val_metrics.csv`、`val_per_query.csv`、`val_selection.json`。
- `val_metrics.csv` 含 `state` 列：`state=0/1/2` 对应最后三个 adapter 状态；**final state 为 primary，last-3 只作辅助**。
- primary 固定，不在 val 上选 γ/rank/loss；val 只作 sanity（val ID 已被 V2 见过，≈0.93 是天花板，不作 go/no-go）。
- 若 `residual_ratio` <0.02，先视为 **inconclusive（adapter inactive）**，不要宣布 TIR ceiling。

### 1.4 V2 test（只跑一次）

```bash
python tools/tir_calib_probe.py --stage test \
  --tag v2 --output_dir ./logs/tir_calib/v2 \
  --device cuda --seeds 1234,5678,9012 \
  --epochs 120 --p 16 --k 4 --rank 64 --gamma 0.1 \
  --bootstrap 10000
```

- 要求 `val_metrics.csv` 已存在；test 已存在时拒绝第二次，除非 `--force`（不建议）。
- 输出 `test_metrics.csv`、`test_per_query.csv`、`classmean_cos.csv`、`classmean_cos_last3.csv`、`decision.json`。
- `decision.json` 的 primary 是 **final adapter state**（`final_state`）；`last3` 字段只作辅助。
- `decision.json` 的 `status`：`GO` / `STRONG_GO` / `NO_GO` / `INCONCLUSIVE`。

### 1.5 V4 val-only robustness（可选，V2 primary 通过后）

```bash
python tools/tir_calib_probe.py --stage cache \
  --config_file configs/GAC_UAD.yml \
  --checkpoint /project/code/WHU_base/WEV4/logs/GAC_1000_DDP/transformer_120.pth \
  --tag v4 --output_dir ./logs/tir_calib/v4 \
  --dataset_root /project/code/data --dataset_dir WHU-MARS \
  --device cuda --batch_size 512 --num_workers 8

python tools/tir_calib_probe.py --stage split \
  --tag v4 --output_dir ./logs/tir_calib/v4 --folds 5 --seed 1234

python tools/tir_calib_probe.py --stage val \
  --tag v4 --output_dir ./logs/tir_calib/v4 \
  --device cuda --seeds 1234,5678,9012 \
  --epochs 120 --p 16 --k 4 --rank 64 --gamma 0.1 \
  --arms primary
```

V4 只 val；不跑 `--stage test`，不进 gate。

## 2. Adapter / loss

```
f_n = f_TIR (raw pre-BN CLS)
h = LayerNorm(f_n, affine=False)
d = W2(GELU(W1(h)))                 # 768 -> 64 -> 768
res = gamma * d; gamma init 0.1
res = res * r_max / (r_max + ||res||)
f_TIR' = f_n + res
```

- `r_max = 0.25 * median(||f_TIR||)` from train features，约 2.35；残差上限约 25%。
- **不要用 `f_TIR + γd` with r_max=0.5**：V2 norm≈9.99，会 dead（0.3–0.5%）。
- `γ=0` 用 identity 分支；`torch.equal` 与 V2 一致。
- CE 用 V2 `bottleneck`（eval）+ 400 行 classifier；GPD 400 frozen prototypes；WRT/ProCA V2 recipe。
- 120 epochs on cached features；AdamW lr 2e-3、wd 1e-4（γ/LN 不 decay）；clip 1.0；保存 last-3 states（final primary，last-3 auxiliary）。

## 3. Go / No-go（decision.json）

- **统计对象**：final adapter state（epoch 120）的 test；last-3 均值只作辅助，不参与 GO/NO_GO 主判定。
- **Active 前置**：`residual_ratio` 中位数 ≥0.02；否则 `INCONCLUSIVE`，不是 NO_GO。
- **GO**：test TIR cross ΔmAP ≥+0.15pp；identity-cluster bootstrap 95% CI lower ≥−0.05pp；≥4/5 folds 正；至少 1 个 tail 信号（ΔR5 ≥+0.5pp / ΔR10 ≥+0.8pp / Δ>100 ≤−1pp / Δ>500 ≤−0.5pp）；TIR same mAP ≥−0.5pp、R1 ≥−1.0pp；mixed mAP ≥−0.10pp；TIR cross class-mean cos Δ≥+0.005。
- **STRONG_GO**：ΔmAP ≥+0.25pp 且 CI lower >0；≥2 个 tail 信号；cos Δ≥+0.01。
- **NO_GO**：test ΔmAP 不达标且无 tail；或 same 硬护栏破；或 cos 两方向都降且无 tail。
- **INCONCLUSIVE**：adapter inactive/优化失败；或 CI 太宽/tail 混合；或 val/test 强不一致。

## 4. 输出文件

| 文件 | 内容 |
|---|---|
| `features.npz` | raw pre-BN CLS + pids/cams/mods/names + V2 BN/classifier |
| `cache_meta.json` | checkpoint/config/counts/hash |
| `split.json` / `split_audit.csv` | 5-fold manifests / val query-gallery audit |
| `val_metrics.csv` / `val_per_query.csv` | val arm/fold/seed/**state** metrics + per-query AP/rank |
| `val_selection.json` | fixed primary config（无 val 调参；final state primary） |
| `test_metrics.csv` / `test_per_query.csv` | test primary/gamma0 per fold/seed/state + per-query |
| `classmean_cos.csv` | final-state 3×3 Q→G class-mean cos |
| `classmean_cos_last3.csv` | last-3 均值，仅辅助 |
| `decision.json` | status + final-state delta + bootstrap + guards + last3 辅助 |

## 5. 自检

```bash
python -I -m compileall -q .
python -I tools/tir_calib_probe.py --stage selftest \
  --output_dir /tmp/tir_calib_selftest --device cpu \
  --epochs 3 --p 4 --k 2 --rank 4 --folds 5 --seeds 1234 --bootstrap 20
```

`SELFTEST OK` 表示 split/adapter/AP/rank/bootstrap 的纯逻辑通过；它不替代真实数据 probe。
