# `analyse_gac.py`：WEV4/GAC 免训练诊断

这个脚本只做**前向特征提取、rank audit、单 batch 梯度 proxy**，不训练、不建 optimizer、不改 checkpoint、不改 `processor.do_train` / `test.py` 的训练/推理路径。

## 1. 输入

- `--config_file`：与训练/测试一致的配置，例如 `configs/GAC_UAD.yml`。
- `--checkpoint`：
  - `checkpoint_latest.pth`：full state，含精确 HPD bank，优先；
  - `transformer_120.pth`：model-only，含 `aux_heads`，但无 bank；脚本会用 train split 按 warm-start 同样的 class-mean 规则重算 `P_scene/P_id`。
- `--output_dir`：输出目录。
- 可选 `DATASETS.FILTER_GD True`（1000-GD；仍是同一个 `WHU-MARS`，不要改成 `WHU-MARS_1000GD`）。

## 2. 输出

### rank audit（test split）

| 文件 | 内容 |
|---|---|
| `summary.csv` | mixed / mixed_platform / pair / AG 的 `rank1/2-5/6-10/>10`、R1、mean/median rank、margin |
| `rank_query.csv` | 每个 query 的 mixed + AG rank/margin/top-1 |
| `pair_rank_query.csv` | 每个 query × 每个 gallery modality 的 rank/margin/top-1；R5/R10 可从 rank 离线算 |

same-camera gallery 按 `utils/metrics.py` 排除；特征为 pre-BN CLS + L2。

### head behaviour（train split）

| 文件 | 内容 |
|---|---|
| `gac_head_ce_acc.csv` | 行=query modality q，列=head h：CE、top-1 acc、n |
| `gac_head_home_cross.csv` | 每个 q 的 home（h=q）与 cross（h≠q）平均 CE/acc |
| `gac_head_logits.csv` | 每个 q,h 的 pos/neg/neg-max logits 均值、entropy |
| `feature_norm.csv` | 每个 modality 的 pre-BN feature norm mean/std |

head CE/acc 只在 train split 计算：query/gallery 身份与训练 500 类不重叠，test pid 不是合法分类目标。

### geometry

| 文件 | 内容 |
|---|---|
| `gac_head_geometry.csv` | 完整 `cos(W_m, P_scene[:,m'])` 矩阵：head m × scene column m' 的 mean/min/max |
| `gac_head_pair_cos.csv` | `cos(W_m,W_n)` per-class mean |
| `gac_head_summary.csv` | 每个 head 的 norm、对角 `cos(W_m,P_scene[:,m])`、best scene column、`cos(W_m,P_id)` |
| `bank_geometry.json` | `P_id` vs `P_scene` 列、`P_scene` pair cos、`cos(P_id,mean P_scene)`、norm 范围 |
| `train_geometry.json` | 重算 class-mean 的 geometry；若用 full checkpoint，还给出 bank vs recomputed cos |

### gradient proxy（一个 train batch）

| 文件 | 内容 |
|---|---|
| `gac_grad.json` | `share_id/scene/base/pca/aux`、`cos_aux_id/scene/base/pca`、`eff_*`、`head_scene/id/pair_cos`、`own_ratio`、`usage`、id/scene pos/neg/entropy |
| `gac_aux_head_grad.csv` | 每个 head 的 grad norm、raw share、effective share（`aux_w/M * n_h / total_eff`） |

梯度只取 last ViT block + final norm（与 `diagnose_gpd_gradients` 相同），是 proxy，不是 full-ViT。

## 3. 解释要点

- **`gac_head_geometry.csv` 是判断 gallery conditioning 的核心**：如果 head m 在 scene column m 上不是最高、且非对角 cos 也高，head 与 gallery modality 的对应关系就不成立。
- **`gac_head_home_cross.csv`**：home 高、cross 低说明 heads 主要在复制 own；cross 也高才说明跨模态判别。
- **`gac_aux_head_grad.csv`**：per-head effective share 是按 per-head loss 的 proxy；总和不必严格等于 `share_aux`（总 aux 是 mean loss 的梯度）。如果 cross head 的 share 很小，名义 2/3 cross 没有变成有效梯度。
- **`summary.csv`** 的 `rank6_10`/`rank_gt10` 用来解释“R1 涨、mAP 不涨”：GAC 常是 top-1 改善 + 6–10 回退。
- **`gac_grad.json`** 的 `cos_aux_*` 高说明 aux 与 GPD 同向，冗余；`cos_aux_pca` 用于判断与 ProCA 的方向关系。
- **bank source**：model-only checkpoint 的 `P_scene` 是重算的，不是训练时精确 bank；head 几何的精确性以 full checkpoint 为准。`--max_train_batches` 只适合 smoke；正式 head CE/acc 和 bank 重算请用 full checkpoint 或 `max_train_batches=0`。

## 4. 正式运行顺序（先做 1000，再做 GD）

**第 1 步：1000，full checkpoint（首选；bank 精确）**

```bash
python tools/analyse_gac.py \
  --config_file configs/GAC_UAD.yml \
  --checkpoint ./logs/GAC_1000_DDP/checkpoint_latest.pth \
  --output_dir ./logs/gac_diag_1000_full --tag v4_1000_full \
  --grad_batch_size 32 \
  DATASETS.ROOT_DIR /project/code/data
```

**第 2 步：1000-GD，full checkpoint（同一 `WHU-MARS` + `FILTER_GD`）**

```bash
python tools/analyse_gac.py \
  --config_file configs/GAC_UAD.yml \
  --checkpoint ./logs/GAC_1000_DDP/checkpoint_latest.pth \
  --output_dir ./logs/gac_diag_gd_full --tag v4_gd_full \
  --grad_batch_size 32 \
  DATASETS.ROOT_DIR /project/code/data \
  DATASETS.FILTER_GD True
```

**第 3 步：检查 `meta.json`**

```bash
python - <<'PY'
import json
for d in ['logs/gac_diag_1000_full', 'logs/gac_diag_gd_full']:
    m = json.load(open(d + '/meta.json'))
    print(d, 'bank_source=', m['bank_source'],
          'query=', m['query_counts'], 'gallery=', m['gallery_counts'],
          'grad_done=', m['grad_done'])
PY
```

full checkpoint 的 `bank_source` 应为 `checkpoint`。如果只有 `transformer_120.pth`，把 `--checkpoint` 换成它，`bank_source` 会是 `recomputed_train`，head geometry 只能当近似。

## 5. 分段 / smoke（不需要作为正式步骤）

- `--skip_rank` / `--skip_head` / `--skip_grad` / `--max_train_batches 1` 只用于排错；正式运行不要加。
- 正式运行默认就是全量 head pass + rank + gradient。
- gradient proxy 默认只取 one training batch 的前 **32 个 PKM 样本**（`--grad_batch_size 32`，3×32=96 张图），不是完整 256×3=768 张；24G 卡上通常够用。显存不足时优先降到 16 或 8；仍不足再用 `--batch_size 128`（需被 `DATALOADER.NUM_INSTANCE=4` 整除）。

CPU smoke（只在环境/代码排错时用）：

```bash
python tools/analyse_gac.py \
  --config_file configs/GAC_UAD.yml \
  --checkpoint /path/to/transformer_120.pth \
  --output_dir /tmp/gac_diag --device cpu --num_workers 0 \
  --max_train_batches 1 --skip_rank
```
