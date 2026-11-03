# WEV4 / GAC：V2 复位 + Gallery-Conditioned Auxiliary Classifier

代码基线：`WEV2/WHU`（HPD）。保留 `P_id`/`P_scene`、warm start、count EMA、DDP、诊断；新增 3 个 gallery 条件辅助头。

## 损失

```
L_total = CE + WRT + 0.01*ProCA + 1.0*GPD + aux_w * L_aux
L_aux   = mean_{query n, head m} CE( cos(f_n, W_m[c]) / 0.03 )
```

- `W ∈ [M,C,D]`，无 bias；warm start 后从 `P_scene[:,m]` 初始化；每个头作用在所有 query 模态上。
- `aux_w`: epoch 1 `0.05` → epoch 5 `0.10`，之后 constant。
- 训练-only；推理仍是 pre-BN CLS + L2 + 欧氏；不加载 heads/bank。

## 运行

4 卡 DDP：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
  --config_file configs/GAC_UAD.yml \
  MODEL.DIST_TRAIN True \
  DATASETS.ROOT_DIR /project/code/data \
  SOLVER.IMS_PER_BATCH 256 \
  SOLVER.BASE_LR 0.032 \
  OUTPUT_DIR ./logs/GAC_1000_DDP
```

单卡：

```bash
python train.py --config_file configs/GAC_UAD.yml \
  DATASETS.ROOT_DIR /project/code/data \
  SOLVER.IMS_PER_BATCH 64 \
  SOLVER.BASE_LR 0.008
```

## 1ep smoke（必须先跑）

`GPD_DIAG_PERIOD=1` 让 HPD/GAC 诊断在第一个 iteration 就执行，避免诊断日志/诊断路径的问题拖到 120ep 才发现。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
  --config_file configs/GAC_UAD.yml \
  MODEL.DIST_TRAIN True \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR /project/code/data \
  SOLVER.IMS_PER_BATCH 256 \
  SOLVER.BASE_LR 0.032 \
  SOLVER.MAX_EPOCHS 1 \
  SOLVER.CHECKPOINT_PERIOD 1 \
  MODEL.GPD_DIAG_PERIOD 1 \
  OUTPUT_DIR ./logs/GAC_smoke
```

检查：`GAC diag` 正常打印；aux loss finite；DDP/clip 正常；`checkpoint_latest.pth` 可保存。

## 中断续训

每个 `CHECKPOINT_PERIOD`（配置为 5）以及最后一个 epoch 会保存：

- `checkpoint_latest.pth`：model + optimizer + scheduler + scaler + bank + RNG + epoch；
- `transformer_{epoch}.pth`：model-only，供 test.py 使用。

从最近一次完整状态继续：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
  --config_file configs/GAC_UAD.yml \
  --resume ./logs/GAC_1000_DDP/checkpoint_latest.pth \
  MODEL.DIST_TRAIN True \
  DATASETS.ROOT_DIR /project/code/data \
  SOLVER.IMS_PER_BATCH 256 \
  SOLVER.BASE_LR 0.032 \
  OUTPUT_DIR ./logs/GAC_1000_DDP
```

也可以 `SOLVER.RESUME ./logs/GAC_1000_DDP/checkpoint_latest.pth`，不用 `--resume`。

- 续训从 checkpoint 的 `epoch+1` 开始；DataLoader 从 epoch 边界恢复，不保证 mid-epoch 精确。
- `checkpoint_latest.pth` 每 5 epoch 写一次（`CHECKPOINT_PERIOD`），崩溃最多丢 4 个 epoch；要更细可设 `SOLVER.CHECKPOINT_PERIOD 1`。
- 完整 checkpoint 会恢复 torch/cuda/numpy/python RNG，sampler 序列从 checkpoint 继续；bank type 必须匹配，否则报错。
- 若传入 `transformer_{epoch}.pth`（model-only），只恢复模型，optimizer/scheduler/bank 重置，从 epoch 1 开始；如果 key 不兼容会报错，不再静默从零开始；不推荐用于正式续训。
- `AUX_WEIGHT=0` 时严格 V2 路径：heads 已注册但跳过 forward/loss；不改变 V2 轨迹。

## 测试

```bash
python test.py --config_file configs/GAC_UAD.yml \
  TEST.WEIGHT ./logs/GAC_1000_DDP/transformer_120.pth \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  TEST.NECK_FEAT before TEST.FEAT_NORM yes \
  DATASETS.ROOT_DIR /project/code/data
```

GD 用同一 `WHU-MARS` + `DATASETS.FILTER_GD True`，不是单独的 `WHU-MARS_1000GD` 目录。

## 免训练 GAC 诊断（不训练、不改训练路径）

`tools/analyse_gac.py` 加载 V4 checkpoint，提取 pre-BN CLS，输出：

- `summary.csv` / `rank_query.csv` / `pair_rank_query.csv`：mixed、九对、AG 的 rank bins（same-camera 排除）；
- `gac_head_ce_acc.csv` / `gac_head_home_cross.csv`：per-head CE/acc、home/cross；
- `gac_head_logits.csv` / `feature_norm.csv`：aux pos/neg logits、entropy、特征 norm；
- `gac_head_geometry.csv` / `gac_head_pair_cos.csv` / `gac_head_summary.csv`：完整 `cos(W_m,P_scene[:,m'])`、head pair cos、head 近邻；
- `gac_grad.json` / `gac_aux_head_grad.csv`：`share_aux`、`cos_aux_id/scene/base/pca`、per-head grad share；
- `bank_geometry.json` / `train_geometry.json` / `meta.json`。

### 先分清两种 checkpoint

| 文件 | 是什么 | 对诊断的影响 |
|---|---|---|
| `checkpoint_latest.pth` | **full checkpoint**：model + optimizer + scheduler + scaler + **HPD bank** + RNG + epoch | 首选；bank（`P_scene/P_id`）和 heads 都是训练时精确状态 |
| `transformer_120.pth` | **model-only**：只保存 `model.state_dict()`（含 `aux_heads`、`classifier`、backbone），没有 optimizer/bank | 可用；bank 会用 train split 按 warm-start 规则重算，head geometry 是近似值 |

- `checkpoint_latest.pth` 的 `epoch` 应确认是 120；如果它是 115、而 `transformer_120.pth` 是 120，二者不是同一模型，不能混用。优先用与最终指标一致的 epoch。
- 两个文件都含 `aux_heads`；`test.py` 的 `load_param` 会跳过它，但 `analyse_gac.py` 会显式加载。

### 正式运行顺序（必须做）

**第 1 步：1000，full checkpoint（主结果，优先）**

```bash
python tools/analyse_gac.py \
  --config_file configs/GAC_UAD.yml \
  --checkpoint ./logs/GAC_1000_DDP/checkpoint_latest.pth \
  --output_dir ./logs/gac_diag_1000_full \
  --tag v4_1000_full \
  --grad_batch_size 32 \
  DATASETS.ROOT_DIR /project/code/data
```

**第 2 步：1000-GD，full checkpoint（同一 `WHU-MARS` + `FILTER_GD`）**

```bash
python tools/analyse_gac.py \
  --config_file configs/GAC_UAD.yml \
  --checkpoint ./logs/GAC_1000_DDP/checkpoint_latest.pth \
  --output_dir ./logs/gac_diag_gd_full \
  --tag v4_gd_full \
  --grad_batch_size 32 \
  DATASETS.ROOT_DIR /project/code/data \
  DATASETS.FILTER_GD True
```

**第 3 步：检查输出是否完整**

```bash
python - <<'PY'
import json, os
for d in ['logs/gac_diag_1000_full', 'logs/gac_diag_gd_full']:
    m = json.load(open(os.path.join(d, 'meta.json')))
    print(d, 'bank_source=', m['bank_source'],
          'query=', m['query_counts'], 'gallery=', m['gallery_counts'],
          'grad_done=', m['grad_done'])
PY
```

full checkpoint 的 `bank_source` 应为 `checkpoint`；如果是 `recomputed_train`，说明没有读到 bank，应检查 checkpoint 是否真的是 full checkpoint。

### 没有 full checkpoint 时（备选，不是并行做）

只有 `transformer_120.pth` 时，把上面两条命令的 `--checkpoint` 换掉，输出目录改成 `logs/gac_diag_1000_modelonly` / `logs/gac_diag_gd_modelonly`。此时 `meta.json` 的 `bank_source` 应为 `recomputed_train`，head geometry 只能当近似。

### 分段 / smoke 不需要做

- `--skip_rank`、`--skip_head`、`--skip_grad`、`--max_train_batches 1` **只用于排错**；正式诊断不要加，默认就是全量。
- CPU smoke 已在本地用合成小数据集跑过；服务器正式跑不需要再 smoke，除非正式命令报错/显存不足，才用分段定位。
- gradient proxy 默认只用 one training batch 的前 **32 个 PKM 样本**（3×32=96 张图），不是完整的 256×3=768 张；这是为了避免 24G 卡在 grad 阶段 OOM。显存不足时优先把 `--grad_batch_size` 降到 16 或 8；仍不足再调 `--batch_size 128`（需被 `DATALOADER.NUM_INSTANCE=4` 整除）。
- 不要用 `--max_train_batches` 代替正式 head pass；它只限制 head 统计的 batch 数。

说明：

- head CE/acc 只在 **train split** 有意义（query/gallery 身份与 500 个训练类不重叠）；test 只做 rank bins。
- `gac_grad.json` 是 last ViT block + final norm 的 proxy，和 `diagnose_gpd_gradients` 同口径，不是 full-ViT。
- 该脚本不创建 optimizer、不更新权重、不写 checkpoint；训练逻辑不变。

更详细的输出字段见 `tools/README_GAC_DIAGNOSTICS.md`。

## 2337

2337 是**独立数据集、独立训练**：`configs/GAC_2337.yml` 已设 `DATASET_DIR: WHU-MARS_2337`、`OUTPUT_DIR: ./logs/GAC_2337_DDP`。不要加载 1000 的 checkpoint，也不要对 2337 加 `DATASETS.FILTER_GD`。

**`GAC_UAD.yml` 本身可以复用**；`GAC_2337.yml` 只是把 2337 的 dataset/output 固定下来，防止训练/续训/测试时漏加 override。复用 `GAC_UAD.yml` 的完整训练命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
  --config_file configs/GAC_UAD.yml \
  MODEL.DIST_TRAIN True \
  DATASETS.DATASET_DIR WHU-MARS_2337 \
  DATASETS.ROOT_DIR /project/code/data \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  SOLVER.IMS_PER_BATCH 256 \
  SOLVER.BASE_LR 0.032 \
  SOLVER.MAX_EPOCHS 120 \
  OUTPUT_DIR ./logs/GAC_2337_DDP
```

先检查 2337 是否有缺模态 identity（当前 PKM sampler 假定每个 pid 三模态齐全；`cur=0` 会 `divmod(...,0)`）：

```bash
python - <<'PY'
from datasets.whu_mars import WHU_MARS
from collections import defaultdict
d = WHU_MARS(root='/project/code/data', dataset_dir='WHU-MARS_2337',
             modalities=['RGB', 'IR', 'Thermal'], verbose=True)
cnt = defaultdict(lambda: defaultdict(int))
for m, rows in d.train.items():
    for _, pid, _, _ in rows:
        cnt[pid][m] += 1
bad = [(pid, [m for m in d.modality_ls if cnt[pid][m] == 0])
       for pid in cnt if any(cnt[pid][m] == 0 for m in d.modality_ls)]
print('train pids', len(cnt), 'missing-modality pids', len(bad))
print(bad[:20])
PY
```

若 `missing-modality pids > 0`，先停，当前 sampler 会失败；需要加 skip/fallback 再跑。

训练（从 ImageNet 从零训练 2337）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
  --config_file configs/GAC_2337.yml \
  MODEL.DIST_TRAIN True \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR /project/code/data \
  SOLVER.IMS_PER_BATCH 256 \
  SOLVER.BASE_LR 0.032 \
  OUTPUT_DIR ./logs/GAC_2337_DDP
```

测试：

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --config_file configs/GAC_2337.yml \
  MODEL.DIST_TRAIN False \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR /project/code/data \
  TEST.WEIGHT ./logs/GAC_2337_DDP/transformer_120.pth \
  TEST.NECK_FEAT before \
  TEST.FEAT_NORM yes \
  OUTPUT_DIR ./logs/GAC_2337_test
```

## 服务器兼容性（PyTorch 1.12.1 + CUDA 11.3）

- 代码只使用 PyTorch >=1.12 API：`torch.cuda.amp.autocast/GradScaler`、`torch.einsum`、`torch.autograd.grad`、`torch.cuda.get/set_rng_state`、`DDP`、`torch.load/save`。
- `vit_pytorch.py` 的 `torch.load(..., weights_only=False)` 有 `try/except TypeError`，1.12 会回退到无该参数的形式。
- `torch.cuda.set_rng_state` 使用 `get_rng_state` 返回的 CPU ByteTensor，不 `.to('cuda')`；1.12/2.x 都兼容。
- `requirements.txt` 已去掉 `torch==2.2.2`/`torchvision==0.17.2` 的强制 pin；服务器请保留预装 `torch==1.12.1+cu113`、`torchvision==0.13.1`。`timm` 只用于 `RandomErasing`，`timm>=0.9,<1.1` 即可（服务器已有版本可直接用）。
- 未在 1.12.1 实机验证；本地是 torch 2.8.0+cpu。建议 1ep smoke 先在服务器确认 DDP/NCCL/AMP。

## 关键约束

- `GPD_TYPE='hierarchical'`、`GPD_WARMSTART=True`、`GPD_SUB_CENTERS==M`、`TEST.NECK_FEAT='before'`；否则报错。
- `aux_heads` 在 `build_transformer.__init__` 零初始化注册；warm start + bank broadcast 后所有 rank 从 `P_scene` 填充。
- `return_aux=False` 默认，`diagnose_gpd_gradients` 的 3 元组不变；mode≥1 推理路径不变。
- `aux_heads` 在 `load_param` 中跳过；test 不受影响。
