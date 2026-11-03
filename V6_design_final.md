# V6-A 完整设计（定稿）

> 版本：2026-11-03，S77 定稿
> 状态：三路终审 APPROVED WITH MINOR FIXES；本文件已把全部 P0/P1/P2 修入。未改代码、未训练、未 test。
> 范围：只做 A（直接 120ep）；不做 B（cached-feature probe gate）。
> 约束：单 UAD/ViT-B/16；无 CLIP；无跨模型分数融合；无多模型 ensemble；单 checkpoint；256×128。
> 重要口径：`L_corr` 主要是 **class-level cross-modal alignment**，不是强 instance-level correspondence；预期收益 0 到 +0.2pp。null/弱结果不否定 instance-level 假设，因为 centered/projection 变体未测。
>
> **取代关系：** 本文件取代 `CC/WEV5/V6_design_A_latest.md` 与 `CC/WEV5/V6_design_core.md` 中的 V6 设计；旧文件的 same-frame/camera-sampler/V2 guard 不再是最终方案。实现与报告以本文件为准。

---

## 0. 一句话

**V6-A = V4 recipe + `L_corr`（RGB–NIR / RGB–TIR / NIR–TIR 三个 pair 的 cross-camera cross-modal InfoNCE，pair-equal 1/3）+ 120ep from scratch；raw `global_feat`；无 frame/same-frame/camera sampler/all-gather；推理仍 pre-BN CLS + L2 + Euclidean。**

---

## 1. 已敲定的设计选择

| 问题 | 决定 | 依据 |
|---|---|---|
| Scope | all-cross-modal，三个 unordered pair | 覆盖 6 个 cross pair，兼顾 TIR/NIR/RGB |
| Pair 权重 | **pair-equal 1/3** | 三 pair anchors/positives/negatives 相同；TIR 两 pair 天然更难；69–71% TIR gradient 是结构结果，不是刻意加权 |
| 特征 | **raw `global_feat`** | 最小改动；与推理特征一致；承认它主要是 class-level |
| same-frame | **不做** | 当前 sampler 只有 ~12.85% same-frame 覆盖；激活需第二机制；同相机 shortcut |
| camera-stratified PKM | **不做** | 当前 per-pair 覆盖 89.5–90.2%，足够；sampler 是第二机制 |
| frame plumbing | **不做** | V6-A 不需要 frame；`camids` 已在 batch；frame 碰撞 1.9% 与本设计无关 |
| all-gather | **不做** | per-rank 120 negatives 足够；跨 rank 同 ID 不会成为 false negative，因为没有 all-gather |
| denominator | **union**：不同 ID 的全部行 | 60 same-modality different-ID + 60 cross-modality different-ID；joint intra+cross-modal discriminative denominator |
| 多正对归一化 | **mean-over-positives**，pair loss 除以固定 `2B` | per-anchor 尺度可比；DDP 归一化精确 |
| λ / warmup | **λ=0.1，10 epoch linear warmup** | epoch0 L_corr≈4.3，需要更稳 warmup |
| Base | **V4 recipe** | 历史整体指标最好；metrics-first |
| Test | **一次官方 test，final 120ep checkpoint** | 不做 B；不 fine-tune |
| Gate | **TIR mAP/R1 co-primary +0.20；STRONG +0.50** | 单一阈值，避免范围；九对 +0.05 descriptive |
| NIR→NIR | **non-target** | cross-modal loss 不直接覆盖；有 guard，无 GO endpoint |

---

## 2. Base：V4 recipe

V4 = ViT-B/16 + V2 HPD + V4 GAC + 标准 120ep DDP。

| 项目 | 设置 |
|---|---|
| Backbone | ViT-B/16，ImageNet `jx_vit_base_p16_224-80ecf9dd.pth` |
| 输入 | 256×128，mean/std=0.5 |
| 模态 | RGB / NIR(TIR) / Thermal |
| Sampler | PKM，K=4；P=16 IDs/rank × K=4 × M=3 |
| Batch | 256 tuples，4 卡 DDP；每 rank 64 tuples / 192 rows |
| Epoch | 120 |
| Optimizer | SGD，LR 0.032，warmup 5，cosine |
| Weight decay | 1e-4；bias LR ×2 |
| Seed | 1234 |
| Loss | CE + WRT + 0.01 ProCA + 1.0 GPD_HPD + 0.1 GAC_aux |
| HPD | 0.25·CE(P_id) + 0.75·CE(P_scene)，K=3，τ=0.03 |
| GAC | 3 个 train-only aux heads `[3,500,768]`，初始化自 P_scene；推理丢弃 |
| Checkpoint | 每 5 epoch |
| 推理 | pre-BN CLS + L2 + Euclidean；`NECK_FEAT=before`；`RE_RANKING=False` |

V4 历史结果：1000 11.56/30.80/47.98/55.93；九对 R1/mAP 19.97/12.88；TIR cross 7.85/17.61/24.07/5.64。

### 2.1 归因边界（必须写进报告）

- V4 是 **reference recipe，不是 clean baseline**：它包含 train-only GAC，GAC 自己的 mAP gate 未过；V4−V2 在单 seed 噪声内（1000 +0.09/+0.28；九对 +0.50/+0.12）。
- 本实验 **不跑 λ=0 的 120ep control**；1ep λ=0 smoke 只验证代码路径，不是 120ep null。
- `V6−V4` 是 **recipe-level comparison**，不是“L_corr alone”的因果结论。
- V5 corrected probe 的 class-mean 警告是 V2-based；V4 val-only 提示类似的 seen-ID drop，但 V4 test 未跑；V6 是 V4 的第一次 test。
- 不做 V2 归因主跑；如需严格归因，另立 V2+corr 或 V4 λ=0 120ep control。

---

## 3. 数据流（精确）

```text
dataset.train[m]: list[(img_path, pid, camid, mindex)]
  mindex 1-based；camid 0..6；pid 0..499
  -> PKMSampler_DDP:
       每 rank 64 tuples = 16 IDs × K=4
       每个 tuple = (idx_RGB, idx_NIR, idx_TIR)
  -> ImageDataset.__getitem__(tuple):
       imgs[M] × [3,256,128]
       pid: scalar
       camids[M]: list of scalar
  -> train_collate_fn:
       imgs[M][B,3,256,128]
       pids[B]
       camids[M][B]
  -> model.forward(mode=0):
       x = torch.cat(imgs, dim=0)  # [M*B,3,H,W]
       global_feat = base(x)       # [M*B,768], pre-BN CLS
       feat = bottleneck(global_feat)
       cls_score = classifier(feat)
  -> processor:
       target_rep = target.repeat(M)                  # [M*B]
       cams       = torch.cat(camids)                 # [M*B]
       mods       = torch.arange(M).repeat_interleave(B)  # [0]*B+[1]*B+[2]*B
```

**绝不能用的写法：**

- `mods = torch.arange(M).repeat(M)`；
- `cams = torch.stack(camids).t().reshape(-1)`；
- 使用 dataset 的 1-based `mindex` 作为 `mods`；
- 对 `global_feat` 做 transpose/reshape 后再切 pair；
- 用 Python `|` 连接未加括号的 `mods==a`/`mods==b`。

行 `r` 的映射固定为：

```text
m = r // B
i = r % B
pid = target_rep[r]
cam = cams[r]
mod = mods[r]
```

`IR = NIR`，`Thermal = TIR`；顺序由 `DATASETS.MODALITIES` 决定。

---

## 4. `L_corr` 定义

### 4.1 Pair

```text
P01 = (mod 0, mod 1)  # RGB–NIR
P02 = (mod 0, mod 2)  # RGB–TIR
P12 = (mod 1, mod 2)  # NIR–TIR
```

对每个 pair `(a,b)`：

```text
cand_rows = rows where mod in {a,b}      # 2B rows
z = L2_normalize(global_feat[cand_rows].float(), dim=1)
```

### 4.2 Mask

```text
same_pid = pid[:,None] == pid[None,:]
same_mod = mod[:,None] == mod[None,:]
same_cam = cam[:,None] == cam[None,:]
eye      = torch.eye(2B, dtype=bool, device=z.device)

pos  = same_pid & ~same_mod & ~same_cam
excl = eye | (same_pid & same_cam) | (same_pid & same_mod)
den  = ~excl
valid = pos.any(dim=1)
```

- `pos`：同 ID、跨模态、cross-camera；
- `excl`：self、same-ID same-camera、same-ID same-modality；
- `den`：positives + 60 same-modality different-ID + 60 cross-modality different-ID = **positives + 120 negatives**；
- `den` 是 joint intra+cross-modal discriminative denominator，不是纯 cross-modal 对应；
- `valid`：至少一个正对的 anchor；约 89.5–90.2%；空正对 anchor 的 loss 置 0，**但仍计入 2B 分母**；不做 valid-count 重归一化；
- 本设计使用 **no same-camera cross-modal positives**（proxy；不使用 frame metadata；frame 号碰撞 1.9% 与本设计无关）。

### 4.3 Loss（NaN-safe）

```python
pos_f   = pos.to(scores.dtype)
pos_cnt = pos_f.sum(dim=1)
pos_mean = (scores * pos_f).sum(dim=1) / pos_cnt.clamp_min(1)
den_scores = scores.masked_fill(~den, -1e9)
l = torch.logsumexp(den_scores, dim=1) - pos_mean
l = torch.where(pos_cnt > 0, l, torch.zeros_like(l))
pair_loss = l.sum() / float(2 * B)
```

- `scores = z @ z.t() / tau`，`tau=0.1`；
- `pos_cnt.clamp_min(1)` 是 P0：避免空 anchor 的 0/0 在 backward 产生 NaN；
- `torch.where` 只 mask forward，不能替代 clamp；
- `den_scores` 用 fp32 与大负数，不用 `-inf`；
- `l_i` 对正对取 mean；空 anchor 为 0；
- `pair_loss` 固定除以 `2B`（B=64/rank；2B=128）；
- `L_corr = (1/3)(pair_loss_01 + pair_loss_02 + pair_loss_12)`；
- 如果某个 pair 全部无效（概率极低），该 pair loss=0；不重新分配权重；
- 全 loss：`L_total = L_V4 + λ_corr * L_corr`，`λ_corr = 0.1 * min(1, epoch/10)`；
- `epoch` 与 `gac_aux_weight` 同一 1-based 语义；resume 从 `last_epoch+1` 继续。

### 4.4 多正对/尺度

- `logsumexp(den) - mean(pos)` 是标准 SupCon mean-over-positives；
- 每个 anchor 的梯度尺度与 `|pos|` 无关；
- 固定 2B 在 DDP 下是精确的 rank-slot 均值；空 anchor 计 0；
- 不需要 all-reduce/all-gather。

### 4.5 梯度

```text
L_corr -> global_feat -> ViT
```

- 只回 `global_feat` → ViT；
- 不经过 classifier/BN/GAC aux heads/GPD bank；
- `global_feat` 不 detach；
- `L_corr` 内 `float()` 且 **autocast 关闭**（见 §5）；
- per-rank，不 all-gather；
- 与 CE/WRT/ProCA/GPD/GAC 共享 ViT 梯度；监控 share/clip。

### 4.6 科学口径：class-level，不是强 instance-level

- 实测 raw `L_corr` 的 class-center 梯度约 **60.7–61.6%**（all-modality pid center；定义依赖；per-(pid,mod) 中心约 36.6%）；
- 训练后 raw loss 在 V4 特征上约 1.3，epoch0 约 4.3；不同 checkpoint/feature 定义下约 0.5–1.3；
- V5 corrected probe：TIR cross class-mean cos +0.0143，但 TIR cross mAP −0.104、TIR same −1.059、mixed −0.265；
- 因此：
  - 本实验主要测试 **class-level cross-modal alignment**；
  - 预期收益 **0 到 +0.2pp**；
  - null/弱结果 **不否定** instance-level correspondence，因为 centered/projection 变体未测（A2，不在本轮）；
  - 报告中不得把 class-mean 改善写成 instance matching 改善。

---

## 5. 实现

### 5.1 新文件

```text
CC/WEV6/WHU/loss/corr_loss.py
```

函数：

```python
def corr_loss(global_feat, target_rep, camids, num_modalities, tau):
    # returns loss, diag
```

- stateless：无参数、无 buffer、无 RNG；
- `camids` 为 list[M][B]；
- 内部构造 `cams = torch.cat(camids).to(device)`、`mods = torch.arange(M, device=device).repeat_interleave(B)`；
- 内部 `with torch.cuda.amp.autocast(enabled=False):`；
- `scores.dtype == torch.float32` 断言；
- 返回 `loss`、`diag`（每 pair loss、valid count、pos count、pos/neg cosine）。

### 5.2 `processor/processor.py`

- 在 `with amp.autocast(enabled=True):` 块 **之后**、`scaler.scale(loss).backward()` **之前**，12-space loop scope 插入；
- **不要**放在 `if current_aux_weight > 0:` 或 `if use_aux_modules:` 内部；
- 代码：

```python
if corr_enabled and corr_w > 0:
    with torch.cuda.amp.autocast(enabled=False):
        loss_corr, corr_diag = corr_loss(
            global_feat.float(), target_rep, camids,
            num_modalities, tau)
    loss = loss + corr_w * loss_corr
```

- `corr_w = cfg.MODEL.CORR_LOSS.WEIGHT * warmup_factor(epoch, cfg.MODEL.CORR_LOSS.WARMUP_EPOCHS)`；
- `corr_enabled = cfg.MODEL.CORR_LOSS.ENABLED`；
- `warmup_factor` 用 1-based epoch；`WARMUP_EPOCHS<=0` 时 guard 为 1；
- 记录 `corr_diag` 到 logger（每 LOG_PERIOD）；
- `ENABLED=False` 或 `WEIGHT=0` 完全跳过，不分配 tensor。

### 5.3 `config/defaults.py`

```python
_C.MODEL.CORR_LOSS = CN()
_C.MODEL.CORR_LOSS.ENABLED = False
_C.MODEL.CORR_LOSS.WEIGHT = 0.0
_C.MODEL.CORR_LOSS.TAU = 0.1
_C.MODEL.CORR_LOSS.WARMUP_EPOCHS = 10
```

### 5.4 V6 config

复制 `WEV5/WHU/configs/GAC_UAD.yml` → `WEV6/WHU/configs/V6_CORR.yml`，加：

```yaml
MODEL:
  CORR_LOSS:
    ENABLED: True
    WEIGHT: 0.1
    TAU: 0.1
    WARMUP_EPOCHS: 10

OUTPUT_DIR: './logs/V6_CORR_1000_DDP'
```

- 不在 YAML 写 `MODEL.DIST_TRAIN True`（test.py 共享 config；DDP 由命令行/torchrun 控制）；
- 不修改 `model/make_model.py`；
- 只修改 `datasets/sampler.py`、`datasets/sampler_ddp.py`、`datasets/make_dataloader.py`（2337 skip 支持）；其他 `datasets/*` 不动；
- 不修改 `test.py`、`train.py`；
- `train.py` 不改；
- 新实验目录 `CC/WEV6/WHU` 从 `CC/WEV5/WHU` 复制，保留 V5 历史。

### 5.5 服务器同步清单

**新增/修改：**

- `CC/WEV6/WHU/loss/corr_loss.py`（新增）
- `CC/WEV6/WHU/config/defaults.py`（加 `MODEL.CORR_LOSS`、`DATALOADER.SKIP_INCOMPLETE_IDS`）
- `CC/WEV6/WHU/processor/processor.py`（插入 loss 调用）
- `CC/WEV6/WHU/datasets/sampler.py`（2337 缺模态 skip；1000 无缺模态时 no-op）
- `CC/WEV6/WHU/datasets/sampler_ddp.py`（同上）
- `CC/WEV6/WHU/datasets/make_dataloader.py`（传 skip flag）
- `CC/WEV6/WHU/configs/V6_CORR.yml`（新增，1000）
- `CC/WEV6/WHU/configs/V6_CORR_2337.yml`（新增，2337；DATASET_DIR/OUTPUT_DIR 独立）
- `CC/WEV6/WHU/tools/v6_data_preflight.py`（新增，1000/2337/GD 适配性预检）
- `CC/WEV6/WHU/tools/v6_eval.py`（新增，per-query AP / bootstrap / AG / nine-pair）
- `CC/WEV6/WHU/tools/v6_delta_ci.py`（新增，V6 vs V4 paired identity-cluster ΔmAP/ΔR1 CI）

**不要覆盖：**

- `model/make_model.py`
- `datasets/` 中除 `sampler.py`、`sampler_ddp.py`、`make_dataloader.py` 之外的文件
- `test.py`
- `train.py`
- 旧 `WEV5/logs/*`

**服务器路径：**

- `/project/code/WHU_base/WEV6/WHU`
- 数据/预训练路径沿用 V4/V5。

---

## 6. 训练、测试、gate

### 6.1 训练

- 从零 120ep；
- 4 卡 DDP，batch 256，LR 0.032，warmup 5，cosine，seed 1234；
- checkpoint 每 5 epoch；
- 支持 resume；**必须用 full checkpoint**（model-only resume 会把 `start_epoch` 重置为 1，λ warmup 会重来）；
- `CORR_LOSS.ENABLED=False` / `WEIGHT=0` 必须严格复现 V4 loss 路径；
- 1ep smoke 使用临时目录，正式 120ep 从 epoch 0 开始；不做 fine-tune。

### 6.2 测试

- 最终 120ep checkpoint；
- 官方 test 一次：500 unseen IDs，query/gallery，九对 + nine_pair + mixed + AG A→G/G→A；
- 模型是 V6 完整模型，不是 V4 checkpoint，不是 head，不是 probe；
- 不在 test 上调 λ/τ/层/采样；
- 第二 seed 只在 GO 后补；
- validation 用官方 test query/gallery，仅监控，不选模型/λ/τ。

### 6.3 Gate（单一阈值）

**Primary（co-primary）：**

- TIR cross 4 对 mean mAP ≥ **+0.20pp**（绝对 5.84）；
- TIR cross 4 对 mean R1 ≥ **+0.20pp**（绝对 8.05）；
- paired identity-cluster bootstrap 95% CI lower >0，**mAP 和 R1 分别计算**；
- ≥4/5 identity folds 同向（descriptive，不是显著性）；fold 定义固定为 sorted unique test q_pid，index mod 5，由 `v6_delta_ci.py` 输出 `fold_delta_mAP/R1` 和 `n_positive_folds_*`。

**STRONG_GO：**

- TIR cross mAP ≥ **+0.50pp**（6.14）；
- TIR cross R1 ≥ **+0.50pp**（8.35）。

**WEAK / INCONCLUSIVE：**

- Δ 在 (0, +0.20)；
- 或 mAP/R1 只有一项为正；
- 或 CI 包含 0；
- 不算 mechanism failure；不宣布 GO。

**Per-pair TIR guard：**

- 四个 TIR cross pair 的单对 mAP 不低于 V4−0.5：
  - RGB→TIR ≥5.21；NIR→TIR ≥4.87；TIR→RGB ≥5.53；TIR→NIR ≥4.93。
- 防止单一方向回退被 4-pair mean 掩盖。

**NIR→NIR（non-target）：**

- mAP ≥17.44（V4 17.94−0.5）；
- R1 ≥28.56（V4 29.56−1.0）；
- 不设 GO endpoint；报告。

**其他 guards：**

- same 3 对 mAP ≥ V4−0.5；
- TIR same mAP ≥14.06、R1 ≥21.01（V4 guard）；
- mixed mAP ≥ V4−0.10；mixed R1 ≥ V4−1.0；
- 1000 mAP/R1 ≥ V4−0.3（R1 30.50；注意这落在历史单 seed R1 噪声 1.13 内，是 practical red line，不是统计显著回归）；
- AG A→G/G→A mAP ≥11.90/13.70；GD mAP ≥14.64；R1 报告；
- 单对 R1 不低于 V4−1.0。

**统计：**

- 单 seed provisional；
- 第二 seed **5678**；GO confirmed 仅当 two-seed mean mAP/R1 ≥ +0.20 且两 seed 同向；否则 provisional/inconclusive；
- V4 reference 仍是单 seed；两 seed V6 不能修正 V4 baseline noise；paper+1 不能由单 seed 宣布；
- 官方 test 已被 V1–V5 复用，不是 fresh holdout；不做 paper+1 声明；
- 一次官方 test per final checkpoint；
- 不用 last-3/8-12 当官方指标。

---

## 7. 监控 / 诊断

每次 LOG_PERIOD：

- 每 pair：valid anchor count / 2B、positives/valid anchor、loss mean；
- 每 pair：pos/neg cosine mean；
- 全局：`L_corr` 值、`λ_corr`；
- 每 pair gradient energy share；TIR pair share；
- 每 modality gradient energy share；
- same-modality vs cross-modality negative gradient decomposition；
- class-center vs residual gradient proxy（定义写清楚：all-modality pid center vs per-(pid,mod) center）；
- **实现说明**：`corr_diag` 只带 loss/coverage/cosine；gradient share 与 class-center 需要单独的 periodic corr-only diagnostic backward（main rank, retain_graph），或者只报告 loss share 并明确“gradient share 未测”。不要声称未实现的监控。
- `clip_count`、pre-clip grad norm；
- GAC `share_aux`、HPD bank geometry；
- validation：TIR cross/same、same、mixed、RGB–NIR、NIR–RGB；
- TIR cross/same class-mean cos（Q→G）；
- 离线（不是 training-time）：per-pair >100/>500 rank bins；从 test per-query CSV 计算。

**Active check：**

- 如果实现了 periodic corr-only diagnostic：`L_corr` gradient share ≥1% of pre-clip grad norm；
- 如果没有实现该 diagnostic：只报告 `L_corr` loss share，并明确写“gradient share 未测”；**不得用未测的 gradient share 作为 NO_GO 依据**；
- `L_corr` 在前几个 epoch 下降；
- 如果 loss share 近似 0 或完全不降：INCONCLUSIVE，不是 NO_GO；先查 mask/coverage/λ/autocast。

---

## 8. 1ep smoke / 自检

### 8.1 关闭路径

- `CORR_LOSS.ENABLED=False`：
  - 模型参数/optimizer/state_dict keys 与 V4 完全一致；
  - loss 路径不调用 `corr_loss`；
  - 1ep 无 NaN，DDP 正常。
- `ENABLED=True, WEIGHT=0`：同样完全跳过。

### 8.2 开启路径

- `L_corr` 有限、>0；
- 每 pair valid coverage 约 89.5–90.2%（epoch/running mean；不是每 batch >0.8）；
- 每 pair positives/valid 约 2.2；
- pos/neg cosine 合理；
- **backward 后所有非 None 参数梯度 finite**；
- `L_corr` gradient share 非零；
- `mods/cams/pids` 行序断言通过；
- `scores.dtype == torch.float32`；
- `DEN` 每行 **120 different-ID negatives + positives**；
- DDP 4 卡 per-rank loss 有限、同量级；**不要断言 loss 相等**；梯度 all-reduced；
- full checkpoint resume 正常；model-only resume 会重置 λ，不作为 resume 测试；
- 1ep 后 `transformer_1.pth` 可加载。

### 8.3 断言 / 单元测试

```python
assert global_feat.shape[0] == M * B
assert cams.shape[0] == M * B
assert mods.shape[0] == M * B
assert torch.equal(mods, torch.arange(M, device=mods.device).repeat_interleave(B))
assert ((den & ~pos).sum(1) == 2 * B - 2 * K).all()  # 120 different-ID negatives
assert (den.sum(1) == 2 * B - 2 * K + pos.sum(1)).all()  # den includes positives
assert (pos & den).eq(pos).all()
assert den.any(1).all()
assert scores.dtype == torch.float32
assert torch.isfinite(loss_corr)
```

- 每 pair 断言 `mods[:B]==a`、`mods[B:]==b`、`pid[:B]==pid[B:]`；
- 合成 reference test：**M=3**（代码只接受 3 模态）、B 小、已知 pid/cam；显式 loop 计算 mask/loss，与函数一致；
- 至少一个 empty-pos anchor + backward，断言梯度 finite；
- 覆盖率断言用 epoch/running mean 在 [0.85, 0.95]；单 batch 可放宽到 >0.6；
- 不在 loss 内写 `assert valid.mean()>0.8`（会随机失败）。

---

## 9. 服务器命令（实现后）

### 9.1 建树

```bash
cd /project/code/WHU_base
rsync -a --exclude 'logs' --exclude '__pycache__' --exclude '*.pth' \
  WEV5/WHU/ WEV6/WHU/
# 然后覆盖/新增：
#   WEV6/WHU/loss/corr_loss.py
#   WEV6/WHU/config/defaults.py
#   WEV6/WHU/processor/processor.py
#   WEV6/WHU/datasets/sampler.py
#   WEV6/WHU/datasets/sampler_ddp.py
#   WEV6/WHU/datasets/make_dataloader.py
#   WEV6/WHU/configs/V6_CORR.yml
#   WEV6/WHU/configs/V6_CORR_2337.yml
#   WEV6/WHU/tools/v6_data_preflight.py
#   WEV6/WHU/tools/v6_eval.py
#   WEV6/WHU/tools/v6_delta_ci.py
```

### 9.2 1ep smoke

```bash
cd /project/code/WHU_base/WEV6/WHU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
  --config_file configs/V6_CORR.yml \
  MODEL.DIST_TRAIN True \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR /project/code/data \
  SOLVER.MAX_EPOCHS 1 \
  SOLVER.CHECKPOINT_PERIOD 1 \
  OUTPUT_DIR ./logs/V6_CORR_1000_smoke
```

可选 `CORR_LOSS` off smoke：`MODEL.CORR_LOSS.ENABLED False` + `OUTPUT_DIR ./logs/V6_CORR_1000_smoke_off`。

### 9.3 120ep

```bash
cd /project/code/WHU_base/WEV6/WHU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
  --config_file configs/V6_CORR.yml \
  MODEL.DIST_TRAIN True \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR /project/code/data \
  OUTPUT_DIR ./logs/V6_CORR_1000_DDP
```

### 9.4 官方 test（一次；用 v6_eval，不再另跑 test.py）

先 V4 sanity（不是 V6 test）：

```bash
cd /project/code/WHU_base/WEV6/WHU
python tools/v6_eval.py \
  --config_file configs/GAC_UAD.yml \
  --weight /project/code/WHU_base/WEV4/logs/GAC_1000_DDP/transformer_120.pth \
  --output ./logs/V4_SANITY \
  --bootstrap 10000 \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth
```

V6 官方 test（一次；不跑 test.py）：

```bash
python tools/v6_eval.py \
  --config_file configs/V6_CORR.yml \
  --weight ./logs/V6_CORR_1000_DDP/transformer_120.pth \
  --output ./logs/V6_CORR_1000_DDP \
  --bootstrap 10000 \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth
```

- `v6_eval.py` 同时输出九对 + nine_pair + same3 + cross6 + TIR cross + mixed + AG A→G/G→A、`v6_per_query.csv`、`v6_summary.json` 和 absolute identity-cluster CI；
- `v6_summary.json` 的 CI 是 **absolute mAP CI**，不是 gate 的 paired Δ CI；gate 的 ΔmAP/ΔR1 CI 由 `v6_delta_ci.py` 对 V6/V4 两份 per-query CSV 计算：
```bash
python tools/v6_delta_ci.py \
  --v6_csv ./logs/V6_CORR_1000_DDP/v6_per_query.csv \
  --v4_csv ./logs/V4_SANITY/v6_per_query.csv \
  --scope tir_cross \
  --output ./logs/V6_CORR_1000_DDP/tir_cross_delta_ci.json
```
- 它是 V6 的**唯一官方 test**；不要再对同一 checkpoint 跑 test.py，避免重复 test；
- 实现后先用 V4 checkpoint 跑一次 sanity，确认 v6_eval 的九对/mixed 与 V4 官方指标一致，再用 V6 final checkpoint 跑正式 test；V4 sanity 不是 V6 test。

### 9.5 GD

```bash
python tools/v6_eval.py --config_file configs/V6_CORR.yml \
  --weight ./logs/V6_CORR_1000_DDP/transformer_120.pth \
  --output ./logs/V6_CORR_1000_DDP_gd \
  --bootstrap 10000 \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.FILTER_GD True
```

- `v6_eval.py` 支持尾部 yacs opts；GD 只作护栏，不参与 GO。

---

## 9.5 数据集适配：1000 / 2337 / 1000-GD

V6 的代码/方案必须同时支持三个数据集；当前实现已做通用化，但 **2337 必须先在服务器做数据 preflight，不能直接训练**。

| 项目 | 1000 | 2337 | 1000-GD |
|---|---|---|---|
| 角色 | 主实验 / 论文 Table 6 | 独立数据规模/泛化实验 | 1000 权重的 GD 评测 |
| train IDs / classes | 500 | **1,000** | 不训练（500 权重） |
| train images/模态 | 30,711 | 约 75k（296 iters ×256） | — |
| iters/epoch | 117 | **296** | — |
| query/gallery | 2,135 / 31,203 per modality | 由 2337 数据集动态读取 | FILTER_GD 子集；984 valid overall query |
| 相机 | 7（0–4 ground，5–6 aerial） | 需 preflight 确认 | 同 1000 |
| 缺失模态 | 无（500 IDs 三模态齐全） | **训练集实测 0**；test TIR 8 IDs 缺失 | 不训练，不影响 loss |
| V6 训练 | 是 | **是；训练集三模态齐全** | 否 |
| V6 loss | 直接适用 | 直接适用；sampler skip 仅作安全阀 | 不计算 |
| Gate | V4 1000 baseline +0.20 | 单独 baseline/gate，不能用 1000 | V4 GD baseline；只护栏 |
| 能否混表 | 基准 | **不能混入 1000** | 不能混入 1000 |

### 2337 已知风险（必须在 preflight 确认）

历史记录/README 明确提示：

- 2337 某些 `(pid, modality)` 组稀疏，TIR 更少；
- `PKMSampler_DDP.sample_list` 对 `cur=0` 会 `divmod(need, 0)`；
- GPD warm start 需要 valid mask/身份回退；
- 当前 `ImageDataset` tuple 和 `corr_loss` 都假设 **M=3 且每个 tuple 三模态齐全**。

V6 已加 **`DATALOADER.SKIP_INCOMPLETE_IDS=True`**（V6 两个 config 均开启）：
- sampler 构造时过滤掉任一模态为 0 的 train ID；
- 1000 三模态齐全时是 no-op，pids/length/RNG 不变；
- 2337 若存在缺模态 ID，则只训练三模态齐全的 IDs，并在 rank0 打印 skipped 数量；
- class count 仍按 dataset 全部 train IDs 构建（1000/2337 的 GPD/AUX 形状不变），被 skip 的类不参与采样；GPD warm start 对缺失模态用 id_mean 填充，因此 GAC head 不会全零。

### Preflight 实测（2026-11-03）

已运行并下载：

- `logs/v6_preflight_1000.json`
- `logs/v6_preflight_2337.json`
- `logs/v6_preflight_gd.json`

关键结果：

- **1000**：train 500 IDs；`missing_modality_pids=0`；cameras 0–6；2.392 cameras/ID；query/gallery 2,135/31,203。
- **2337**：train **1,000 IDs**；`missing_modality_pids=0`；`zero_image_groups=0`；2.444 cameras/ID；train RGB/NIR/TIR 77,253/76,975/63,276；query RGB/NIR/TIR 5,254/5,275/4,491；gallery 78,639/78,680/59,797。
  - **2337 训练集没有 missing modality**；`SKIP_INCOMPLETE_IDS` 在 2337 训练中是 no-op。
  - **2337 test 侧 TIR 有 8 个 IDs 缺失**：query/gallery RGB/NIR 1,337 IDs，TIR 1,329 IDs；这是 query/gallery，不影响训练 loss；`v6_eval` 按模态动态读取，TIR pair 的 n 会少 8 IDs。
- **1000-GD**：query/gallery 112 IDs；399/6,836 per modality；cameras 3–4；**无 aerial**；AG 不可用/会跳过；GD 只作 overall/九对护栏。

因此 2337 的训练集判定已从“待确认”变为：

- `missing_modality_pids=0` → **2337 训练路径成立**；使用 `V6_CORR_2337.yml` 直接训练；
- `SKIP_INCOMPLETE_IDS=True` 保留为未来数据副本/其他 2337 版本的安全阀；
- 2337 test TIR 缺 8 IDs，不阻塞训练；评测时不要假设 TIR pair 的 n=2,135/31,203。

preflight 的 `pkm_coverage` 只用于看数量级/相对差异：旧 JSON 的 `valid_ratio` 是 tuple-level（约 0.49–0.52）；更新后的工具输出 `anchor_valid_ratio`/`positives_per_anchor`/`tuple_cross_ratio`，但仍是 with-replacement 近似。**真正的 anchor-level coverage 以 1ep smoke 的 `V6 corr diag valid x/128` 为准（预期 ~0.89–0.90，positives/valid ~2.2）**；不要用 preflight 数值直接作为 gate。

### 2337 preflight 命令（供复核）

```bash
python tools/v6_data_preflight.py \
  --root /project/code/data \
  --dataset_dir WHU-MARS-2337 \
  --modalities RGB,IR,Thermal \
  --out ./logs/v6_preflight_2337.json
```

输出必须检查：

- `train_structure.missing_modality_pids`；
- `train_structure.zero_image_groups`；
- `train_structure.cameras_per_id_*`；
- `pkm_coverage.per_pair` / `all_cross_modal`；
- `splits` / `ag_camera_counts` / `raw_id_overlap`。

**判定：**

- `missing_modality_pids == 0`：V6 2337 可直接用 `V6_CORR_2337.yml` 训练；仍需确认 per-pair coverage；
- `missing_modality_pids > 0`：先停；选择：
  1. 仅保留三模态齐全的 train IDs（会改变 class count，需要重建 GPD/AUX/classifier）；
  2. 修 sampler 跳过缺模态 ID，并同步 `num_classes`/bank；
  3. 若缺模态很多，V6 当前 M=3 设计不适用，需要单独 modality-missing 方案。
- **禁止**直接改 `corr_loss` 去“补零模态”；零向量会污染 InfoNCE。

### 1000-GD

- 1000-GD **不是训练集**；用 1000 训练的 V6 final checkpoint，`DATASETS.FILTER_GD True` 评测；
- `L_corr` 完全不参与；
- `v6_eval.py` 通过尾部 yacs opts 支持 GD；
- GD 仅作护栏，不作为 GO endpoint；
- 不要训练 V6-GD，不要用 GD 过滤训练集；
- 实测 GD 只有 cameras 3–4、**没有 aerial**，所以 AG A→G/G→A 在 GD 上不可用/会跳过；GD 只报告 overall/九对。

### 1000 preflight

```bash
python tools/v6_data_preflight.py \
  --root /project/code/data --dataset_dir WHU-MARS \
  --modalities RGB,IR,Thermal --out ./logs/v6_preflight_1000.json
```

实测：`missing_modality_pids=0`，500 IDs，2.392 cameras/ID，cameras 0–6；query/gallery 2,135/31,203。anchor-level coverage 以 1ep smoke 的 `V6 corr diag` 为准。

### 2337 训练/评测

训练：`configs/V6_CORR_2337.yml`，从 ImageNet 从零训练，**不加载 1000 checkpoint**，不加 `FILTER_GD`。该 config 使用 `DATASET_DIR='WHU-MARS-2337'`（与历史 2337 训练日志一致）；若服务器实际目录是 `WHU-MARS_2337`，用命令行 `DATASETS.DATASET_DIR WHU-MARS_2337` 覆盖。

```bash
# 1ep smoke
cd /project/code/WHU_base/WEV6/WHU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
  --config_file configs/V6_CORR_2337.yml \
  MODEL.DIST_TRAIN True \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR /project/code/data \
  SOLVER.MAX_EPOCHS 1 SOLVER.CHECKPOINT_PERIOD 1 \
  OUTPUT_DIR ./logs/V6_CORR_2337_smoke

# 120ep
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
  --config_file configs/V6_CORR_2337.yml \
  MODEL.DIST_TRAIN True \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR /project/code/data \
  OUTPUT_DIR ./logs/V6_CORR_2337_DDP
```

评测：用 `v6_eval.py` 的 2337 config；必须先有 UAD 2337 或 V4 2337 的 per-query CSV 才能算 paired Δ；不能套 1000 的 +0.20。

```bash
# UAD 2337 baseline（路径按服务器实际调整）
python tools/v6_eval.py --config_file configs/V6_CORR_2337.yml \
  --weight /project/code/WHU_base/WHU/logs/UAD_2337_DDP/transformer_120.pth \
  --output ./logs/UAD_2337_SANITY --bootstrap 10000 \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth

# V6 2337 official test（一次）
python tools/v6_eval.py --config_file configs/V6_CORR_2337.yml \
  --weight ./logs/V6_CORR_2337_DDP/transformer_120.pth \
  --output ./logs/V6_CORR_2337_DDP --bootstrap 10000 \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth

# paired delta（示例；baseline CSV 必须是 2337 模型）
python tools/v6_delta_ci.py \
  --v6_csv ./logs/V6_CORR_2337_DDP/v6_per_query.csv \
  --v4_csv ./logs/UAD_2337_SANITY/v6_per_query.csv \
  --scope tir_cross --output ./logs/V6_CORR_2337_DDP/tir_cross_delta_ci.json
```

2337 TIR test 只有 1,329 IDs（不是 1,337），TIR pair 的 n 会小于 RGB/NIR；不要套 1000 的 n。2337 gate 单独定，不能套 1000 的 +0.20。

## 10. 明确不做

- 不做 B / cached-feature probe gate；
- 不做 fine-tune；
- 不做 same-frame / same-camera cross-modal positives；
- 不做 camera-stratified PKM；
- 不做 frame plumbing；
- 不做 all-gather；
- 不做 V6b；
- 不做 rerank/bank/heads 推理；
- 不做 class-level 新 head/原型；
- 不做 2337/288×144/test 调参；
- 不把 raw loss 宣称为强 instance-level。

---

## 11. 已闭合问题

| 问题 | 最终决定 | 依据 |
|---|---|---|
| Pair 权重 | pair-equal 1/3 | 三 pair 数量相同；TIR 70.6% 是结构结果；0.2/0.4/0.4 未验证 |
| Denominator | union 120 negatives（60 same-mod + 60 cross-mod） | joint intra+cross；文档；梯度分解诊断 |
| 多正对 | mean-over-positives + fixed 2B | SupCon 标准；DDP 精确 |
| 空 anchor | loss=0，仍计入 2B；clamp_min(1) | 防 NaN backward |
| Raw vs residual | raw A；class-level；A2 另立 | 60.7–61.6% class-center；预期 0–0.2 |
| V4 base / GAC | V4；reference；monitor | metrics-first；归因边界已写 |
| Gate | TIR mAP/R1 +0.20；STRONG +0.50；INCONCLUSIVE | 单一阈值；CI 两项；per-pair guard |
| NIR→NIR | non-target；guard 17.44/28.56 | cross-modal 不直接覆盖 |
| same-frame/sampler | 不做 | 覆盖/第二机制/shortcut |
| frame | 不做 | 不需要；camids 已有 |
| all-gather | 不做 | per-rank 120 negatives；same-ID 跨 rank 不构成 false negative |
| B | 不做 | 用户指定只 A |
| 1ep smoke | NaN-safe + 行序 reference + finite grad + coverage epoch mean | P0/P1 修复 |
| 命令/复制 | §9 | 完整 |

---

## 12. 终审签收

- **Design audits**：Code / Scientific / Edge 三路均 APPROVED WITH MINOR FIXES；P0/P1 已修入本文件。
- **Implementation audits（WEV6）**：Code / Scientific 为 APPROVED WITH MINOR FIXES，Edge 最终 **APPROVED**；训练核心无 P0；
  - `v6_eval.py` 增加 AG A→G/G→A、nine_pair、class-mean cos；
  - 新增 `v6_delta_ci.py`，提供 V6−V4 paired identity-cluster ΔmAP/ΔR1 CI 和 5 identity folds；
  - `den` 断言修正为 `(den & ~pos).sum == 120`、`den.sum == 120 + pos.sum`；
  - 日志改为 `positives_per_valid`；代码注释改为 class-level alignment；
  - 混合 CSV 的 `g_mod` 改为 -1；AG 行同样 -1；`cross6` 过滤只取九对；
  - AG 方向已 AST 级核对：A→G = `cams>=5` query → `cams<5` gallery；G→A 反之；
  - `--bootstrap 0` 的 aggregate guard 已加。
- **2337 适配补充**：新增 `V6_CORR_2337.yml`、`v6_data_preflight.py`、`DATALOADER.SKIP_INCOMPLETE_IDS` 与 sampler skip；1000/2337 训练集 preflight 均 `missing_modality_pids=0`，skip 为 no-op；2337 train 1,000 IDs/296 iters，test TIR 8 IDs 缺失（1,329 vs 1,337），训练不受影响；1000-GD 112 query/gallery IDs、无 aerial、AG 跳过；class count/GPD/AUX 形状不变。
- **当前状态**：`WEV6/WHU` 代码已写入；`compileall`、`corr_loss` selftest、真实缓存特征 loss、config merge、sampler skip 合成测试、`v6_eval` helper 合成测试通过；未训练、未 test、未跑 120ep。

**本文件为 V6-A 定稿。实现时必须按 §5/§8/§9 执行；服务器 1ep smoke 通过后才跑 120ep。**
