# V6-A 最新设计（审查稿；已被 V6_design_final.md 取代）

> **注意：本文件是 S76 审查稿，已被 `CC/WEV5/V6_design_final.md` 取代。最终实现与报告以 `V6_design_final.md` 为准。**
>
> 版本：2026-11-03，S76
> 状态：审查中；未改代码、未训练、未 test。
> 范围：只考虑 A（直接 120ep），不做 B（cached-feature probe gate）。
> 目标：在 V4 recipe 上加入 **all-cross-modal pair-balanced instance correspondence**，同时改善 TIR cross 与 NIR/RGB cross；推理不变。

---

## 0. 一句话

**V6-A = V4 recipe + `L_corr`（RGB–NIR、RGB–TIR、NIR–TIR 三个 pair 的 cross-camera cross-modal InfoNCE，pair 等权）+ 120ep from scratch；无 same-frame、无 camera sampler；推理仍为 pre-BN CLS + L2 + Euclidean。**

---

## 1. Base：V4 recipe

V4 = ViT-B/16 + V2 HPD + V4 GAC + 标准 120ep DDP。

| 项目 | 设置 |
|---|---|
| Backbone | ViT-B/16，ImageNet |
| 输入 | 256×128，mean/std=0.5 |
| 模态 | RGB / NIR / TIR |
| Sampler | PKM，K=4；P=16 IDs/rank × K=4 × M=3 |
| Batch | 256 tuples，4 卡 DDP；每 rank 64 tuples / 192 rows |
| Epoch | 120 |
| Optimizer | SGD，LR 0.032，warmup 5，cosine |
| Loss | CE + WRT + 0.01 ProCA + 1.0 GPD_HPD + 0.1 GAC_aux |
| HPD | 0.25·CE(P_id) + 0.75·CE(P_scene)，K=3，τ=0.03 |
| GAC | 3 个 train-only aux heads；推理丢弃 |
| 推理 | pre-BN CLS + L2 + Euclidean；不加载新 head/bank |

---

## 2. `L_corr`

### 2.1 作用域

三个 unordered 跨模态 pair：

```text
RGB–NIR
RGB–TIR
NIR–TIR
```

每个 pair 独立计算 InfoNCE，再等权平均：

```text
L_corr = (1/3) * [ L(RGB,NIR) + L(RGB,TIR) + L(NIR,TIR) ]
```

### 2.2 每个 pair 内

- anchor：该 pair 的两个模态的所有行；
- 正对：同 ID、另一模态、**cross-camera**；
- 负对：不同 ID；
- denominator 排除：self、same-ID same-camera、same-ID same-modality；
- 无 same-frame；
- 无 camera sampler；
- 无 cross-pair 混合 denominator；
- `τ=0.1`；
- `λ_corr=0.1`，warmup 5 epoch；
- 多正对用 logsumexp；
- 空正对 anchor 跳过，记录覆盖率。

### 2.3 输入输出

- 输入：`global_feat [M*B,768]`（pre-BN CLS）、`target_rep [M*B]`、`cams [M*B]`、`mods [M*B]`；
- 输出：标量 `L_corr`；
- 梯度：只回 `global_feat` → ViT；不经过 classifier/BN/aux heads/bank；
- `λ=0` 完全跳过，不计算、不乘 0。

### 2.4 训练/推理

- 120ep from scratch，V4 recipe + `L_corr`；
- 推理不变；
- 官方 test 一次，final checkpoint；
- 不做 fine-tune；不做 B。

---

## 3. P0/P1 修正（来自 S75 审查）

### P0

1. `CORR_LOSS.*` 必须注册到 `config/defaults.py`；`SAMPLER.CAMERA_STRATIFIED` 不合法，且本设计不需要 sampler 开关。
2. frame 元数据：不要改 dataset 4 元组；在 `ImageDataset.__getitem__` 解析 frame，返回 `(imgs,pid,camids,frames)`；`train_collate_fn` 第 4 项；`processor` unpack 4。
3. denominator：严格排除 self/same-ID same-camera/same-ID same-modality；多正对 logsumexp。
4. `λ=0` 完全跳过；loss stateless；epoch-based warmup；resume 不重置。
5. `global_feat` 必须 `float32` + L2；mask 用大负数，不用 `-inf`。
6. per-rank 计算，不做 all-gather。

### P1

- 当前 V4 sampler 的 cross-camera 覆盖：97.45% anchors；无需 camera sampler。
- same-frame 当前仅 12.85% anchors；本设计不用。
- raw InfoNCE 的 60.8% 梯度在 class-center；39.2% 在 residual；pair-balanced 是否改善需审查。
- V4 base 的 GAC 弱；TIR same 护栏按 V4：14.06/21.01。
- Gate：TIR cross mAP +0.2–0.25 GO；+0.5 strong；九对平均 +0.1 secondary；单 seed provisional。

---

## 4. Gate / 监控

- Primary：TIR cross 4 对 mAP/R1；
- Secondary：RGB–NIR、NIR–RGB 不退化；九对平均 mAP ≥ +0.1；
- GO：TIR cross mAP ≥ +0.2–0.25，CI lower >0，≥4/5 folds 正；
- STRONG_GO：TIR cross mAP ≥ +0.5；
- 护栏：same 3 对 ≥V4−0.5；TIR same mAP ≥14.06、R1 ≥21.01；mixed ≥V4−0.10；1000/AG/GD ≥V4−0.3；
- 监控：每 pair loss/梯度 share、class-center vs residual、cross-camera 覆盖、TIR/RGB/NIR 分 pair 指标。

---

## 5. 明确不做

- 不做 B（cached-feature probe gate）；
- 不做 fine-tune；
- 不做 same-frame；
- 不做 camera-stratified PKM；
- 不做 all-gather；
- 不做 V6b；
- 不做 rerank/bank/heads 推理；
- 不做 class-level 新 head/原型；
- 不做 2337/288×144/test 调参。

---

## 6. 审查重点

1. all-cross-modal pair-balanced 是否真的比 TIR-only 更符合整体指标，而不是稀释 TIR；
2. 每个 pair 的 mask/denominator/权重是否正确；三个 pair 的 loss 尺度是否可比；
3. RGB–NIR 的正对/负对数量远大于 TIR pair，等权平均是否足够；
4. 60.8% class-center 梯度在 pair-balanced 下是否仍主导；是否需要 centered/projection；
5. frame/cam 元数据与 row order 是否可实现；
6. V4 base + GAC + L_corr 的梯度交互；
7. 120ep 从零、gate +0.2–0.25、九对 +0.1 是否现实；
8. 是否有 P0 工程问题。
