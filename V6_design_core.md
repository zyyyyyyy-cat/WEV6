# V6 方案核心（审查稿）

> 版本：2026-11-03，S75
> 状态：设计审查中；未改代码、未训练、未 test。
> 目标：在 V4 recipe 上加入 **cross-modal instance correspondence** 训练信号，改善 TIR cross 4 对与九对平均；推理不变。
> 约束：单 UAD/ViT-B/16；无 CLIP；无跨模型分数融合；无多模型 ensemble；单 checkpoint；256×128。
>
> **S75 审查：MODIFY。** 原稿的 same-frame/camera-stratified PKM 应移出主 V6；主 V6 收敛为 TIR-involving cross-camera InfoNCE，无 sampler；配置/元数据有 P0 修复；class-center 主导需 centered/projection 或 probe；详见 `CC/CHAT_LOG.md` S75。以下为审查稿原文，保留供对照。
>
> **S77 定稿：本文件已被 `CC/WEV5/V6_design_final.md` 取代。** 最终设计为 all-cross-modal pair-equal 1/3，raw global_feat，无 same-frame/sampler/frame/all-gather；以 final 文件为准。

---

## 0. 一句话

**V6 = V4 recipe + `L_corr`（frame/camera-aware cross-modal instance InfoNCE）+ camera-stratified PKM；训练-only；推理仍是 pre-BN CLS + L2 + Euclidean。**

---

## 1. Base：V4 recipe

V4 recipe = ViT-B/16 + V2 HPD + V4 GAC + 标准 120ep DDP。

| 项目 | 设置 |
|---|---|
| Backbone | ViT-B/16，ImageNet `jx_vit_base_p16_224-80ecf9dd.pth` |
| 输入 | 256×128，mean/std=0.5 |
| 模态 | RGB / IR / Thermal |
| Sampler | PKM，K=4；P=16 × K=4 × M=3 |
| Batch | 256，4 卡 DDP |
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

---

## 2. 新机制：`L_corr`

### 2.1 目的

直接监督同 ID、跨模态、跨相机实例特征靠近，补上 CE/GPD/WRT 没有提供的 instance-level 跨模态对应信号。

### 2.2 依据（已复核的缓存特征）

- 同帧跨模态 raw cos：RGB–TIR 0.579、NIR–TIR 0.576；
- 同 ID 随机 raw cos：0.521 / 0.517；
- 去掉每 ID/模态 class mean 后的同帧 residual：0.099 / 0.100；
- 同 ID cross-camera residual：−0.033 / −0.034。

结论：模型主要编码 class mean，instance-level 跨模态对应很弱。

### 2.3 Loss 定义

```text
L_V6 = L_V4 + λ_corr * L_corr
```

对每个 anchor `i`：

- `P_i = { j | pid_j=pid_i, mod_j!=mod_i, cam_j!=cam_i }`：主正对；
- `F_i = { j | pid_j=pid_i, frame_j=frame_i, mod_j!=mod_i }`：辅助正对（same-frame，通常同相机）；
- `N_i = { j | pid_j!=pid_i }`：负对；
- 若 `P_i` 为空，降级到 cross-modal 正对；若仍为空，跳过并记录；
- `L_corr = InfoNCE(P_i) + β * InfoNCE(F_i)`；
- `τ=0.1`；`λ_corr=0.1`，warmup 5–10 epoch；`β=0.1–0.3`；
- 不引入 memory queue、不做 batch-local ranking。

### 2.4 Camera-stratified PKM

- 默认开启：每个 ID/模态尽量保证至少一对 cross-camera 样本；
- 若某 ID 没有足够 cross-camera 样本，记录并降级，不伪造；
- 监控 batch 内 cross-camera 正对覆盖率；
- 若离线覆盖率已很高（例如 >70% anchors 已有 cross-camera 正对），可关闭 sampler，只加 `L_corr`，作为 config 变体。

### 2.5 推理

完全不改：same 3 对、RGB/NIR cross、TIR cross、mixed、AG/GD 全部使用 f12；不加载新 head/bank；不 rerank。

---

## 3. 训练与评估

### 3.1 训练

- 从零训练 120ep，不做 fine-tune；
- V4 recipe + `L_corr` + camera-stratified PKM；
- 4 卡 DDP，batch 256，LR 0.032，warmup 5，cosine，seed 1234；
- checkpoint 每 5 epoch；支持 resume；
- `L_corr.ENABLED=False + sampler off` 必须严格复现 V4 路径；
- 1ep smoke 仅工程自检（临时目录），正式 120ep 从 epoch 0 开始，LR/loss schedule 不受 smoke 影响。

### 3.2 Test

- 一次官方 test：500 unseen IDs，query/gallery，九对 + mixed；
- 模型：V6 最终 120ep checkpoint；
- 不做 probe test；不在 test 上调参；第二 seed 只在 GO 后补。

### 3.3 Gate

主 endpoint：TIR cross 4 对 mAP/R1。

- GO：TIR cross mAP ≥ +0.5pp，R1 ≥ +0.5pp；
- STRONG_GO：mAP ≥ +1.0pp，R1 ≥ +1.0pp；
- 护栏：same 3 对不低于 V4−0.5；TIR same mAP ≥14.05、R1 ≥20.12；mixed mAP ≥V4−0.10；1000 mAP/R1 不低于 V4−0.3；AG/GD 不低于 V4−0.3；
- 只有 +0.15–0.5pp：partial signal，不宣布 paper+1，先分析；
- 无增益或护栏破：停止，不重复 test。

### 3.4 监控

- `L_corr` 值、梯度 share；
- cross-camera / same-frame 正对覆盖率；
- HPD/GAC/backbone 梯度 share；
- TIR cross/same、same、mixed validation 曲线（只监控，不选模）；
- checkpoint 特征诊断：class-mean residual、cross-camera same-ID cos。

---

## 4. 实现触点（待审查）

- `datasets/whu_mars.py`：返回 `frame_id`（从文件名解析）；确认 `camid`/`modid` 语义。
- `datasets/sampler_ddp.py`：可选 camera-stratified PKM。
- `loss/corr_loss.py`：新增 `L_corr`。
- `processor/processor.py`：在原有 losses 后加 `λ_corr * L_corr`；保持 CE/WRT/ProCA/GPD/GAC 顺序与权重。
- `configs/GAC_UAD.yml` → V6 config：新增 `CORR_LOSS.*`、`SAMPLER.CAMERA_STRATIFIED`。
- 不修改 `model/make_model.py` 推理路径；不修改 `test.py`。

---

## 5. 明确不做

- fine-tune；frozen-head probe；V2/V4 双主跑；
- rerank/bank/heads 推理；class-level 新 head/原型；
- fixed f6/f9/α/γ 扫描；batch-local ranking/TTA/global calibration；
- 2337 混 1000；288×144；test 调参；
- 失败后预设 MixER-lite/CSLS/anchor。

---

## 6. 审查重点

1. `L_corr` 的 mask/正负对/温度/权重是否与现有 V4 loss 数据流和梯度流兼容；
2. `frame_id`/`camid`/`modid` 是否可可靠解析，same-frame 是否真同帧；
3. camera-stratified PKM 是否破坏 V4 batch 结构/DDP/resume；
4. `L_corr` 与 GAC/HPD/WRT 的梯度交互，是否会伤 same/TIR same；
5. 120ep 从零训练是否是合理验证，gate 是否现实；
6. V4 base 是否合适，是否应改用 V2 或组合 base。
