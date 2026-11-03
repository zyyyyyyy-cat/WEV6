# Layer-route probe（V5 training-free）

`layer_route_probe.py` 是 **零训练** 的层路由探针：缓存 V2 的 f6/f9/f12，在 5-fold identity-disjoint val 上屏幕固定层/α/role 路由，选一个配置，再在 test 上评一次。它不训练 backbone/head，不修改 `model/processor/datasets/loss/train/test`。

## 0. 关键约定

- **same pair 永远用 f12**；只改 cross pair。
- 主候选：`z_s(α)=L2((1-α)L2(f12)+αL2(f9))`；α 包含 0.01–0.25，历史最佳 α≈0.01；不扫 test。
- 固定 all-pair 融合只作负对照；不选它。
- 5-fold 400/100；每折 100 val IDs；selection 只读 `features_train.npz`，test 只读 `features_test.npz`。
- **test 只一次**；`selection.json` 不是 `TEST_READY` 时 test 拒绝。
- val 是 V2 已见 ID 天花板；val 只做 screen，test 才是 unseen-ID 证据。
- 训练-free 结果不是 PASD-C head 的最终证据；GO 只说明“固定层路由有 headroom”。

## 1. 顺序

```text
cache → screen (val only) → test (selected config, once)
```

`cache` 与 `screen` 不能并行；`screen` 必须先完成；`test` 只能在 `selection.json` 为 `TEST_READY` 后跑一次。

## 2. 命令

```bash
cd /project/code/WHU_base/WEV5/WHU

# 0) 自检（可选但建议）
python -I tools/layer_route_probe.py --stage selftest \
  --output_dir /tmp/layer_route_selftest --device cpu

# 1) cache f6/f9/f12
python tools/layer_route_probe.py --stage cache \
  --config_file configs/HPD_UAD.yml \
  --checkpoint /project/code/WHU_base/WEV2/logs/HPD_1000_DDP/transformer_120.pth \
  --tag v2 --output_dir ./logs/layer_route/v2 \
  --dataset_root /project/code/data --dataset_dir WHU-MARS \
  --layers 6,9,12 --device cuda --batch_size 512 --num_workers 8

# 2) screen on val only (reuse the corrected probe split)
python tools/layer_route_probe.py --stage screen \
  --tag v2 --output_dir ./logs/layer_route/v2 \
  --split_json ./logs/tir_calib/v2/split.json \
  --alphas 0,0.01,0.1,0.25,0.5,1.0 \
  --role_alphas 0,0.1,0.25,0.5 \
  --device cuda --bootstrap 10000

# 3) test selected route, once
python tools/layer_route_probe.py --stage test \
  --tag v2 --output_dir ./logs/layer_route/v2 \
  --device cuda --bootstrap 10000
```

若 `selection.json` 的 `status` 是 `NO_TEST`，**不要跑 test**；这表示没有稳定的固定层/α 路由。

## 3. 输出

| 文件 | 内容 |
|---|---|
| `features_train.npz` / `features_test.npz` | f6/f9/f12 + pids/cams；物理隔离 |
| `cache_meta.json` | checkpoint SHA、counts、layers |
| `screen_metrics.csv` | 5 folds × candidates × scopes |
| `selection.json` | selected config、fold deltas、stability、mixed val |
| `test_metrics.csv` | baseline/selected × nine/TIR same/mixed |
| `test_per_query.csv` | per-query AP/rank |
| `decision.json` | GO/NO_GO/INCONCLUSIVE + guards/bootstrap |

## 4. Go/No-go

- test TIR cross mAP Δ ≥ +0.15pp；
- identity-cluster bootstrap 95% CI lower ≥ −0.05pp；
- TIR same mAP ≥ baseline−0.5pp、R1 ≥ baseline−1.0pp；
- mixed mAP ≥ baseline−0.10pp；
- same 3 对必须与 f12 **逐位一致**（代码断言）；
- 至少 1 个 tail 信号（R5/R10/>100/>500）。
- GO 只说明固定层路由有 headroom；下一步才是 PASD-C head probe/120ep。NO_GO 则不做 PASD-C 训练，转 independent anchor/数据诊断。

## 5. 服务器同步

- **必须新增**：`tools/layer_route_probe.py` → `/project/code/WHU_base/WEV5/WHU/tools/`。
- **可选**：`tools/README_LAYER_ROUTE_PROBE.md`、`../V5_training_free_probe_design.md`。
- **不要覆盖**：`model/processor/datasets/loss/train/test`；旧 `logs/tir_calib/*`。
- 新 cache 在服务器生成；不需要上传现有 features.npz。
