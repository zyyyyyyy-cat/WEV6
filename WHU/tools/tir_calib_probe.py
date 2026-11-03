#!/usr/bin/env python
"""WEV5 TIR calibration probe (frozen V2/V4 features, TIR-only adapter).

Stages
------
cache : extract raw pre-BN CLS features from a V2/V4 checkpoint and cache them
split : build 5-fold 400/100 identity-disjoint manifests and val query/gallery
val   : train TIR adapter variants on cached features and evaluate val splits
test  : train the preregistered primary adapter on each 400-ID fold
        (5 folds x 3 seeds), evaluate the official test split once, and write
        a decision table; final state is primary, last-3 is auxiliary
selftest : synthetic in-memory check of split/adapter/eval/bootstrap logic

This tool does NOT modify model/processor/datasets/loss/train/test.  It only
reads checkpoints and cached features.  The feature stage uses the inference
feature (pre-BN CLS, TEST.NECK_FEAT='before', 256x128).

Typical order (from WEV5/WHU):

  python tools/tir_calib_probe.py --stage cache \
    --config_file configs/HPD_UAD.yml \
    --checkpoint /path/to/WEV2/logs/HPD_1000_DDP/transformer_120.pth \
    --tag v2 --output_dir logs/tir_calib/v2 \
    --dataset_root /project/code/data --dataset_dir WHU-MARS

  python tools/tir_calib_probe.py --stage split --tag v2 \
    --output_dir logs/tir_calib/v2

  python tools/tir_calib_probe.py --stage val --tag v2 \
    --output_dir logs/tir_calib/v2 --device cuda

  python tools/tir_calib_probe.py --stage test --tag v2 \
    --output_dir logs/tir_calib/v2 --device cuda

  # Optional V4 val-only robustness (after V2 primary passes):
  python tools/tir_calib_probe.py --stage cache ... --tag v4 ...
  python tools/tir_calib_probe.py --stage split ... --tag v4 ...
  python tools/tir_calib_probe.py --stage val --tag v4 ...

The test stage refuses to run twice unless --force is given.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import cfg  # noqa: E402
from datasets import make_dataloader  # noqa: E402
from model import make_model  # noqa: E402
from loss.wrt_loss import TripletLoss_WRT  # noqa: E402


MOD_NAMES = {1: "RGB", 2: "NIR", 3: "TIR"}
TIR = 3


def mod_name(m):
    return MOD_NAMES.get(int(m), "M{}".format(int(m)))


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _sha(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _file_sha256(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _load_npz(path):
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    z.close()
    return out


# ---------------------------------------------------------------------------
# cache stage
# ---------------------------------------------------------------------------

def _cache_cfg(args):
    cfg.merge_from_file(args.config_file)
    opts = [
        "MODEL.PRETRAIN_CHOICE", "self",
        "MODEL.DIST_TRAIN", "False",
        "TEST.NECK_FEAT", "before",
        "TEST.FEAT_NORM", "yes",
        "INPUT.SIZE_TRAIN", "[256,128]",
        "INPUT.SIZE_TEST", "[256,128]",
        "DATASETS.ROOT_DIR", str(args.dataset_root),
        "DATASETS.DATASET_DIR", str(args.dataset_dir),
        "DATASETS.FILTER_GD", "False",
        "DATALOADER.NUM_WORKERS", str(int(args.num_workers)),
        "TEST.IMS_PER_BATCH", str(int(args.batch_size)),
        "SOLVER.IMS_PER_BATCH", str(max(64, int(args.batch_size))),
    ]
    if args.opts:
        opts.extend(args.opts)
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


def _load_model_state(model, path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    state = {str(k).replace("module.", "", 1) if str(k).startswith("module.")
             else str(k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if "aux_heads" not in k]
    if bad_missing:
        raise RuntimeError(
            "checkpoint is missing model keys: {}".format(bad_missing[:12]))
    if unexpected:
        print("warning: unexpected checkpoint keys: {}".format(unexpected[:12]))
    return len(state)


def _extract_loader(model, loader, mode, device):
    feats, pids, cams, mods, names = [], [], [], [], []
    for batch in loader:
        img = batch[0].to(device, non_blocking=True)
        with torch.no_grad():
            feat = model(img, mode=mode)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        feats.append(feat.detach().float().cpu())
        pids.append(_np(batch[1]).astype(np.int64))
        cams.append(_np(batch[2]).astype(np.int64))
        mods.append(_np(batch[4]).astype(np.int64))
        names.extend([str(x) for x in batch[5]])
    if not feats:
        return (torch.empty(0, 768), np.empty(0, np.int64),
                np.empty(0, np.int64), np.empty(0, np.int64), [])
    return (torch.cat(feats, 0), np.concatenate(pids),
            np.concatenate(cams), np.concatenate(mods), names)


def stage_cache(args):
    if not args.config_file or not args.checkpoint:
        raise ValueError("--config_file and --checkpoint are required for cache")
    os.makedirs(args.output_dir, exist_ok=True)
    _cache_cfg(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    (train_loader, train_normal_loaders, val_loaders, num_querys,
     num_classes, camera_num, view_num) = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes,
                       camera_num=camera_num, view_num=view_num)
    nkeys = _load_model_state(model, args.checkpoint)
    model.to(device).eval()
    print("loaded {} state keys; num_classes={} queries={}".format(
        nkeys, num_classes, num_querys))

    arrays = {}
    for m, loader in enumerate(train_normal_loaders, start=1):
        f, p, c, md, names = _extract_loader(model, loader, m, device)
        arrays["train_feat_{}".format(m)] = _np(f)
        arrays["train_pid_{}".format(m)] = p.astype(np.int32)
        arrays["train_cam_{}".format(m)] = c.astype(np.int32)
        arrays["train_mod_{}".format(m)] = md.astype(np.int32)
        arrays["train_name_{}".format(m)] = np.asarray(names)
        print("train m{}: {} x {}".format(m, f.shape[0], f.shape[1]))

    for m, (loader, nq) in enumerate(zip(val_loaders, num_querys), start=1):
        f, p, c, md, names = _extract_loader(model, loader, m, device)
        nq = int(nq)
        arrays["q_feat_{}".format(m)] = _np(f[:nq])
        arrays["g_feat_{}".format(m)] = _np(f[nq:])
        arrays["q_pid_{}".format(m)] = p[:nq].astype(np.int32)
        arrays["g_pid_{}".format(m)] = p[nq:].astype(np.int32)
        arrays["q_cam_{}".format(m)] = c[:nq].astype(np.int32)
        arrays["g_cam_{}".format(m)] = c[nq:].astype(np.int32)
        arrays["q_name_{}".format(m)] = np.asarray(names[:nq])
        arrays["g_name_{}".format(m)] = np.asarray(names[nq:])
        print("val m{}: query {} gallery {}".format(m, nq, f.shape[0] - nq))

    arrays["classifier_weight"] = _np(model.classifier.weight.detach().float())
    b = model.bottleneck
    arrays["bn_weight"] = _np(b.weight.detach().float())
    arrays["bn_bias"] = _np(b.bias.detach().float())
    arrays["bn_mean"] = _np(b.running_mean.detach().float())
    arrays["bn_var"] = _np(b.running_var.detach().float())
    arrays["bn_eps"] = np.asarray([float(b.eps)], dtype=np.float32)

    np.savez(os.path.join(args.output_dir, "features.npz"), **arrays)
    meta = dict(
        tag=args.tag, config_file=args.config_file,
        checkpoint=args.checkpoint, checkpoint_state_keys=nkeys,
        num_classes=num_classes, num_querys=[int(x) for x in num_querys],
        modalities=list(cfg.DATASETS.MODALITIES),
        train_counts=[int(arrays["train_feat_{}".format(m)].shape[0])
                      for m in range(1, len(cfg.DATASETS.MODALITIES) + 1)],
        query_counts=[int(arrays["q_feat_{}".format(m)].shape[0])
                      for m in range(1, len(cfg.DATASETS.MODALITIES) + 1)],
        gallery_counts=[int(arrays["g_feat_{}".format(m)].shape[0])
                        for m in range(1, len(cfg.DATASETS.MODALITIES) + 1)],
        checkpoint_sha256=_file_sha256(args.checkpoint),
        feature_note="raw pre-BN CLS, eval/no_grad, 256x128, no L2 applied",
    )
    with open(os.path.join(args.output_dir, "cache_meta.json"), "w") as f:
        json.dump(_jsonable(meta), f, indent=2)
    print("cached features to {}".format(args.output_dir))


# ---------------------------------------------------------------------------
# split stage
# ---------------------------------------------------------------------------

def _pid_to_indices(pids):
    d = defaultdict(list)
    for i, p in enumerate(pids):
        d[int(p)].append(int(i))
    return d


def _stable_indices(indices, seed, tag):
    return sorted([int(i) for i in indices],
                  key=lambda i: _sha(seed, tag, i))


def _choose_queries(indices, cams, q_count, seed, m, pid):
    groups = defaultdict(list)
    for i in indices:
        groups[int(cams[i])].append(int(i))
    for c in groups:
        groups[c] = _stable_indices(groups[c], seed, "q-{}-{}-{}".format(m, pid, c))
    cam_order = sorted(groups.keys(),
                       key=lambda c: (0 if int(c) >= 5 else 1, _sha(seed, pid, c)))
    chosen = []
    # Prefer up to 2 aerial images, then ground, round-robin across cameras.
    aerial_cams = [c for c in cam_order if int(c) >= 5]
    ground_cams = [c for c in cam_order if int(c) < 5]
    priority = aerial_cams[:2] + ground_cams if aerial_cams else ground_cams
    while len(chosen) < q_count:
        progress = False
        for c in priority:
            if groups[c] and len(chosen) < q_count:
                chosen.append(groups[c].pop(0))
                progress = True
        if not progress:
            break
    return chosen


def stage_split(args):
    cache = _load_npz(os.path.join(args.output_dir, "features.npz"))
    M = 3
    ids = sorted(set(int(x) for x in cache["train_pid_1"]))
    if len(ids) != 500:
        print("warning: expected 500 train IDs, got {}".format(len(ids)))
    # stratify by aerial presence / camera count / stable hash
    stats = {}
    for pid in ids:
        cams = [int(cache["train_cam_1"][i]) for i, p in
                enumerate(cache["train_pid_1"]) if int(p) == pid]
        stats[pid] = (1 if any(c >= 5 for c in cams) else 0,
                      len(set(cams)))
    ordered = sorted(ids, key=lambda p: (
        stats[p][0], stats[p][1], _sha(args.seed, "fold", p)))
    folds = [[] for _ in range(int(args.folds))]
    for i, pid in enumerate(ordered):
        folds[i % int(args.folds)].append(pid)
    extra_ids = set(sorted(ids, key=lambda p: _sha(args.seed, "extra", p))[:135])

    all_rows = []
    split = dict(seed=int(args.seed), folds=[], extra_ids=sorted(extra_ids))
    for fi, val_ids in enumerate(folds):
        train_ids = sorted(set(ids) - set(val_ids))
        fold = dict(fold=fi, train_ids=train_ids, val_ids=sorted(val_ids),
                    q_idx={}, g_idx={})
        for m in range(1, M + 1):
            pid2idx = _pid_to_indices(cache["train_pid_{}".format(m)])
            cams = cache["train_cam_{}".format(m)]
            val_all = sorted(i for p in val_ids for i in pid2idx[p])
            q_idx = []
            for pid in sorted(val_ids):
                q_count = 5 if pid in extra_ids else 4
                picked = _choose_queries(
                    pid2idx[pid], cams, q_count, args.seed, m, pid)
                q_idx.extend(picked)
            qset = set(q_idx)
            g_idx = [i for i in val_all if i not in qset]
            # prune queries with no same-pid, different-camera positive for any
            # gallery modality; this mirrors the evaluation mask.
            gcams = {}
            for gm in range(1, M + 1):
                gpids = cache["train_pid_{}".format(gm)]
                gc = cache["train_cam_{}".format(gm)]
                mm = defaultdict(set)
                for i in g_idx:
                    mm[int(gpids[i])].add(int(gc[i]))
                gcams[gm] = mm
            keep = []
            for i in q_idx:
                pid = int(cache["train_pid_{}".format(m)][i])
                cam = int(cams[i])
                ok = all(
                    any(cc != cam for cc in gcams[gm].get(pid, set()))
                    for gm in range(1, M + 1))
                if ok:
                    keep.append(i)
                else:
                    all_rows.append(dict(
                        fold=fi, mod=m, pid=pid, reason="no_cross_cam_positive"))
            q_idx = keep
            fold["q_idx"][str(m)] = [int(i) for i in q_idx]
            fold["g_idx"][str(m)] = [int(i) for i in g_idx]
            all_rows.append(dict(
                fold=fi, mod=m, pid=-1, reason="ok",
                n_q=len(q_idx), n_g=len(g_idx)))
            print("fold {} m{}: val IDs {} query {} gallery {}".format(
                fi, m, len(val_ids), len(q_idx), len(g_idx)))
        split["folds"].append(fold)

    with open(os.path.join(args.output_dir, "split.json"), "w") as f:
        json.dump(split, f, indent=2)
    _write_csv(os.path.join(args.output_dir, "split_audit.csv"), all_rows,
               ["fold", "mod", "pid", "reason", "n_q", "n_g"])
    print("wrote split.json")


# ---------------------------------------------------------------------------
# adapters / cached-feature training
# ---------------------------------------------------------------------------

class LowRankAdapter(nn.Module):
    def __init__(self, dim, rank, gamma=0.1, rmax=2.0, w2_std=1e-3):
        super().__init__()
        self.dim = int(dim)
        self.rank = int(rank)
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.w1 = nn.Linear(dim, rank, bias=False)
        self.w2 = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_normal_(self.w1.weight, a=0, mode="fan_out")
        nn.init.normal_(self.w2.weight, std=float(w2_std))
        self.gamma = nn.Parameter(torch.full((dim,), float(gamma)))
        self.rmax = float(rmax)

    def forward(self, f):
        d = self.w2(F.gelu(self.w1(self.ln(f))))
        res = self.gamma.unsqueeze(0) * d
        res = res * self.rmax / (self.rmax + res.norm(dim=1, keepdim=True) + 1e-12)
        return f + res


class AffineAdapter(nn.Module):
    def __init__(self, dim, **kw):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, f):
        return f * self.scale.unsqueeze(0) + self.bias.unsqueeze(0)


class LinearAdapter(nn.Module):
    def __init__(self, dim, gamma=0.1, rmax=2.0, w_std=1e-3, **kw):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.w = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.w.weight, std=float(w_std))
        self.gamma = nn.Parameter(torch.full((dim,), float(gamma)))
        self.rmax = float(rmax)

    def forward(self, f):
        d = self.w(self.ln(f))
        res = self.gamma.unsqueeze(0) * d
        res = res * self.rmax / (self.rmax + res.norm(dim=1, keepdim=True) + 1e-12)
        return f + res


def _make_adapter(arm, dim, rank, gamma, rmax):
    if arm == "gamma0":
        return None, False
    if arm == "affine":
        return AffineAdapter(dim), False
    if arm == "linear":
        return LinearAdapter(dim, gamma=gamma, rmax=rmax), False
    if arm == "r32":
        return LowRankAdapter(dim, 32, gamma, rmax), False
    if arm == "shared":
        return LowRankAdapter(dim, rank, gamma, rmax), True
    return LowRankAdapter(dim, rank, gamma, rmax), False


def _bn_eval(x, cache, device):
    return F.batch_norm(
        x, torch.as_tensor(cache["bn_mean"], device=device),
        torch.as_tensor(cache["bn_var"], device=device),
        torch.as_tensor(cache["bn_weight"], device=device),
        torch.as_tensor(cache["bn_bias"], device=device),
        training=False, eps=float(cache["bn_eps"][0]))


def _fc_rows(cache, train_ids, device):
    w = torch.as_tensor(cache["classifier_weight"], device=device)
    return w[torch.as_tensor(train_ids, dtype=torch.long, device=device)]


def _prototypes(cache, train_ids, device):
    # class means from frozen train features; train_ids are relabelled 0..499.
    pids = cache["train_pid_1"]
    p_scene = []
    for m in range(1, 4):
        f = torch.as_tensor(cache["train_feat_{}".format(m)], device=device)
        f = F.normalize(f.float(), dim=1)
        p = []
        for pid in train_ids:
            mask = torch.as_tensor(pids == pid, device=device)
            if bool(mask.any()):
                p.append(F.normalize(f[mask].mean(0), dim=0))
            else:
                p.append(torch.zeros(f.shape[1], device=device))
        p_scene.append(torch.stack(p, 0))
    p_scene = torch.stack(p_scene, 1)  # [C,3,D]
    p_scene = F.normalize(p_scene, dim=2)
    p_id = F.normalize(p_scene.mean(1), dim=1)
    return p_scene, p_id


def _sample_batch(cache, pid2idx, pids, rng, p_count, k_count, device):
    chosen = list(rng.choice(pids, size=min(p_count, len(pids)), replace=False))
    labels = torch.arange(len(chosen), dtype=torch.long, device=device)
    # Row index into the frozen classifier/prototype tables, which are stored
    # in sorted train_ids order.  labels are batch-local for WRT; row_idx is
    # the matching global row for CE/GPD.  Do not index fc/prototypes by
    # labels: chosen is in RNG order, not sorted train_ids order.
    pid_to_row = {int(p): i for i, p in enumerate(pids)}
    row_idx = torch.as_tensor([pid_to_row[int(p)] for p in chosen],
                              dtype=torch.long, device=device)
    feats = []
    for m in range(1, 4):
        rows = []
        for pid in chosen:
            idx = pid2idx[m][int(pid)]
            if len(idx) >= k_count:
                take = rng.choice(idx, size=k_count, replace=False)
            else:
                take = rng.choice(idx, size=k_count, replace=True)
            rows.append(cache["train_feat_{}".format(m)][take])
        f = np.concatenate(rows, 0).reshape(len(chosen), k_count, -1)
        feats.append(torch.as_tensor(f, dtype=torch.float32, device=device))
    target = labels.repeat_interleave(k_count)
    return feats, target, labels, row_idx, len(chosen)


def _proca_loss(feats):
    # feats: list of [P,K,D]; normalized modality centers vs their mean.
    centers = [F.normalize(f.mean(1), dim=1) for f in feats]
    g_cent = torch.stack(centers, 0).mean(0).detach()
    loss = sum((c - g_cent).pow(2).sum(1) for c in centers).mean() / len(centers)
    return loss


def _train_adapter(cache, fold, arm, seed, args, device):
    train_ids = [int(x) for x in fold["train_ids"]]
    pid2idx = {m: _pid_to_indices(cache["train_pid_{}".format(m)])
               for m in range(1, 4)}
    dim = int(cache["train_feat_1"].shape[1])
    tir_norm = torch.as_tensor(cache["train_feat_3"], device=device).norm(dim=1)
    rmax = float(tir_norm.median().item() * args.rmax_factor)
    adapter, apply_all = _make_adapter(
        arm, dim, args.rank, args.gamma, rmax)
    if adapter is None:
        return [None], dict(arm=arm, seed=seed, fold=fold["fold"],
                            residual_ratio=0.0, loss=float("nan"))
    adapter.to(device)
    decay, no_decay = [], []
    for n, p in adapter.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if ("gamma" in n or "ln." in n) else decay).append(p)
    groups = [{"params": decay, "weight_decay": args.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.999))
    rng = np.random.RandomState(abs(int(seed) + 1000 * (int(fold["fold"]) + 1)))
    p_count = min(int(args.p), len(train_ids))
    steps = max(1, len(train_ids) // p_count)
    epochs = int(args.epochs)
    warmup = max(1, int(args.warmup))
    last_epochs = sorted(set([max(1, epochs - 20), max(1, epochs - 10), epochs]))
    states, ratios, losses = [], [], []
    triplet = TripletLoss_WRT()
    fc_w = _fc_rows(cache, train_ids, device)
    p_scene = p_id = None
    if arm != "ce_only":
        p_scene, p_id = _prototypes(cache, train_ids, device)
    for epoch in range(1, epochs + 1):
        if epoch <= warmup:
            lr = args.lr * epoch / warmup
        else:
            frac = (epoch - warmup) / max(1, epochs - warmup)
            lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = lr
        for _ in range(steps):
            feats, target, labels, row_idx, P = _sample_batch(
                cache, pid2idx, train_ids, rng, p_count, args.k, device)
            if apply_all:
                adapted = [adapter(f.reshape(-1, dim)).reshape_as(f)
                           for f in feats]
            else:
                adapted = [feats[0], feats[1],
                           adapter(feats[2].reshape(-1, dim)).reshape_as(feats[2])]
            tir = adapted[2].reshape(-1, dim)
            ce = F.cross_entropy(
                F.linear(_bn_eval(tir, cache, device), fc_w[row_idx]), target)
            gpd = torch.zeros((), device=device)
            if arm != "ce_only":
                fn = F.normalize(tir, dim=1)
                logits_id = fn @ p_id[row_idx].t() / 0.03
                logits_scene = torch.einsum("nd,ckd->nck", fn,
                                            p_scene[row_idx])
                logits_scene = logits_scene.max(2).values / 0.03
                gpd = (0.25 * F.cross_entropy(logits_id, target)
                       + 0.75 * F.cross_entropy(logits_scene, target))
            if arm in ("ce_only", "ce_gpd"):
                loss = ce + gpd
            else:
                allf = torch.cat([x.reshape(-1, dim) for x in adapted], 0)
                wrt = triplet(allf, labels.repeat_interleave(args.k).repeat(3))[0]
                pca = _proca_loss(adapted)
                loss = ce + gpd + wrt + 0.01 * pca
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                base_tir = feats[2].reshape(-1, dim)
                adapt_tir = adapted[2].reshape(-1, dim)
                ratio = float(((adapt_tir - base_tir).norm(dim=1) /
                               (base_tir.norm(dim=1) + 1e-12)).mean())
            ratios.append(ratio)
            losses.append(float(loss.detach()))
        if epoch in last_epochs:
            states.append(copy.deepcopy(adapter.state_dict()))
    return states, dict(arm=arm, seed=seed, fold=fold["fold"],
                        residual_ratio=float(np.mean(ratios)) if ratios else 0.0,
                        loss=float(np.mean(losses)) if losses else float("nan"))


# ---------------------------------------------------------------------------
# evaluation / bootstrap
# ---------------------------------------------------------------------------

def _apply(feat, adapter, apply_all, m):
    if adapter is None:
        return feat
    if apply_all or int(m) == TIR:
        return adapter(feat)
    return feat


def _eval_pair(qf, gf, qpid, qcam, gpid, gcam, q_mod, g_mod):
    qf = F.normalize(qf.float(), dim=1)
    gf = F.normalize(gf.float(), dim=1)
    rows = []
    for qi in range(qf.shape[0]):
        keep = gcam != qcam[qi]
        if not bool(keep.any()):
            continue
        s = qf[qi] @ gf.t()
        s = s[keep]
        gp = gpid[keep]
        pos = gp == qpid[qi]
        if not bool(pos.any()):
            continue
        pos_best = s[pos].max()
        rank = 1 + int((s > pos_best).sum().item())
        order = torch.argsort(s, descending=True)
        m = (gp[order] == qpid[qi]).float()
        cum = torch.cumsum(m, 0)
        ranks = torch.arange(1, m.numel() + 1,
                             dtype=m.dtype, device=m.device)
        precision = cum / ranks
        ap = float((precision * m).sum() / (m.sum() + 1e-12))
        rows.append(dict(q_index=int(qi), q_pid=int(qpid[qi]),
                         q_cam=int(qcam[qi]), q_mod=mod_name(q_mod),
                         g_mod=mod_name(g_mod), rank=int(rank),
                         ap=ap, valid=1))
    return rows


def _metrics(rows):
    if not rows:
        return dict(n=0, valid=0, mAP=float("nan"), R1=float("nan"),
                    R5=float("nan"), R10=float("nan"),
                    gt100=float("nan"), gt500=float("nan"))
    n = len(rows)
    ranks = np.asarray([r["rank"] for r in rows], dtype=np.float64)
    aps = np.asarray([r["ap"] for r in rows], dtype=np.float64)
    return dict(n=n, valid=n, mAP=float(aps.mean()),
                R1=float((ranks == 1).mean()),
                R5=float((ranks <= 5).mean()),
                R10=float((ranks <= 10).mean()),
                gt100=float((ranks > 100).mean()),
                gt500=float((ranks > 500).mean()))


def _eval_all(cache, fold_or_none, adapter, apply_all, device, split):
    M = 3
    pair_rows = {}
    for q in range(1, M + 1):
        for g in range(1, M + 1):
            if fold_or_none is None:
                qf = torch.as_tensor(cache["q_feat_{}".format(q)], device=device)
                gf = torch.as_tensor(cache["g_feat_{}".format(g)], device=device)
                qp = torch.as_tensor(cache["q_pid_{}".format(q)], device=device)
                qc = torch.as_tensor(cache["q_cam_{}".format(q)], device=device)
                gp = torch.as_tensor(cache["g_pid_{}".format(g)], device=device)
                gc = torch.as_tensor(cache["g_cam_{}".format(g)], device=device)
            else:
                qi = fold_or_none["q_idx"][str(q)]
                gi = fold_or_none["g_idx"][str(g)]
                qf = torch.as_tensor(cache["train_feat_{}".format(q)][qi], device=device)
                gf = torch.as_tensor(cache["train_feat_{}".format(g)][gi], device=device)
                qp = torch.as_tensor(cache["train_pid_{}".format(q)][qi], device=device)
                qc = torch.as_tensor(cache["train_cam_{}".format(q)][qi], device=device)
                gp = torch.as_tensor(cache["train_pid_{}".format(g)][gi], device=device)
                gc = torch.as_tensor(cache["train_cam_{}".format(g)][gi], device=device)
            qf = _apply(qf, adapter, apply_all, q)
            gf = _apply(gf, adapter, apply_all, g)
            pair_rows[(q, g)] = _eval_pair(qf, gf, qp, qc, gp, gc, q, g)

    tir_cross = [r for (q, g), rs in pair_rows.items()
                 if (q == TIR) ^ (g == TIR) for r in rs]
    tir_same = pair_rows[(TIR, TIR)]
    all_rows = [r for rs in pair_rows.values() for r in rs]

    # mixed: all query/gallery modalities concatenated; TIR adapter applied.
    if fold_or_none is None:
        qf = [torch.as_tensor(cache["q_feat_{}".format(m)], device=device)
              for m in range(1, M + 1)]
        gf = [torch.as_tensor(cache["g_feat_{}".format(m)], device=device)
              for m in range(1, M + 1)]
        qp = [torch.as_tensor(cache["q_pid_{}".format(m)], device=device)
              for m in range(1, M + 1)]
        qc = [torch.as_tensor(cache["q_cam_{}".format(m)], device=device)
              for m in range(1, M + 1)]
        gp = [torch.as_tensor(cache["g_pid_{}".format(m)], device=device)
              for m in range(1, M + 1)]
        gc = [torch.as_tensor(cache["g_cam_{}".format(m)], device=device)
              for m in range(1, M + 1)]
    else:
        qf, gf, qp, qc, gp, gc = [], [], [], [], [], []
        for m in range(1, M + 1):
            qi = fold_or_none["q_idx"][str(m)]
            gi = fold_or_none["g_idx"][str(m)]
            qf.append(torch.as_tensor(cache["train_feat_{}".format(m)][qi], device=device))
            gf.append(torch.as_tensor(cache["train_feat_{}".format(m)][gi], device=device))
            qp.append(torch.as_tensor(cache["train_pid_{}".format(m)][qi], device=device))
            qc.append(torch.as_tensor(cache["train_cam_{}".format(m)][qi], device=device))
            gp.append(torch.as_tensor(cache["train_pid_{}".format(m)][gi], device=device))
            gc.append(torch.as_tensor(cache["train_cam_{}".format(m)][gi], device=device))
    qf = torch.cat([_apply(x, adapter, apply_all, m)
                    for m, x in enumerate(qf, 1)], 0)
    gf = torch.cat([_apply(x, adapter, apply_all, m)
                    for m, x in enumerate(gf, 1)], 0)
    mixed_rows = _eval_pair(qf, gf, torch.cat(qp), torch.cat(qc),
                            torch.cat(gp), torch.cat(gc), 1, 1)
    # class-mean cos
    cos_matrix = np.full((M, M), np.nan)
    for q in range(M):
        for g in range(M):
            if fold_or_none is None:
                qx = torch.as_tensor(cache["q_feat_{}".format(q + 1)], device=device)
                gx = torch.as_tensor(cache["g_feat_{}".format(g + 1)], device=device)
                qp_m = torch.as_tensor(cache["q_pid_{}".format(q + 1)], device=device)
                gp_m = torch.as_tensor(cache["g_pid_{}".format(g + 1)], device=device)
            else:
                qi = fold_or_none["q_idx"][str(q + 1)]
                gi = fold_or_none["g_idx"][str(g + 1)]
                qx = torch.as_tensor(cache["train_feat_{}".format(q + 1)][qi], device=device)
                gx = torch.as_tensor(cache["train_feat_{}".format(g + 1)][gi], device=device)
                qp_m = torch.as_tensor(cache["train_pid_{}".format(q + 1)][qi], device=device)
                gp_m = torch.as_tensor(cache["train_pid_{}".format(g + 1)][gi], device=device)
            qx = F.normalize(_apply(qx, adapter, apply_all, q + 1), dim=1)
            gx = F.normalize(_apply(gx, adapter, apply_all, g + 1), dim=1)
            common = sorted(set(qp_m.tolist()) & set(gp_m.tolist()))
            if not common:
                continue
            qmean = torch.stack([qx[qp_m == pid].mean(0) for pid in common])
            gmean = torch.stack([gx[gp_m == pid].mean(0) for pid in common])
            qmean = F.normalize(qmean, dim=1)
            gmean = F.normalize(gmean, dim=1)
            cos_matrix[q, g] = float((qmean * gmean).sum(1).mean().item())
    return dict(pair_rows=pair_rows, tir_cross=tir_cross, tir_same=tir_same,
                all_rows=all_rows, mixed=mixed_rows, cos_matrix=cos_matrix)


def _rows_for_csv(rows, split, fold, seed, state, arm, scope=""):
    out = []
    for r in rows:
        d = dict(r)
        d.update(split=split, fold=int(fold), seed=int(seed),
                 state=int(state), arm=arm, scope=scope)
        out.append(d)
    return out


def _bootstrap_delta(base_rows, arm_rows, n=10000, seed=1234):
    def by_pid(rows):
        d = defaultdict(list)
        for r in rows:
            d[int(r["q_pid"])].append(float(r["ap"]))
        return {k: float(np.mean(v)) for k, v in d.items()}
    b, a = by_pid(base_rows), by_pid(arm_rows)
    ids = sorted(set(b) & set(a))
    if not ids:
        return dict(n_ids=0, delta_mAP=float("nan"), ci_low=float("nan"),
                    ci_high=float("nan"))
    rng = np.random.RandomState(int(seed))
    deltas = []
    for _ in range(int(n)):
        s = rng.choice(ids, size=len(ids), replace=True)
        deltas.append(np.mean([a[i] - b[i] for i in s]))
    deltas = np.asarray(deltas)
    return dict(n_ids=len(ids), delta_mAP=float(np.mean(
        [a[i] - b[i] for i in ids])),
        ci_low=float(np.percentile(deltas, 2.5)),
        ci_high=float(np.percentile(deltas, 97.5)))


# ---------------------------------------------------------------------------
# val / test stages
# ---------------------------------------------------------------------------

def _load_split(args):
    with open(os.path.join(args.output_dir, "split.json")) as f:
        return json.load(f)


def _arms(args):
    arms = [x.strip() for x in str(args.arms).split(",") if x.strip()]
    if "primary" not in arms:
        arms.insert(0, "primary")
    return arms


def _fold_eval(cache, fold, adapter, apply_all, device):
    return _eval_all(cache, fold, adapter, apply_all, device, "val")


def stage_val(args):
    cache = _load_npz(os.path.join(args.output_dir, "features.npz"))
    split = _load_split(args)
    device = torch.device(args.device)
    arms = _arms(args)
    primary_seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    val_rows, perq = [], []
    for fold in split["folds"]:
        # gamma0 baseline
        base = _fold_eval(cache, fold, None, False, device)
        bm = _metrics(base["tir_cross"])
        val_rows.append(dict(arm="gamma0", seed=0, fold=fold["fold"],
                             state=0,
                             mAP=bm["mAP"], R1=bm["R1"], R5=bm["R5"],
                             R10=bm["R10"], gt100=bm["gt100"],
                             gt500=bm["gt500"], n=bm["n"]))
        perq.extend(_rows_for_csv(base["tir_cross"], "val", fold["fold"],
                                  0, 0, "gamma0", "tir_cross"))
        for arm in arms:
            seeds = primary_seeds if arm == "primary" else [primary_seeds[0]]
            for seed in seeds:
                states, stat = _train_adapter(cache, fold, arm, seed, args, device)
                for si, st in enumerate(states):
                    if st is None:
                        adapter, apply_all = None, False
                    else:
                        adapter, apply_all = _make_adapter(
                            arm, int(cache["train_feat_1"].shape[1]),
                            args.rank, args.gamma,
                            float(torch.as_tensor(cache["train_feat_3"]).norm(dim=1).median().item()
                                  * args.rmax_factor))
                        adapter.load_state_dict(st)
                        adapter.to(device).eval()
                    ev = _fold_eval(cache, fold, adapter, apply_all, device)
                    mm = _metrics(ev["tir_cross"])
                    val_rows.append(dict(arm=("primary" if arm == "primary" else arm),
                                         seed=seed, fold=fold["fold"],
                                         state=si,
                                         mAP=mm["mAP"], R1=mm["R1"],
                                         R5=mm["R5"], R10=mm["R10"],
                                         gt100=mm["gt100"], gt500=mm["gt500"],
                                         n=mm["n"],
                                         residual_ratio=stat.get("residual_ratio")))
                    perq.extend(_rows_for_csv(ev["tir_cross"], "val",
                                              fold["fold"], seed, si,
                                              ("primary" if arm == "primary" else arm),
                                              "tir_cross"))
    _write_csv(os.path.join(args.output_dir, "val_metrics.csv"), val_rows,
               ["arm", "seed", "fold", "state", "mAP", "R1", "R5", "R10",
                "gt100", "gt500", "n", "residual_ratio"])
    _write_csv(os.path.join(args.output_dir, "val_per_query.csv"), perq,
               ["arm", "seed", "state", "fold", "split", "scope", "q_index",
                "q_pid", "q_cam", "q_mod", "g_mod", "rank", "ap", "valid"])
    # selection is fixed (primary, r=64, gamma=.1); this file records it.
    sel = dict(primary_arm="primary", rank=int(args.rank),
               gamma=float(args.gamma), epochs=int(args.epochs),
               seeds=primary_seeds, primary_state="final",
               last3_auxiliary=True,
               note="no val hyperparameter selection; final state is primary, "
                    "last-3 is auxiliary")
    with open(os.path.join(args.output_dir, "val_selection.json"), "w") as f:
        json.dump(_jsonable(sel), f, indent=2)
    print("wrote val_metrics.csv / val_per_query.csv")


def _eval_test_with_adapter(cache, adapter, apply_all, device):
    return _eval_all(cache, None, adapter, apply_all, device, "test")


def _adapter_from_state(cache, state, arm, args, device):
    if state is None:
        return None, False
    dim = int(cache["train_feat_1"].shape[1])
    adapter, apply_all = _make_adapter(
        arm, dim, args.rank, args.gamma,
        float(torch.as_tensor(cache["train_feat_3"]).norm(dim=1).median().item()
              * args.rmax_factor))
    adapter.load_state_dict(state)
    return adapter.to(device).eval(), apply_all


def stage_test(args):
    out_test = os.path.join(args.output_dir, "test_metrics.csv")
    if os.path.exists(out_test) and not args.force:
        raise RuntimeError("test already exists; refusing a second test "
                           "(use --force only for a pre-registered rerun)")
    if not os.path.exists(os.path.join(args.output_dir, "val_metrics.csv")):
        raise RuntimeError("run --stage val before --stage test")
    cache = _load_npz(os.path.join(args.output_dir, "features.npz"))
    split = _load_split(args)
    device = torch.device(args.device)
    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    folds = split["folds"]
    # gamma0 baseline; test is evaluated once, but fold adapters are only
    # trained on their 400-ID training folds; test IDs never enter training.
    base = _eval_test_with_adapter(cache, None, False, device)
    rows, perq = [], []
    for scope in ("tir_cross", "tir_same", "mixed"):
        rs = base[scope if scope != "mixed" else "mixed"]
        if scope == "tir_same":
            rs = base["tir_same"]
        elif scope == "tir_cross":
            rs = base["tir_cross"]
        mm = _metrics(rs)
        rows.append(dict(arm="gamma0", seed=0, fold=-1, state=0,
                         scope=scope, **mm))
        perq.extend(_rows_for_csv(rs, "test", -1, 0, 0, "gamma0", scope))
    cos_by_state = defaultdict(list)
    residuals = []
    for fold in folds:
        for seed in seeds:
            states, stat = _train_adapter(cache, fold, "primary", seed, args, device)
            residuals.append(stat.get("residual_ratio", 0.0))
            for si, st in enumerate(states):
                adapter, apply_all = _adapter_from_state(
                    cache, st, "primary", args, device)
                ev = _eval_test_with_adapter(cache, adapter, apply_all, device)
                for scope in ("tir_cross", "tir_same", "mixed"):
                    rs = ev["mixed"] if scope == "mixed" else ev[scope]
                    mm = _metrics(rs)
                    rows.append(dict(arm="primary", seed=seed,
                                     fold=fold["fold"], state=si,
                                     scope=scope, **mm,
                                     residual_ratio=stat.get("residual_ratio")))
                    perq.extend(_rows_for_csv(
                        rs, "test", fold["fold"], seed, si, "primary", scope))
                cos_by_state[si].append(ev["cos_matrix"])
    final_state = max(cos_by_state) if cos_by_state else 0
    if cos_by_state:
        cos_final = np.nanmean(np.stack(cos_by_state[final_state], 0), 0)
        np.savetxt(os.path.join(args.output_dir, "classmean_cos.csv"),
                   cos_final, delimiter=",")
        cos_l3 = np.nanmean(np.stack([m for ms in cos_by_state.values()
                                      for m in ms], 0), 0)
        np.savetxt(os.path.join(args.output_dir, "classmean_cos_last3.csv"),
                   cos_l3, delimiter=",")
    _write_csv(out_test, rows, ["arm", "seed", "fold", "state", "scope",
                                "n", "valid", "mAP", "R1", "R5", "R10",
                                "gt100", "gt500", "residual_ratio"])
    _write_csv(os.path.join(args.output_dir, "test_per_query.csv"), perq,
               ["arm", "seed", "state", "split", "fold", "scope", "q_index",
                "q_pid", "q_cam", "q_mod", "g_mod", "rank", "ap", "valid"])

    def avg(arm, scope, key, state=None):
        vals = [r[key] for r in rows
                if r["arm"] == arm and r["scope"] == scope
                and (state is None or int(r["state"]) == int(state))
                and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    base_c = {k: avg("gamma0", "tir_cross", k) for k in
              ("mAP", "R1", "R5", "R10", "gt100", "gt500")}
    prim_c = {k: avg("primary", "tir_cross", k, state=final_state) for k in
              ("mAP", "R1", "R5", "R10", "gt100", "gt500")}
    prim_c_l3 = {k: avg("primary", "tir_cross", k) for k in
                 ("mAP", "R1", "R5", "R10", "gt100", "gt500")}
    base_s = {k: avg("gamma0", "tir_same", k) for k in
              ("mAP", "R1", "R5", "R10")}
    prim_s = {k: avg("primary", "tir_same", k, state=final_state) for k in
              ("mAP", "R1", "R5", "R10")}
    prim_s_l3 = {k: avg("primary", "tir_same", k) for k in
                 ("mAP", "R1", "R5", "R10")}
    base_m = avg("gamma0", "mixed", "mAP")
    prim_m = avg("primary", "mixed", "mAP", state=final_state)
    prim_m_l3 = avg("primary", "mixed", "mAP")
    d = {k: prim_c[k] - base_c[k] for k in base_c}
    d_l3 = {k: prim_c_l3[k] - base_c[k] for k in base_c}
    d_same = {k: prim_s[k] - base_s[k] for k in base_s}
    d_same_l3 = {k: prim_s_l3[k] - base_s[k] for k in base_s}
    d_mixed = prim_m - base_m
    d_mixed_l3 = prim_m_l3 - base_m
    # TIR cross cos delta (TIR->RGB, TIR->NIR, RGB->TIR, NIR->TIR)
    def _cos_delta(path):
        try:
            cos = np.loadtxt(path, delimiter=",")
            cos0 = _eval_test_with_adapter(cache, None, False, device)["cos_matrix"]
            return float(np.nanmean([
                cos[2, 0] - cos0[2, 0], cos[2, 1] - cos0[2, 1],
                cos[0, 2] - cos0[0, 2], cos[1, 2] - cos0[1, 2]]))
        except Exception:
            return float("nan")

    cos_delta = _cos_delta(os.path.join(args.output_dir, "classmean_cos.csv"))
    cos_delta_l3 = _cos_delta(os.path.join(args.output_dir,
                                           "classmean_cos_last3.csv"))
    # per-fold sign consistency on TIR cross mAP, final state only
    fold_delta = []
    for fold in folds:
        b = [r["mAP"] for r in rows if r["arm"] == "gamma0"
             and r["scope"] == "tir_cross"]
        p = [r["mAP"] for r in rows if r["arm"] == "primary"
             and r["scope"] == "tir_cross" and r["fold"] == fold["fold"]
             and int(r["state"]) == int(final_state)]
        if b and p:
            fold_delta.append(float(np.mean(p) - np.mean(b)))
    n_pos_folds = int(sum(x > 0 for x in fold_delta))
    b_rows = [r for r in perq if r["arm"] == "gamma0"
              and r["scope"] == "tir_cross"]
    a_rows = [r for r in perq if r["arm"] == "primary"
              and r["scope"] == "tir_cross"
              and int(r["state"]) == int(final_state)]
    boot = _bootstrap_delta(b_rows, a_rows, n=args.bootstrap, seed=args.seed)
    tail_ok = (d["R5"] >= 0.005 or d["R10"] >= 0.008
               or d["gt100"] <= -0.01 or d["gt500"] <= -0.005)
    same_ok = (d_same["mAP"] >= -0.005 and d_same["R1"] >= -0.010)
    mixed_ok = d_mixed >= -0.001
    cos_ok = (np.isfinite(cos_delta) and cos_delta >= 0.005)
    active = (np.mean(residuals) >= 0.02 if residuals else False)
    go = (active and d["mAP"] >= 0.0015 and boot["ci_low"] >= -0.0005
          and n_pos_folds >= 4 and tail_ok and same_ok and mixed_ok and cos_ok)
    strong = (go and d["mAP"] >= 0.0025 and boot["ci_low"] > 0
              and (d["R5"] >= 0.005 and d["R10"] >= 0.008))
    status = ("STRONG_GO" if strong else "GO" if go else
              ("INCONCLUSIVE" if not active else "NO_GO"))
    decision = dict(
        status=status,
        active=active,
        primary_state="final",
        final_state=int(final_state),
        residual_ratio_mean=float(np.mean(residuals)) if residuals else 0.0,
        baseline_cross=base_c, primary_cross=prim_c, delta=d,
        primary_cross_last3=prim_c_l3, delta_last3=d_l3,
        baseline_same=base_s, primary_same=prim_s, delta_same=d_same,
        delta_same_last3=d_same_l3,
        delta_mixed_mAP=d_mixed, delta_mixed_mAP_last3=d_mixed_l3,
        cos_delta_tir_cross=cos_delta,
        cos_delta_tir_cross_last3=cos_delta_l3,
        n_positive_folds=n_pos_folds, fold_deltas=fold_delta,
        bootstrap=boot,
        tail_ok=bool(tail_ok), same_ok=bool(same_ok),
        mixed_ok=bool(mixed_ok), cos_ok=bool(cos_ok),
        note="mAP/R1/R5/R10 are fractions; primary endpoint is the final "
             "adapter state (official-test analogue). last-3 mean is auxiliary. "
             "GO uses +0.15pp mAP, CI lower >= -0.05pp, >=4/5 positive folds, "
             "at least one tail signal, TIR-same hard guard "
             "(mAP >= -0.5pp, R1 >= -1.0pp), mixed mAP >= -0.10pp, and "
             "TIR-cross class-mean cos +0.005.",
        val_selection=os.path.join(args.output_dir, "val_selection.json"),
    )
    with open(os.path.join(args.output_dir, "decision.json"), "w") as f:
        json.dump(_jsonable(decision), f, indent=2)
    print(json.dumps(_jsonable(decision), indent=2))


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def _synthetic_cache(n_ids=30, dim=16, per_id=12):
    rng = np.random.RandomState(7)
    cache = {}
    centers = rng.randn(n_ids, dim).astype(np.float32)
    for m in range(1, 4):
        rows, pids, cams = [], [], []
        for pid in range(n_ids):
            for j in range(per_id):
                rows.append(centers[pid] + 0.3 * rng.randn(dim))
                pids.append(pid)
                cams.append(j % 3)
        f = np.stack(rows).astype(np.float32)
        cache["train_feat_{}".format(m)] = f
        cache["train_pid_{}".format(m)] = np.asarray(pids, np.int32)
        cache["train_cam_{}".format(m)] = np.asarray(cams, np.int32)
        cache["train_mod_{}".format(m)] = np.full(len(pids), m, np.int32)
        cache["q_feat_{}".format(m)] = f[:20]
        cache["g_feat_{}".format(m)] = f[:20]
        cache["q_pid_{}".format(m)] = np.asarray(pids[:20], np.int32)
        cache["g_pid_{}".format(m)] = np.asarray(pids[:20], np.int32)
        cache["q_cam_{}".format(m)] = np.asarray(cams[:20], np.int32)
        cache["g_cam_{}".format(m)] = np.asarray(cams[:20], np.int32)
    cache["classifier_weight"] = rng.randn(n_ids, dim).astype(np.float32)
    cache["bn_weight"] = np.ones(dim, np.float32)
    cache["bn_bias"] = np.zeros(dim, np.float32)
    cache["bn_mean"] = np.zeros(dim, np.float32)
    cache["bn_var"] = np.ones(dim, np.float32)
    cache["bn_eps"] = np.asarray([1e-5], np.float32)
    return cache


def run_selftest(args):
    set_seed(1234)
    cache = _synthetic_cache()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        np.savez(os.path.join(d, "features.npz"), **cache)
        args.output_dir = d
        args.folds = 5
        args.seed = 1234
        # build split without reading ids=500 assumption
        stage_split(args)
        split = _load_split(args)
        assert len(split["folds"]) == 5
        for fold in split["folds"]:
            for m in range(1, 4):
                assert fold["q_idx"][str(m)]
                assert fold["g_idx"][str(m)]
        # eval baseline
        device = torch.device("cpu")
        base = _eval_all(cache, split["folds"][0], None, False, device, "val")
        assert base["tir_cross"]
        # adapter train smoke
        args.rank = 4
        args.p_k = 2
        args.epochs = 3
        args.p = 4
        args.k = 2
        # Regression guard for the S64 label-row bug: row_idx must point to the
        # sorted-train_ids row of the identity whose cached images were sampled.
        train_ids = split["folds"][0]["train_ids"]
        pid2idx = {m: _pid_to_indices(cache["train_pid_{}".format(m)])
                   for m in range(1, 4)}
        feats, target, labels, row_idx, n_p = _sample_batch(
            cache, pid2idx, train_ids, np.random.RandomState(1234),
            args.p, args.k, device)
        assert int(n_p) == len(row_idx)
        assert len(set(int(x) for x in row_idx.tolist())) == int(n_p)
        assert torch.equal(target, labels.repeat_interleave(args.k))
        for i in range(int(n_p)):
            gp = int(train_ids[int(row_idx[i])])
            bank = cache["train_feat_1"][pid2idx[1][gp]]
            for j in range(args.k):
                assert np.any(np.all(
                    bank == feats[0][i, j].detach().cpu().numpy(), axis=1))
        states, stat = _train_adapter(cache, split["folds"][0], "primary",
                                      1234, args, device)
        assert states
        adapter, apply_all = _adapter_from_state(cache, states[-1], "primary", args, device)
        ev = _eval_all(cache, split["folds"][0], adapter, apply_all, device, "val")
        assert ev["tir_cross"]
        # gamma0 must be exact
        f = torch.as_tensor(cache["q_feat_1"][:4])
        assert torch.equal(f, _apply(f, None, False, 1))
        # bootstrap smoke
        boot = _bootstrap_delta(base["tir_cross"], ev["tir_cross"], n=50)
        assert "ci_low" in boot
    print("SELFTEST OK")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="WEV5 TIR calibration probe")
    p.add_argument("--stage", required=True,
                   choices=["cache", "split", "val", "test", "selftest"])
    p.add_argument("--config_file", default="")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--tag", default="v2")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset_root", default="/project/code/data")
    p.add_argument("--dataset_dir", default="WHU-MARS")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--p", type=int, default=16)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--rmax_factor", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seeds", default="1234,5678,9012")
    p.add_argument("--arms", default="primary,affine,linear,r32,shared,ce_only,ce_gpd")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--force", action="store_true")
    p.add_argument("opts", nargs=argparse.REMAINDER)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.stage == "selftest":
        run_selftest(args)
        return
    if args.stage == "cache":
        stage_cache(args)
    elif args.stage == "split":
        stage_split(args)
    elif args.stage == "val":
        stage_val(args)
    elif args.stage == "test":
        stage_test(args)


if __name__ == "__main__":
    main()
