#!/usr/bin/env python
"""Training-free GAC diagnostics for WEV4 (WEV4/WHU).

This script never trains, never creates an optimizer, and never touches the
training path.  It loads a V4/GAC checkpoint, extracts the pre-BN CLS feature
used at inference, and writes the diagnostics needed to decide whether GAC is
gallery-conditioned or just an extra classifier:

  * rank audit (same protocol as utils/metrics.py):
      summary.csv, rank_query.csv, pair_rank_query.csv
  * training-set head behaviour (labels are in classifier space):
      gac_head_ce_acc.csv, gac_head_home_cross.csv,
      gac_head_logits.csv, feature_norm.csv
  * head/bank geometry:
      gac_head_geometry.csv, gac_head_pair_cos.csv,
      gac_head_summary.csv, bank_geometry.json, train_geometry.json
  * gradient diagnostics on one training proxy batch:
      gac_grad.json, gac_aux_head_grad.csv
  * meta.json

Important:
  * Head CE/accuracy are computed on the *training* split.  Query/gallery
    identities are disjoint from the 500 training identities, so test labels
    are not valid classifier targets.
  * Rank bins use the same-camera exclusion and the pre-BN CLS + L2 protocol.
  * `gac_grad.json` is a proxy over the last ViT block + final norm, matching
    `diagnose_gpd_gradients`; it is not a full-ViT gradient.
  * If a full `checkpoint_latest.pth` is supplied, its HPD bank is used.  A
    model-only `transformer_*.pth` has no bank, so P_scene/P_id are recomputed
    from the training split with the same class-mean rule as warm start.

Usage (from WEV4/WHU):

  python tools/analyse_gac.py \
    --config_file configs/GAC_UAD.yml \
    --checkpoint logs/GAC_1000_DDP/transformer_120.pth \
    --output_dir logs/gac_diag_1000 --tag v4_1000

  # 1000-GD: same config/dataset as the test log, plus FILTER_GD.
  python tools/analyse_gac.py \
    --config_file configs/GAC_UAD.yml \
    --checkpoint logs/GAC_1000_DDP/transformer_120.pth \
    --output_dir logs/gac_diag_gd --tag v4_gd \
    DATASETS.FILTER_GD True

  # Full checkpoint: exact bank + aux heads.
  python tools/analyse_gac.py \
    --config_file configs/GAC_UAD.yml \
    --checkpoint logs/GAC_1000_DDP/checkpoint_latest.pth \
    --output_dir logs/gac_diag_1000_full --tag v4_1000_full

  # Quick smoke on CPU (head/rank may be slow; gradient proxy works too).
  python tools/analyse_gac.py --device cpu --num_workers 0 \
    --config_file configs/GAC_UAD.yml \
    --checkpoint /path/to/transformer_120.pth \
    --output_dir /tmp/gac_diag --max_train_batches 1 --skip_rank

Notes:
  * `TEST.NECK_FEAT` must be 'before' and `TEST.FEAT_NORM` must be 'yes'.
  * `--skip_rank` skips the test-set feature pass.
  * `--skip_head` skips the training-set head pass; a full checkpoint bank is
    then required for head/bank geometry.
  * `--skip_grad` skips the gradient proxy.
  * `--max_train_batches N` limits the head pass (0 = all).
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import cfg  # noqa: E402
from datasets import make_dataloader  # noqa: E402
from model import make_model  # noqa: E402
from processor.processor import (  # noqa: E402
    HierarchicalPrototypeBank,
    PCA,
    diagnose_gpd_gradients,
    gac_aux_loss,
)
from loss.wrt_loss import TripletLoss_WRT  # noqa: E402


MOD_NAMES = {}


def set_mod_names(modalities):
    global MOD_NAMES
    MOD_NAMES = {i + 1: str(name) for i, name in enumerate(modalities)}


def mod_name(m):
    return MOD_NAMES.get(int(m), "M{}".format(int(m)))


def platform_name(cam):
    return "ground" if int(cam) < 5 else "aerial"


def _np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _rank_bin(rank):
    if rank <= 1:
        return "1"
    if rank <= 5:
        return "2-5"
    if rank <= 10:
        return "6-10"
    return ">10"


def _audit_from_keep(scores, keep, q_pid, q_cam, q_mod, q_platform,
                     g_pids, g_cams, g_mods, g_platforms):
    """Rank/margin/top-1 for one query given the same-camera keep mask."""
    idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return None
    s = scores[idx]
    gp = g_pids[idx]
    gm = g_mods[idx]
    gc = g_cams[idx]
    gpl = g_platforms[idx]

    pos = gp == q_pid
    neg = ~pos
    if not bool(pos.any()) or not bool(neg.any()):
        return None

    pos_best = s[pos].max()
    neg_best = s[neg].max()
    # Rank of the best positive, ties in favour of the positive.
    rank = 1 + int((s > pos_best).sum().item())
    top1 = int(s.argmax().item())
    top1_pid = int(gp[top1].item())
    top1_mod = int(gm[top1].item())
    top1_cam = int(gc[top1].item())
    top1_pl = platform_name(top1_cam)
    top1_correct = (top1_pid == int(q_pid))
    return {
        "q_pid": int(q_pid),
        "q_cam": int(q_cam),
        "q_mod": int(q_mod),
        "q_platform": q_platform,
        "rank": rank,
        "rank_bin": _rank_bin(rank),
        "margin": float(pos_best.item() - neg_best.item()),
        "top1_pid": top1_pid,
        "top1_mod": top1_mod,
        "top1_platform": top1_pl,
        "top1_correct": bool(top1_correct),
    }


def extract_loader_features(model, loader, mode, device):
    feats, pids, cams, mods, norms = [], [], [], [], []
    for batch in loader:
        img = batch[0].to(device, non_blocking=True)
        with torch.no_grad():
            feat = model(img, mode=mode)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        raw = feat.detach().float().cpu()
        feats.append(raw)
        norms.append(raw.norm(dim=1))
        pids.append(_np(batch[1]).astype(np.int64))
        cams.append(_np(batch[2]).astype(np.int64))
        mods.append(_np(batch[4]).astype(np.int64))
    if not feats:
        return (torch.empty(0, 0), np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                torch.empty(0))
    return (torch.cat(feats, dim=0), np.concatenate(pids),
            np.concatenate(cams), np.concatenate(mods),
            torch.cat(norms, dim=0))


def load_split_features(model, val_loaders, num_querys, device, logger=print):
    """Return per-modality and mixed query/gallery features (L2-normalized)."""
    per_mod = []
    for mode, (loader, nq) in enumerate(zip(val_loaders, num_querys), start=1):
        feats, pids, cams, mods, norms = extract_loader_features(
            model, loader, mode, device)
        nq = int(nq)
        if nq <= 0 or nq >= feats.shape[0]:
            raise RuntimeError("bad query count mode={} nq={} total={}".format(
                mode, nq, feats.shape[0]))
        feats = F.normalize(feats.float(), dim=1)
        qf, gf = feats[:nq], feats[nq:]
        qp, gp = pids[:nq], pids[nq:]
        qc, gc = cams[:nq], cams[nq:]
        qm = mods[:nq] if mods.size else np.full(nq, mode, dtype=np.int64)
        gm = mods[nq:] if mods.size else np.full(gf.shape[0], mode, dtype=np.int64)
        qnorm, gnorm = norms[:nq], norms[nq:]
        per_mod.append(dict(loader=loader, mode=mode, nq=nq,
                            qf=qf, gf=gf, qp=qp, gp=gp, qc=qc, gc=gc,
                            qm=qm, gm=gm, qnorm=qnorm, gnorm=gnorm))
        logger("mode {} {}: query {} gallery {}; feature norm q/g {:.3f}/{:.3f}".format(
            mode, mod_name(mode), qf.shape[0], gf.shape[0],
            float(qnorm.mean().item()), float(gnorm.mean().item())))

    qf = torch.cat([x["qf"] for x in per_mod], dim=0)
    gf = torch.cat([x["gf"] for x in per_mod], dim=0)
    qp = np.concatenate([x["qp"] for x in per_mod])
    gp = np.concatenate([x["gp"] for x in per_mod])
    qc = np.concatenate([x["qc"] for x in per_mod])
    gc = np.concatenate([x["gc"] for x in per_mod])
    qm = np.concatenate([x["qm"] for x in per_mod])
    gm = np.concatenate([x["gm"] for x in per_mod])
    qnorm = torch.cat([x["qnorm"] for x in per_mod], dim=0)
    gnorm = torch.cat([x["gnorm"] for x in per_mod], dim=0)
    return per_mod, dict(qf=qf, gf=gf, qp=qp, gp=gp, qc=qc, gc=gc,
                         qm=qm, gm=gm, qnorm=qnorm, gnorm=gnorm)


def run_audit(mixed, per_mod, device, query_chunk=256, logger=print):
    """One score pass over mixed gallery; derive mixed/pair/AG audits."""
    qf = mixed["qf"].to(device)
    gf = mixed["gf"].to(device)
    qp = mixed["qp"]
    gp = mixed["gp"]
    qc = mixed["qc"]
    gc = mixed["gc"]
    qm = mixed["qm"]
    gm = mixed["gm"]

    gp_t = torch.as_tensor(gp, dtype=torch.long, device=device)
    gc_t = torch.as_tensor(gc, dtype=torch.long, device=device)
    gm_t = torch.as_tensor(gm, dtype=torch.long, device=device)
    gpl_t = torch.as_tensor([0 if c < 5 else 1 for c in gc],
                            dtype=torch.long, device=device)

    mixed_rows, pair_rows, ag_rows = [], [], []
    rank_rows = {}
    mod_ids = sorted(set(int(x) for x in qm))
    for s in range(0, qf.shape[0], int(query_chunk)):
        e = min(s + int(query_chunk), qf.shape[0])
        scores_all = qf[s:e] @ gf.t()
        for off in range(e - s):
            qi = s + off
            q_pid = int(qp[qi])
            q_cam = int(qc[qi])
            q_mod = int(qm[qi])
            q_pl = platform_name(q_cam)
            scores = scores_all[off]
            keep_cam = gc_t != q_cam

            # mixed gallery
            r = _audit_from_keep(scores, keep_cam, q_pid, q_cam, q_mod, q_pl,
                                 gp_t, gc_t, gm_t, gpl_t)
            if r is not None:
                rank_rows[qi] = dict(
                    mixed_rank=r["rank"], mixed_margin=r["margin"],
                    mixed_top1_pid=r["top1_pid"],
                    mixed_top1_mod=mod_name(r["top1_mod"]),
                    mixed_top1_platform=r["top1_platform"],
                    mixed_rank_bin=r["rank_bin"],
                )
                r_mixed = dict(r)
                r_mixed.update(scope="mixed", q_index=qi,
                               q_mod_name=mod_name(q_mod),
                               g_mod_name="ALL",
                               q_platform="ALL", g_platform="ALL")
                mixed_rows.append(r_mixed)
                r_pl = dict(r)
                r_pl.update(scope="mixed_platform", q_index=qi,
                            q_mod_name="ALL", g_mod_name="ALL",
                            q_platform=q_pl, g_platform="ALL")
                mixed_rows.append(r_pl)

            # pair-space: query modality -> each gallery modality
            for g_mod in mod_ids:
                keep = keep_cam & (gm_t == int(g_mod))
                pr = _audit_from_keep(scores, keep, q_pid, q_cam, q_mod, q_pl,
                                      gp_t, gc_t, gm_t, gpl_t)
                if pr is None:
                    continue
                pr.update(scope="pair", q_index=qi,
                          q_mod_name=mod_name(q_mod),
                          g_mod_name=mod_name(g_mod))
                pair_rows.append(pr)

            # AG: all modalities, platform-only
            if q_pl == "aerial":
                ag_name, g_pl_id, g_pl_name = "ag_A->G", 0, "ground"
            else:
                ag_name, g_pl_id, g_pl_name = "ag_G->A", 1, "aerial"
            keep_ag = keep_cam & (gpl_t == g_pl_id)
            ar = _audit_from_keep(scores, keep_ag, q_pid, q_cam, q_mod, q_pl,
                                  gp_t, gc_t, gm_t, gpl_t)
            if ar is not None:
                ar.update(scope=ag_name, q_index=qi,
                          q_mod_name=mod_name(q_mod),
                          g_mod_name="ALL", g_platform=g_pl_name)
                ag_rows.append(ar)
                rank_rows.setdefault(qi, {}).update(
                    ag_dir=ag_name, ag_rank=ar["rank"],
                    ag_margin=ar["margin"], ag_top1_pid=ar["top1_pid"],
                    ag_top1_mod=mod_name(ar["top1_mod"]),
                    ag_top1_platform=ar["top1_platform"],
                    ag_rank_bin=ar["rank_bin"],
                )

    logger("audit rows: mixed={} pair={} ag={}".format(
        len(mixed_rows), len(pair_rows), len(ag_rows)))
    return mixed_rows, pair_rows, ag_rows, rank_rows


def _agg_key(r):
    return (r.get("scope", ""), r.get("q_mod_name", ""),
            r.get("g_mod_name", ""), r.get("q_platform", "ALL"),
            r.get("g_platform", "ALL"))


def aggregate_rows(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[_agg_key(r)].append(r)
    out = []
    for key, vals in sorted(groups.items()):
        ranks = np.asarray([v["rank"] for v in vals], dtype=np.float64)
        margins = np.asarray([v["margin"] for v in vals], dtype=np.float64)
        margins = margins[np.isfinite(margins)]
        n = len(vals)
        wrong_mod = {1: 0, 2: 0, 3: 0}
        wrong_pl = {"ground": 0, "aerial": 0}
        for v in vals:
            if not v["top1_correct"]:
                wrong_mod[int(v["top1_mod"])] = wrong_mod.get(
                    int(v["top1_mod"]), 0) + 1
                wrong_pl[v["top1_platform"]] = wrong_pl.get(
                    v["top1_platform"], 0) + 1
        out.append(dict(
            scope=key[0], q_mod=key[1], g_mod=key[2],
            q_platform=key[3], g_platform=key[4], n=n,
            rank1=int((ranks == 1).sum()),
            rank2_5=int(((ranks >= 2) & (ranks <= 5)).sum()),
            rank6_10=int(((ranks >= 6) & (ranks <= 10)).sum()),
            rank_gt10=int((ranks > 10).sum()),
            r1=float((ranks == 1).mean()) if n else float("nan"),
            mean_rank=float(ranks.mean()) if n else float("nan"),
            median_rank=float(np.median(ranks)) if n else float("nan"),
            mean_margin=float(margins.mean()) if margins.size else float("nan"),
            median_margin=float(np.median(margins)) if margins.size else float("nan"),
            top1_correct=int(sum(1 for v in vals if v["top1_correct"])),
            wrong_mod_RGB=wrong_mod.get(1, 0),
            wrong_mod_NIR=wrong_mod.get(2, 0),
            wrong_mod_TIR=wrong_mod.get(3, 0),
            wrong_platform_ground=wrong_pl.get("ground", 0),
            wrong_platform_aerial=wrong_pl.get("aerial", 0),
        ))
    return out


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary(out_dir, mixed_rows, pair_rows, ag_rows):
    rows = mixed_rows + pair_rows + ag_rows
    summary = aggregate_rows(rows)
    fields = ["scope", "q_mod", "g_mod", "q_platform", "g_platform", "n",
              "rank1", "rank2_5", "rank6_10", "rank_gt10", "r1",
              "mean_rank", "median_rank", "mean_margin", "median_margin",
              "top1_correct", "wrong_mod_RGB", "wrong_mod_NIR",
              "wrong_mod_TIR", "wrong_platform_ground",
              "wrong_platform_aerial"]
    write_csv(os.path.join(out_dir, "summary.csv"), summary, fields)
    return summary


def write_query_csvs(out_dir, mixed, rank_rows, pair_rows):
    q_fields = ["q_index", "q_pid", "q_cam", "q_mod", "q_platform",
                "mixed_rank", "mixed_rank_bin", "mixed_margin",
                "mixed_top1_pid", "mixed_top1_mod", "mixed_top1_platform",
                "ag_dir", "ag_rank", "ag_rank_bin", "ag_margin",
                "ag_top1_pid", "ag_top1_mod", "ag_top1_platform"]
    qrows = []
    for qi in range(mixed["qf"].shape[0]):
        d = rank_rows.get(qi, {})
        qrows.append(dict(
            q_index=qi, q_pid=int(mixed["qp"][qi]),
            q_cam=int(mixed["qc"][qi]),
            q_mod=mod_name(int(mixed["qm"][qi])),
            q_platform=platform_name(int(mixed["qc"][qi])),
            **d
        ))
    write_csv(os.path.join(out_dir, "rank_query.csv"), qrows, q_fields)

    p_fields = ["q_index", "q_pid", "q_cam", "q_mod", "q_platform",
                "g_mod", "rank", "rank_bin", "margin",
                "top1_pid", "top1_mod", "top1_platform", "top1_correct"]
    prows = []
    for r in pair_rows:
        prows.append(dict(
            q_index=r["q_index"], q_pid=r["q_pid"], q_cam=r["q_cam"],
            q_mod=r["q_mod_name"], q_platform=r["q_platform"],
            g_mod=r["g_mod_name"], rank=r["rank"], rank_bin=r["rank_bin"],
            margin=r["margin"], top1_pid=r["top1_pid"],
            top1_mod=mod_name(r["top1_mod"]),
            top1_platform=r["top1_platform"],
            top1_correct=r["top1_correct"],
        ))
    write_csv(os.path.join(out_dir, "pair_rank_query.csv"), prows, p_fields)


def _class_means_from_sums(sums, counts):
    """sums/counts: [C,D]/[C] for one modality; returns L2 means + valid mask."""
    valid = counts > 0
    means = torch.zeros_like(sums)
    if bool(valid.any()):
        means[valid] = F.normalize(
            sums[valid] / counts[valid].unsqueeze(1), dim=1)
    return means, valid


def analyze_train_split(model, loaders, num_classes, cfg, device,
                        max_batches=0, logger=print):
    """One training-split pass: head CE/acc/logits + recomputed P_scene/P_id.

    Head CE/acc are valid here because training pids are relabelled to
    [0, num_classes).  Query/gallery pids are not used as classifier targets.
    """
    M = len(loaders)
    D = int(getattr(model, "in_planes", 768))
    W = None
    if hasattr(model, "aux_heads"):
        W = F.normalize(model.aux_heads.detach().float(), dim=2).to(device)
    tau = float(getattr(cfg.MODEL, "AUX_TAU", 0.03))

    sums = [torch.zeros(num_classes, D, device=device) for _ in range(M)]
    counts = [torch.zeros(num_classes, device=device) for _ in range(M)]
    stats = defaultdict(lambda: dict(
        n=0, ce=0.0, acc=0.0, pos=0.0, neg=0.0, neg_max=0.0,
        entropy=0.0))
    norm_stats = defaultdict(lambda: dict(n=0, sum=0.0, sumsq=0.0))
    per_q_n = [0 for _ in range(M)]
    n_images = 0

    was_training = model.training
    model.eval()
    try:
        for q, loader in enumerate(loaders, start=1):
            n_batches = 0
            for batch in loader:
                if max_batches and n_batches >= int(max_batches):
                    break
                n_batches += 1
                img = batch[0].to(device, non_blocking=True)
                pid = torch.as_tensor(batch[1], dtype=torch.long, device=device)
                with torch.no_grad():
                    feat = model(img, mode=q)
                    if isinstance(feat, (tuple, list)):
                        feat = feat[0]
                    feat = feat.float()
                    f = F.normalize(feat, dim=1)
                    sums[q - 1].index_add_(0, pid, f)
                    counts[q - 1].index_add_(
                        0, pid,
                        torch.ones_like(pid, dtype=sums[q - 1].dtype))

                    raw_norm = feat.norm(dim=1)
                    ns = norm_stats[q]
                    ns["n"] += int(pid.numel())
                    ns["sum"] += float(raw_norm.sum().item())
                    ns["sumsq"] += float(raw_norm.pow(2).sum().item())

                    if W is not None:
                        logits = torch.einsum("nd,mcd->nmc", f, W) / tau
                        B = int(pid.numel())
                        idx = torch.arange(B, device=device)
                        for h in range(M):
                            lh = logits[:, h, :]
                            ce = F.cross_entropy(lh, pid)
                            acc = (lh.argmax(dim=1) == pid).float().mean()
                            pos = lh[idx, pid]
                            neg_mask = torch.ones_like(lh, dtype=torch.bool)
                            neg_mask[idx, pid] = False
                            neg = lh[neg_mask].reshape(B, -1)
                            neg_mean = neg.mean(dim=1)
                            neg_max = neg.max(dim=1).values
                            prob = lh.softmax(dim=1)
                            entropy = -(prob * lh.log_softmax(dim=1)).sum(dim=1)
                            st = stats[(q, h + 1)]
                            st["n"] += B
                            st["ce"] += float(ce.item()) * B
                            st["acc"] += float(acc.item()) * B
                            st["pos"] += float(pos.mean().item()) * B
                            st["neg"] += float(neg_mean.mean().item()) * B
                            st["neg_max"] += float(neg_max.mean().item()) * B
                            st["entropy"] += float(entropy.mean().item()) * B
                per_q_n[q - 1] += int(pid.numel())
                n_images += int(pid.numel())
            logger("train head pass: modality {} {} batches={} images={}".format(
                q, mod_name(q), n_batches, per_q_n[q - 1]))
    finally:
        model.train(was_training)

    means, valids = [], []
    for q in range(M):
        mean, valid = _class_means_from_sums(sums[q], counts[q])
        means.append(mean)
        valids.append(valid)

    id_sum = torch.zeros(num_classes, D, device=device)
    id_cnt = torch.zeros(num_classes, device=device)
    for mean, valid in zip(means, valids):
        id_sum += mean * valid.unsqueeze(1).float()
        id_cnt += valid.float()
    id_mean = F.normalize(
        id_sum / torch.clamp(id_cnt, min=1.0).unsqueeze(1), dim=1)

    p_scene = torch.stack(means, dim=1)          # [C,M,D]
    for q in range(M):
        missing = ~valids[q]
        if bool(missing.any()):
            p_scene[missing, q] = id_mean[missing]
    p_scene = F.normalize(p_scene, dim=2)

    return dict(
        p_scene=p_scene.detach().cpu(),
        p_id=id_mean.detach().cpu(),
        valids=[v.detach().cpu() for v in valids],
        counts=torch.stack(counts, dim=1).detach().cpu(),  # [C,M]
        head_stats={k: dict(v) for k, v in stats.items()},
        norm_stats={k: dict(v) for k, v in norm_stats.items()},
        per_q_n=per_q_n,
        n_images=n_images,
    )


def _stat_rows(head_stats, M):
    ce_rows, home_rows, logit_rows = [], [], []
    for q in range(1, M + 1):
        for h in range(1, M + 1):
            st = head_stats.get((q, h))
            if not st or st["n"] <= 0:
                continue
            n = float(st["n"])
            ce_rows.append(dict(
                q_mod=mod_name(q), head=h, head_mod=mod_name(h),
                n=int(st["n"]), ce=st["ce"] / n, acc=st["acc"] / n))
            logit_rows.append(dict(
                q_mod=mod_name(q), head=h, head_mod=mod_name(h),
                n=int(st["n"]), pos_mean=st["pos"] / n,
                neg_mean=st["neg"] / n, neg_max_mean=st["neg_max"] / n,
                entropy=st["entropy"] / n))
        home = head_stats.get((q, q))
        cross = [head_stats[(q, h)] for h in range(1, M + 1) if h != q
                 and (q, h) in head_stats]
        if home and cross:
            home_n = float(home["n"])
            cross_n = float(sum(x["n"] for x in cross))
            home_rows.append(dict(
                q_mod=mod_name(q), n=int(home["n"]),
                home_ce=home["ce"] / home_n,
                home_acc=home["acc"] / home_n,
                cross_ce=sum(x["ce"] for x in cross) / cross_n,
                cross_acc=sum(x["acc"] for x in cross) / cross_n))
    return ce_rows, home_rows, logit_rows


def _norm_rows(norm_stats):
    rows = []
    for q in sorted(norm_stats):
        st = norm_stats[q]
        n = max(1, int(st["n"]))
        mean = st["sum"] / n
        var = max(0.0, st["sumsq"] / n - mean * mean)
        rows.append(dict(mod=mod_name(q), n=int(st["n"]),
                         mean_norm=mean, std_norm=math.sqrt(var)))
    return rows


def write_head_behaviour(out_dir, train_info, M):
    ce_rows, home_rows, logit_rows = _stat_rows(
        train_info["head_stats"], M)
    write_csv(os.path.join(out_dir, "gac_head_ce_acc.csv"), ce_rows,
              ["q_mod", "head", "head_mod", "n", "ce", "acc"])
    write_csv(os.path.join(out_dir, "gac_head_home_cross.csv"), home_rows,
              ["q_mod", "n", "home_ce", "home_acc", "cross_ce", "cross_acc"])
    write_csv(os.path.join(out_dir, "gac_head_logits.csv"), logit_rows,
              ["q_mod", "head", "head_mod", "n", "pos_mean", "neg_mean",
               "neg_max_mean", "entropy"])
    write_csv(os.path.join(out_dir, "feature_norm.csv"),
              _norm_rows(train_info["norm_stats"]),
              ["mod", "n", "mean_norm", "std_norm"])
    return ce_rows, home_rows, logit_rows


def head_geometry_tables(W_raw, P_scene, P_id):
    """Full head x P_scene cos matrix, head pair cos, head summary."""
    W = F.normalize(W_raw.detach().float().cpu(), dim=2)
    Ps = F.normalize(P_scene.detach().float().cpu(), dim=2)
    Pi = F.normalize(P_id.detach().float().cpu(), dim=1)
    M = int(W.shape[0])
    C = int(W.shape[1])
    if int(Ps.shape[1]) != M:
        raise ValueError(
            "GAC head count ({}) != P_scene columns ({}); check "
            "MODEL.GPD_SUB_CENTERS == len(DATASETS.MODALITIES)".format(
                M, int(Ps.shape[1])))

    w_valid = W.norm(dim=2) > 1e-6
    s_valid = Ps.norm(dim=2) > 1e-6
    i_valid = Pi.norm(dim=1) > 1e-6

    matrix_rows, pair_rows, summary_rows = [], [], []
    cos_matrix = torch.full((M, M), float("nan"))
    for m in range(M):
        for k in range(M):
            mask = w_valid[m] & s_valid[:, k]
            if bool(mask.any()):
                cos = (W[m][mask] * Ps[mask, k]).sum(dim=1)
                cos_matrix[m, k] = cos.mean()
                matrix_rows.append(dict(
                    head=m + 1, head_mod=mod_name(m + 1),
                    scene_col=k + 1, scene_mod=mod_name(k + 1),
                    valid_classes=int(mask.sum().item()),
                    mean_cos=float(cos.mean().item()),
                    min_cos=float(cos.min().item()),
                    max_cos=float(cos.max().item())))
            else:
                matrix_rows.append(dict(
                    head=m + 1, head_mod=mod_name(m + 1),
                    scene_col=k + 1, scene_mod=mod_name(k + 1),
                    valid_classes=0, mean_cos=float("nan"),
                    min_cos=float("nan"), max_cos=float("nan")))

        mask_id = w_valid[m] & i_valid
        id_cos = (float((W[m][mask_id] * Pi[mask_id]).sum(dim=1).mean().item())
                  if bool(mask_id.any()) else float("nan"))
        row = cos_matrix[m]
        best_col = int(torch.nan_to_num(row, nan=-2.0).argmax().item())
        summary_rows.append(dict(
            head=m + 1, head_mod=mod_name(m + 1),
            head_norm_mean=float(W_raw[m].norm(dim=1).mean().item()),
            head_norm_min=float(W_raw[m].norm(dim=1).min().item()),
            head_norm_max=float(W_raw[m].norm(dim=1).max().item()),
            scene_diag_cos=float(row[m].item()),
            scene_best_col=best_col + 1,
            scene_best_mod=mod_name(best_col + 1),
            scene_best_cos=float(row[best_col].item()),
            id_cos=id_cos,
        ))

    for m in range(M):
        for n in range(m + 1, M):
            mask = w_valid[m] & w_valid[n]
            if bool(mask.any()):
                cos = (W[m][mask] * W[n][mask]).sum(dim=1)
                pair_rows.append(dict(
                    head1=m + 1, head2=n + 1,
                    mod1=mod_name(m + 1), mod2=mod_name(n + 1),
                    valid_classes=int(mask.sum().item()),
                    mean_cos=float(cos.mean().item()),
                    min_cos=float(cos.min().item()),
                    max_cos=float(cos.max().item())))
            else:
                pair_rows.append(dict(
                    head1=m + 1, head2=n + 1,
                    mod1=mod_name(m + 1), mod2=mod_name(n + 1),
                    valid_classes=0, mean_cos=float("nan"),
                    min_cos=float("nan"), max_cos=float("nan")))

    return matrix_rows, pair_rows, summary_rows, cos_matrix


def write_head_geometry(out_dir, model, bank):
    W_raw = model.aux_heads.detach().float().cpu()
    P_scene = bank.scene_bank.sub_centers.detach().float().cpu()
    P_id = bank.id_bank.prototypes.detach().float().cpu()
    matrix_rows, pair_rows, summary_rows, cos_matrix = head_geometry_tables(
        W_raw, P_scene, P_id)
    write_csv(os.path.join(out_dir, "gac_head_geometry.csv"), matrix_rows,
              ["head", "head_mod", "scene_col", "scene_mod", "valid_classes",
               "mean_cos", "min_cos", "max_cos"])
    write_csv(os.path.join(out_dir, "gac_head_pair_cos.csv"), pair_rows,
              ["head1", "head2", "mod1", "mod2", "valid_classes",
               "mean_cos", "min_cos", "max_cos"])
    write_csv(os.path.join(out_dir, "gac_head_summary.csv"), summary_rows,
              ["head", "head_mod", "head_norm_mean", "head_norm_min",
               "head_norm_max", "scene_diag_cos", "scene_best_col",
               "scene_best_mod", "scene_best_cos", "id_cos"])
    return matrix_rows, pair_rows, summary_rows, cos_matrix


def bank_geometry(bank):
    P_scene = F.normalize(
        bank.scene_bank.sub_centers.detach().float().cpu(), dim=2)
    P_id = F.normalize(
        bank.id_bank.prototypes.detach().float().cpu(), dim=1)
    M = int(P_scene.shape[1])
    s_valid = P_scene.norm(dim=2) > 1e-6
    i_valid = P_id.norm(dim=1) > 1e-6
    rows = []
    for m in range(M):
        mask = s_valid[:, m] & i_valid
        if bool(mask.any()):
            cos = (P_id[mask] * P_scene[mask, m]).sum(dim=1)
            rows.append(dict(
                mod=mod_name(m + 1), valid_classes=int(mask.sum().item()),
                pid_scene_mean_cos=float(cos.mean().item()),
                pid_scene_min_cos=float(cos.min().item()),
                pid_scene_max_cos=float(cos.max().item())))
        else:
            rows.append(dict(
                mod=mod_name(m + 1), valid_classes=0,
                pid_scene_mean_cos=float("nan"),
                pid_scene_min_cos=float("nan"),
                pid_scene_max_cos=float("nan")))
    pair_rows = []
    for m in range(M):
        for n in range(m + 1, M):
            mask = s_valid[:, m] & s_valid[:, n]
            if bool(mask.any()):
                cos = (P_scene[mask, m] * P_scene[mask, n]).sum(dim=1)
                pair_rows.append(dict(
                    mod1=mod_name(m + 1), mod2=mod_name(n + 1),
                    valid_classes=int(mask.sum().item()),
                    mean_cos=float(cos.mean().item()),
                    min_cos=float(cos.min().item()),
                    max_cos=float(cos.max().item())))
            else:
                pair_rows.append(dict(
                    mod1=mod_name(m + 1), mod2=mod_name(n + 1),
                    valid_classes=0, mean_cos=float("nan"),
                    min_cos=float("nan"), max_cos=float("nan")))
    mean_scene = F.normalize(P_scene.mean(dim=1), dim=1)
    mask = i_valid & (mean_scene.norm(dim=1) > 0)
    pid_mean_scene_cos = (
        float((P_id[mask] * mean_scene[mask]).sum(dim=1).mean().item())
        if bool(mask.any()) else float("nan"))
    return dict(
        pid_scene=rows,
        scene_pairs=pair_rows,
        pid_mean_scene_cos=pid_mean_scene_cos,
        id_norm_min=float(P_id.norm(dim=1).min().item()),
        id_norm_max=float(P_id.norm(dim=1).max().item()),
        scene_norm_min=float(P_scene.norm(dim=2).min().item()),
        scene_norm_max=float(P_scene.norm(dim=2).max().item()),
    )


def train_geometry(train_info, bank):
    """Compare recomputed training class means with the loaded bank."""
    rec_scene = F.normalize(train_info["p_scene"].float(), dim=2)
    rec_id = F.normalize(train_info["p_id"].float(), dim=1)
    out = dict(
        recomputed_pid_mean_scene_cos=float("nan"),
        recomputed_scene_pairs=[],
        bank_vs_recomputed_scene_cos=[],
    )
    mean_scene = F.normalize(rec_scene.mean(dim=1), dim=1)
    valid = rec_id.norm(dim=1) > 0
    if bool(valid.any()):
        out["recomputed_pid_mean_scene_cos"] = float(
            (rec_id[valid] * mean_scene[valid]).sum(dim=1).mean().item())
    M = int(rec_scene.shape[1])
    for m in range(M):
        for n in range(m + 1, M):
            mask = (rec_scene[:, m].norm(dim=1) > 0) & (
                rec_scene[:, n].norm(dim=1) > 0)
            if bool(mask.any()):
                cos = (rec_scene[mask, m] * rec_scene[mask, n]).sum(dim=1)
                out["recomputed_scene_pairs"].append(dict(
                    mod1=mod_name(m + 1), mod2=mod_name(n + 1),
                    valid_classes=int(mask.sum().item()),
                    mean_cos=float(cos.mean().item())))
    if bank is not None:
        b_scene = F.normalize(
            bank.scene_bank.sub_centers.detach().float().cpu(), dim=2)
        for m in range(M):
            mask = (b_scene[:, m].norm(dim=1) > 0) & (
                rec_scene[:, m].norm(dim=1) > 0)
            if bool(mask.any()):
                cos = (b_scene[mask, m] * rec_scene[mask, m]).sum(dim=1)
                out["bank_vs_recomputed_scene_cos"].append(dict(
                    mod=mod_name(m + 1), valid_classes=int(mask.sum().item()),
                    mean_cos=float(cos.mean().item())))
    return out


def _grad_norm(grads):
    total = 0.0
    for g in grads:
        if g is not None:
            total += float(g.detach().float().pow(2).sum().item())
    return total ** 0.5


def _grad_dot(a, b):
    total = 0.0
    for x, y in zip(a, b):
        if x is not None and y is not None:
            total += float((x.detach().float() * y.detach().float()).sum().item())
    return total


def _make_loss_fn(cfg):
    triplet = TripletLoss_WRT()

    def loss_fn(score, feat, target):
        ce = F.cross_entropy(score, target)
        tri = triplet(feat, target)[0]
        return (cfg.MODEL.ID_LOSS_WEIGHT * ce
                + cfg.MODEL.TRIPLET_LOSS_WEIGHT * tri, ce, tri)

    return loss_fn


def gradient_diagnostics(model, train_loader, bank, cfg, device,
                         grad_batch_size=32, logger=print):
    """Total GAC/GPD gradient shares + per-head GAC gradient shares.

    Uses only the first `grad_batch_size` PKM samples of one training batch.
    The full training batch (SOLVER.IMS_PER_BATCH=256) is 3*256 images for
    three modalities; running it with gradients on one GPU can OOM even on
    24 GB.  Gradients are taken over the last ViT block and final norm,
    exactly like diagnose_gpd_gradients; this is a proxy, not full-ViT.
    """
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("train_loader is empty; cannot compute gradients")
    raw_b = int(batch[1].numel())
    k = int(cfg.DATALOADER.NUM_INSTANCE)
    if grad_batch_size and int(grad_batch_size) > 0:
        g = min(int(grad_batch_size), raw_b)
        # PCA/GPD require complete PK groups.
        g = max(k, (g // k) * k)
    else:
        g = raw_b
    if g > raw_b:
        g = raw_b

    # Slice before moving to GPU: the DataLoader batch itself can be large.
    imgs = [x[:g].to(device) for x in batch[0]]
    target = batch[1][:g].to(device).long()
    camids = [x[:g].to(device) for x in batch[2]]
    M = len(imgs)
    aux_w = float(getattr(cfg.MODEL, "AUX_WEIGHT", 0.0))
    loss_fn = _make_loss_fn(cfg)

    # Keep this call compatible with the original processor.py signature
    # (V4 checkpoints were produced before this tool existed).  Do not pass
    # tool-only keyword arguments here; the server may have the original
    # processor.py even when tools/analyse_gac.py is updated.
    diag = diagnose_gpd_gradients(
        model, imgs, target, camids, bank, cfg, device,
        cfg.DATALOADER.NUM_INSTANCE, loss_fn, aux_weight=aux_w)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    diag["grad_batch_size"] = int(target.numel())
    diag["grad_batch_raw"] = raw_b
    diag["grad_images"] = int(target.numel() * M)
    diag["grad_proxy"] = "last ViT block + final norm"
    diag["aux_weight_used"] = aux_w

    per_head_rows = []
    if aux_w > 0:
        was_training = model.training
        model.eval()
        try:
            with torch.enable_grad(), torch.cuda.amp.autocast(
                    enabled=(device.type == "cuda")):
                _, global_feat, _, aux_scores = model(
                    imgs, target, camids, return_aux=True)
                loss_pca, _, _, _ = PCA(
                    global_feat, target, num_modalities=M,
                    group_size=cfg.DATALOADER.NUM_INSTANCE)
                target_rep = target.repeat(M)
                loss_aux = gac_aux_loss(aux_scores, target_rep, M)
                params = list(model.base.blocks[-1].parameters()) + list(
                    model.base.norm.parameters())
                head_grads = []
                for h in range(M):
                    loss_h = F.cross_entropy(aux_scores[:, h, :], target_rep)
                    head_grads.append(torch.autograd.grad(
                        loss_h, params, retain_graph=True, allow_unused=True))
                g_aux = torch.autograd.grad(
                    loss_aux, params, retain_graph=True, allow_unused=True)
                g_pca = torch.autograd.grad(
                    loss_pca, params, retain_graph=False, allow_unused=True)
            n_aux = _grad_norm(g_aux)
            n_pca = _grad_norm(g_pca)
            diag["cos_aux_pca"] = _grad_dot(g_aux, g_pca) / (
                n_aux * n_pca + 1e-12)
            total_eff = sum(float(diag.get(k, 0.0)) for k in (
                "eff_id", "eff_scene", "eff_base", "eff_pca", "eff_aux")) + 1e-12
            raw_norms = [_grad_norm(g) for g in head_grads]
            raw_total = sum(raw_norms) + 1e-12
            for h, n_h in enumerate(raw_norms, start=1):
                per_head_rows.append(dict(
                    head=h, head_mod=mod_name(h), grad_norm=n_h,
                    raw_share=n_h / raw_total,
                    effective_share=(aux_w / M) * n_h / total_eff))
            diag["aux_head_grad_norms"] = raw_norms
            diag["aux_head_effective_share"] = [
                (aux_w / M) * n_h / total_eff for n_h in raw_norms]
        finally:
            model.train(was_training)
    else:
        diag["cos_aux_pca"] = float("nan")
        diag["aux_head_grad_norms"] = []
        diag["aux_head_effective_share"] = []
        logger("AUX_WEIGHT=0: per-head GAC gradients skipped")

    return diag, per_head_rows


def load_checkpoint_state(path):
    ckpt = torch.load(path, map_location="cpu")
    bank_state = None
    epoch = None
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        bank_state = ckpt.get("bank_state_dict")
        epoch = ckpt.get("epoch")
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise RuntimeError("unsupported checkpoint format: {}".format(path))
    state = _strip_module(state)
    return state, bank_state, epoch


def _strip_module(state):
    out = {}
    for k, v in state.items():
        k = str(k)
        if k.startswith("module."):
            k = k[len("module."):]
        out[k] = v
    return out


def load_model_weights(model, state, path):
    if "aux_heads" not in state:
        raise RuntimeError(
            "checkpoint {} has no aux_heads; this is not a V4/GAC checkpoint"
            .format(path))
    if tuple(state["aux_heads"].shape) != tuple(model.aux_heads.shape):
        raise RuntimeError(
            "aux_heads shape mismatch: checkpoint {} vs model {} (config/num_classes?)"
            .format(tuple(state["aux_heads"].shape),
                    tuple(model.aux_heads.shape)))
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            "failed to load {}: {} (check DATASETS.DATASET_DIR/num_classes)"
            .format(path, exc))
    bad_missing = [k for k in missing
                   if "aux_heads" not in k and "classifier" not in k]
    if bad_missing:
        raise RuntimeError(
            "checkpoint {} is missing model keys: {}".format(
                path, bad_missing[:8]))
    if unexpected:
        print("warning: unexpected checkpoint keys ignored: {}".format(
            unexpected[:8]))


def build_bank(cfg, num_classes, feat_dim, device):
    return HierarchicalPrototypeBank(
        num_classes=num_classes,
        feat_dim=feat_dim,
        num_subcenters=cfg.MODEL.GPD_SUB_CENTERS,
        momentum=cfg.MODEL.GPD_MOMENTUM,
        device=device,
        count_ema=cfg.MODEL.GPD_COUNT_EMA,
        ema_min=cfg.MODEL.GPD_EMA_MIN,
        ema_max=cfg.MODEL.GPD_EMA_MAX,
        shared_momentum=cfg.MODEL.GPD_SHARED_MOMENTUM,
    )


def restore_bank(bank, bank_state):
    if not bank_state:
        return False
    if bank_state.get("type") != "hierarchical":
        raise RuntimeError(
            "expect hierarchical bank, got {}".format(bank_state.get("type")))
    needed = ("id_prototypes", "scene_sub_centers")
    if any(k not in bank_state for k in needed):
        raise RuntimeError("hierarchical bank state is incomplete")
    bank.id_bank.prototypes.copy_(
        bank_state["id_prototypes"].to(bank.id_bank.prototypes.device))
    bank.scene_bank.sub_centers.copy_(
        bank_state["scene_sub_centers"].to(bank.scene_bank.sub_centers.device))
    if "scene_counts" in bank_state:
        bank.scene_bank.counts.copy_(
            bank_state["scene_counts"].to(bank.scene_bank.counts.device))
    return True


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
    return x


def parse_args():
    p = argparse.ArgumentParser(
        description="Training-free GAC diagnostics for WEV4")
    p.add_argument("--config_file", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--query_chunk", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=0,
                   help="override TEST/SOLVER batch size (0 = config)")
    p.add_argument("--num_workers", type=int, default=-1,
                   help="-1 = config; 0 is useful for CPU smoke")
    p.add_argument("--max_train_batches", type=int, default=0,
                   help="0 = all training batches for head diagnostics")
    p.add_argument("--grad_batch_size", type=int, default=32,
                   help="PKM samples used for the gradient proxy (default 32; "
                        "0 = full training batch; must be a multiple of "
                        "DATALOADER.NUM_INSTANCE)")
    p.add_argument("--pretrain_choice", default="self",
                   help="'self' avoids needing ImageNet weights; checkpoint "
                        "provides the model")
    p.add_argument("--aux_weight", type=float, default=None,
                   help="override MODEL.AUX_WEIGHT for gradient diagnostics "
                        "only (default: config)")
    p.add_argument("--bank_source", default="auto",
                   choices=["auto", "checkpoint", "recompute"],
                   help="auto: full checkpoint bank else recompute from train")
    p.add_argument("--skip_rank", action="store_true")
    p.add_argument("--skip_head", action="store_true")
    p.add_argument("--skip_grad", action="store_true")
    p.add_argument("opts", nargs=argparse.REMAINDER,
                   help="YACS overrides, e.g. DATASETS.FILTER_GD True")
    return p.parse_args()


def main():
    args = parse_args()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    if str(cfg.TEST.NECK_FEAT).lower() != "before":
        raise ValueError("TEST.NECK_FEAT must be 'before', got {}".format(
            cfg.TEST.NECK_FEAT))
    if str(cfg.TEST.FEAT_NORM).lower() not in ("yes", "true", "1"):
        raise ValueError("TEST.FEAT_NORM must be 'yes' for this analysis")
    cfg.MODEL.PRETRAIN_CHOICE = args.pretrain_choice
    cfg.MODEL.DIST_TRAIN = False
    if args.batch_size and args.batch_size > 0:
        cfg.TEST.IMS_PER_BATCH = args.batch_size
        cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    if args.num_workers >= 0:
        cfg.DATALOADER.NUM_WORKERS = args.num_workers
    if args.aux_weight is not None:
        cfg.MODEL.AUX_WEIGHT = args.aux_weight
    cfg.freeze()

    set_mod_names(cfg.DATASETS.MODALITIES)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; use --device cpu")
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()
    print("loading dataloaders: config={} checkpoint={}".format(
        args.config_file, args.checkpoint))
    (train_loader, train_normal_loaders, val_loaders, num_querys,
     num_classes, camera_num, view_num) = make_dataloader(cfg)
    print("num_classes={} modalities={} query_counts={}".format(
        num_classes, len(cfg.DATASETS.MODALITIES), num_querys))

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num,
                       view_num=view_num)
    state, bank_state, ckpt_epoch = load_checkpoint_state(args.checkpoint)
    load_model_weights(model, state, args.checkpoint)
    if float(model.aux_heads.detach().abs().sum().item()) == 0.0:
        print("warning: aux_heads are all zero; head CE/geometry/gradients are "
              "not meaningful (is this a lambda=0 or non-GAC checkpoint?)")
    model.to(device)
    model.eval()

    train_info = None
    if not args.skip_head:
        print("training-split head pass (CE/acc/logits + class means)...")
        train_info = analyze_train_split(
            model, train_normal_loaders, num_classes, cfg, device,
            max_batches=args.max_train_batches)
        write_head_behaviour(
            args.output_dir, train_info, len(cfg.DATASETS.MODALITIES))
    else:
        print("skip_head=True: no training-split head diagnostics")

    bank = None
    bank_source = "none"
    if args.bank_source == "checkpoint" and not bank_state:
        raise RuntimeError(
            "--bank_source checkpoint requested, but {} has no bank_state_dict"
            .format(args.checkpoint))
    if bank_state and args.bank_source in ("auto", "checkpoint"):
        bank = build_bank(cfg, num_classes, model.in_planes, device)
        if restore_bank(bank, bank_state):
            bank_source = "checkpoint"
    if bank is None:
        if train_info is None:
            raise RuntimeError(
                "no checkpoint bank and --skip_head=True; cannot build bank")
        bank = build_bank(cfg, num_classes, model.in_planes, device)
        bank.initialize(
            train_info["p_scene"].to(device), train_info["p_id"].to(device))
        bank.scene_bank.counts.copy_(train_info["counts"].to(device))
        bank_source = "recomputed_train"
    print("bank source: {}".format(bank_source))

    if train_info is not None:
        geo = train_geometry(train_info, bank)
        with open(os.path.join(args.output_dir, "train_geometry.json"), "w") as f:
            json.dump(_jsonable(geo), f, indent=2)
    bgeo = bank_geometry(bank)
    with open(os.path.join(args.output_dir, "bank_geometry.json"), "w") as f:
        json.dump(_jsonable(bgeo), f, indent=2)
    write_head_geometry(args.output_dir, model, bank)

    grad_diag = None
    grad_rows = []
    if not args.skip_grad:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print("gradient proxy on one training batch "
              "(grad_batch_size={})...".format(args.grad_batch_size))
        grad_diag, grad_rows = gradient_diagnostics(
            model, train_loader, bank, cfg, device,
            grad_batch_size=args.grad_batch_size)
        with open(os.path.join(args.output_dir, "gac_grad.json"), "w") as f:
            json.dump(_jsonable(grad_diag), f, indent=2, allow_nan=True)
        write_csv(os.path.join(args.output_dir, "gac_aux_head_grad.csv"),
                  grad_rows, ["head", "head_mod", "grad_norm",
                              "raw_share", "effective_share"])
    else:
        print("skip_grad=True: no gradient diagnostics")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    summary = []
    if not args.skip_rank:
        print("test-set feature pass and rank audit...")
        per_mod, mixed = load_split_features(
            model, val_loaders, num_querys, device)
        mixed_rows, pair_rows, ag_rows, rank_rows = run_audit(
            mixed, per_mod, device, query_chunk=args.query_chunk)
        summary = write_summary(args.output_dir, mixed_rows, pair_rows, ag_rows)
        write_query_csvs(args.output_dir, mixed, rank_rows, pair_rows)
        query_norm_mean = [float(x["qnorm"].mean().item()) for x in per_mod]
        gallery_norm_mean = [float(x["gnorm"].mean().item()) for x in per_mod]
        gallery_counts = [int(x["gf"].shape[0]) for x in per_mod]
    else:
        print("skip_rank=True: no rank audit")
        query_norm_mean = []
        gallery_norm_mean = []
        gallery_counts = []

    meta = dict(
        tag=args.tag,
        config_file=args.config_file,
        checkpoint=args.checkpoint,
        checkpoint_epoch=ckpt_epoch,
        output_dir=args.output_dir,
        device=str(device),
        num_classes=num_classes,
        modalities=list(cfg.DATASETS.MODALITIES),
        query_counts=[int(x) for x in num_querys],
        gallery_counts=gallery_counts,
        query_feature_norm_mean=query_norm_mean,
        gallery_feature_norm_mean=gallery_norm_mean,
        bank_source=bank_source,
        aux_weight=float(cfg.MODEL.AUX_WEIGHT),
        aux_tau=float(cfg.MODEL.AUX_TAU),
        summary_rows=len(summary),
        grad_done=bool(grad_diag is not None),
        train_images=int(train_info["n_images"]) if train_info else 0,
        wall_time_sec=round(time.time() - t0, 2),
        note="rank uses same-camera exclusion; features are pre-BN CLS, L2; "
             "head CE/acc are on the relabelled training split; gradient is "
             "a last-block+norm proxy; no training/optimizer is created",
    )
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(_jsonable(meta), f, indent=2)
    print("wrote {} in {:.1f}s".format(args.output_dir, time.time() - t0))


if __name__ == "__main__":
    main()
