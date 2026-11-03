#!/usr/bin/env python
"""Training-free layer-route probe for V5 PASD-C.

Stages
------
cache  : extract f5/f8/f12 pre-BN CLS from a V2 (or V4) checkpoint.
screen : evaluate training-free layer/alpha routes on the 5-fold val split;
         select at most one route; no test evaluation.
test   : evaluate the selected route on the official test split once.
selftest : synthetic checks of routing/eval/selection.

No parameters are trained.  No model/processor/train/test code is modified.
The tool reuses the evaluation helpers in tir_calib_probe.py.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from config import cfg  # noqa: E402
from datasets import make_dataloader  # noqa: E402
from model import make_model  # noqa: E402

import tir_calib_probe as base  # noqa: E402

MOD_NAMES = {1: "RGB", 2: "NIR", 3: "TIR"}
TIR = 3
PAIRS = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3),
         (3, 1), (3, 2), (3, 3)]
TIR_CROSS = [(1, 3), (2, 3), (3, 1), (3, 2)]
SAME = [(1, 1), (2, 2), (3, 3)]


def _jsonable(x):
    return base._jsonable(x)


def _file_sha256(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write_csv(path, rows, fields):
    base._write_csv(path, rows, fields)


def _load_npz(path):
    return base._load_npz(path)


def _norm(x):
    return F.normalize(x.float(), dim=1)


def _parse_layers(s):
    vals = sorted(set(int(x) for x in str(s).split(",") if x.strip()))
    if not vals:
        raise ValueError("empty --layers")
    if 12 not in vals:
        vals.append(12)
        vals = sorted(set(vals))
    return vals


def _parse_floats(s):
    return [float(x) for x in str(s).split(",") if x.strip()]


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

def _register_layer_hooks(model, layers):
    caps = {}
    handles = []
    base_model = model.base
    for L in layers:
        if L == 12:
            continue
        if L < 1 or L > len(base_model.blocks):
            raise ValueError("bad layer {}".format(L))
        def make_hook(layer):
            def hook(module, inp, out):
                x = out[0] if isinstance(out, (tuple, list)) else out
                # The ViT applies self.norm after the last block.  Apply the
                # same LayerNorm to intermediate CLS tokens so all cached
                # layers are in the same normalized space as f12.
                caps[layer] = base_model.norm(x[:, 0]).detach()
            return hook
        handles.append(base_model.blocks[L - 1].register_forward_hook(make_hook(L)))
    return caps, handles


def _extract_layer_loader(model, loader, mode, device, layers):
    caps, handles = _register_layer_hooks(model, layers)
    feats = {L: [] for L in layers}
    pids, cams, mods, names = [], [], [], []
    model.eval()
    try:
        for batch in loader:
            img = batch[0].to(device, non_blocking=True)
            caps.clear()
            with torch.no_grad():
                out = model(img, mode=mode)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.ndim != 2 or out.shape[1] != 768:
                raise RuntimeError("unexpected f12 shape {}".format(tuple(out.shape)))
            for L in layers:
                if L == 12:
                    feats[L].append(out.detach().float().cpu())
                else:
                    if L not in caps or caps[L].ndim != 2 or caps[L].shape[1] != 768:
                        raise RuntimeError("unexpected layer {} shape".format(L))
                    feats[L].append(caps[L].float().cpu())
            pids.append(base._np(batch[1]).astype(np.int64))
            cams.append(base._np(batch[2]).astype(np.int64))
            mods.append(base._np(batch[4]).astype(np.int64))
            names.extend([str(x) for x in batch[5]])
    finally:
        for h in handles:
            h.remove()
    if not pids:
        empty = torch.empty(0, 768)
        return {L: empty for L in layers}, (np.empty(0, np.int64),
                np.empty(0, np.int64), np.empty(0, np.int64), [])
    return ({L: torch.cat(feats[L], 0) for L in layers},
            (np.concatenate(pids), np.concatenate(cams),
             np.concatenate(mods), names))


def stage_cache(args):
    if not args.config_file or not args.checkpoint:
        raise ValueError("--config_file and --checkpoint are required for cache")
    layers = _parse_layers(args.layers)
    os.makedirs(args.output_dir, exist_ok=True)
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
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    (train_loader, train_normal_loaders, val_loaders, num_querys,
     num_classes, camera_num, view_num) = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes,
                       camera_num=camera_num, view_num=view_num)
    nkeys = base._load_model_state(model, args.checkpoint)
    model.to(device).eval()
    print("loaded {} state keys; num_classes={} queries={}".format(
        nkeys, num_classes, num_querys))

    arrays = {}
    for m, loader in enumerate(train_normal_loaders, start=1):
        fdict, (p, c, md, names) = _extract_layer_loader(
            model, loader, m, device, layers)
        for L in layers:
            arrays["train_feat_{}_{}".format(L, m)] = base._np(fdict[L])
        arrays["train_pid_{}".format(m)] = p.astype(np.int32)
        arrays["train_cam_{}".format(m)] = c.astype(np.int32)
        arrays["train_mod_{}".format(m)] = md.astype(np.int32)
        arrays["train_name_{}".format(m)] = np.asarray(names)
        print("train m{}: {} x {} (layers {})".format(
            m, fdict[12].shape[0], fdict[12].shape[1], layers))

    for m, (loader, nq) in enumerate(zip(val_loaders, num_querys), start=1):
        fdict, (p, c, md, names) = _extract_layer_loader(
            model, loader, m, device, layers)
        nq = int(nq)
        for L in layers:
            arrays["q_feat_{}_{}".format(L, m)] = base._np(fdict[L][:nq])
            arrays["g_feat_{}_{}".format(L, m)] = base._np(fdict[L][nq:])
        arrays["q_pid_{}".format(m)] = p[:nq].astype(np.int32)
        arrays["g_pid_{}".format(m)] = p[nq:].astype(np.int32)
        arrays["q_cam_{}".format(m)] = c[:nq].astype(np.int32)
        arrays["g_cam_{}".format(m)] = c[nq:].astype(np.int32)
        arrays["q_name_{}".format(m)] = np.asarray(names[:nq])
        arrays["g_name_{}".format(m)] = np.asarray(names[nq:])
        print("val m{}: query {} gallery {}".format(m, nq, fdict[12].shape[0] - nq))

    arrays["classifier_weight"] = base._np(model.classifier.weight.detach().float())
    b = model.bottleneck
    arrays["bn_weight"] = base._np(b.weight.detach().float())
    arrays["bn_bias"] = base._np(b.bias.detach().float())
    arrays["bn_mean"] = base._np(b.running_mean.detach().float())
    arrays["bn_var"] = base._np(b.running_var.detach().float())
    arrays["bn_eps"] = np.asarray([float(b.eps)], dtype=np.float32)
    # Split train/test caches so the selection stage physically cannot read
    # test features.  Norm/classifier metadata is shared.
    train_arrays = {k: v for k, v in arrays.items()
                    if k.startswith("train_")}
    test_arrays = {k: v for k, v in arrays.items()
                   if k.startswith("q_") or k.startswith("g_")}
    shared = {k: arrays[k] for k in
              ("classifier_weight", "bn_weight", "bn_bias", "bn_mean",
               "bn_var", "bn_eps")}
    train_arrays.update(shared)
    test_arrays.update(shared)
    np.savez(os.path.join(args.output_dir, "features_train.npz"), **train_arrays)
    np.savez(os.path.join(args.output_dir, "features_test.npz"), **test_arrays)
    meta = dict(
        tag=args.tag, config_file=args.config_file,
        checkpoint=args.checkpoint, checkpoint_state_keys=nkeys,
        num_classes=num_classes, num_querys=[int(x) for x in num_querys],
        modalities=list(cfg.DATASETS.MODALITIES), layers=layers,
        train_counts=[int(arrays["train_feat_12_{}".format(m)].shape[0])
                      for m in range(1, len(cfg.DATASETS.MODALITIES) + 1)],
        query_counts=[int(arrays["q_feat_12_{}".format(m)].shape[0])
                      for m in range(1, len(cfg.DATASETS.MODALITIES) + 1)],
        gallery_counts=[int(arrays["g_feat_12_{}".format(m)].shape[0])
                        for m in range(1, len(cfg.DATASETS.MODALITIES) + 1)],
        checkpoint_sha256=_file_sha256(args.checkpoint),
        feature_note="raw pre-BN CLS per layer; base.norm applied to intermediate CLS; 256x128",
    )
    with open(os.path.join(args.output_dir, "cache_meta.json"), "w") as f:
        json.dump(_jsonable(meta), f, indent=2)
    print("cached layer features to {}".format(args.output_dir))


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------

def _layer_feat(cache, part, layer, mod):
    # part is one of "train", "q", "g"; do not use split[0] for "test".
    if part == "train":
        return np.asarray(cache["train_feat_{}_{}".format(layer, mod)])
    return np.asarray(cache["{}_feat_{}_{}".format(part, layer, mod)])


def _mix_feat(f12, f_layer, alpha):
    a = float(alpha)
    if a <= 0:
        return _norm(torch.as_tensor(f12))
    if a >= 1:
        return _norm(torch.as_tensor(f_layer))
    z12 = _norm(torch.as_tensor(f12))
    zl = _norm(torch.as_tensor(f_layer))
    return _norm((1.0 - a) * z12 + a * zl)


def _mean_feat(fs):
    zs = [_norm(torch.as_tensor(x)) for x in fs]
    return _norm(torch.stack(zs, 0).mean(0))


def _route_spec_feat(cache, part, mod, spec):
    kind = spec["kind"]
    if kind == "f12":
        return _norm(torch.as_tensor(_layer_feat(cache, part, 12, mod)))
    if kind == "layer":
        return _norm(torch.as_tensor(_layer_feat(cache, part, spec["layer"], mod)))
    if kind == "mix":
        return _mix_feat(_layer_feat(cache, part, 12, mod),
                         _layer_feat(cache, part, spec["layer"], mod),
                         spec["alpha"])
    if kind == "mean":
        return _mean_feat([_layer_feat(cache, part, L, mod)
                           for L in spec["layers"]])
    raise ValueError("bad spec {}".format(spec))


def _pair_specs(cfg_dict, q, g):
    """Return (q_spec, g_spec) for one query/gallery modality pair."""
    if q == g:
        return cfg_dict["same"], cfg_dict["same"]
    if cfg_dict["kind"] == "fixed_all":
        return cfg_dict["cross"], cfg_dict["cross"]
    if cfg_dict["kind"] == "role" and (q == TIR or g == TIR):
        if g == TIR:
            return cfg_dict["cross"], cfg_dict["tir_gallery"]
        return cfg_dict["tir_query"], cfg_dict["cross"]
    return cfg_dict["cross"], cfg_dict["cross"]


def _eval_config_pairs(cache, split, fold, cfg_dict, device=None):
    q_idx = {m: fold["q_idx"][str(m)] for m in (1, 2, 3)}
    g_idx = {m: fold["g_idx"][str(m)] for m in (1, 2, 3)}
    pids = {m: np.asarray(cache["train_pid_{}".format(m)]) for m in (1, 2, 3)}
    cams = {m: np.asarray(cache["train_cam_{}".format(m)]) for m in (1, 2, 3)}
    rows = {}
    for q, g in PAIRS:
        qs, gs = _pair_specs(cfg_dict, q, g)
        qf = _route_spec_feat(cache, "train", q, qs)[q_idx[q]]
        gf = _route_spec_feat(cache, "train", g, gs)[g_idx[g]]
        qp = torch.as_tensor(pids[q][q_idx[q]], dtype=torch.long)
        qc = torch.as_tensor(cams[q][q_idx[q]], dtype=torch.long)
        gp = torch.as_tensor(pids[g][g_idx[g]], dtype=torch.long)
        gc = torch.as_tensor(cams[g][g_idx[g]], dtype=torch.long)
        if device is not None:
            qf, gf = qf.to(device), gf.to(device)
            qp, qc, gp, gc = qp.to(device), qc.to(device), gp.to(device), gc.to(device)
        rr = base._eval_pair(qf, gf, qp, qc, gp, gc, q, g)
        rows[(q, g)] = rr
    return rows


def _eval_test_pairs(cache, cfg_dict, device=None):
    rows = {}
    for q, g in PAIRS:
        qs, gs = _pair_specs(cfg_dict, q, g)
        qf = _route_spec_feat(cache, "q", q, qs)
        gf = _route_spec_feat(cache, "g", g, gs)
        qp = torch.as_tensor(cache["q_pid_{}".format(q)], dtype=torch.long)
        qc = torch.as_tensor(cache["q_cam_{}".format(q)], dtype=torch.long)
        gp = torch.as_tensor(cache["g_pid_{}".format(g)], dtype=torch.long)
        gc = torch.as_tensor(cache["g_cam_{}".format(g)], dtype=torch.long)
        if device is not None:
            qf, gf = qf.to(device), gf.to(device)
            qp, qc, gp, gc = qp.to(device), qc.to(device), gp.to(device), gc.to(device)
        rows[(q, g)] = base._eval_pair(qf, gf, qp, qc, gp, gc, q, g)
    return rows


def _rows_scope(rows, scope):
    if scope == "tir_cross":
        return [r for (q, g), rs in rows.items() if (q == TIR) ^ (g == TIR) for r in rs]
    if scope == "tir_same":
        return rows[(TIR, TIR)]
    if scope == "same":
        return [r for (q, g), rs in rows.items() if q == g for r in rs]
    if scope == "all":
        return [r for rs in rows.values() for r in rs]
    raise ValueError(scope)


def _scope_metrics(rows, scope):
    return base._metrics(_rows_scope(rows, scope))


# ---------------------------------------------------------------------------
# candidate configs
# ---------------------------------------------------------------------------

def _make_configs(args):
    alphas = _parse_floats(args.alphas)
    inter = [L for L in _parse_layers(args.layers) if L != 12]
    mean_name = "mean_L" + "L".join(str(L) for L in inter + [12])
    cfgs = [
        dict(name="baseline", kind="baseline", same=dict(kind="f12"),
             cross=dict(kind="f12")),
        dict(name=mean_name, kind="mean", same=dict(kind="f12"),
             cross=dict(kind="mean", layers=[L for L in inter] + [12])),
    ]
    for L in inter:
        cfgs.append(dict(name="single_L{}".format(L), kind="single",
                         same=dict(kind="layer", layer=L),
                         cross=dict(kind="layer", layer=L)))
        for a in alphas:
            cfgs.append(dict(name="global_L{}_a{}".format(L, a),
                             kind="global", same=dict(kind="f12"),
                             cross=dict(kind="mix", layer=L, alpha=a)))
            cfgs.append(dict(name="fixed_all_L{}_a{}".format(L, a),
                             kind="fixed_all", same=dict(kind="mix", layer=L, alpha=a),
                             cross=dict(kind="mix", layer=L, alpha=a)))
    # TIR role-specific: RGB/NIR cross stays f12; same stays f12.
    # Compact pre-registered asymmetric candidates; no full 8x8 grid.
    role_pairs = [(0.0, 0.1), (0.1, 0.0), (0.25, 0.0),
                  (0.0, 0.25), (0.25, 0.25), (0.5, 0.5)]
    for L in inter:
        for ag, aq in role_pairs:
            cfgs.append(dict(
                name="role_L{}_ag{}_aq{}".format(L, ag, aq),
                kind="role", same=dict(kind="f12"),
                cross=dict(kind="f12"),
                tir_gallery=dict(kind="mix", layer=L, alpha=ag),
                tir_query=dict(kind="mix", layer=L, alpha=aq),
            ))
    return cfgs


def _cfg_to_specs(c):
    return {
        "name": c["name"], "kind": c["kind"],
        "same": c["same"], "cross": c["cross"],
        "tir_gallery": c.get("tir_gallery", c["cross"]),
        "tir_query": c.get("tir_query", c["cross"]),
    }


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------

def _fold_eval_screen(cache, fold, cfg_dict, device=None):
    rows = _eval_config_pairs(cache, "train", fold, cfg_dict, device)
    return {s: _scope_metrics(rows, s) for s in ("tir_cross", "tir_same", "same", "all")}


def _selection_from_screen(screen, baseline, cfgs, args):
    # screen: list of dict config/fold/scope metrics
    by_cfg = defaultdict(lambda: defaultdict(dict))
    for r in screen:
        by_cfg[r["config"]][r["fold"]][r["scope"]] = r
    base_by_fold = defaultdict(dict)
    for r in baseline:
        base_by_fold[r["fold"]][r["scope"]] = r
    cands = []
    for c in cfgs:
        name = c["name"]
        if c["kind"] in ("fixed_all", "single") or name == "baseline":
            continue
        vals = []
        guard = True
        deltas = []
        for fi in range(int(args.folds)):
            r = by_cfg.get(name, {}).get(fi, {})
            b = base_by_fold.get(fi, {})
            if not r or not b or "tir_cross" not in r or "tir_cross" not in b:
                guard = False
                break
            vals.append(r["tir_cross"]["mAP"])
            deltas.append(r["tir_cross"]["mAP"] - b["tir_cross"]["mAP"])
            if r["tir_same"]["mAP"] < b["tir_same"]["mAP"] - 0.005:
                guard = False
            if r["tir_same"]["R1"] < b["tir_same"]["R1"] - 0.010:
                guard = False
            if r["same"]["mAP"] < b["same"]["mAP"] - 0.005:
                guard = False
        if not guard or not vals:
            continue
        mean_map = float(np.mean(vals))
        mean_delta = float(np.mean(deltas))
        n_pos = int(sum(x > 0 for x in deltas))
        std = float(np.std(deltas))
        cands.append(dict(name=name, cfg=c, mean_tir_mAP=mean_map,
                          mean_delta=mean_delta, n_pos_folds=n_pos,
                          std_delta=std, deltas=deltas))
    if not cands:
        return dict(status="NO_TEST", reason="no candidate passed guards",
                    candidates=[])
    # Effect size and stability are pre-registered.
    eligible = [c for c in cands
                if c["mean_delta"] >= 0.0010 and c["n_pos_folds"] >= 4
                and c["std_delta"] <= 0.0015]
    eligible.sort(key=lambda x: (x["mean_tir_mAP"], x["mean_delta"]), reverse=True)
    if not eligible:
        return dict(status="NO_TEST", reason="no stable candidate >= +0.10pp",
                    candidates=sorted(cands, key=lambda x: x["mean_tir_mAP"], reverse=True))
    best = eligible[0]
    return dict(status="TEST_READY", selected=best,
                candidates=sorted(cands, key=lambda x: x["mean_tir_mAP"], reverse=True))


def stage_screen(args):
    cache = _load_npz(os.path.join(args.output_dir, "features_train.npz"))
    with open(args.split_json) as f:
        split = json.load(f)
    folds = split["folds"]
    device = torch.device(args.device)
    cfgs = _make_configs(args)
    screen = []
    baseline = []
    for fold in folds:
        fi = int(fold["fold"])
        b = _fold_eval_screen(cache, fold, _cfg_to_specs(
            dict(name="baseline", kind="baseline", same=dict(kind="f12"),
                 cross=dict(kind="f12"))), device)
        for scope, m in b.items():
            baseline.append(dict(config="baseline", fold=fi, scope=scope, **m))
        for c in cfgs:
            if c["name"] == "baseline":
                continue
            m = _fold_eval_screen(cache, fold, _cfg_to_specs(c), device)
            for scope, mm in m.items():
                screen.append(dict(config=c["name"], fold=fi, scope=scope, **mm))
        print("screen fold {} done ({} configs)".format(fi, len(cfgs)))
    _write_csv(os.path.join(args.output_dir, "screen_metrics.csv"),
               baseline + screen,
               ["config", "fold", "scope", "n", "valid", "mAP", "R1", "R5",
                "R10", "gt100", "gt500"])
    sel = _selection_from_screen(screen, baseline, cfgs, args)
    if sel.get("status") == "TEST_READY":
        selected_cfg = _cfg_to_specs(sel["selected"]["cfg"])
        baseline_cfg = _cfg_to_specs(dict(
            name="baseline", kind="baseline", same=dict(kind="f12"),
            cross=dict(kind="f12")))
        bmix, smix = [], []
        for fold in folds:
            bmix.append(base._metrics(_eval_mixed_routed(
                cache, "train", baseline_cfg, fold))["mAP"])
            smix.append(base._metrics(_eval_mixed_routed(
                cache, "train", selected_cfg, fold))["mAP"])
        sel["mixed_val"] = dict(
            baseline_frac=[float(x) for x in bmix],
            selected_frac=[float(x) for x in smix],
            mean_baseline=float(np.mean(bmix)),
            mean_selected=float(np.mean(smix)),
            mean_delta=float(np.mean(smix) - np.mean(bmix)),
        )
        if sel["mixed_val"]["mean_delta"] < -0.001:
            sel["status"] = "NO_TEST"
            sel["reason"] = "selected route mixed val drop > 0.10pp"
    with open(os.path.join(args.output_dir, "selection.json"), "w") as f:
        json.dump(_jsonable(sel), f, indent=2)
    print(json.dumps(_jsonable(sel), indent=2)[:4000])


# ---------------------------------------------------------------------------
# mixed-gallery routed evaluation
# ---------------------------------------------------------------------------

def _eval_mixed_routed(cache, split, cfg_dict, fold=None):
    """Pair-routed mixed gallery: same pair uses f12, cross uses routed spec."""
    if split == "train":
        if fold is None:
            raise ValueError("fold required for train mixed")
        qpids = {m: np.asarray(cache["train_pid_{}".format(m)]) for m in (1, 2, 3)}
        qcams = {m: np.asarray(cache["train_cam_{}".format(m)]) for m in (1, 2, 3)}
        gpids = qpids
        gcams = qcams
        q_idx = {m: fold["q_idx"][str(m)] for m in (1, 2, 3)}
        g_idx = {m: fold["g_idx"][str(m)] for m in (1, 2, 3)}
        qpart, gpart = "train", "train"
    else:
        qpids = {m: np.asarray(cache["q_pid_{}".format(m)]) for m in (1, 2, 3)}
        qcams = {m: np.asarray(cache["q_cam_{}".format(m)]) for m in (1, 2, 3)}
        gpids = {m: np.asarray(cache["g_pid_{}".format(m)]) for m in (1, 2, 3)}
        gcams = {m: np.asarray(cache["g_cam_{}".format(m)]) for m in (1, 2, 3)}
        q_idx = {m: np.arange(len(qpids[m])) for m in (1, 2, 3)}
        g_idx = {m: np.arange(len(gpids[m])) for m in (1, 2, 3)}
        qpart, gpart = "q", "g"
    rows = []
    for q in (1, 2, 3):
        qp = qpids[q][q_idx[q]]
        qc = qcams[q][q_idx[q]]
        qfeats = {}
        for g in (1, 2, 3):
            qs, _ = _pair_specs(cfg_dict, q, g)
            qfeats[g] = _route_spec_feat(cache, qpart, q, qs)[q_idx[q]]
        for g in (1, 2, 3):
            _, gs = _pair_specs(cfg_dict, q, g)
            gf = _route_spec_feat(cache, gpart, g, gs)[g_idx[g]]
            gp = gpids[g][g_idx[g]]
            gc = gcams[g][g_idx[g]]
            scores = (qfeats[g] @ gf.t()).cpu().numpy()
            if g == 1:
                all_scores = [scores]
                all_gp = [gp]
                all_gc = [gc]
            else:
                all_scores.append(scores)
                all_gp.append(gp)
                all_gc.append(gc)
        # Concatenate gallery modalities per query.
        for qi in range(len(qp)):
            ss, pp = [], []
            for si in range(3):
                mask = all_gc[si] != qc[qi]
                if not np.any(mask):
                    continue
                ss.append(all_scores[si][qi][mask])
                pp.append(all_gp[si][mask])
            if not ss:
                continue
            s = np.concatenate(ss)
            gp = np.concatenate(pp)
            pos = gp == qp[qi]
            if not np.any(pos):
                continue
            pos_best = s[pos].max()
            rank = 1 + int((s > pos_best).sum())
            order = np.argsort(-s)
            m = (gp[order] == qp[qi]).astype(np.float64)
            cum = np.cumsum(m)
            ranks = np.arange(1, len(m) + 1, dtype=np.float64)
            ap = float((cum / ranks * m).sum() / (m.sum() + 1e-12))
            rows.append(dict(q_index=int(qi), q_pid=int(qp[qi]),
                             q_cam=int(qc[qi]), q_mod=MOD_NAMES[q],
                             g_mod="MIX", rank=int(rank), ap=ap, valid=1))
    return rows


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------

def stage_test(args):
    out_test = os.path.join(args.output_dir, "test_metrics.csv")
    if os.path.exists(out_test) and not args.force:
        raise RuntimeError("test already exists; refusing a second test")
    sel_path = os.path.join(args.output_dir, "selection.json")
    if not os.path.exists(sel_path):
        raise RuntimeError("run --stage screen first")
    with open(sel_path) as f:
        sel = json.load(f)
    if sel.get("status") != "TEST_READY":
        raise RuntimeError("selection status is {}; refusing test".format(sel.get("status")))
    cache = _load_npz(os.path.join(args.output_dir, "features_test.npz"))
    device = torch.device(args.device)
    selected_cfg = sel["selected"]["cfg"]
    selected_cfg = _cfg_to_specs(selected_cfg)
    baseline_cfg = _cfg_to_specs(dict(name="baseline", kind="baseline",
                                     same=dict(kind="f12"), cross=dict(kind="f12")))
    rows, perq = [], []
    for arm, cfg_dict in (("baseline", baseline_cfg),
                          ("selected", selected_cfg)):
        rows_pair = _eval_test_pairs(cache, cfg_dict, device)
        for scope in ("tir_cross", "tir_same", "same", "all"):
            m = _scope_metrics(rows_pair, scope)
            rows.append(dict(arm=arm, scope=scope, **m))
            perq.extend(base._rows_for_csv(
                _rows_scope(rows_pair, scope), "test", -1, 0, 0, arm, scope))
        mixed = _eval_mixed_routed(cache, "test", cfg_dict)
        mm = base._metrics(mixed)
        rows.append(dict(arm=arm, scope="mixed", **mm))
        perq.extend(base._rows_for_csv(mixed, "test", -1, 0, 0, arm, "mixed"))
        print("test {}: TIR cross mAP {:.4f}, mixed mAP {:.4f}".format(
            arm, m["mAP"], mm["mAP"]))
    _write_csv(out_test, rows,
               ["arm", "scope", "n", "valid", "mAP", "R1", "R5", "R10",
                "gt100", "gt500"])
    _write_csv(os.path.join(args.output_dir, "test_per_query.csv"), perq,
               ["arm", "seed", "state", "split", "fold", "scope", "q_index",
                "q_pid", "q_cam", "q_mod", "g_mod", "rank", "ap", "valid"])

    def get(arm, scope, key):
        for r in rows:
            if r["arm"] == arm and r["scope"] == scope:
                return float(r[key])
        return float("nan")

    b = {k: get("baseline", "tir_cross", k) for k in ("mAP", "R1", "R5", "R10", "gt100", "gt500")}
    if args.tag == "v2":
        expected = dict(mAP=0.05593003215632495, R1=0.07482435597189696,
                        R5=0.17330210772833723, R10=0.23676814988290398,
                        gt100=0.4195550351288056, gt500=0.15585480093676815)
        for k, v in expected.items():
            if abs(b[k] - v) > 1e-4:
                raise RuntimeError(
                    "f12 baseline mismatch {}: got {:.8f} expected {:.8f}; "
                    "check cache/config/checkpoint".format(k, b[k], v))
    p = {k: get("selected", "tir_cross", k) for k in ("mAP", "R1", "R5", "R10", "gt100", "gt500")}
    d = {k: p[k] - b[k] for k in b}
    bs = {k: get("baseline", "tir_same", k) for k in ("mAP", "R1", "R5", "R10")}
    ps = {k: get("selected", "tir_same", k) for k in bs}
    ds = {k: ps[k] - bs[k] for k in bs}
    # Same pairs must be byte-identical to f12; a non-zero delta is a routing
    # or evaluator bug, not a method effect.
    if max(abs(v) for v in ds.values()) > 1e-6:
        raise RuntimeError("TIR-same-pair f12 mismatch: {}".format(ds))
    ba = {k: get("baseline", "same", k) for k in ("mAP", "R1", "R5", "R10")}
    pa = {k: get("selected", "same", k) for k in ba}
    da = {k: pa[k] - ba[k] for k in ba}
    if max(abs(v) for v in da.values()) > 1e-6:
        raise RuntimeError("same-3-pair f12 mismatch: {}".format(da))
    bm = get("baseline", "mixed", "mAP")
    if args.tag == "v2":
        expected_same = dict(mAP=0.14545523695593462, R1=0.21124121779859484,
                             R5=0.3695550351288056, R10=0.468384074941452)
        for k, v in expected_same.items():
            if abs(bs[k] - v) > 1e-4:
                raise RuntimeError(
                    "f12 TIR-same mismatch {}: got {:.8f} expected {:.8f}"
                    .format(k, bs[k], v))
        if abs(bm - 0.11465081710524115) > 1e-4:
            raise RuntimeError(
                "f12 mixed mismatch: got {:.8f} expected {:.8f}"
                .format(bm, 0.11465081710524115))
    pm = get("selected", "mixed", "mAP")
    dm = pm - bm
    b_rows = [r for r in perq if r["arm"] == "baseline" and r["scope"] == "tir_cross"]
    a_rows = [r for r in perq if r["arm"] == "selected" and r["scope"] == "tir_cross"]
    boot = base._bootstrap_delta(b_rows, a_rows, n=args.bootstrap, seed=args.seed)
    tail_ok = (d["R5"] >= 0.005 or d["R10"] >= 0.008
               or d["gt100"] <= -0.01 or d["gt500"] <= -0.005)
    same_ok = ds["mAP"] >= -0.005 and ds["R1"] >= -0.010
    mixed_ok = dm >= -0.001
    go = (d["mAP"] >= 0.0015 and boot["ci_low"] >= -0.0005
          and same_ok and mixed_ok and tail_ok)
    status = "GO" if go else ("INCONCLUSIVE" if boot["n_ids"] < 50 else "NO_GO")
    decision = dict(
        status=status, selected=sel["selected"],
        selected_cfg=selected_cfg,
        baseline_cross=b, selected_cross=p, delta=d,
        baseline_same=bs, selected_same=ps, delta_same=ds,
        baseline_same3=ba, selected_same3=pa, delta_same3=da,
        delta_mixed_mAP=dm, bootstrap=boot,
        tail_ok=bool(tail_ok), same_ok=bool(same_ok), mixed_ok=bool(mixed_ok),
        note="training-free route; GO requires +0.15pp TIR-cross mAP, "
             "CI lower >= -0.05pp, TIR-same hard guard, mixed mAP >= -0.10pp, "
             "and at least one tail signal; test evaluated once.",
    )
    with open(os.path.join(args.output_dir, "decision.json"), "w") as f:
        json.dump(_jsonable(decision), f, indent=2)
    print(json.dumps(_jsonable(decision), indent=2)[:4000])


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def _synthetic_layers(n_ids=30, dim=16, per_id=12):
    rng = np.random.RandomState(7)
    centers = rng.randn(n_ids, dim).astype(np.float32)
    cache = {}
    for m in (1, 2, 3):
        rows, pids, cams = [], [], []
        for pid in range(n_ids):
            for j in range(per_id):
                rows.append(centers[pid] + 0.3 * rng.randn(dim))
                pids.append(pid)
                cams.append(j % 3)
        f = np.stack(rows).astype(np.float32)
        for L in (6, 9, 12):
            cache["train_feat_{}_{}".format(L, m)] = f + rng.randn(*f.shape).astype(np.float32) * 0.05
            cache["q_feat_{}_{}".format(L, m)] = cache["train_feat_{}_{}".format(L, m)][:20]
            cache["g_feat_{}_{}".format(L, m)] = cache["train_feat_{}_{}".format(L, m)][:20]
        cache["train_pid_{}".format(m)] = np.asarray(pids, np.int32)
        cache["train_cam_{}".format(m)] = np.asarray(cams, np.int32)
        cache["q_pid_{}".format(m)] = np.asarray(pids[:20], np.int32)
        cache["g_pid_{}".format(m)] = np.asarray(pids[:20], np.int32)
        cache["q_cam_{}".format(m)] = np.asarray(cams[:20], np.int32)
        cache["g_cam_{}".format(m)] = np.asarray(cams[:20], np.int32)
    return cache


def run_selftest(args):
    cache = _synthetic_layers()
    # Build a tiny fold: first 24 IDs, 4 q per modality and the rest gallery.
    fold = dict(fold=0, train_ids=list(range(24)), val_ids=list(range(24)),
                q_idx={}, g_idx={})
    for m in (1, 2, 3):
        pids = cache["train_pid_{}".format(m)]
        q = []
        for pid in range(24):
            q.extend(np.where(pids == pid)[0][:4].tolist())
        q = sorted(set(q))
        fold["q_idx"][str(m)] = q
        fold["g_idx"][str(m)] = [i for i in range(len(pids)) if i not in set(q)]
    cfg_b = _cfg_to_specs(dict(name="baseline", kind="baseline",
                               same=dict(kind="f12"), cross=dict(kind="f12")))
    cfg_m = _cfg_to_specs(dict(name="mix", kind="global", same=dict(kind="f12"),
                               cross=dict(kind="mix", layer=9, alpha=0.5)))
    r1 = _eval_config_pairs(cache, "train", fold, cfg_b)
    r2 = _eval_config_pairs(cache, "train", fold, cfg_m)
    m1 = base._metrics(_rows_scope(r1, "tir_cross"))
    m2 = base._metrics(_rows_scope(r2, "tir_cross"))
    assert m1["n"] > 0 and m2["n"] > 0
    assert np.isfinite(m1["mAP"]) and np.isfinite(m2["mAP"])
    rq1 = _rows_scope(r1, "tir_cross")
    rq2 = _rows_scope(r2, "tir_cross")
    boot = base._bootstrap_delta(rq1, rq2, n=50, seed=1)
    assert "ci_low" in boot
    # Mixed routed must produce rows and finite metrics.
    mixed = _eval_mixed_routed(cache, "train", cfg_b, fold)
    assert mixed
    mm = base._metrics(mixed)
    assert np.isfinite(mm["mAP"])
    print("SELFTEST OK")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="V5 training-free layer-route probe")
    p.add_argument("--stage", required=True,
                   choices=["cache", "screen", "test", "selftest"])
    p.add_argument("--config_file", default="")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--tag", default="v2")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split_json", default="")
    p.add_argument("--dataset_root", default="/project/code/data")
    p.add_argument("--dataset_dir", default="WHU-MARS")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--layers", default="6,9,12")
    p.add_argument("--alphas", default="0,0.01,0.1,0.25,0.5,1.0")
    p.add_argument("--role_alphas", default="0,0.1,0.25,0.5")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--force", action="store_true")
    p.add_argument("opts", nargs=argparse.REMAINDER)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.stage == "selftest":
        run_selftest(args)
    elif args.stage == "cache":
        stage_cache(args)
    elif args.stage == "screen":
        stage_screen(args)
    elif args.stage == "test":
        stage_test(args)


if __name__ == "__main__":
    main()
