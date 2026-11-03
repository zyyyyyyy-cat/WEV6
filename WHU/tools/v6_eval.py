"""V6 official-test evaluation with per-query AP dump and bootstrap CI.

This tool is intentionally separate from test.py: test.py does not persist
per-query APs, while the V6 gate needs an identity-cluster bootstrap on the
TIR-cross mAP/R1 delta.  Run this tool once on the final checkpoint; it is the
official test for V6 (do not also run test.py on the same checkpoint).

No training logic is touched.  The model is loaded with model.load_param(), so
the classifier/GAC heads remain excluded exactly as in test.py.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import cfg  # noqa: E402
from datasets import make_dataloader  # noqa: E402
from model import make_model  # noqa: E402


MODALITY_NAMES = {1: "RGB", 2: "NIR", 3: "TIR"}


def _as_list(x):
    if torch.is_tensor(x):
        return x.tolist()
    return list(x)


def _collect_features(model, val_loaders, num_querys, device):
    feats, pids, cams, mods, paths, is_query = [], [], [], [], [], []
    for mode, (loader, num_q) in enumerate(zip(val_loaders, num_querys), start=1):
        local_feats = []
        local_pids, local_cams, local_paths = [], [], []
        for img, vid, camid, _, _, imgpath in loader:
            with torch.no_grad():
                feat = model(img.to(device), mode=mode)
            local_feats.append(feat.detach().cpu())
            local_pids.extend(_as_list(vid))
            local_cams.extend(_as_list(camid))
            local_paths.extend(list(imgpath))
        f = torch.cat(local_feats, dim=0).float()
        n = f.shape[0]
        if n != len(local_pids) or n != len(local_cams) or n != len(local_paths):
            raise RuntimeError(
                "modality {} feature/meta length mismatch: {} vs {}/{}/{}".format(
                    mode, n, len(local_pids), len(local_cams), len(local_paths)))
        if num_q < 0 or num_q > n:
            raise RuntimeError(
                "modality {} num_query {} out of range {}".format(mode, num_q, n))
        feats.append(f)
        pids.extend(local_pids)
        cams.extend(local_cams)
        mods.extend([mode] * n)
        paths.extend(local_paths)
        is_query.extend([True] * num_q + [False] * (n - num_q))

    return (
        torch.cat(feats, dim=0),
        np.asarray(pids, dtype=np.int64),
        np.asarray(cams, dtype=np.int64),
        np.asarray(mods, dtype=np.int64),
        np.asarray(paths),
        np.asarray(is_query, dtype=bool),
    )


def _scope_ap_matrix(qf, gf, q_pids, q_cams, g_pids, g_cams, q_chunk=512):
    """Per-query AP/rank/n_rel for one query-gallery scope.

    qf/gf are L2-normalized.  Same-camera gallery rows are removed exactly as
    in utils/metrics.py (sysu protocol).
    """
    num_q = qf.shape[0]
    aps, ranks, n_rels = [], [], []
    gf_t = gf.t().contiguous()
    for start in range(0, num_q, q_chunk):
        end = min(start + q_chunk, num_q)
        scores = (qf[start:end] @ gf_t).cpu().numpy()
        for i in range(end - start):
            qi = start + i
            q_pid = int(q_pids[qi])
            q_cam = int(q_cams[qi])
            keep = g_cams != q_cam
            gp = g_pids[keep]
            ss = scores[i][keep]
            num_rel = int(np.sum(gp == q_pid))
            if num_rel == 0:
                aps.append(np.nan)
                ranks.append(np.nan)
                n_rels.append(0)
                continue
            order = np.argsort(-ss, kind="stable")
            matches = (gp[order] == q_pid).astype(np.float32)
            tmp = np.cumsum(matches) / np.arange(1, matches.size + 1)
            aps.append(float(np.sum(tmp * matches) / num_rel))
            hit = np.flatnonzero(matches)
            ranks.append(float(hit[0] + 1) if hit.size else float(matches.size + 1))
            n_rels.append(num_rel)
    return np.asarray(aps), np.asarray(ranks), np.asarray(n_rels)


def _summarize(aps, ranks):
    valid = np.isfinite(aps) & np.isfinite(ranks)
    aps, ranks = aps[valid], ranks[valid]
    if aps.size == 0:
        return {"n": 0}
    return {
        "n": int(aps.size),
        "mAP": float(aps.mean()),
        "R1": float(np.mean(ranks <= 1)),
        "R5": float(np.mean(ranks <= 5)),
        "R10": float(np.mean(ranks <= 10)),
        "gt100": float(np.mean(ranks > 100)),
        "gt500": float(np.mean(ranks > 500)),
    }


def _bootstrap_ci(aps, pids, n_boot=10000, seed=1234):
    valid = np.isfinite(aps)
    aps, pids = aps[valid], pids[valid]
    if aps.size == 0:
        return [float("nan"), float("nan")]
    uniq = np.unique(pids)
    idx_by_pid = {p: np.where(pids == p)[0] for p in uniq}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([idx_by_pid[p] for p in sample])
        means.append(float(aps[idx].mean()))
    return [float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5))]


def _classmean_cos(qf, gf, q_pids, g_pids):
    """Q->G class-mean cosine; diagnostic only (class-level alignment)."""
    common = np.intersect1d(np.unique(q_pids), np.unique(g_pids))
    if common.size == 0:
        return float("nan")
    q_means, g_means = [], []
    for pid in common:
        q_means.append(qf[q_pids == pid].mean(dim=0))
        g_means.append(gf[g_pids == pid].mean(dim=0))
    qm = F.normalize(torch.stack(q_means, dim=0), dim=1)
    gm = F.normalize(torch.stack(g_means, dim=0), dim=1)
    return float((qm * gm).sum(dim=1).mean().item())


def _eval_scope(name, qf, gf, q_pids, q_cams, g_pids, g_cams,
                q_mods, g_mods, q_paths, n_boot):
    aps, ranks, n_rels = _scope_ap_matrix(
        qf, gf, q_pids, q_cams, g_pids, g_cams)
    summary = _summarize(aps, ranks)
    if summary.get("n", 0):
        summary["classmean_cos"] = _classmean_cos(
            qf, gf, q_pids, g_pids)
    if summary.get("n", 0) and n_boot > 0:
        summary["mAP_ci95"] = _bootstrap_ci(aps, q_pids, n_boot=n_boot)
    rows = []
    for i in range(len(aps)):
        if not np.isfinite(aps[i]):
            continue
        rows.append({
            "scope": name,
            "q_mod": int(q_mods[i]),
            "g_mod": int(g_mods[i]),
            "q_pid": int(q_pids[i]),
            "q_cam": int(q_cams[i]),
            "path": q_paths[i],
            "ap": float(aps[i]),
            "rank": float(ranks[i]),
            "n_rel": int(n_rels[i]),
        })
    return summary, rows


def _aggregate_rows(scope, rows, n_boot):
    if not rows:
        return None
    aps = np.asarray([r["ap"] for r in rows])
    ranks = np.asarray([r["rank"] for r in rows])
    pids = np.asarray([r["q_pid"] for r in rows])
    summary = _summarize(aps, ranks)
    if summary.get("n", 0) and n_boot > 0:
        summary["mAP_ci95"] = _bootstrap_ci(aps, pids, n_boot=n_boot)
    return summary


def main():
    parser = argparse.ArgumentParser(description="V6 official-test eval")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    _, _, val_loaders, num_querys, num_classes, camera_num, view_num = \
        make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes,
                       camera_num=camera_num, view_num=view_num)
    model.load_param(args.weight)
    model.to(device).eval()

    feats, pids, cams, mods, paths, is_query = _collect_features(
        model, val_loaders, num_querys, device)
    feats = F.normalize(feats, dim=1, p=2)

    out_dir = args.output or (cfg.OUTPUT_DIR or ".")
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []
    summaries = {}

    for q_mod in (1, 2, 3):
        for g_mod in (1, 2, 3):
            q_idx = np.where(is_query & (mods == q_mod))[0]
            g_idx = np.where((~is_query) & (mods == g_mod))[0]
            if q_idx.size == 0 or g_idx.size == 0:
                continue
            name = "{}->{}".format(
                MODALITY_NAMES[q_mod], MODALITY_NAMES[g_mod])
            summary, rows = _eval_scope(
                name, feats[q_idx], feats[g_idx],
                pids[q_idx], cams[q_idx], pids[g_idx], cams[g_idx],
                mods[q_idx], np.full(q_idx.size, g_mod, dtype=np.int64),
                paths[q_idx], 0)
            summaries[name] = summary
            all_rows.extend(rows)

    tir_names = ("RGB->TIR", "NIR->TIR", "TIR->RGB", "TIR->NIR")
    tir_rows = [r for r in all_rows if r["scope"] in tir_names]
    tir_summary = _aggregate_rows("tir_cross", tir_rows, args.bootstrap)
    if tir_summary is not None:
        summaries["tir_cross"] = tir_summary

    same_names = ("RGB->RGB", "NIR->NIR", "TIR->TIR")
    same_rows = [r for r in all_rows if r["scope"] in same_names]
    same_summary = _aggregate_rows("same3", same_rows, args.bootstrap)
    if same_summary is not None:
        summaries["same3"] = same_summary

    cross_rows = [r for r in all_rows if r["q_mod"] != r["g_mod"]]
    cross_summary = _aggregate_rows("cross6", cross_rows, args.bootstrap)
    if cross_summary is not None:
        summaries["cross6"] = cross_summary

    nine_names = [f"{a}->{b}" for a in ("RGB", "NIR", "TIR")
                  for b in ("RGB", "NIR", "TIR")]
    nine_rows = [r for r in all_rows if r["scope"] in nine_names]
    nine_summary = _aggregate_rows("nine_pair", nine_rows, args.bootstrap)
    if nine_summary is not None:
        summaries["nine_pair"] = nine_summary

    # Macro means (pair-equal) are the protocol means when query counts differ.
    def _macro(names):
        vals = [summaries[n] for n in names
                if n in summaries and summaries[n].get("n", 0)]
        if not vals:
            return None
        return {
            "n_pairs": len(vals),
            "mAP": float(np.mean([v["mAP"] for v in vals])),
            "R1": float(np.mean([v["R1"] for v in vals])),
            "R5": float(np.mean([v["R5"] for v in vals])),
            "R10": float(np.mean([v["R10"] for v in vals])),
            "classmean_cos": float(np.nanmean(
                [v.get("classmean_cos", np.nan) for v in vals])),
        }

    tir_macro = _macro(tir_names)
    if tir_macro is not None:
        summaries["tir_cross_macro"] = tir_macro
        if "tir_cross" in summaries:
            summaries["tir_cross"]["classmean_cos"] = tir_macro.get(
                "classmean_cos", float("nan"))
    same_macro = _macro(same_names)
    if same_macro is not None:
        summaries["same3_macro"] = same_macro
        if "same3" in summaries:
            summaries["same3"]["classmean_cos"] = same_macro.get(
                "classmean_cos", float("nan"))
    nine_macro = _macro(tuple(nine_names))
    if nine_macro is not None:
        summaries["nine_pair_macro"] = nine_macro
        if "nine_pair" in summaries:
            summaries["nine_pair"]["classmean_cos"] = nine_macro.get(
                "classmean_cos", float("nan"))

    q_idx = np.where(is_query)[0]
    g_idx = np.where(~is_query)[0]
    if q_idx.size and g_idx.size:
        summary, rows = _eval_scope(
            "mixed", feats[q_idx], feats[g_idx],
            pids[q_idx], cams[q_idx], pids[g_idx], cams[g_idx],
            mods[q_idx], np.full(q_idx.size, -1, dtype=np.int64),
            paths[q_idx], args.bootstrap)
        summaries["mixed"] = summary
        all_rows.extend(rows)

    # AG-ReID Table 4 protocol: all modalities; cams 0-4 ground, 5-6 aerial.
    # Same-camera removal inside _eval_scope matches utils/metrics.py.
    viewpoint_specs = (
        ("A->G", is_query & (cams >= 5), (~is_query) & (cams < 5)),
        ("G->A", is_query & (cams < 5), (~is_query) & (cams >= 5)),
    )
    for name, q_mask, g_mask in viewpoint_specs:
        q_idx = np.where(q_mask)[0]
        g_idx = np.where(g_mask)[0]
        if q_idx.size == 0 or g_idx.size == 0:
            continue
        summary, rows = _eval_scope(
            name, feats[q_idx], feats[g_idx],
            pids[q_idx], cams[q_idx], pids[g_idx], cams[g_idx],
            mods[q_idx], np.full(q_idx.size, -1, dtype=np.int64),
            paths[q_idx], args.bootstrap)
        summaries[name] = summary
        all_rows.extend(rows)

    csv_path = os.path.join(out_dir, "v6_per_query.csv")
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    with open(os.path.join(out_dir, "v6_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)

    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
