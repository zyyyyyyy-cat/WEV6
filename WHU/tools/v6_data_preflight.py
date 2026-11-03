"""V6 dataset adaptation preflight for 1000 / 2337 / 1000-GD.

Read-only: loads the dataset with the same WHU_MARS class used by training and
reports the assumptions that V6's cross-modal loss depends on:

  - train/query/gallery IDs and images per modality;
  - train IDs missing any modality;
  - cameras per ID/modality and zero-image groups;
  - AG camera counts;
  - simulated PKM batch cross-camera positive coverage per modality pair.

Run on the server (data is not local):
  python tools/v6_data_preflight.py \
    --root /project/code/data --dataset_dir WHU-MARS \
    --modalities RGB,IR,Thermal --out logs/v6_preflight_1000.json

  python tools/v6_data_preflight.py \
    --root /project/code/data --dataset_dir WHU-MARS_2337 \
    --modalities RGB,IR,Thermal --out logs/v6_preflight_2337.json

  python tools/v6_data_preflight.py \
    --root /project/code/data --dataset_dir WHU-MARS \
    --modalities RGB,IR,Thermal --filter_gd --out logs/v6_preflight_gd.json

This tool does not train, evaluate, or write checkpoints.
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.whu_mars import WHU_MARS  # noqa: E402


RAW_PID = re.compile(r"(\d+)_c(\d+)")


def _split_stats(dataset, modalities):
    out = {}
    for split_name in ("train", "query", "gallery"):
        split = getattr(dataset, split_name)
        split_out = {}
        all_pids = set()
        raw_pids = set()
        for m in modalities:
            rows = split.get(m, [])
            pids = set()
            cams = set()
            for path, pid, cam, _ in rows:
                pids.add(pid)
                cams.add(cam)
                all_pids.add(pid)
                hit = RAW_PID.search(os.path.basename(path))
                if hit:
                    raw_pids.add(int(hit.group(1)))
            split_out[m] = {
                "images": len(rows),
                "pids": len(pids),
                "cameras": sorted(int(c) for c in cams),
            }
        split_out["all_pids"] = len(all_pids)
        split_out["raw_pids"] = len(raw_pids)
        out[split_name] = split_out
    return out


def _train_structure(dataset, modalities):
    by = defaultdict(lambda: defaultdict(list))
    for m in modalities:
        for path, pid, cam, _ in dataset.train.get(m, []):
            by[pid][m].append((path, cam))

    pids = sorted(by)
    missing = []
    zero_groups = []
    cameras_per_id = {}
    images_per_id_mod = {}
    for pid in pids:
        present = [m for m in modalities if len(by[pid][m]) > 0]
        if len(present) != len(modalities):
            missing.append({
                "pid": int(pid),
                "missing": [m for m in modalities if m not in present],
            })
        cameras_per_id[int(pid)] = {
            m: sorted({int(cam) for _, cam in by[pid][m]})
            for m in modalities
        }
        images_per_id_mod[int(pid)] = {
            m: len(by[pid][m]) for m in modalities
        }
        for m in modalities:
            if len(by[pid][m]) == 0:
                zero_groups.append({"pid": int(pid), "modality": m})

    cam_set = set()
    for pid in pids:
        for m in modalities:
            cam_set.update(cameras_per_id[int(pid)][m])

    # Per-ID camera-count summary.
    cam_counts = []
    for pid in pids:
        if all(len(cameras_per_id[int(pid)][m]) > 0 for m in modalities):
            cam_counts.append(max(len(cameras_per_id[int(pid)][m])
                                  for m in modalities))
    return {
        "train_pids": len(pids),
        "missing_modality_pids": len(missing),
        "missing_examples": missing[:30],
        "zero_image_groups": len(zero_groups),
        "zero_examples": zero_groups[:30],
        "camera_ids": sorted(int(c) for c in cam_set),
        "cameras_per_id_max_mean": (
            float(np.mean(cam_counts)) if cam_counts else None),
        "cameras_per_id_max_min": (
            int(np.min(cam_counts)) if cam_counts else None),
        "cameras_per_id_max_median": (
            float(np.median(cam_counts)) if cam_counts else None),
    }


def _simulate_coverage(dataset, modalities, num_instance, batch_size,
                       n_batches, seed, world_size=1):
    """Approximate the current PKM sampler for one rank.

    The sampler groups each pid/modality's images into K-sized groups and
    pops one group per modality per pid.  This simulation reports the
    cross-camera coverage V6's `L_corr` would see.
    """
    K = int(num_instance)
    mini_batch = int(batch_size) // int(world_size)
    pids_per_batch = mini_batch // K
    if pids_per_batch <= 0:
        raise ValueError("batch_size/world_size too small for K")

    index_dic = defaultdict(lambda: defaultdict(list))
    for m in modalities:
        for path, pid, cam, _ in dataset.train.get(m, []):
            index_dic[pid][m].append(int(cam))

    pids = sorted(index_dic)
    valid_pids = [
        p for p in pids
        if all(len(index_dic[p][m]) > 0 for m in modalities)
    ]
    if len(valid_pids) < pids_per_batch:
        raise ValueError("not enough complete-modality train IDs")

    rng = random.Random(seed)
    per_pair = {
        "{}--{}".format(a, b): {
            "anchors": 0, "valid": 0, "positives": 0,
            "tuple_anchors": 0, "tuple_cross": 0,
        }
        for i, a in enumerate(modalities) for b in modalities[i + 1:]
    }
    all_cross = {"anchors": 0, "valid": 0, "positives": 0}

    for _ in range(int(n_batches)):
        chosen = rng.sample(valid_pids, pids_per_batch)
        by_pid = {}
        for pid in chosen:
            by_pid[pid] = {
                m: [rng.choice(index_dic[pid][m]) for _ in range(K)]
                for m in modalities
            }

        for pid, cams_by in by_pid.items():
            for i, a in enumerate(modalities):
                for b in modalities[i + 1:]:
                    key = "{}--{}".format(a, b)
                    ca, cb = cams_by[a], cams_by[b]
                    # 2K anchors for this pair: every row of a and every
                    # row of b.  Each anchor sees the K rows of the other
                    # modality; positives are the cross-camera rows.
                    for x in ca:
                        pos = sum(1 for y in cb if y != x)
                        per_pair[key]["anchors"] += 1
                        if pos:
                            per_pair[key]["valid"] += 1
                            per_pair[key]["positives"] += pos
                    for y in cb:
                        pos = sum(1 for x in ca if x != y)
                        per_pair[key]["anchors"] += 1
                        if pos:
                            per_pair[key]["valid"] += 1
                            per_pair[key]["positives"] += pos
                    per_pair[key]["tuple_anchors"] += K
                    per_pair[key]["tuple_cross"] += sum(
                        1 for x, y in zip(ca, cb) if x != y)

            # All-cross-modal anchor: one row sees all K rows of the other
            # two modalities (2K slots).
            for a in modalities:
                others = [m for m in modalities if m != a]
                for x in cams_by[a]:
                    pos = sum(1 for b in others for y in cams_by[b] if y != x)
                    all_cross["anchors"] += 1
                    if pos:
                        all_cross["valid"] += 1
                        all_cross["positives"] += pos

    for v in per_pair.values():
        v["anchor_valid_ratio"] = (
            v["valid"] / v["anchors"]) if v["anchors"] else 0.0
        v["valid_ratio"] = v["anchor_valid_ratio"]  # backward-compatible alias
        v["positives_per_anchor"] = (
            v["positives"] / v["anchors"]) if v["anchors"] else 0.0
        v["tuple_cross_ratio"] = (
            v["tuple_cross"] / v["tuple_anchors"]
            if v["tuple_anchors"] else 0.0)
    all_cross["anchor_valid_ratio"] = (
        all_cross["valid"] / all_cross["anchors"]
        if all_cross["anchors"] else 0.0)
    all_cross["valid_ratio"] = all_cross["anchor_valid_ratio"]
    all_cross["positives_per_anchor"] = (
        all_cross["positives"] / all_cross["anchors"]
        if all_cross["anchors"] else 0.0)
    return {
        "n_batches": int(n_batches),
        "pids_per_batch": int(pids_per_batch),
        "K": K,
        "coverage_level": "anchor-level; valid_ratio aliases anchor_valid_ratio",
        "per_pair": per_pair,
        "all_cross_modal": all_cross,
    }


def _ag_counts(dataset):
    out = {}
    for split in ("query", "gallery"):
        rows = []
        for m, items in getattr(dataset, split).items():
            rows.extend(items)
        cams = [int(r[2]) for r in rows]
        out[split] = {
            "total": len(cams),
            "ground_cam_lt5": int(sum(c < 5 for c in cams)),
            "aerial_cam_ge5": int(sum(c >= 5 for c in cams)),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--modalities", default="RGB,IR,Thermal")
    parser.add_argument("--filter_gd", action="store_true")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    dataset = WHU_MARS(
        root=args.root,
        dataset_dir=args.dataset_dir,
        modalities=modalities,
        filter_gd=args.filter_gd,
        verbose=False,
    )

    result = {
        "root": args.root,
        "dataset_dir": args.dataset_dir,
        "modalities": modalities,
        "filter_gd": bool(args.filter_gd),
        "splits": _split_stats(dataset, modalities),
        "train_structure": _train_structure(dataset, modalities),
        "ag_camera_counts": _ag_counts(dataset),
    }
    try:
        result["pkm_coverage"] = _simulate_coverage(
            dataset, modalities, args.num_instance, args.batch_size,
            args.num_batches, args.seed, world_size=args.world_size)
    except Exception as exc:
        result["pkm_coverage_error"] = repr(exc)

    # Train/query/gallery raw ID overlap (filenames carry raw pids).
    raw = {}
    for split in ("train", "query", "gallery"):
        ids = set()
        for rows in getattr(dataset, split).values():
            for path, _, _, _ in rows:
                hit = RAW_PID.search(os.path.basename(path))
                if hit:
                    ids.add(int(hit.group(1)))
        raw[split] = ids
    result["raw_id_overlap"] = {
        "train_query": len(raw["train"] & raw["query"]),
        "train_gallery": len(raw["train"] & raw["gallery"]),
        "query_gallery": len(raw["query"] & raw["gallery"]),
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
