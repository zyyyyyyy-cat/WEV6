"""Paired identity-cluster bootstrap for V6 vs V4 per-query CSVs.

This is an offline analysis tool.  It does NOT run the model or touch the test
set; it consumes the `v6_per_query.csv` files produced by `tools/v6_eval.py`
for the V6 final checkpoint and the V4 reference checkpoint.

Usage:
  python tools/v6_delta_ci.py \
    --v6_csv  ./logs/V6_CORR_1000_DDP/v6_per_query.csv \
    --v4_csv  ./logs/V4_SANITY/v6_per_query.csv \
    --scope   tir_cross \
    --bootstrap 10000

Scopes: tir_cross, same3, cross6, mixed, AG_A2G, AG_G2A, or an exact
pair name such as RGB->TIR.
"""

import argparse
import csv
import json

import numpy as np


TIR = ("RGB->TIR", "NIR->TIR", "TIR->RGB", "TIR->NIR")
SAME = ("RGB->RGB", "NIR->NIR", "TIR->TIR")
NINE = tuple("{}->{}".format(a, b)
             for a in ("RGB", "NIR", "TIR")
             for b in ("RGB", "NIR", "TIR"))


def _keep_scope(row, scope):
    s = row["scope"]
    if scope == "tir_cross":
        return s in TIR
    if scope == "same3":
        return s in SAME
    if scope == "cross6":
        return s in NINE and row["q_mod"] != row["g_mod"]
    if scope == "nine_pair":
        return s in NINE
    if scope == "AG_A2G":
        return s == "A->G"
    if scope == "AG_G2A":
        return s == "G->A"
    return s == scope


def _load(path, scope):
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if not _keep_scope(row, scope):
                continue
            key = (row["scope"], row["q_mod"], row["q_pid"], row["path"])
            out[key] = (float(row["ap"]), float(row["rank"]))
    if not out:
        raise RuntimeError("no rows for scope {} in {}".format(scope, path))
    return out


def _paired_ci(keys, v6, v4, n_boot, seed=1234):
    ap_d = np.asarray([v6[k][0] - v4[k][0] for k in keys], dtype=np.float64)
    r1_v6 = np.asarray([1.0 if v6[k][1] <= 1 else 0.0 for k in keys])
    r1_v4 = np.asarray([1.0 if v4[k][1] <= 1 else 0.0 for k in keys])
    r1_d = r1_v6 - r1_v4
    pids = np.asarray([int(k[2]) for k in keys], dtype=np.int64)

    uniq = np.unique(pids)
    idx_by_pid = {p: np.where(pids == p)[0] for p in uniq}
    pid_ap = np.asarray([ap_d[idx_by_pid[p]].mean() for p in uniq])
    pid_r1 = np.asarray([r1_d[idx_by_pid[p]].mean() for p in uniq])

    rng = np.random.default_rng(seed)
    boot_ap, boot_r1 = [], []
    for _ in range(n_boot):
        sample = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([idx_by_pid[p] for p in sample])
        boot_ap.append(float(ap_d[idx].mean()))
        boot_r1.append(float(r1_d[idx].mean()))

    # Deterministic 5 identity folds: sorted unique pids, index mod 5.
    fold_ap, fold_r1 = [], []
    for fold in range(5):
        fold_pids = uniq[fold::5]
        mask = np.isin(pids, fold_pids)
        fold_ap.append(float(ap_d[mask].mean()) if mask.any() else float("nan"))
        fold_r1.append(float(r1_d[mask].mean()) if mask.any() else float("nan"))

    return {
        "n_rows": int(len(keys)),
        "n_ids": int(uniq.size),
        "query_weighted_delta_mAP": float(ap_d.mean()),
        "id_weighted_delta_mAP": float(pid_ap.mean()),
        "delta_mAP_ci95": [
            float(np.percentile(boot_ap, 2.5)),
            float(np.percentile(boot_ap, 97.5)),
        ],
        "query_weighted_delta_R1": float(r1_d.mean()),
        "id_weighted_delta_R1": float(pid_r1.mean()),
        "delta_R1_ci95": [
            float(np.percentile(boot_r1, 2.5)),
            float(np.percentile(boot_r1, 97.5)),
        ],
        "fold_definition": "sorted unique q_pid, index mod 5",
        "fold_delta_mAP": fold_ap,
        "fold_delta_R1": fold_r1,
        "n_positive_folds_mAP": int(np.sum(np.asarray(fold_ap) > 0)),
        "n_positive_folds_R1": int(np.sum(np.asarray(fold_r1) > 0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v6_csv", required=True)
    parser.add_argument("--v4_csv", required=True)
    parser.add_argument("--scope", default="tir_cross")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    v6 = _load(args.v6_csv, args.scope)
    v4 = _load(args.v4_csv, args.scope)
    keys = sorted(set(v6) & set(v4))
    if not keys:
        raise RuntimeError("no paired rows between V6 and V4 for {}".format(
            args.scope))
    missing_v6 = len(set(v4) - set(v6))
    missing_v4 = len(set(v6) - set(v4))
    result = _paired_ci(keys, v6, v4, args.bootstrap)
    result["scope"] = args.scope
    result["missing_v6_rows"] = int(missing_v6)
    result["missing_v4_rows"] = int(missing_v4)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
