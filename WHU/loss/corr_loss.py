"""V6 class-level cross-modal alignment loss (SupCon-style).

This is a SupCon-style loss over three unordered modality pairs:
    RGB-NIR, RGB-TIR, NIR-TIR

It operates on the raw pre-BN global CLS and is primarily a class-level
cross-modal alignment objective, not a strong instance-level correspondence
objective.  The explicit cross-modal positive pairing is the incremental signal
over CE/GPD/ProCA/WRT.

For each pair:
  - positives: same identity, other modality, different camera;
  - negatives: all different-identity rows;
  - same-identity same-camera and same-identity same-modality rows are
    excluded from both positives and the denominator.

The loss is computed on the pre-BN global CLS feature.  It is intentionally
stateless: no parameters, buffers, or RNG, so enabling/disabling it does not
change the model, optimizer, checkpoint schema, or resume behavior.
"""

import argparse

import torch
import torch.nn.functional as F


PAIRS = ((0, 1), (0, 2), (1, 2))
PAIR_NAMES = ("RGB-NIR", "RGB-TIR", "NIR-TIR")


def _validate_inputs(global_feat, target_rep, camids, num_modalities):
    if num_modalities != 3:
        raise ValueError(
            "V6 corr_loss expects 3 modalities, got {}".format(num_modalities))
    if global_feat.dim() != 2:
        raise ValueError(
            "global_feat must be [M*B, D], got {}".format(tuple(global_feat.shape)))
    if target_rep.dim() != 1:
        raise ValueError(
            "target_rep must be [M*B], got {}".format(tuple(target_rep.shape)))
    if global_feat.shape[0] != target_rep.numel():
        raise ValueError(
            "global_feat rows ({}) != target_rep rows ({})".format(
                global_feat.shape[0], target_rep.numel()))
    if target_rep.numel() % num_modalities != 0:
        raise ValueError(
            "target_rep rows ({}) not divisible by modalities ({})".format(
                target_rep.numel(), num_modalities))
    if len(camids) != num_modalities:
        raise ValueError(
            "camids must have one tensor per modality, got {}".format(len(camids)))
    B = target_rep.numel() // num_modalities
    for m, c in enumerate(camids):
        if c.numel() != B:
            raise ValueError(
                "camids[{}] has {} rows, expected {}".format(m, c.numel(), B))
    return B


def _pair_masks(pid, cam, mod):
    """Return pos/den masks for one 2B-row modality pair."""
    same_pid = pid[:, None].eq(pid[None, :])
    same_mod = mod[:, None].eq(mod[None, :])
    same_cam = cam[:, None].eq(cam[None, :])
    eye = torch.eye(pid.numel(), dtype=torch.bool, device=pid.device)

    pos = same_pid & ~same_mod & ~same_cam
    excl = eye | (same_pid & same_cam) | (same_pid & same_mod)
    den = ~excl
    return pos, den


def _infonce_pair(z, pos, den, tau):
    """Multi-positive SupCon with mean-over-positives and fixed 2B scaling."""
    scores = z @ z.t() / tau
    if scores.dtype != torch.float32:
        raise RuntimeError(
            "V6 corr_loss scores must be float32, got {}".format(scores.dtype))

    pos_f = pos.to(dtype=scores.dtype)
    pos_cnt = pos_f.sum(dim=1)
    # clamp_min(1) is required: an empty-positive anchor must not produce
    # 0/0 in backward (torch.where alone masks forward, not the gradient).
    pos_mean = (scores * pos_f).sum(dim=1) / pos_cnt.clamp_min(1.0)

    den_scores = scores.masked_fill(~den, -1e9)
    l = torch.logsumexp(den_scores, dim=1) - pos_mean
    l = torch.where(pos_cnt > 0, l, torch.zeros_like(l))
    return l, scores, pos_f, pos_cnt


def corr_loss(global_feat, target_rep, camids, num_modalities, tau=0.1):
    """Compute V6 cross-modal correspondence loss and diagnostics.

    Args:
        global_feat: [M*B, D] pre-BN CLS, modality-major.
        target_rep:  [M*B] identity labels, target.repeat(M).
        camids:      list[M] of [B] camera ids.
        num_modalities: 3.
        tau: temperature.

    Returns:
        loss: scalar tensor.
        diag: dict with per-pair loss/coverage/cosine diagnostics.
    """
    if tau <= 0:
        raise ValueError("tau must be positive, got {}".format(tau))
    camids = [torch.as_tensor(c) for c in camids]
    B = _validate_inputs(global_feat, target_rep, camids, num_modalities)
    device = global_feat.device

    cams = torch.cat(
        [c.to(device=device, dtype=torch.long).reshape(-1) for c in camids],
        dim=0)
    mods = torch.arange(num_modalities, device=device).repeat_interleave(B)
    if cams.numel() != num_modalities * B:
        raise ValueError(
            "flattened camids rows ({}) != M*B ({})".format(
                cams.numel(), num_modalities * B))

    with torch.cuda.amp.autocast(enabled=False):
        z = F.normalize(global_feat.float(), dim=1)
        if z.dtype != torch.float32:
            raise RuntimeError(
                "V6 corr_loss z must be float32, got {}".format(z.dtype))

        total = z.new_zeros(())
        diag = {}
        for (a, b), name in zip(PAIRS, PAIR_NAMES):
            idx = torch.cat([
                torch.arange(a * B, (a + 1) * B, device=device),
                torch.arange(b * B, (b + 1) * B, device=device),
            ])
            z_pair = z.index_select(0, idx)
            pid_pair = target_rep.index_select(0, idx)
            cam_pair = cams.index_select(0, idx)
            mod_pair = mods.index_select(0, idx)

            pos, den = _pair_masks(pid_pair, cam_pair, mod_pair)
            l, scores, pos_f, pos_cnt = _infonce_pair(z_pair, pos, den, tau)

            pair_loss = l.sum() / float(2 * B)
            total = total + pair_loss

            with torch.no_grad():
                neg = den & ~pos
                neg_f = neg.to(dtype=scores.dtype)
                pos_cos = (
                    (scores * pos_f).sum() / pos_f.sum().clamp_min(1.0)) * tau
                neg_cos = (
                    (scores * neg_f).sum() / neg_f.sum().clamp_min(1.0)) * tau
                valid = pos_cnt > 0
                diag[name] = {
                    "loss": float(pair_loss.detach().item()),
                    "valid": int(valid.sum().item()),
                    "anchors": int(2 * B),
                    "valid_ratio": float(valid.float().mean().item()),
                    "positives_per_anchor": float(pos_cnt.mean().item()),
                    "positives_per_valid": float(
                        (pos_f.sum() / valid.float().sum().clamp_min(1.0)).item()),
                    "pos_cos": float(pos_cos.item()),
                    "neg_cos": float(neg_cos.item()),
                }

        loss = total / float(len(PAIRS))
    return loss, diag


def _reference_loss(global_feat, target_rep, camids, num_modalities, tau):
    """Explicit-loop reference for the synthetic self-test."""
    B = target_rep.numel() // num_modalities
    device = global_feat.device
    cams = torch.cat(
        [c.to(device=device, dtype=torch.long).reshape(-1) for c in camids], 0)
    mods = torch.arange(num_modalities, device=device).repeat_interleave(B)
    z = F.normalize(global_feat.float(), dim=1)
    total = 0.0
    for a, b in PAIRS:
        idx = torch.cat([
            torch.arange(a * B, (a + 1) * B, device=device),
            torch.arange(b * B, (b + 1) * B, device=device),
        ])
        zp = z.index_select(0, idx)
        pid = target_rep.index_select(0, idx)
        cam = cams.index_select(0, idx)
        mod = mods.index_select(0, idx)
        pair_total = 0.0
        for i in range(2 * B):
            pos_idx = []
            den_idx = []
            for j in range(2 * B):
                if i == j:
                    continue
                same_pid = bool(pid[i].item() == pid[j].item())
                same_cam = bool(cam[i].item() == cam[j].item())
                same_mod = bool(mod[i].item() == mod[j].item())
                if same_pid and same_cam:
                    continue
                if same_pid and same_mod:
                    continue
                den_idx.append(j)
                if same_pid and not same_cam and not same_mod:
                    pos_idx.append(j)
            if not pos_idx:
                continue
            sims = (zp[i] @ zp[den_idx].t()) / tau
            pos_sims = (zp[i] @ zp[pos_idx].t()) / tau
            pair_total += float(
                torch.logsumexp(sims, dim=0).item()
                - pos_sims.mean().item())
        total += pair_total / float(2 * B)
    return total / float(len(PAIRS))


def _selftest():
    torch.manual_seed(0)
    M, B, D = 3, 4, 16
    feat = torch.randn(M * B, D, requires_grad=True)
    pids = torch.tensor([0, 1, 2, 3] * M, dtype=torch.long)
    # Cross-camera structure: alternate camera order by modality so every
    # identity has a cross-camera positive in each modality pair.
    camids = [
        torch.tensor([0, 1, 2, 0] if m == 0 else
                     ([1, 2, 0, 1] if m == 1 else [2, 0, 1, 2]),
                     dtype=torch.long)
        for m in range(M)
    ]

    loss, diag = corr_loss(feat, pids, camids, M, tau=0.1)
    ref = _reference_loss(feat.detach(), pids, camids, M, 0.1)
    if abs(float(loss.detach()) - ref) > 1e-4:
        raise AssertionError(
            "corr_loss {:.6f} != reference {:.6f}".format(
                float(loss.detach()), ref))
    loss.backward()
    if feat.grad is None or not torch.isfinite(feat.grad).all():
        raise AssertionError("corr_loss produced non-finite/None gradient")

    # Empty-positive path: all cameras identical -> every pos mask is empty.
    empty_cams = [torch.zeros(B, dtype=torch.long) for _ in range(M)]
    empty_feat = torch.randn(M * B, D, requires_grad=True)
    empty_loss, empty_diag = corr_loss(
        empty_feat, pids, empty_cams, M, tau=0.1)
    if float(empty_loss.detach()) != 0.0:
        raise AssertionError("empty-positive loss should be zero")
    empty_loss.backward()
    if empty_feat.grad is None or not torch.isfinite(empty_feat.grad).all():
        raise AssertionError("empty-positive path produced non-finite gradient")

    print("corr_loss selftest OK")
    print("loss", float(loss.detach()), "reference", ref)
    for name, d in diag.items():
        print(name, d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V6 corr loss self-test")
    parser.parse_args()
    _selftest()
