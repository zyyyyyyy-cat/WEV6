import logging
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from loss.corr_loss import corr_loss
from torch.cuda import amp
import torch.distributed as dist
import torch.nn.functional as F


def PCA(feat, target, num_modalities, group_size=4):
    """Progressive Center Alignment"""
    D = feat.shape[1]
    B = target.shape[0]

    feat_by_modality = feat.reshape(num_modalities, B, D)
    centers = []
    for m in range(num_modalities):
        fm = F.normalize(feat_by_modality[m], dim=1)
        cm = fm.reshape(-1, group_size, D).mean(1)
        cm = F.normalize(cm, dim=1)
        centers.append(cm)

    g_cent = F.normalize(torch.stack(centers, dim=0).mean(0), dim=1).detach()
    loss = sum((center - g_cent).pow(2).sum(1) for center in centers).mean() / num_modalities

    ids = target.reshape(-1, group_size)[:, 0].long()
    # `centers[m]` is the per-identity center of modality m, shape [P, D].
    # GPD-SC needs these modality-specific centers for its sub-center update.
    return loss, ids, g_cent, centers


class PrototypeBank(nn.Module):
    def __init__(self, num_classes, feat_dim, momentum=0.2, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.m = momentum
        self.register_buffer('prototypes', torch.zeros(num_classes, feat_dim, device=device))

    @torch.no_grad()
    def update(self, ids_local, gcent_local, ddp=True):
        ids_local = ids_local.to(self.prototypes.device).contiguous()
        gcent_local = gcent_local.to(self.prototypes.device).contiguous()

        if ddp and dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            ids_list = [torch.empty_like(ids_local) for _ in range(world)]
            gcent_list = [torch.empty_like(gcent_local) for _ in range(world)]
            dist.all_gather(ids_list, ids_local)
            dist.all_gather(gcent_list, gcent_local)
            ids_all = torch.cat(ids_list, dim=0)
            gc_all = torch.cat(gcent_list, dim=0)
        else:
            ids_all, gc_all = ids_local, gcent_local

        C, _ = self.prototypes.shape
        proto_sum = torch.zeros_like(self.prototypes)
        proto_count = torch.zeros(C, device=self.prototypes.device)

        proto_sum.index_add_(0, ids_all, gc_all)
        proto_count.index_add_(0, ids_all, torch.ones_like(ids_all, dtype=proto_count.dtype))

        mask = proto_count > 0
        new_centers = torch.zeros_like(self.prototypes)
        new_centers[mask] = proto_sum[mask] / proto_count[mask].unsqueeze(1)
        new_centers = F.normalize(new_centers, dim=1)

        old = self.prototypes[mask]
        upd = F.normalize((1 - self.m) * old + self.m * new_centers[mask], dim=1)
        self.prototypes[mask] = upd


class SubCenterPrototypeBank(nn.Module):
    """GPD-SC prototype memory: K sub-centers per identity.

    The K sub-centers are intended to carry the RGB/NIR/TIR modality centers
    of the same identity.  Updates use the modality center of the current
    samples (not the cross-modal identity center `g_cent`), so the sub-centers
    keep their modality semantics instead of collapsing to one point.
    """

    def __init__(self, num_classes, feat_dim, num_subcenters=3, momentum=0.2,
                 device='cuda', count_ema=False, ema_min=0.1, ema_max=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.K = int(num_subcenters)
        self.m = momentum
        self.count_ema = count_ema
        self.ema_min = ema_min
        self.ema_max = ema_max
        self.register_buffer('sub_centers',
                             torch.zeros(num_classes, self.K, feat_dim, device=device))
        self.register_buffer('counts', torch.zeros(num_classes, self.K, device=device))

    @torch.no_grad()
    def initialize(self, centers):
        """centers: [C, K, D], already L2-normalized."""
        if centers.shape != self.sub_centers.shape:
            raise ValueError('sub-center shape mismatch: {} vs {}'.format(
                tuple(centers.shape), tuple(self.sub_centers.shape)))
        self.sub_centers.copy_(centers.float())
        self.counts.fill_(1.0)

    @torch.no_grad()
    def update(self, ids_local, mod_ids_local, centers_local, ddp=True):
        """Update one sub-center per (identity, modality) pair.

        ids_local:    [G] identity ids
        mod_ids_local:[G] modality index in [0, K)
        centers_local:[G, D] modality center for each pair
        """
        ids_local = ids_local.to(self.sub_centers.device).contiguous()
        mod_ids_local = mod_ids_local.to(self.sub_centers.device).contiguous()
        centers_local = centers_local.to(self.sub_centers.device).float().contiguous()

        if ddp and dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            ids_list = [torch.empty_like(ids_local) for _ in range(world)]
            mod_list = [torch.empty_like(mod_ids_local) for _ in range(world)]
            center_list = [torch.empty_like(centers_local) for _ in range(world)]
            dist.all_gather(ids_list, ids_local)
            dist.all_gather(mod_list, mod_ids_local)
            dist.all_gather(center_list, centers_local)
            ids_all = torch.cat(ids_list, dim=0).long()
            mod_all = torch.cat(mod_list, dim=0).long()
            centers_all = torch.cat(center_list, dim=0).float()
        else:
            ids_all = ids_local.long()
            mod_all = mod_ids_local.long()
            centers_all = centers_local.float()

        C, K, D = self.sub_centers.shape
        flat = self.sub_centers.view(C * K, D)
        count_flat = self.counts.view(C * K)
        idx = ids_all * K + mod_all      # [G_all]

        sums = torch.zeros_like(flat)
        counts = torch.zeros(C * K, device=flat.device)
        sums.index_add_(0, idx, centers_all)
        counts.index_add_(0, idx, torch.ones_like(idx, dtype=counts.dtype))

        mask = counts > 0
        new_centers = F.normalize(sums[mask] / counts[mask].unsqueeze(1), dim=1)
        old = flat[mask]
        bank_count = count_flat[mask]
        if self.count_ema:
            m_eff = torch.clamp(1.0 / (bank_count + 1.0),
                                min=self.ema_min, max=self.ema_max)
            # An uninitialized sub-center should be set directly, not blended
            # with a zero vector.
            m_eff = torch.where(bank_count <= 0,
                                torch.ones_like(m_eff), m_eff)
        else:
            m_eff = torch.full_like(bank_count, float(self.m))
        upd = F.normalize((1.0 - m_eff).unsqueeze(1) * old
                          + m_eff.unsqueeze(1) * new_centers, dim=1)
        flat[mask] = upd
        count_flat[mask] += 1.0


class HierarchicalPrototypeBank(nn.Module):
    """HPD memory: one shared identity prototype plus K modality sub-centers.

    The shared prototype `P_id[c]` is the original GPD anchor and is updated
    with the cross-modal identity center `g_cent`.  The K sub-centers
    `P_scene[c,k]` are the GPD-SC modality centers and are updated with the
    ProCA modality centers.  Both banks are buffers; neither receives gradients.
    """

    def __init__(self, num_classes, feat_dim, num_subcenters=3, momentum=0.2,
                 device='cuda', count_ema=False, ema_min=0.1, ema_max=0.5,
                 shared_momentum=0.2):
        super().__init__()
        self.scene_bank = SubCenterPrototypeBank(
            num_classes=num_classes,
            feat_dim=feat_dim,
            num_subcenters=num_subcenters,
            momentum=momentum,
            device=device,
            count_ema=count_ema,
            ema_min=ema_min,
            ema_max=ema_max,
        )
        self.id_bank = PrototypeBank(
            num_classes=num_classes,
            feat_dim=feat_dim,
            momentum=shared_momentum,
            device=device,
        )

    @property
    def num_classes(self):
        return self.id_bank.num_classes

    @property
    def feat_dim(self):
        return self.id_bank.feat_dim

    @property
    def K(self):
        return self.scene_bank.K

    @torch.no_grad()
    def initialize(self, scene_centers, id_centers):
        """scene_centers: [C,K,D]; id_centers: [C,D]; both L2-normalized."""
        self.scene_bank.initialize(scene_centers)
        if id_centers.shape != self.id_bank.prototypes.shape:
            raise ValueError('id-center shape mismatch: {} vs {}'.format(
                tuple(id_centers.shape), tuple(self.id_bank.prototypes.shape)))
        self.id_bank.prototypes.copy_(id_centers.float())

    @torch.no_grad()
    def update(self, ids, g_cent, scene_ids, mod_ids, scene_centers, ddp=True):
        """Update the shared anchor and the modality sub-centers.

        ids/g_cent:        [P] / [P,D]        per-identity cross-modal center
        scene_ids/mod_ids: [M*P] / [M*P]      modality-major identity/modality ids
        scene_centers:     [M*P,D]            ProCA modality centers
        """
        self.id_bank.update(ids, g_cent, ddp=ddp)
        self.scene_bank.update(scene_ids, mod_ids, scene_centers, ddp=ddp)

    def broadcast(self, src=0):
        dist.broadcast(self.id_bank.prototypes, src=src)
        dist.broadcast(self.scene_bank.sub_centers, src=src)
        dist.broadcast(self.scene_bank.counts, src=src)


def GPD(feat, target, bank: PrototypeBank, ids, g_cent, num_modalities, tau=0.03):
    """Global Prototype Discrimination"""
    target_rep = target.repeat(num_modalities)
    f = F.normalize(feat, dim=1)
    with torch.no_grad():
        P = bank.prototypes.clone()
        norms = P.norm(dim=1)
        need = norms[ids] < 1e-6
        if need.any():
            P[ids[need]] = g_cent[need]
        P = F.normalize(P, dim=1)

    logits = (f @ P.t()) / tau
    loss = F.cross_entropy(logits, target_rep)
    return loss


@torch.no_grad()
def _warm_start_centers(model, loaders, num_classes, feat_dim, device):
    """Compute per-modality class means and the valid-modality identity mean.

    Returns (centers, id_mean, valids, n_valid_ids):
      centers: list of M tensors [C,D], missing modalities filled with id_mean
      id_mean: [C,D], the identity mean over *valid* modalities (before fill)
      valids:  list of M bool tensors [C]
    """
    model.eval()
    centers, valids = [], []
    for mode, loader in enumerate(loaders, start=1):
        sums = torch.zeros(num_classes, feat_dim, device=device)
        cnts = torch.zeros(num_classes, device=device)
        for batch in loader:
            img = batch[0].to(device)
            pid = torch.as_tensor(batch[1], dtype=torch.long, device=device)
            feat = model(img, mode=mode)
            feat = F.normalize(feat.float(), dim=1)
            sums.index_add_(0, pid, feat)
            cnts.index_add_(0, pid, torch.ones_like(pid, dtype=cnts.dtype))
        valid = cnts > 0
        center = torch.zeros_like(sums)
        center[valid] = F.normalize(sums[valid] / cnts[valid].unsqueeze(1), dim=1)
        centers.append(center)
        valids.append(valid)

    id_sum = torch.zeros(num_classes, feat_dim, device=device)
    id_cnt = torch.zeros(num_classes, device=device)
    for center, valid in zip(centers, valids):
        id_sum += center * valid.unsqueeze(1).float()
        id_cnt += valid.float()
    id_mean = F.normalize(
        id_sum / torch.clamp(id_cnt, min=1.0).unsqueeze(1), dim=1)
    for m in range(len(centers)):
        missing = ~valids[m]
        if missing.any():
            centers[m][missing] = id_mean[missing]
    return centers, id_mean, valids, int((id_cnt > 0).sum().item())


@torch.no_grad()
def warm_start_subcenters(model, bank, loaders, device, logger=None):
    """Initialize K sub-centers per identity with per-modality class means.

    `loaders` is the tri-modal deterministic training loader list, ordered as
    [RGB, NIR, TIR].  Missing (identity, modality) groups fall back to the
    identity mean over valid modalities.
    """
    if not loaders:
        return 0
    if len(loaders) != bank.K:
        raise ValueError('warm-start loaders ({}) != sub-centers ({})'.format(
            len(loaders), bank.K))
    centers, _, _, n_ids = _warm_start_centers(
        model, loaders, bank.num_classes, bank.feat_dim, device)
    bank.initialize(torch.stack(centers, dim=1))
    model.train()
    if logger is not None:
        logger.info('GPD-SC warm start: {}/{} identities, K={}'.format(
            n_ids, bank.num_classes, bank.K))
    return n_ids


@torch.no_grad()
def warm_start_hierarchical(model, bank, loaders, device, logger=None):
    """Initialize both HPD banks from one tri-modal pass.

    P_scene uses the per-(pid,modality) class mean; P_id uses the valid-modality
    identity mean (computed before missing modalities are filled).
    """
    if not loaders:
        return 0
    if len(loaders) != bank.K:
        raise ValueError('warm-start loaders ({}) != sub-centers ({})'.format(
            len(loaders), bank.K))
    centers, id_mean, _, n_ids = _warm_start_centers(
        model, loaders, bank.num_classes, bank.feat_dim, device)
    bank.initialize(torch.stack(centers, dim=1), id_mean)
    if not (torch.isfinite(bank.id_bank.prototypes).all()
            and torch.isfinite(bank.scene_bank.sub_centers).all()):
        raise RuntimeError('HPD warm start produced non-finite prototypes')
    model.train()
    if logger is not None:
        logger.info('HPD warm start: {}/{} identities, K={}, '
                    'P_id/P_scene initialized'.format(
                        n_ids, bank.num_classes, bank.K))
    return n_ids


@torch.no_grad()
def initialize_gac_heads(model, bank, logger=None):
    """Fill the GAC heads from the warmed-up P_scene bank.

    Must run on every rank after the bank broadcast.  The heads are registered
    as zero parameters in build_transformer.__init__ so the optimizer/DDP see
    them before this point; here they are overwritten with the modality class
    means.  Every head must be non-zero, otherwise the three heads are
    permutation-identical and GAC silently degenerates to shared W_aux.
    """
    if not hasattr(model, 'aux_heads'):
        raise RuntimeError('GAC requires model.aux_heads')
    M = int(model.aux_heads.shape[0])
    if bank is None:
        raise RuntimeError('GAC requires a warmed-up HierarchicalPrototypeBank')
    if int(bank.K) != M:
        raise ValueError(
            'GAC heads ({}) must equal GPD sub-centers ({})'.format(M, bank.K))
    P_scene = bank.scene_bank.sub_centers.float()   # [C, K, D]
    P_id = bank.id_bank.prototypes.float()          # [C, D]
    if P_scene.shape[1] != M:
        raise ValueError('P_scene columns ({}) != GAC heads ({})'.format(
            P_scene.shape[1], M))
    fallback_rows = 0
    for m in range(M):
        W = P_scene[:, m].clone()
        zero = W.norm(dim=1) < 1e-6
        if zero.any():
            W[zero] = P_id[zero]
            fallback_rows += int(zero.sum().item())
        W = F.normalize(W, dim=1)
        if not torch.isfinite(W).all() or float(W.norm()) < 1e-6:
            raise RuntimeError(
                'GAC head {} is all-zero/non-finite after P_scene init; '
                'check GPD_WARMSTART and the bank'.format(m))
        model.aux_heads.data[m].copy_(W)
    if logger is not None:
        logger.info(
            'GAC heads initialized from P_scene: M={}, P_id fallback rows={}'
            .format(M, fallback_rows))


def gac_aux_weight(cfg, epoch):
    """Linear ramp: 0.5*target at epoch 1 -> target at AUX_RAMP_EPOCHS."""
    target = float(getattr(cfg.MODEL, 'AUX_WEIGHT', 0.0))
    if target <= 0.0:
        return 0.0
    ramp = int(getattr(cfg.MODEL, 'AUX_RAMP_EPOCHS', 5))
    if ramp <= 1 or epoch >= ramp:
        return target
    start = 0.5 * target
    return start + (target - start) * float(epoch - 1) / float(ramp - 1)


def corr_weight(cfg, epoch):
    """V6 linear warmup: 0 -> WEIGHT over WARMUP_EPOCHS.

    Uses the same 1-based epoch as gac_aux_weight, so resume does not reset
    the schedule.  A non-positive/disabled config returns exactly 0.
    """
    corr_cfg = getattr(cfg.MODEL, 'CORR_LOSS', None)
    if corr_cfg is None or not bool(corr_cfg.ENABLED):
        return 0.0
    target = float(corr_cfg.WEIGHT)
    if target <= 0.0:
        return 0.0
    warmup = int(corr_cfg.WARMUP_EPOCHS)
    if warmup <= 0:
        return target
    return target * min(1.0, float(epoch) / float(warmup))


def gac_aux_loss(aux_scores, target_rep, num_modalities):
    """Mean CE over every query row x every gallery-conditioned head.

    aux_scores: [N, M, C] with N = M*B; target_rep: [N].
    Flatten is n-major, so use unsqueeze/expand, not target_rep.repeat(M).
    """
    N, M, C = aux_scores.shape
    if int(M) != int(num_modalities):
        raise ValueError('aux head count ({}) != modalities ({})'.format(
            M, num_modalities))
    targets = target_rep.unsqueeze(1).expand(-1, M).reshape(-1)
    return F.cross_entropy(aux_scores.reshape(N * M, C), targets)


def format_gac_diag(diag):
    """Format the GAC diagnostic line (unit-testable; no logger dependency)."""
    return (
        'GAC diag: loss_aux {:.4f}, share_aux {:.3f}, '
        'cos aux-id/aux-scene/aux-base {:.3f}/{:.3f}/{:.3f}, '
        'head_pair_cos {}, head_scene_cos {}, head_id_cos {}'
        .format(
            diag['loss_aux'], diag['share_aux'],
            diag['cos_aux_id'], diag['cos_aux_scene'], diag['cos_aux_base'],
            [round(x, 3) for x in diag['head_pair_cos']],
            [round(x, 3) for x in diag['head_scene_cos']],
            [round(x, 3) for x in diag['head_id_cos']]))


def _bank_type(bank):
    if bank is None:
        return 'none'
    if hasattr(bank, 'id_bank') and hasattr(bank, 'scene_bank'):
        return 'hierarchical'
    if hasattr(bank, 'sub_centers') and hasattr(bank, 'counts'):
        return 'subcenter'
    if hasattr(bank, 'prototypes'):
        return 'single'
    return 'unknown'


def save_resume_checkpoint(path, epoch, model, optimizer, scheduler, scaler,
                           proto_bank):
    """Full training state for interruption/resume. Model-only files remain
    available separately for test.py."""
    model_to_save = model.module if hasattr(model, 'module') else model
    np_state = np.random.get_state()
    state = {
        'epoch': int(epoch),
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': (
            scaler.state_dict() if hasattr(scaler, 'state_dict') else None),
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': (
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None),
        'numpy_rng_state': (
            np_state[0], np_state[1].tolist(), np_state[2],
            np_state[3], np_state[4]),
        'python_rng_state': random.getstate(),
    }
    if proto_bank is not None:
        bank_state = {}
        if hasattr(proto_bank, 'id_bank') and hasattr(proto_bank, 'scene_bank'):
            bank_state['type'] = 'hierarchical'
            bank_state['id_prototypes'] = (
                proto_bank.id_bank.prototypes.detach().cpu())
            bank_state['scene_sub_centers'] = (
                proto_bank.scene_bank.sub_centers.detach().cpu())
            bank_state['scene_counts'] = (
                proto_bank.scene_bank.counts.detach().cpu())
        elif hasattr(proto_bank, 'sub_centers') and hasattr(proto_bank, 'counts'):
            bank_state['type'] = 'subcenter'
            bank_state['sub_centers'] = proto_bank.sub_centers.detach().cpu()
            bank_state['counts'] = proto_bank.counts.detach().cpu()
        elif hasattr(proto_bank, 'prototypes'):
            bank_state['type'] = 'single'
            bank_state['prototypes'] = proto_bank.prototypes.detach().cpu()
        else:
            bank_state['type'] = 'unknown'
        state['bank_state_dict'] = bank_state
    tmp_path = path + '.tmp'
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load_resume_checkpoint(path, model, optimizer, scheduler, scaler,
                           proto_bank, device, logger=None):
    """Load a full checkpoint (model+optimizer+scheduler+scaler+bank+RNG).

    A model-only `transformer_*.pth` is also accepted for convenience; in that
    case optimizer/scheduler/scaler/bank are not restored and the caller starts
    at epoch 1.  Returns (last_epoch, is_full_checkpoint).
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError('resume checkpoint not found: {}'.format(path))
    ckpt = torch.load(path, map_location='cpu')
    model_to_load = model.module if hasattr(model, 'module') else model
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model_to_load.load_state_dict(ckpt['model_state_dict'])
        if ckpt.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if (ckpt.get('scaler_state_dict') is not None
                and hasattr(scaler, 'load_state_dict')):
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        bank_state = ckpt.get('bank_state_dict')
        if proto_bank is not None and bank_state:
            expected_type = bank_state.get('type')
            actual_type = _bank_type(proto_bank)
            if expected_type and expected_type != actual_type:
                raise RuntimeError(
                    'resume bank type mismatch: checkpoint={}, model={}'
                    .format(expected_type, actual_type))
            bank_loaded = False
            if ('id_prototypes' in bank_state
                    and hasattr(proto_bank, 'id_bank')
                    and hasattr(proto_bank, 'scene_bank')):
                proto_bank.id_bank.prototypes.copy_(
                    bank_state['id_prototypes'].to(
                        proto_bank.id_bank.prototypes.device))
                proto_bank.scene_bank.sub_centers.copy_(
                    bank_state['scene_sub_centers'].to(
                        proto_bank.scene_bank.sub_centers.device))
                if 'scene_counts' in bank_state:
                    proto_bank.scene_bank.counts.copy_(
                        bank_state['scene_counts'].to(
                            proto_bank.scene_bank.counts.device))
                bank_loaded = True
            elif ('sub_centers' in bank_state
                    and hasattr(proto_bank, 'sub_centers')):
                proto_bank.sub_centers.copy_(
                    bank_state['sub_centers'].to(
                        proto_bank.sub_centers.device))
                if 'counts' in bank_state:
                    proto_bank.counts.copy_(
                        bank_state['counts'].to(proto_bank.counts.device))
                bank_loaded = True
            elif ('prototypes' in bank_state
                    and hasattr(proto_bank, 'prototypes')):
                proto_bank.prototypes.copy_(
                    bank_state['prototypes'].to(
                        proto_bank.prototypes.device))
                bank_loaded = True
            if not bank_loaded:
                raise RuntimeError(
                    'resume bank_state could not be restored into {}'
                    .format(actual_type))
        if ckpt.get('rng_state') is not None:
            torch.set_rng_state(ckpt['rng_state'])
        if ckpt.get('cuda_rng_state') is not None and torch.cuda.is_available():
            # torch.cuda.set_rng_state expects the CPU ByteTensor returned by
            # torch.cuda.get_rng_state(); do not move it to CUDA.
            torch.cuda.set_rng_state(ckpt['cuda_rng_state'])
        np_state = ckpt.get('numpy_rng_state')
        if np_state is not None:
            np.random.set_state((
                np_state[0],
                np.array(np_state[1], dtype=np.uint32),
                np_state[2], np_state[3], np_state[4]))
        if ckpt.get('python_rng_state') is not None:
            random.setstate(tuple(ckpt['python_rng_state']))
        epoch = int(ckpt.get('epoch', 0))
        if logger is not None:
            logger.info('Resumed full checkpoint {} at epoch {}'.format(
                path, epoch))
        return epoch, True

    if (isinstance(ckpt, dict) and ckpt
            and all(str(k).startswith('module.') for k in ckpt)):
        ckpt = {str(k).replace('module.', ''): v for k, v in ckpt.items()}
    result = model_to_load.load_state_dict(ckpt, strict=False)
    model_keys = set(model_to_load.state_dict().keys())
    matched = sum(1 for k in ckpt if k in model_keys)
    bad_missing = [k for k in result.missing_keys
                   if 'aux_heads' not in k and 'classifier' not in k]
    bad_unexpected = [k for k in result.unexpected_keys
                      if 'aux_heads' not in k and 'classifier' not in k]
    if matched == 0:
        raise RuntimeError(
            'model-only resume {} matched no model keys; refusing to '
            'restart from scratch'.format(path))
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            'model-only resume {} is incompatible: missing {}, unexpected {}'
            .format(path, bad_missing[:8], bad_unexpected[:8]))
    if logger is not None:
        logger.warning(
            'Resumed model-only checkpoint {}; optimizer/scheduler/scaler/bank '
            'not restored, restarting at epoch 1'.format(path))
    return 0, False


def GPD_SC(feat, target, bank: SubCenterPrototypeBank, ids, g_cent,
           num_modalities, tau=0.03):
    """GPD with K sub-centers per identity (max over sub-centers)."""
    target_rep = target.repeat(num_modalities)
    f = F.normalize(feat.float(), dim=1)
    with torch.no_grad():
        P = bank.sub_centers.float().clone()
        norms = P.norm(dim=2)                    # [C, K]
        need = norms[ids].sum(dim=1) < 1e-6      # [P]
        if need.any():
            P[ids[need]] = g_cent[need].float().unsqueeze(1).expand(
                -1, bank.K, -1)
        P = F.normalize(P, dim=2)

    # [N, C, K] -> max over K -> [N, C]
    logits = torch.einsum('nd,ckd->nck', f, P) / tau
    logits = logits.max(dim=2).values
    return F.cross_entropy(logits, target_rep)


def GPD_HPD(feat, target, bank: HierarchicalPrototypeBank, ids, g_cent,
            num_modalities, tau=0.03, alpha=0.75, centers=None):
    """HPD: shared identity CE + modality sub-center CE.

    alpha is the scene weight:
        loss = (1 - alpha) * CE(shared) + alpha * CE(scene)
    alpha=1 is GPD-SC; alpha=0 is GPD with a warm-started shared prototype.
    `centers` is the optional ProCA modality-center list; when given, the
    pre-normalization norm of the identity mean is reported as a collapse
    diagnostic.
    """
    target_rep = target.repeat(num_modalities)
    f = F.normalize(feat.float(), dim=1)

    with torch.no_grad():
        P_id = bank.id_bank.prototypes.float().clone()          # [C,D]
        norms_id = P_id.norm(dim=1)                             # [C]
        need_id = norms_id[ids] < 1e-6                          # [P]
        if need_id.any():
            P_id[ids[need_id]] = g_cent[need_id].float()
        P_id = F.normalize(P_id, dim=1)

        P_scene = bank.scene_bank.sub_centers.float().clone()   # [C,K,D]
        norms_scene = P_scene.norm(dim=2)                       # [C,K]
        need_scene = norms_scene[ids].sum(dim=1) < 1e-6         # [P]
        if need_scene.any():
            P_scene[ids[need_scene]] = g_cent[need_scene].float().unsqueeze(1).expand(
                -1, bank.K, -1)
        P_scene = F.normalize(P_scene, dim=2)

    logits_id = (f @ P_id.t()) / tau
    logits_scene_all = torch.einsum('nd,ckd->nck', f, P_scene) / tau
    logits_scene = logits_scene_all.max(dim=2).values

    loss_id = F.cross_entropy(logits_id, target_rep)
    loss_scene = F.cross_entropy(logits_scene, target_rep)
    loss = (1.0 - alpha) * loss_id + alpha * loss_scene

    with torch.no_grad():
        B = target.shape[0]
        selected = logits_scene_all.max(dim=2).indices          # [M*B, C]
        pos_selected = selected[
            torch.arange(selected.shape[0], device=selected.device),
            target_rep]                                          # [M*B]
        own_ratio = []
        for m in range(num_modalities):
            seg = pos_selected[m * B:(m + 1) * B]
            own_ratio.append(float((seg == m).float().mean().item()))

        def _stats(logits):
            N = logits.shape[0]
            idx = torch.arange(N, device=logits.device)
            pos = logits[idx, target_rep]
            neg_mask = torch.ones_like(logits, dtype=torch.bool)
            neg_mask[idx, target_rep] = False
            neg = logits[neg_mask].reshape(N, -1)
            prob = logits.softmax(dim=1)
            entropy = -(prob * logits.log_softmax(dim=1)).sum(dim=1)
            return {
                'pos_mean': float(pos.mean().item()),
                'neg_mean': float(neg.mean().item()),
                'neg_max_mean': float(neg.max(dim=1).values.mean().item()),
                'entropy': float(entropy.mean().item()),
            }

        usage = torch.bincount(pos_selected, minlength=bank.K).float()
        diag = {
            'own_ratio': own_ratio,
            'usage': [int(x) for x in usage.tolist()],
            'id': _stats(logits_id),
            'scene': _stats(logits_scene),
        }
        if centers is not None:
            raw_mean = torch.stack(centers, dim=0).mean(0)     # [P,D]
            diag['g_cent_norm'] = float(raw_mean.norm(dim=1).mean().item())
    return loss, loss_id, loss_scene, diag


def diagnose_gpd_gradients(model, imgs, target, camids, bank, cfg, device,
                           group_size, loss_fn, aux_weight=0.0):
    """Measure CE_id/CE_scene/base/PCA (+GAC) gradient shares on a small proxy.

    One extra eval-mode forward on the unwrapped module.  Eval mode disables
    dropout/drop-path; the GPD terms use the pre-BN CLS, so no BN buffers are
    updated.  The gradient is taken over the last ViT block + final norm to
    keep the cost low; it is a proxy, not the full-ViT gradient.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad(), amp.autocast(enabled=True):
            if aux_weight > 0:
                cls_score, global_feat, _, aux_scores = model(
                    imgs, target, camids, return_aux=True)
            else:
                cls_score, global_feat, _ = model(imgs, target, camids)
            loss_pca, ids, g_cent, centers = PCA(
                global_feat, target,
                num_modalities=len(imgs),
                group_size=group_size,
            )
            loss_gpd, loss_id, loss_scene, diag = GPD_HPD(
                global_feat, target, bank, ids, g_cent,
                num_modalities=len(imgs),
                tau=cfg.MODEL.GPD_TAU,
                alpha=cfg.MODEL.GPD_SCENE_WEIGHT,
                centers=centers,
            )
            loss_base, _, _ = loss_fn(
                cls_score, global_feat, target.repeat(len(imgs)))
            loss_aux = None
            if aux_weight > 0:
                loss_aux = gac_aux_loss(
                    aux_scores, target.repeat(len(imgs)), len(imgs))
            params = list(model.base.blocks[-1].parameters()) + list(
                model.base.norm.parameters())
            g_id = torch.autograd.grad(
                loss_id, params, retain_graph=True, allow_unused=True)
            g_scene = torch.autograd.grad(
                loss_scene, params, retain_graph=True, allow_unused=True)
            g_base = torch.autograd.grad(
                loss_base, params, retain_graph=True, allow_unused=True)
            g_pca = torch.autograd.grad(
                loss_pca, params, retain_graph=(aux_weight > 0),
                allow_unused=True)
            if aux_weight > 0:
                g_aux = torch.autograd.grad(
                    loss_aux, params, retain_graph=False, allow_unused=True)

        def _norm(grads):
            total = 0.0
            for g in grads:
                if g is not None:
                    total += float(g.detach().float().pow(2).sum().item())
            return total ** 0.5

        def _dot(a, b):
            total = 0.0
            for x, y in zip(a, b):
                if x is not None and y is not None:
                    total += float((x.detach().float()
                                    * y.detach().float()).sum().item())
            return total

        alpha = cfg.MODEL.GPD_SCENE_WEIGHT
        n_id, n_scene = _norm(g_id), _norm(g_scene)
        n_base, n_pca = _norm(g_base), _norm(g_pca)
        n_aux = _norm(g_aux) if aux_weight > 0 else 0.0
        eff_id = (1.0 - alpha) * n_id
        eff_scene = alpha * n_scene
        eff_base = n_base
        eff_pca = cfg.MODEL.PCA_LOSS_WEIGHT * n_pca
        eff_aux = aux_weight * n_aux
        total_eff = eff_id + eff_scene + eff_base + eff_pca + eff_aux + 1e-12

        # GAC head geometry: per-class mean cos, matching P_scene pair_cos.
        with torch.no_grad():
            W = F.normalize(model.aux_heads.float(), dim=2)
            P_scene = F.normalize(bank.scene_bank.sub_centers.float(), dim=2)
            P_id = F.normalize(bank.id_bank.prototypes.float(), dim=1)
            M = W.shape[0]
            head_scene_cos, head_id_cos, head_pair_cos = [], [], []
            for m in range(M):
                head_scene_cos.append(float((W[m] * P_scene[:, m]).sum(1).mean().item()))
                head_id_cos.append(float((W[m] * P_id).sum(1).mean().item()))
            for m in range(M):
                for n in range(m + 1, M):
                    head_pair_cos.append(float((W[m] * W[n]).sum(1).mean().item()))

        result = {
            'g_id': n_id,
            'g_scene': n_scene,
            'g_base': n_base,
            'g_pca': n_pca,
            'g_aux': n_aux,
            'eff_id': eff_id,
            'eff_scene': eff_scene,
            'eff_base': eff_base,
            'eff_pca': eff_pca,
            'eff_aux': eff_aux,
            'share_id': eff_id / total_eff,
            'share_scene': eff_scene / total_eff,
            'share_base': eff_base / total_eff,
            'share_pca': eff_pca / total_eff,
            'share_aux': eff_aux / total_eff,
            'cos': _dot(g_id, g_scene) / (n_id * n_scene + 1e-12),
            'cos_aux_id': (
                _dot(g_aux, g_id) / (n_aux * n_id + 1e-12)
                if aux_weight > 0 else float('nan')),
            'cos_aux_scene': (
                _dot(g_aux, g_scene) / (n_aux * n_scene + 1e-12)
                if aux_weight > 0 else float('nan')),
            'cos_aux_base': (
                _dot(g_aux, g_base) / (n_aux * n_base + 1e-12)
                if aux_weight > 0 else float('nan')),
            'loss_id': float(loss_id.detach().item()),
            'loss_scene': float(loss_scene.detach().item()),
            'loss_base': float(loss_base.detach().item()),
            'loss_pca': float(loss_pca.detach().item()),
            'loss_aux': (float(loss_aux.detach().item())
                         if loss_aux is not None else float('nan')),
            'own_ratio': diag['own_ratio'],
            'usage': diag['usage'],
            'g_cent_norm': diag.get('g_cent_norm', float('nan')),
            'id_stats': diag['id'],
            'scene_stats': diag['scene'],
            'head_scene_cos': head_scene_cos,
            'head_id_cos': head_id_cos,
            'head_pair_cos': head_pair_cos,
        }
        return result
    finally:
        model.train(was_training)


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             train_normal_loaders,
             val_loaders,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_querys, local_rank, resume_path=''):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = 'cuda'
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger('transreid.train')
    dist_on = (
        cfg.MODEL.DIST_TRAIN
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )
    rank = dist.get_rank() if dist_on else 0
    is_main = rank == 0
    if is_main:
        logger.info('start training')
    if device:
        model.to(local_rank)
        if dist_on:
            if is_main:
                logger.info('Using {} GPUs for DDP training'.format(dist.get_world_size()))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                # The ViT backbone retains its ImageNet classification head
                # (base.fc), while ReID training uses the outer classifier.
                # Consequently base.fc never participates in the loss.
                find_unused_parameters=True,
            )
            if local_rank == 0:
                torch.set_num_threads(16)
            else:
                torch.set_num_threads(4)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    meter_ls = [AverageMeter() for _ in range(4)]
    hpd_id_meter = AverageMeter()
    hpd_scene_meter = AverageMeter()
    gpd_diag = None
    clip_count = 0
    last_grad_norm = 0.0

    evaluator = R1_mAP_eval(
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING,
        top_k=cfg.TEST.TOP_K_EVAL,
        logger=logger,
    )
    scaler = amp.GradScaler()
    model_meta = model.module if hasattr(model, 'module') else model
    feat_dim = getattr(model_meta, 'in_planes', 768)
    num_classes = getattr(model_meta, 'num_classes', 500)
    aux_target = float(getattr(cfg.MODEL, 'AUX_WEIGHT', 0.0))
    aux_meter = AverageMeter()
    corr_cfg = getattr(cfg.MODEL, 'CORR_LOSS', None)
    corr_enabled = bool(corr_cfg is not None and corr_cfg.ENABLED)
    corr_tau = float(corr_cfg.TAU) if corr_enabled else 0.1
    if corr_enabled and corr_tau <= 0:
        raise ValueError('MODEL.CORR_LOSS.TAU must be positive')
    corr_meter = AverageMeter()
    resume_full = False
    start_epoch = 1
    loss_type = cfg.SOLVER.LOSS_TYPE.lower().strip()
    use_aux_modules = loss_type != 'base'
    gpd_type = getattr(cfg.MODEL, 'GPD_TYPE', 'single').lower().strip()
    if gpd_type not in ('single', 'subcenter', 'hierarchical'):
        raise ValueError(
            "MODEL.GPD_TYPE must be 'single', 'subcenter', or "
            "'hierarchical', got {}".format(gpd_type))
    if gpd_type != 'single' and 'gpd' not in loss_type:
        raise ValueError(
            "GPD_TYPE={} requires SOLVER.LOSS_TYPE to contain 'gpd', "
            "got {}".format(gpd_type, cfg.SOLVER.LOSS_TYPE))
    if is_main:
        logger.info('Feature dimension: {}; identities: {}; GPD type: {}'.format(
            feat_dim, num_classes, gpd_type))
    if aux_target > 0:
        if not use_aux_modules:
            raise ValueError('GAC requires SOLVER.LOSS_TYPE to include pca+gpd')
        if gpd_type != 'hierarchical':
            raise ValueError('GAC requires GPD_TYPE=hierarchical')
        if not cfg.MODEL.GPD_WARMSTART:
            raise ValueError('GAC requires GPD_WARMSTART=True')
        if cfg.MODEL.GPD_SUB_CENTERS != len(cfg.DATASETS.MODALITIES):
            raise ValueError('GAC heads require GPD_SUB_CENTERS == modalities')
        if cfg.TEST.NECK_FEAT != 'before':
            raise ValueError("GAC requires TEST.NECK_FEAT='before'")
        if not hasattr(model_meta, 'aux_heads'):
            raise ValueError('GAC requires model.aux_heads')
        if is_main:
            logger.info(
                'GAC enabled: weight target {:.3f}, ramp {} epochs, M={}'.format(
                    aux_target, cfg.MODEL.AUX_RAMP_EPOCHS,
                    len(cfg.DATASETS.MODALITIES)))
    if use_aux_modules:
        if gpd_type in ('subcenter', 'hierarchical'):
            num_modalities_cfg = len(cfg.DATASETS.MODALITIES)
            if cfg.MODEL.GPD_SUB_CENTERS != num_modalities_cfg:
                raise ValueError(
                    "GPD-SC/HPD use modality-indexed sub-centers: "
                    "GPD_SUB_CENTERS ({}) must equal the number of modalities "
                    "({}). Use GPD_TYPE='single' for the K=1 baseline.".format(
                        cfg.MODEL.GPD_SUB_CENTERS, num_modalities_cfg))
        if gpd_type == 'hierarchical':
            proto_bank = HierarchicalPrototypeBank(
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
        elif gpd_type == 'subcenter':
            proto_bank = SubCenterPrototypeBank(
                num_classes=num_classes,
                feat_dim=feat_dim,
                num_subcenters=cfg.MODEL.GPD_SUB_CENTERS,
                momentum=cfg.MODEL.GPD_MOMENTUM,
                device=device,
                count_ema=cfg.MODEL.GPD_COUNT_EMA,
                ema_min=cfg.MODEL.GPD_EMA_MIN,
                ema_max=cfg.MODEL.GPD_EMA_MAX,
            )
        else:
            proto_bank = PrototypeBank(
                num_classes=num_classes,
                feat_dim=feat_dim,
                momentum=cfg.MODEL.GPD_MOMENTUM,
                device=device,
            )
        if resume_path:
            last_epoch, resume_full = load_resume_checkpoint(
                resume_path, model, optimizer, scheduler, scaler,
                proto_bank, device, logger)
            start_epoch = last_epoch + 1 if resume_full else 1
            if dist_on:
                dist.barrier()
        if (cfg.MODEL.GPD_WARMSTART and gpd_type in ('subcenter', 'hierarchical')
                and not resume_full):
            if cfg.TEST.NECK_FEAT != 'before':
                raise ValueError(
                    "GPD warm start expects TEST.NECK_FEAT='before' so that "
                    "the prototypes use the same pre-BN CLS as GPD.")
            # DataLoader iterator creation draws the worker base seed from the
            # global torch RNG.  Save/restore it so that warm-start on/off runs
            # keep the same training augmentation stream.
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            if dist_on:
                if is_main:
                    if gpd_type == 'hierarchical':
                        warm_start_hierarchical(model_meta, proto_bank,
                                                train_normal_loaders, device, logger)
                    else:
                        warm_start_subcenters(model_meta, proto_bank,
                                              train_normal_loaders, device, logger)
                dist.barrier()
                if gpd_type == 'hierarchical':
                    proto_bank.broadcast(src=0)
                else:
                    dist.broadcast(proto_bank.sub_centers, src=0)
                    dist.broadcast(proto_bank.counts, src=0)
            else:
                if gpd_type == 'hierarchical':
                    warm_start_hierarchical(model_meta, proto_bank,
                                            train_normal_loaders, device, logger)
                else:
                    warm_start_subcenters(model_meta, proto_bank,
                                          train_normal_loaders, device, logger)
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)

    if (not use_aux_modules) and resume_path:
        last_epoch, resume_full = load_resume_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, None, device,
            logger)
        start_epoch = last_epoch + 1 if resume_full else 1
        if dist_on:
            dist.barrier()

    if aux_target > 0:
        # Initialize after warm start/bank broadcast, on every rank.  A full
        # V2/V4 resume already contains the heads; only fill if they are zero.
        need_init = (not resume_full) or (
            float(model_meta.aux_heads.abs().sum().item()) == 0.0)
        if need_init:
            initialize_gac_heads(
                model_meta, proto_bank, logger if is_main else None)
        if dist_on:
            dist.barrier()

    group_size = cfg.DATALOADER.NUM_INSTANCE

    if start_epoch > epochs:
        if is_main:
            logger.info('Resume start_epoch {} > MAX_EPOCHS {}; nothing to do'
                        .format(start_epoch, epochs))
        return

    for epoch in range(start_epoch, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        for m in meter_ls:
            m.reset()
        hpd_id_meter.reset()
        hpd_scene_meter.reset()
        aux_meter.reset()
        corr_meter.reset()
        current_aux_weight = gac_aux_weight(cfg, epoch)
        current_corr_weight = corr_weight(cfg, epoch)
        corr_diag = None
        clip_count = 0
        last_grad_norm = 0.0
        evaluator.reset()
        scheduler.step(epoch)
        model.train()

        n_iter = 0
        for n_iter, (imgs, vid, camids) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            imgs = [img.to(device) for img in imgs]
            camids = [cam.to(device) for cam in camids]
            target = vid.to(device)

            num_modalities = len(imgs)
            target_rep = target.repeat(num_modalities)

            if (use_aux_modules and gpd_type == 'hierarchical' and is_main
                    and cfg.MODEL.GPD_DIAG_PERIOD > 0
                    and (n_iter + 1) % cfg.MODEL.GPD_DIAG_PERIOD == 0):
                diag = diagnose_gpd_gradients(
                    model_meta, imgs, target, camids, proto_bank, cfg,
                    device, group_size, loss_fn, aux_weight=current_aux_weight)
                logger.info(
                    'GPD-HPD diag: loss_id {:.4f}, loss_scene {:.4f}, '
                    'g_id {:.3e}, g_scene {:.3e}, g_base {:.3e}, '
                    'g_pca {:.3e}, cos {:.3f}, share id/scene/base/pca '
                    '{:.3f}/{:.3f}/{:.3f}/{:.3f}, g_cent_norm {:.3f}, '
                    'own_ratio {}, usage {}'.format(
                        diag['loss_id'], diag['loss_scene'], diag['g_id'],
                        diag['g_scene'], diag['g_base'], diag['g_pca'],
                        diag['cos'], diag['share_id'], diag['share_scene'],
                        diag['share_base'], diag['share_pca'],
                        diag['g_cent_norm'],
                        [round(x, 3) for x in diag['own_ratio']],
                        diag['usage']))
                logger.info(
                    'GPD-HPD diag logits: id pos/neg/entropy '
                    '{:.3f}/{:.3f}/{:.3f}, scene pos/neg/entropy '
                    '{:.3f}/{:.3f}/{:.3f}'.format(
                        diag['id_stats']['pos_mean'],
                        diag['id_stats']['neg_mean'],
                        diag['id_stats']['entropy'],
                        diag['scene_stats']['pos_mean'],
                        diag['scene_stats']['neg_mean'],
                        diag['scene_stats']['entropy']))
                if current_aux_weight > 0:
                    try:
                        logger.info(format_gac_diag(diag))
                    except Exception:
                        # Diagnostics are optional; never abort training on a
                        # diagnostic/log-format failure.
                        logger.exception(
                            'GAC diagnostic log failed; continuing training')

            with amp.autocast(enabled=True):
                if current_aux_weight > 0:
                    cls_score, global_feat, feat, aux_scores = model(
                        imgs, target, camids, return_aux=True)
                else:
                    cls_score, global_feat, feat = model(imgs, target, camids)

                loss, il, tl = loss_fn(cls_score, global_feat, target_rep)
                meter_ls[0].update(il.item(), target_rep.shape[0])
                meter_ls[1].update(tl.item(), target_rep.shape[0])

                if use_aux_modules:
                    loss_pca, ids, g_cent, centers = PCA(
                        global_feat,
                        target,
                        num_modalities=num_modalities,
                        group_size=group_size,
                    )
                    meter_ls[2].update(loss_pca.item(), max(1, target.shape[0] // group_size))

                    if gpd_type == 'hierarchical':
                        loss_gpd, loss_gpd_id, loss_gpd_scene, gpd_diag = GPD_HPD(
                            global_feat,
                            target,
                            proto_bank,
                            ids,
                            g_cent,
                            num_modalities=num_modalities,
                            tau=cfg.MODEL.GPD_TAU,
                            alpha=cfg.MODEL.GPD_SCENE_WEIGHT,
                            centers=centers,
                        )
                        hpd_id_meter.update(
                            loss_gpd_id.item(), max(1, target.shape[0] // group_size))
                        hpd_scene_meter.update(
                            loss_gpd_scene.item(), max(1, target.shape[0] // group_size))
                    elif gpd_type == 'subcenter':
                        loss_gpd = GPD_SC(
                            global_feat,
                            target,
                            proto_bank,
                            ids,
                            g_cent,
                            num_modalities=num_modalities,
                            tau=cfg.MODEL.GPD_TAU,
                        )
                    else:
                        loss_gpd = GPD(
                            global_feat,
                            target,
                            proto_bank,
                            ids,
                            g_cent,
                            num_modalities=num_modalities,
                            tau=cfg.MODEL.GPD_TAU,
                        )
                    meter_ls[3].update(loss_gpd.item(), max(1, target.shape[0] // group_size))

                    if 'pca' in loss_type:
                        loss += loss_pca * cfg.MODEL.PCA_LOSS_WEIGHT
                    if 'gpd' in loss_type:
                        loss += loss_gpd

                    if current_aux_weight > 0:
                        loss_aux = gac_aux_loss(
                            aux_scores, target_rep, num_modalities)
                        loss = loss + current_aux_weight * loss_aux
                        aux_meter.update(
                            loss_aux.item(), target_rep.shape[0])

            # V6: all-cross-modal, pair-equal, class-level alignment loss.
            # Inserted at loop scope, after the autocast block and outside the
            # GAC aux condition; corr_loss itself disables autocast and casts
            # the pre-BN CLS to fp32.
            if corr_enabled and current_corr_weight > 0:
                loss_corr, corr_diag = corr_loss(
                    global_feat, target_rep, camids,
                    num_modalities, tau=corr_tau)
                loss = loss + current_corr_weight * loss_corr
                corr_meter.update(loss_corr.item(), 1)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)
            last_grad_norm = float(grad_norm)
            if last_grad_norm > 1.0:
                clip_count += 1
            scaler.step(optimizer)
            scaler.update()

            if use_aux_modules:
                with torch.no_grad():
                    if gpd_type == 'hierarchical':
                        num_ids = ids.shape[0]
                        centers_all = torch.cat(centers, dim=0)  # [M*P, D]
                        mod_ids = torch.arange(
                            num_modalities, device=device).repeat_interleave(num_ids)
                        proto_bank.update(
                            ids, g_cent,
                            ids.repeat(num_modalities), mod_ids, centers_all,
                            ddp=cfg.MODEL.DIST_TRAIN,
                        )
                    elif gpd_type == 'subcenter':
                        num_ids = ids.shape[0]
                        centers_all = torch.cat(centers, dim=0)  # [M*P, D]
                        mod_ids = torch.arange(
                            num_modalities, device=device).repeat_interleave(num_ids)
                        proto_bank.update(
                            ids.repeat(num_modalities),
                            mod_ids,
                            centers_all,
                            ddp=cfg.MODEL.DIST_TRAIN,
                        )
                    else:
                        proto_bank.update(ids, g_cent, ddp=cfg.MODEL.DIST_TRAIN)

            acc = (cls_score.max(1)[1] == target_rep).float().mean()
            loss_meter.update(loss.item(), target.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize(local_rank)

            if is_main and (n_iter + 1) % log_period == 0:
                logger.info(
                    'Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, '
                    'Base Lr: {:.2e}, pre-clip grad norm: {:.3f}, '
                    'clip_count: {}, aux_w: {:.3f}, corr_w: {:.4f}'.format(
                        epoch,
                        (n_iter + 1),
                        len(train_loader),
                        loss_meter.avg,
                        acc_meter.avg,
                        scheduler._get_lr(epoch)[0],
                        last_grad_norm,
                        clip_count,
                        current_aux_weight,
                        current_corr_weight,
                    )
                )
                metrics = [f'{m.avg:.3f}' for m in meter_ls]
                if current_aux_weight > 0:
                    metrics.append(f'aux {aux_meter.avg:.3f}')
                if corr_enabled and current_corr_weight > 0:
                    metrics.append(f'corr {corr_meter.avg:.3f}')
                metrics_str = ', '.join(metrics)
                logger.info(f'Epoch[{epoch}] {metrics_str}')
                if corr_diag is not None:
                    corr_parts = []
                    for name, d in corr_diag.items():
                        corr_parts.append(
                            '{} loss {:.4f} valid {}/{} pos/valid {:.3f} '
                            'cos {:.3f}/{:.3f}'.format(
                                name, d['loss'], d['valid'], d['anchors'],
                                d['positives_per_valid'],
                                d['pos_cos'], d['neg_cos']))
                    logger.info('V6 corr diag: ' + ' | '.join(corr_parts))
                if gpd_type == 'hierarchical':
                    logger.info(
                        'Epoch[{}] HPD losses: id {:.4f}, scene {:.4f}, '
                        'alpha {:.2f}, own_ratio {}, usage {}, '
                        'g_cent_norm {:.3f}'.format(
                            epoch, hpd_id_meter.avg, hpd_scene_meter.avg,
                            cfg.MODEL.GPD_SCENE_WEIGHT,
                            [round(x, 3) for x in gpd_diag['own_ratio']]
                            if gpd_diag else None,
                            gpd_diag['usage'] if gpd_diag else None,
                            gpd_diag.get('g_cent_norm', float('nan'))
                            if gpd_diag else float('nan')))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if dist_on:
            if is_main:
                logger.info(
                    'Epoch {} done. Time per batch: {:.3f}[s] Total: {:.1f}[s]'.format(
                        epoch, time_per_batch, end_time - start_time
                    )
                )
        else:
            logger.info(
                'Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'.format(
                    epoch, time_per_batch, train_loader.batch_size / time_per_batch
                )
            )

        if is_main and use_aux_modules and gpd_type in ('subcenter', 'hierarchical'):
            with torch.no_grad():
                if gpd_type == 'hierarchical':
                    c = proto_bank.scene_bank.counts
                    updated = (c > 1).sum(dim=0)
                    mean_count = c.mean(dim=0)
                    logger.info(
                        'HPD scene bank: updated per modality {}, mean count per '
                        'modality {}, count min/mean/max {:.1f}/{:.1f}/{:.1f}'.format(
                            updated.tolist(),
                            [round(float(x), 1) for x in mean_count],
                            c.min().item(), c.mean().item(), c.max().item()))

                    P_id = F.normalize(
                        proto_bank.id_bank.prototypes.float(), dim=1)
                    P_scene = F.normalize(
                        proto_bank.scene_bank.sub_centers.float(), dim=2)
                    cos = torch.einsum('cd,ckd->ck', P_id, P_scene)
                    valid = ((P_scene.norm(dim=2) > 0)
                             & (P_id.norm(dim=1, keepdim=True) > 0))
                    per_mod_mean = [
                        float(cos[:, m][valid[:, m]].mean().item())
                        if valid[:, m].any() else float('nan')
                        for m in range(proto_bank.K)
                    ]
                    per_mod_min = [
                        float(cos[:, m][valid[:, m]].min().item())
                        if valid[:, m].any() else float('nan')
                        for m in range(proto_bank.K)
                    ]
                    pair_cos = []
                    for m in range(proto_bank.K):
                        for n in range(m + 1, proto_bank.K):
                            pair_valid = valid[:, m] & valid[:, n]
                            if pair_valid.any():
                                pair_cos.append(float(
                                    (P_scene[:, m] * P_scene[:, n]).sum(dim=1)[pair_valid].mean().item()))
                            else:
                                pair_cos.append(float('nan'))
                    mean_scene = F.normalize(P_scene.mean(dim=1), dim=1)
                    cos_mean = (P_id * mean_scene).sum(dim=1)
                    cos_mean_valid = cos_mean[P_id.norm(dim=1) > 0]
                    logger.info(
                        'HPD geometry: cos(P_id,P_scene) mean/min per '
                        'modality {}/{}, sub-center pair cos {}, '
                        'cos(P_id,mean P_scene) {:.3f}, '
                        'P_id norm min/max {:.3f}/{:.3f}'.format(
                            [round(x, 3) for x in per_mod_mean],
                            [round(x, 3) for x in per_mod_min],
                            [round(x, 3) for x in pair_cos],
                            float(cos_mean_valid.mean().item())
                            if cos_mean_valid.numel() else float('nan'),
                            float(proto_bank.id_bank.prototypes.norm(dim=1).min().item()),
                            float(proto_bank.id_bank.prototypes.norm(dim=1).max().item())))
                else:
                    c = proto_bank.counts
                    # counts start at 1 after warm start, so >1 means "updated at
                    # least once since initialization".
                    updated = (c > 1).sum(dim=0)
                    mean_count = c.mean(dim=0)
                    logger.info(
                        'GPD-SC bank: updated per modality {}, mean count per '
                        'modality {}, count min/mean/max {:.1f}/{:.1f}/{:.1f}'.format(
                            updated.tolist(),
                            [round(float(x), 1) for x in mean_count],
                            c.min().item(), c.mean().item(), c.max().item()))

        if epoch % checkpoint_period == 0 or epoch == epochs:
            if is_main:
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(
                    model_to_save.state_dict(),
                    os.path.join(
                        cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)
                    ),
                )
                # Full state for interruption/resume.  checkpoint_latest.pth
                # is the file passed to train.py --resume.
                save_resume_checkpoint(
                    os.path.join(cfg.OUTPUT_DIR, 'checkpoint_latest.pth'),
                    epoch, model, optimizer, scheduler, scaler,
                    proto_bank if use_aux_modules else None,
                )

        if epoch % eval_period == 0:
            model.eval()

            world_sz = dist.get_world_size() if dist_on else 1
            if dist_on:
                # Validation assigns whole modalities to ranks, so ranks execute
                # different numbers of forward passes (and rank 3 is idle with
                # three modalities). Synchronize rank-0 buffers once and use the
                # unwrapped module to avoid DDP's per-forward buffer collective.
                for buffer in model.module.buffers():
                    dist.broadcast(buffer, src=0)
                dist.barrier()
                eval_model = model.module
            else:
                eval_model = model

            for mode, (val_loader, num_query) in enumerate(zip(val_loaders, num_querys), start=1):
                evaluator.set_query_num(mode, num_query)

                if dist_on and (mode - 1) % world_sz != rank:
                    continue

                for img, vid, camid, camidt, modid, _ in val_loader:
                    with torch.no_grad():
                        img = img.to(device)
                        feat = eval_model(img, mode=mode)
                        evaluator.update((feat, vid, camid, modid), mode)

            evaluator.split_all()
            cmc, mAP, *_ = evaluator.compute()

            if cmc is not None:
                logger.info(f'Validation Results - Epoch: {epoch}')
                logger.info(f'mAP: {mAP:.2%}')
                for r in (1, 5, 10):
                    logger.info(f'CMC curve, Rank-{r:<2}: {cmc[r-1]:.2%}')
            torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loaders,
                 num_querys):
    device = 'cuda'
    logger = logging.getLogger('transreid.test')
    logger.info('Enter inferencing')

    domain_report_path = None
    if cfg.TEST.DETAILED_DOMAIN_EVAL:
        domain_report_path = os.path.join(
            cfg.OUTPUT_DIR or ".", "domain_protocol_results.csv")

    evaluator = R1_mAP_eval(
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING,
        top_k=cfg.TEST.TOP_K_EVAL,
        logger=logger,
        detailed_domain_eval=cfg.TEST.DETAILED_DOMAIN_EVAL,
        domain_report_path=domain_report_path,
    )

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    for mode, (val_loader, num_query) in enumerate(zip(val_loaders, num_querys), start=1):
        evaluator.set_query_num(mode, num_query)

        for img, vid, camid, camidt, modid, imgpath in val_loader:
            with torch.no_grad():
                img = img.to(device)

                feat = model(img, mode=mode)
                evaluator.update((feat, vid, camid, modid), mode)
                img_path_list.extend(imgpath)

    evaluator.split_all()
    cmc, mAP, *_ = evaluator.compute()

    logger.info('Validation Results ')
    logger.info('mAP: {:.2%}'.format(mAP))
    for r in [1, 5, 10]:
        logger.info('CMC curve, Rank-{:<3}:{:.2%}'.format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
