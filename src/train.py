import os, sys
import argparse, torch, re
import numpy as np
from scipy.sparse import load_npz
import h5py
import torch.optim as optim
import time, pickle
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import datetime
import torch.nn.functional as F
import torch.nn as nn
from scipy import sparse
import psutil, gc
import math

from model import build_model


# ============================================================
# 100 bp nascent RNA training with:
#   1. same-locus multi-cell batching
#   2. channel-balanced absolute loss
#   3. channel-balanced delta loss
#   4. per-strand validation logging
#   5. plateau-based LR schedule: 1e-4 -> 5e-5 -> 1e-5
# ============================================================

def parser_args():
    parser = argparse.ArgumentParser()

    # Model args kept from previous script
    parser.add_argument('--nheads', default=4, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--enc_layers', default=2, type=int)
    parser.add_argument('--batchsize', default=1, type=int)
    parser.add_argument('--accum_iter', default=2, type=int)
    parser.add_argument('--dec_layers', default=2, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--bins', type=int, default=600)
    parser.add_argument('--crop', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=58)
    parser.add_argument('-d', '--depth', type=int, default=4)
    parser.add_argument('--embed_dim', default=768, type=int)
    parser.add_argument('--return_embed', default=False, action='store_true')
    parser.add_argument('--return_local', default=True, action='store_false')

    # Checkpoint / output
    parser.add_argument('--init_ckpt', type=str, default='erna_1kb.pt', help='1 kb nascent RNA checkpoint used to initialize 100 bp fine-tuning.')
    parser.add_argument('--save_prefix', type=str, default='erna_100bp_same_locus_delta', help='Prefix for log and checkpoint files.')

    # Same-locus delta loss
    parser.add_argument('--lambda_delta', type=float, default=0.2, help='Weight for same-locus multi-cell delta loss.')
    parser.add_argument('--lr_list', type=str, default='1e-4,5e-5,1e-5', help='Comma-separated learning rates used by plateau scheduler.')
    parser.add_argument('--lr_patience', default=3, type=int, help='Number of non-improving epochs before reducing LR.')
    parser.add_argument('--lr_threshold', default=1e-4, type=float, help='Minimum validation improvement required to reset LR patience.')
    parser.add_argument('--early_stop_patience', default=5, type=int, help='Stop after this many non-improving epochs at the final LR.')
    parser.add_argument('--print_every', default=100, type=int)
    parser.add_argument('--validate_every', default=1, type=int)

    args = parser.parse_args()
    return args


def get_args():
    return parser_args()


def parse_lr_list(lr_string):
    return [float(x.strip()) for x in lr_string.split(',') if x.strip()]


class PlateauLRScheduler:
    def __init__(self, optimizer, lr_list, patience=3, threshold=1e-4, early_stop_patience=5):
        if len(lr_list) == 0:
            raise ValueError("lr_list cannot be empty.")
        self.optimizer = optimizer
        self.lr_list = lr_list
        self.patience = patience
        self.threshold = threshold
        self.early_stop_patience = early_stop_patience

        self.lr_idx = 0
        self.best = -np.inf
        self.bad_epochs = 0
        self.bad_epochs_at_min_lr = 0
        self.set_lr(self.lr_list[self.lr_idx])

    def set_lr(self, lr):
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return {
            "lr_idx": self.lr_idx,
            "best": self.best,
            "bad_epochs": self.bad_epochs,
            "bad_epochs_at_min_lr": self.bad_epochs_at_min_lr,
        }

    def step(self, score):
        """
        Returns:
            improved, lr_reduced, should_stop
        """
        improved = score > (self.best + self.threshold)
        lr_reduced = False
        should_stop = False

        if improved:
            self.best = score
            self.bad_epochs = 0
            self.bad_epochs_at_min_lr = 0
            return improved, lr_reduced, should_stop

        self.bad_epochs += 1

        if self.lr_idx < len(self.lr_list) - 1:
            if self.bad_epochs >= self.patience:
                self.lr_idx += 1
                self.set_lr(self.lr_list[self.lr_idx])
                self.bad_epochs = 0
                lr_reduced = True
        else:
            self.bad_epochs_at_min_lr += 1
            if self.bad_epochs_at_min_lr >= self.early_stop_patience:
                should_stop = True

        return improved, lr_reduced, should_stop


def split_dataset(seed=24):
    input_locs = np.load('/nfs/turbo/umms-drjieliu/usr/zzh/mutimodal_epcot/input_region_dup_250_noX.npy')
    dataset_size = input_locs.shape[0]
    indices = np.arange(dataset_size)
    valid_split = int(np.floor(dataset_size * 0.8))
    test_split = int(np.floor(dataset_size * 0.85))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices, valid_indices = indices[:valid_split], indices[test_split:]
    return train_indices, valid_indices


def norm_smooth(x_f, x_r, alpha=98):
    x_merge = np.vstack((x_f, x_r))
    pos = x_merge[x_merge > 0]
    if len(pos) == 0:
        raise ValueError("No positive values found for normalization.")
    scale = np.percentile(pos, alpha)

    x_f = np.arcsinh(x_f / scale)
    x_r = np.arcsinh(x_r / scale)

    return torch.cat(
        (
            torch.tensor(x_f, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(x_r, dtype=torch.float32).unsqueeze(-1),
        ),
        dim=-1,
    )


def load_bruseq(cl):
    tt_file = '/scratch/lhjiang/bru-seq/%s_bru_fwd_100bp_seq_cov.h5' % cl
    with h5py.File(tt_file, 'r') as hf:
        tt_fwd = np.array(hf['targets']).astype('float32')

    tt_file = '/scratch/lhjiang/bru-seq/%s_bru_rev_100bp_seq_cov.h5' % cl
    with h5py.File(tt_file, 'r') as hf:
        tt_rev = np.array(hf['targets']).astype('float32')

    return norm_smooth(tt_fwd, tt_rev)


def load_ttseq(cl):
    tt_file = '/scratch/lhjiang/tt-seq/data/%s_tt_fwd_100bp_seq_cov.h5' % cl
    with h5py.File(tt_file, 'r') as hf:
        tt_fwd = np.array(hf['targets']).astype('float32')

    tt_file = '/scratch/lhjiang/tt-seq/data/%s_tt_rev_100bp_seq_cov.h5' % cl
    with h5py.File(tt_file, 'r') as hf:
        tt_rev = np.array(hf['targets']).astype('float32')

    return norm_smooth(tt_fwd, tt_rev)


def load_erna(task, cl, required=False):
    try:
        if task == 'tt':
            return load_ttseq(cl)
        if task == 'bru':
            return load_bruseq(cl)
        raise ValueError(f"Unknown task: {task}")
    except Exception as e:
        msg = f"Could not load {task} label for {cl}: {repr(e)}"
        if required:
            raise RuntimeError(msg)
        print("Warning:", msg, flush=True)
        return None


def init_distributed():
    dist_url = "env://"
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    print("World Size:", world_size, flush=True)
    print("Local Rank:", local_rank, flush=True)

    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=5400),
    )
    torch.cuda.set_device(local_rank)
    dist.barrier()


def pad_seq_matrix(matrix, pad_len=300):
    paddings = np.zeros((1, 4, pad_len)).astype('int8')
    dmatrix = np.concatenate((paddings, matrix[:, :, -pad_len:]), axis=0)[:-1, :, :]
    umatrix = np.concatenate((matrix[:, :, :pad_len], paddings), axis=0)[1:, :, :]
    return np.concatenate((dmatrix, matrix, umatrix), axis=2)


def pad_signal_matrix(matrix, pad_len=300):
    paddings = np.zeros(pad_len).astype('float32')
    dmatrix = np.vstack((paddings, matrix[:, -pad_len:]))[:-1, :]
    umatrix = np.vstack((matrix[:, :pad_len], paddings))[1:, :]
    return np.hstack((dmatrix, matrix, umatrix))


def load_dnase(dnase_seq, normalize=False):
    # normalize is kept for API compatibility; previous code did not use it.
    dnase_seq = pad_signal_matrix(dnase_seq.astype('float32').toarray().reshape(-1, 1000))
    return sparse.csr_matrix(dnase_seq)


def load_ref_genome(chr):
    ref_path = '/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/3D/data/ref_genome/'
    ref_file = os.path.join(ref_path, 'chr%s.npz' % chr)
    ref_gen_data = load_npz(ref_file).toarray().reshape(4, -1, 1000).swapaxes(0, 1)
    return torch.tensor(pad_seq_matrix(ref_gen_data))


class indiceDataset(Dataset):
    def __init__(self, indices):
        self.loc_indices = indices
        self.num = self.loc_indices.shape[0]

    def __getitem__(self, index):
        return self.loc_indices[index]

    def __len__(self):
        return self.num


def convert_atac_to_float32(atac_data):
    for cl in atac_data:
        for chrom in atac_data[cl]:
            mat = atac_data[cl][chrom]
            if sparse.issparse(mat):
                if mat.dtype != np.float32:
                    atac_data[cl][chrom] = mat.astype(np.float32)
            else:
                if mat.dtype != np.float32:
                    atac_data[cl][chrom] = mat.astype(np.float32, copy=False)
    return atac_data


def apply_atac_depth_dropout(
    x,
    p_scale=0.2,
    p_mask=0.1,
    scale_range=(0.5, 0.9),
    mask_frac=0.02,
    region_size=5000,
):
    """
    x: (B, 5, L), ATAC is channel 4.
    This is optional and disabled by default in experiment 2.
    """
    # x = x.clone()

    # if torch.rand(1, device=x.device).item() < p_scale:
    #     scale = torch.empty(1, device=x.device).uniform_(*scale_range)
    #     x[:, 4:5, :] *= scale
    if torch.rand(1, device=x.device).item() < p_scale:
        lo, hi = math.log(scale_range[0]), math.log(scale_range[1])
        scale = torch.exp(torch.empty(1, device=x.device).uniform_(lo, hi))
        x[:, 4:5, :] *= scale

    if torch.rand(1, device=x.device).item() < p_mask:
        L = x.shape[-1]
        num_mask = max(1, int(L * mask_frac / region_size))

        for _ in range(num_mask):
            start = torch.randint(
                0, max(1, L - region_size), (1,), device=x.device
            ).item()
            end = start + region_size
            x[:, 4:5, start:end] = 0

    return x


# ============================================================
# Loss functions
# ============================================================

def per_sample_channel_balanced_smooth_l1(pred, target, scale_floor=0.15):
    """
    pred, target: (B, L, C)
    Returns:
        per_sample_loss: (B,)
    Each output channel/strand contributes equally.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

    B = pred.shape[0]
    C = pred.shape[-1]

    channel_losses = []
    for c in range(C):
        loss_map = F.smooth_l1_loss(
            pred[..., c],
            target[..., c],
            reduction='none',
        )
        channel_losses.append(loss_map.reshape(B, -1).mean(dim=1))

    return torch.stack(channel_losses, dim=1).mean(dim=1)
    # per_channel = F.smooth_l1_loss(pred, target, reduction='none').mean(dim=1)   # (B, C)

    # scale = target.detach().std(dim=1)                                            # (B, C)
    # scale = torch.clamp(scale, min=scale_floor)
    # return (per_channel / scale).mean(dim=1)


def channel_balanced_abs_loss_sum(pred, target, sample_weights=None):
    """
    Weighted sum of per-sample, channel-balanced SmoothL1 losses.
    This mirrors the original code that summed losses across cells/tasks
    and normalized by a global denominator.
    """
    per_sample = per_sample_channel_balanced_smooth_l1(pred, target)

    if sample_weights is not None:
        sample_weights = sample_weights.to(per_sample.device, dtype=per_sample.dtype)
        return (per_sample * sample_weights).sum()

    return per_sample.sum()


def channel_balanced_delta_loss_mean(pred, target, sample_weights=None):
    """
    Same-locus pairwise delta loss:
        pred_i - pred_j should match target_i - target_j.

    pred, target: (B, L, C), where B is number of cells for the same locus.
    Returns:
        weighted mean over unordered cell pairs.
    """
    B = pred.shape[0]
    if B < 2:
        return pred.new_tensor(0.0)

    pair_idx = torch.triu_indices(B, B, offset=1, device=pred.device)
    i = pair_idx[0]
    j = pair_idx[1]

    pred_delta = pred[i] - pred[j]
    true_delta = target[i] - target[j]

    # per_pair = F.smooth_l1_loss(pred_delta, true_delta, reduction='none').flatten(1).mean(1)
    per_pair = per_sample_channel_balanced_smooth_l1(pred_delta, true_delta)

    if sample_weights is not None:
        sample_weights = sample_weights.to(pred.device, dtype=per_pair.dtype)
        pair_w = torch.sqrt(sample_weights[i] * sample_weights[j])
        return (per_pair * pair_w).sum() / (pair_w.sum() + 1e-8)

    return per_pair.mean()


# ============================================================
# Streaming Pearson for exact distributed validation
# ============================================================

class RunningPearson:
    def __init__(self):
        self.n = 0
        self.sx = 0.0
        self.sy = 0.0
        self.sx2 = 0.0
        self.sy2 = 0.0
        self.sxy = 0.0

    def update(self, x, y):
        x = x.reshape(-1).astype(np.float64, copy=False)
        y = y.reshape(-1).astype(np.float64, copy=False)

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        self.n += x.size
        self.sx += x.sum()
        self.sy += y.sum()
        self.sx2 += (x * x).sum()
        self.sy2 += (y * y).sum()
        self.sxy += (x * y).sum()

    def add_state(self, state):
        n, sx, sy, sx2, sy2, sxy = state
        self.n += n
        self.sx += sx
        self.sy += sy
        self.sx2 += sx2
        self.sy2 += sy2
        self.sxy += sxy

    def state(self):
        return (self.n, self.sx, self.sy, self.sx2, self.sy2, self.sxy)

    def corr(self):
        n = self.n
        if n == 0:
            return np.nan
        num = n * self.sxy - self.sx * self.sy
        denx = n * self.sx2 - self.sx * self.sx
        deny = n * self.sy2 - self.sy * self.sy
        den = np.sqrt(max(denx, 0.0) * max(deny, 0.0)) + 1e-12
        return num / den


def init_rp_state(valid_cell_dict):
    rp = {}
    for task, cells in valid_cell_dict.items():
        rp[task] = {cl: [RunningPearson(), RunningPearson()] for cl in cells}
    return rp


def rp_to_state_dict(rp):
    out = {}
    for task in rp:
        out[task] = {}
        for cl in rp[task]:
            out[task][cl] = [rp[task][cl][0].state(), rp[task][cl][1].state()]
    return out


def combine_state_dicts(state_dicts, valid_cell_dict):
    rp = init_rp_state(valid_cell_dict)
    for sd in state_dicts:
        if sd is None:
            continue
        for task in sd:
            for cl in sd[task]:
                for strand_i in [0, 1]:
                    rp[task][cl][strand_i].add_state(sd[task][cl][strand_i])
    return rp


def score_dict_from_rp(rp):
    scores = {}
    for task in rp:
        scores[task] = {}
        for cl in rp[task]:
            fwd = rp[task][cl][0].corr()
            rev = rp[task][cl][1].corr()
            mean = np.nanmean([fwd, rev])
            scores[task][cl] = {"fwd": fwd, "rev": rev, "mean": mean}
    return scores


def format_scores(scores, task, cells, key):
    vals = []
    for cl in cells:
        if task in scores and cl in scores[task]:
            vals.append(round(float(scores[task][cl][key]), 4))
        else:
            vals.append(np.nan)
    return ','.join(str(v) for v in vals)


def final_train_score(scores, train_cell_dict):
    vals = []
    for task, cells in train_cell_dict.items():
        for cl in cells:
            if task in scores and cl in scores[task]:
                vals.append(scores[task][cl]["mean"])
    vals = [v for v in vals if np.isfinite(v)]
    if len(vals) == 0:
        return -np.inf
    return float(np.sum(vals))


# ============================================================
# Main
# ============================================================

def main(gpu, args):
    torch.cuda.set_device(gpu)
    rank = gpu
    world_size = int(os.environ['WORLD_SIZE'])

    # Training cells/tasks
    cell_dict = {
        'tt': ['K562', 'pc3', 'Jurkat', 'HeLa', 'H1', 'MCF10A'],
        'bru': ['GM12878', 'K562', 'HepG2', 'MCF-7', 'IMR-90', 'pc3', 'Calu3', 'Caco2','panc1','A673'],
    }

    # Held-out cells for monitoring only.
    heldout_cell_dict_requested = {
        'tt': ['HCT116'],
        'bru': ['HCT116', 'MCF10A'],
    }

    task_to_head = {'tt': 0, 'bru': 1}

    # Load checkpoint.
    ckpt = torch.load(args.init_ckpt, map_location='cpu')
    filtered = {
        k: v for k, v in ckpt.items()
        if not (
            k.startswith("head_tt.")
            or k.startswith("head_bru.")
        )
    }

    model = build_model(args)
    missing, unexpected = model.load_state_dict(filtered, strict=False)

    if rank == 0:
        print("Missing keys:", missing, flush=True)
        print("Unexpected keys:", unexpected, flush=True)

    model.cuda(gpu)
    model = DDP(model, find_unused_parameters=True, device_ids=[gpu])
    model.train()

    lr_list = parse_lr_list(args.lr_list)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_list[0],
        weight_decay=1e-6,
    )
    scheduler = PlateauLRScheduler(
        optimizer=optimizer,
        lr_list=lr_list,
        patience=args.lr_patience,
        threshold=args.lr_threshold,
        early_stop_patience=args.early_stop_patience,
    )

    log_file = f"{args.save_prefix}.txt"
    best_ckpt_file = f"{args.save_prefix}.pt"

    m = model.module
    if rank == 0:
        with open(log_file, 'w') as f:
            f.write("100bp same-locus multi-cell delta training\n")
            f.write(f"init_ckpt: {args.init_ckpt}\n")
            f.write(f"lambda_delta: {args.lambda_delta}\n")
            f.write(f"lr_list: {lr_list}\n")
            f.write(f"lr_patience: {args.lr_patience}\n")
            f.write(f"lr_threshold: {args.lr_threshold}\n")
            f.write(f"early_stop_patience: {args.early_stop_patience}\n")

        for n, p in model.named_parameters():
            print(n, p.requires_grad, flush=True)
        print("Trainable parameters:",
              sum([p.numel() if p.requires_grad else 0 for _, p in model.named_parameters()]),
              flush=True)
        print('load data start', flush=True)

    # Load ATAC data.
    with open('/nfs/turbo/umms-drjieliu/usr/zzh/mutimodal_epcot/atac_bw/train_atac_merge_fp16.pickle', 'rb') as f:
        atac_data = pickle.load(f)
    with open('/nfs/turbo/umms-drjieliu/usr/lhjiang/proj_data/atac-seq/train_HeLa_dnase.pickle', 'rb') as f:
        atac_data['HeLa'] = pickle.load(f)
    with open('/nfs/turbo/umms-drjieliu/usr/lhjiang/proj_data/atac-seq/train_Jurkat_dnase.pickle', 'rb') as f:
        atac_data['Jurkat'] = pickle.load(f)
    with open('/scratch/drjieliu_root/drjieliu/lhjiang/proj_data/atac-seq/MCF10A_atac.pickle','rb') as f:
        atac_data['MCF10A'] = pickle.load(f)
    with open('/scratch/drjieliu_root/drjieliu/lhjiang/proj_data/atac-seq/encode/IMR-90_atac.pickle','rb') as f:
        atac_data['IMR-90'] = pickle.load(f)

    atac_data['MCF10A'] = {
        chrom: load_dnase(atac_data['MCF10A'][chrom])
        for chrom in atac_data['MCF10A'].keys()
    }
    atac_data['IMR-90'] = {
        chrom: load_dnase(atac_data['IMR-90'][chrom])
        for chrom in atac_data['IMR-90'].keys()
    }

    atac_data = convert_atac_to_float32(atac_data)

    if rank == 0:
        print("ATAC cells available:", sorted(list(atac_data.keys())), flush=True)

    # Load reference genome.
    ref_data = {}
    chroms = [i for i in range(1, 23)]
    for chr in chroms:
        ref_data[chr] = load_ref_genome(chr)

    # Load RNA labels.
    rna_data = {'tt': {}, 'bru': {}}

    for task in ['tt', 'bru']:
        for cl in cell_dict[task]:
            rna_data[task][cl] = load_erna(task, cl, required=True)

    # Load held-out labels if available.
    heldout_cell_dict = {'tt': [], 'bru': []}
    for task, cells in heldout_cell_dict_requested.items():
        for cl in cells:
            if cl in rna_data[task]:
                continue
            arr = load_erna(task, cl, required=False)
            if arr is not None:
                rna_data[task][cl] = arr
                heldout_cell_dict[task].append(cl)

    # Held-out validation also requires ATAC.
    for task in list(heldout_cell_dict.keys()):
        kept = []
        for cl in heldout_cell_dict[task]:
            if cl in atac_data:
                kept.append(cl)
            else:
                if rank == 0:
                    print(
                        f"Warning: skipping held-out validation for {task} {cl} "
                        f"because ATAC is not present in atac_data.",
                        flush=True,
                    )
        heldout_cell_dict[task] = kept

    input_locs = np.load('input_region_dup_250_noX.npy')
    region_index = np.load('index_region_dup_250_noX.npy')

    if rank == 0:
        print("input_locs size:", np.shape(input_locs), flush=True)
        print("heldout_cell_dict actually used:", heldout_cell_dict, flush=True)

    def load_data(lidx, cl, task, atac_merge=False):
        chrom, s, e = input_locs[lidx]

        if cl not in atac_data:
            raise KeyError(f"{cl} not found in atac_data.")
        if chrom not in atac_data[cl]:
            raise KeyError(f"chrom {chrom} not found in atac_data[{cl}].")

        mat = atac_data[cl][chrom][s:e]
        if sparse.issparse(mat):
            tmp_atac = torch.tensor(mat.toarray(), dtype=torch.float32).unsqueeze(1)
        else:
            tmp_atac = torch.tensor(np.asarray(mat, dtype=np.float32), dtype=torch.float32).unsqueeze(1)

        input_tensor = torch.cat((ref_data[chrom][s:e], tmp_atac), dim=1).unsqueeze(0)
        input_tensor = input_tensor[:, :, :, 300:1300]

        if atac_merge:
            atac = input_tensor[:, :, 4:5, :]
            B, L, C_atac, W = atac.shape
            atac_100bp = atac.view(B, L, C_atac, W // 10, 10).mean(dim=-1)
            atac_smooth = atac_100bp.repeat_interleave(10, dim=-1)
            input_tensor[:, :, 4:5, :] = atac_smooth

        B, L, C, W = input_tensor.shape
        input_tensor = input_tensor.permute(0, 2, 1, 3).contiguous()
        input_tensor = input_tensor.view(B, C, L * W).cuda(gpu, non_blocking=True)

        rnaidx = region_index[lidx]
        if cl not in rna_data[task]:
            raise KeyError(f"{cl} not found in rna_data[{task}].")

        rna_label = (
            rna_data[task][cl][rnaidx:rnaidx + 1, args.crop * 10:-args.crop * 10, :]
            .float()
            .cuda(gpu, non_blocking=True)
        )

        return input_tensor, rna_label

    def load_task_batch(lidx, task, cells):
        inputs = []
        targets = []
        used_cells = []

        for cl in cells:
            train_input, train_target = load_data(lidx, cl, task)
            inputs.append(train_input)
            targets.append(train_target)
            used_cells.append(cl)

        batch_input = torch.cat(inputs, dim=0)
        batch_target = torch.cat(targets, dim=0)

        return batch_input, batch_target, used_cells

    def prepare_dataloader(dataset: Dataset, shuffle=True):
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        return DataLoader(
            dataset,
            batch_size=args.batchsize,
            pin_memory=True,
            sampler=sampler,
        )

    train_indices, valid_indices = split_dataset()
    if rank == 0:
        print("train/valid sizes:", train_indices.shape[0], valid_indices.shape[0], flush=True)

    train_loader = prepare_dataloader(indiceDataset(train_indices), shuffle=True)
    valid_loader = prepare_dataloader(indiceDataset(valid_indices), shuffle=False)

    tt_cell_depths = {
        'K562': 1.0,
        'HeLa': 0.5,  # low sequencing depth
        'H1': 1.0,
        'pc3': 1.0,
        'Jurkat': 1.0,
        'MCF10A': 1.0,
    }
    total = sum(tt_cell_depths.values())
    n_tt_cells = len(cell_dict['tt'])
    tt_cell_weights = {cl: d * n_tt_cells / total for cl, d in tt_cell_depths.items()}

    task_w = {'tt': 1, 'bru': 1}
    denom = sum(task_w[t] * len(cell_dict[t]) for t in task_w)

    def sample_weights_for_task(task, used_cells, device):
        if task == 'tt':
            w = [tt_cell_weights[cl] for cl in used_cells]
        else:
            w = [1.0 for _ in used_cells]
        return torch.tensor(w, dtype=torch.float32, device=device)

    valid_cell_dict = {}
    for task in ['tt', 'bru']:
        valid_cell_dict[task] = list(cell_dict[task]) + list(heldout_cell_dict[task])

    def model_valid_streaming(model):
        if rank == 0:
            print("validation step: streaming Pearson by task/cell/strand", flush=True)

        rp = init_rp_state(valid_cell_dict)

        model.eval()
        with torch.no_grad():
            for step, idx_x in enumerate(valid_loader):
                for idx_single in idx_x.view(-1):
                    vidx = int(idx_single.item())

                    for task, cells in valid_cell_dict.items():
                        head_i = task_to_head[task]
                        for cl in cells:
                            # Skip if ATAC missing.
                            if cl not in atac_data:
                                continue

                            valid_input, rna_label = load_data(vidx, cl, task)
                            out = model(valid_input)[head_i]

                            pred = out.detach().cpu().numpy()
                            targ = rna_label.detach().cpu().numpy()

                            for strand_i in range(min(2, pred.shape[-1])):
                                rp[task][cl][strand_i].update(pred[..., strand_i], targ[..., strand_i])

        return rp_to_state_dict(rp)

    # Training loop.
    best_score = -np.inf
    optimizer.zero_grad(set_to_none=True)

    should_stop_global = False

    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        dist.barrier()

        epoch_loss = 0.0
        epoch_abs_loss = 0.0
        epoch_delta_loss = 0.0
        n_train_steps = 0

        max_steps = (
            train_indices.shape[0] // world_size + 1
            if train_indices.shape[0] % world_size
            else train_indices.shape[0] // world_size
        )

        if rank == 0:
            print(f"Epoch {epoch} start | lr={scheduler.get_lr():.6g} | max_steps={max_steps}", flush=True)

        for step, idx_x in enumerate(train_loader):
            tts = time.time()

            for idx_single in idx_x.view(-1):
                tidx = int(idx_single.item())

                total_loss_this_locus = torch.tensor(0.0, device=gpu)
                abs_loss_this_locus = torch.tensor(0.0, device=gpu)
                delta_loss_this_locus = torch.tensor(0.0, device=gpu)

                for task in ['tt', 'bru']:
                    cells = cell_dict[task]
                    head_i = task_to_head[task]

                    batch_input, batch_target, used_cells = load_task_batch(
                        tidx,
                        task,
                        cells,
                    )

                    output = model(batch_input)[head_i]

                    sample_w = sample_weights_for_task(task, used_cells, output.device)
                    weight_sum = sample_w.sum()

                    # Channel-balanced absolute loss.
                    abs_sum = channel_balanced_abs_loss_sum(
                        output,
                        batch_target,
                        sample_weights=sample_w,
                    )
                    abs_contrib = task_w[task] * abs_sum / denom

                    # Same-locus multi-cell delta loss.
                    if args.lambda_delta > 0:
                        delta_mean = channel_balanced_delta_loss_mean(
                            output,
                            batch_target,
                            sample_weights=sample_w,
                        )
                        delta_contrib = args.lambda_delta * task_w[task] * weight_sum * delta_mean / denom
                    else:
                        delta_contrib = torch.tensor(0.0, device=output.device)

                    task_loss = abs_contrib + delta_contrib
                    (task_loss / args.accum_iter).backward() 

                    total_loss_this_locus = total_loss_this_locus + task_loss.detach()
                    abs_loss_this_locus = abs_loss_this_locus + abs_contrib.detach()
                    delta_loss_this_locus = delta_loss_this_locus + delta_contrib.detach()


                epoch_loss += float(total_loss_this_locus.detach().cpu())
                epoch_abs_loss += float(abs_loss_this_locus.detach().cpu())
                epoch_delta_loss += float(delta_loss_this_locus.detach().cpu())
                n_train_steps += 1

            if ((step + 1) % args.accum_iter == 0) or (step + 1 == max_steps):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if rank == 0 and args.print_every > 0 and step % args.print_every == 0:
                mean_loss = epoch_loss / max(n_train_steps, 1)
                mean_abs = epoch_abs_loss / max(n_train_steps, 1)
                mean_delta = epoch_delta_loss / max(n_train_steps, 1)
                print(
                    f"Epoch {epoch} step {step}/{max_steps} "
                    f"time={time.time() - tts:.2f}s "
                    f"loss={mean_loss:.6f} abs={mean_abs:.6f} delta={mean_delta:.6f}",
                    flush=True,
                )

            if rank == 0 and step % 2000 == 0:
                with open(log_file, 'a') as f:
                    f.write(
                        f"Epoch: {epoch}, step: {step}, "
                        f"train_loss: {epoch_loss / max(n_train_steps, 1):.8f}, "
                        f"abs_loss: {epoch_abs_loss / max(n_train_steps, 1):.8f}, "
                        f"delta_loss: {epoch_delta_loss / max(n_train_steps, 1):.8f}, "
                        f"lr: {scheduler.get_lr():.8g}\n"
                    )

        dist.barrier()

        if rank == 0:
            print("Finished epoch", epoch, flush=True)

        # Validation.
        run_validation = ((epoch + 1) % args.validate_every == 0) or (epoch == args.epochs - 1)
        if run_validation:
            local_state = model_valid_streaming(model)
        else:
            local_state = None

        gathered_states = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_states, local_state)

        if rank == 0:
            if run_validation:
                rp_combined = combine_state_dicts(gathered_states, valid_cell_dict)
                scores = score_dict_from_rp(rp_combined)
                finalscore = final_train_score(scores, cell_dict)

                epoch_mean_loss = epoch_loss / max(n_train_steps, 1)
                epoch_mean_abs = epoch_abs_loss / max(n_train_steps, 1)
                epoch_mean_delta = epoch_delta_loss / max(n_train_steps, 1)

                improved, lr_reduced, should_stop = scheduler.step(finalscore)

                with open(log_file, 'a') as f:
                    f.write(f"\nEpoch {epoch} summary\n")
                    f.write(f"lr: {scheduler.get_lr():.8g}\n")
                    f.write(f"train_loss: {epoch_mean_loss:.8f}\n")
                    f.write(f"abs_loss: {epoch_mean_abs:.8f}\n")
                    f.write(f"delta_loss: {epoch_mean_delta:.8f}\n")
                    f.write(f"final_train_score: {finalscore:.6f}\n")
                    f.write(f"best_score_scheduler: {scheduler.best:.6f}\n")
                    f.write(f"improved: {improved}, lr_reduced: {lr_reduced}, should_stop: {should_stop}\n")

                    for task, cells in cell_dict.items():
                        f.write(f"{task} train cells: {','.join(cells)}\n")
                        f.write(f"{task} train mean: {format_scores(scores, task, cells, 'mean')}\n")
                        f.write(f"{task} train fwd:  {format_scores(scores, task, cells, 'fwd')}\n")
                        f.write(f"{task} train rev:  {format_scores(scores, task, cells, 'rev')}\n")

                    for task, cells in heldout_cell_dict.items():
                        if len(cells) > 0:
                            f.write(f"{task} heldout cells: {','.join(cells)}\n")
                            f.write(f"{task} heldout mean: {format_scores(scores, task, cells, 'mean')}\n")
                            f.write(f"{task} heldout fwd:  {format_scores(scores, task, cells, 'fwd')}\n")
                            f.write(f"{task} heldout rev:  {format_scores(scores, task, cells, 'rev')}\n")

                    f.write("\n")

                print(f"Epoch {epoch} final_train_score={finalscore:.6f} lr={scheduler.get_lr():.6g}", flush=True)

                if finalscore > best_score + args.lr_threshold:
                    best_score = finalscore
                    torch.save(model.module.state_dict(), best_ckpt_file)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model": model.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "best_score": best_score,
                            "args": vars(args),
                        },
                        f"{args.save_prefix}_full_checkpoint.pt",
                    )
                    print(f"Saved best checkpoint: {best_ckpt_file}", flush=True)

            else:
                should_stop = False

            control = [scheduler.get_lr(), bool(should_stop)]
        else:
            control = [None, None]

        # Broadcast LR and stop decision to all ranks.
        dist.broadcast_object_list(control, src=0)
        new_lr, should_stop_global = control

        for group in optimizer.param_groups:
            group['lr'] = new_lr

        if should_stop_global:
            if rank == 0:
                print("Early stopping triggered.", flush=True)
            break

        dist.barrier()


if __name__ == "__main__":
    args = get_args()
    init_distributed()
    main(int(os.environ['LOCAL_RANK']), args)
