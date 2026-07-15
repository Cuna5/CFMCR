"""Step-1 diagnostics for the degradation-aware noise-bridge CFM.

This script answers two questions WITHOUT changing any training / inference
core code or any hyper-parameter:

  Diagnostic A (oracle upper bound)
    Run 1-step inference twice on the test set:
      * REAL   : noise scale = gamma_hat  (predicted by the gamma-head, the
                 scale actually used at inference)
      * ORACLE : noise scale = gamma_train (computed on the fly from the test
                 set's |x_cloudy - x_clean|, the exact training-time target)
    The test set has clean GT, so gamma_train is computable. This is an oracle
    diagnostic, NOT a reportable metric. It tells you the ceiling of the
    spatial bridge IF gamma were perfect.

  Diagnostic B (visualization + ranges)
    For the thickest-cloud samples, dump gamma_hat and gamma_train heatmaps and
    print the per-pixel gamma range in the clear region vs the cloud region.
    Ideal: clear ~ noise_sigma * noise_sigma_floor (tiny), cloud ~ noise_sigma.

Both diagnostics reuse the model's own components (loss_fn._get_degradation_prob,
sampler._predict_noise_scale, sampler._one_step internals, model.img_metrics,
EMA scope, decode_first_stage), so the numbers are directly comparable to a
normal `--test` run.

Usage (on the GPU server, in the repo root):

    python tools/diagnose_gamma.py \
        --base configs/example_training/cuhk_noise_bridge_cfm.yaml \
        --ckpt /path/to/your_checkpoint.ckpt \
        --device cuda:0 \
        --num-vis 8 \
        --out logs/gamma_diag

You can also append dotlist overrides (same syntax as main.py), e.g. to fix a
data path:
    data.params.test.params.datasets_dir=/data1/.../CUHK-CR1
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch

# Headless plotting.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from omegaconf import OmegaConf  # noqa: E402

# Make the repo root importable when run as `python tools/diagnose_gamma.py`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sgm.util import append_dims, instantiate_from_config  # noqa: E402


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _rgb01(model, img_chw_or_bchw):
    """Return an HxWx3 numpy array in [0, 1] from a [-1, 1] image tensor.

    Uses the first three channels (R, G, B) of the CUHK RGBNIR stack, which is
    intuitive for a heatmap overlay (independent of nir_to_rgb reordering).
    """
    x = img_chw_or_bchw
    if x.ndim == 4:
        x = x[0]
    x01 = model.scale_01(x).detach().float().cpu()
    rgb = x01[:3].clamp(0, 1).numpy()
    return np.transpose(rgb, (1, 2, 0))


class RegionGammaStats:
    """True per-pixel min / mean / max of gamma over a boolean region."""

    def __init__(self):
        self.min = math.inf
        self.max = -math.inf
        self.sum = 0.0
        self.count = 0

    def update(self, values: torch.Tensor):
        if values.numel() == 0:
            return
        v = values.detach().float()
        self.min = min(self.min, float(v.min()))
        self.max = max(self.max, float(v.max()))
        self.sum += float(v.sum())
        self.count += int(v.numel())

    def summary(self):
        mean = self.sum / self.count if self.count else float("nan")
        return {
            "min": self.min if self.count else float("nan"),
            "mean": mean,
            "max": self.max if self.count else float("nan"),
        }


class MetricAccumulator:
    """Running mean of non-NaN metric values (mirrors avg_img_metrics)."""

    def __init__(self):
        self.sum = {}
        self.count = {}

    def add(self, metrics: dict):
        for k, v in metrics.items():
            if not isinstance(v, (int, float)):
                continue
            if isinstance(v, float) and math.isnan(v):
                continue
            self.sum[k] = self.sum.get(k, 0.0) + float(v)
            self.count[k] = self.count.get(k, 0) + 1

    def value(self):
        return {k: self.sum[k] / self.count[k] for k in self.sum}


def _metric_dict(metrics):
    """Convert evaluator outputs to plain floats for CSV/JSON export."""
    out = {}
    for key, value in metrics.items():
        if isinstance(value, tuple):
            value = value[0]
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            value = value.item()
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _safe_array_stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
    }


def _average_ranks(values):
    """Average ranks with deterministic tie handling (SciPy-free)."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _safe_spearman(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


class CalibrationAccumulator:
    """Streaming formal gamma-head calibration statistics."""

    def __init__(self, gamma_max, saturation_threshold=0.095, hist_bins=4096):
        self.gamma_max = max(float(gamma_max), 1e-8)
        self.saturation_threshold = float(saturation_threshold)
        self.hist = np.zeros(int(hist_bins), dtype=np.int64)
        self.hist_edges = np.linspace(
            0.0, self.gamma_max, int(hist_bins) + 1, dtype=np.float64
        )
        self.mae_sum = 0.0
        self.mae_count = 0
        self.saturation_count = 0
        self.saturation_total = 0
        self.image_spearman = []

    def update(self, gamma_hat, gamma_train, clear_mask=None):
        ghat = np.asarray(gamma_hat, dtype=np.float64).reshape(-1)
        gtrain = np.asarray(gamma_train, dtype=np.float64).reshape(-1)
        valid = np.isfinite(ghat) & np.isfinite(gtrain)

        if valid.any():
            abs_error = np.abs(ghat[valid] - gtrain[valid])
            self.mae_sum += float(abs_error.sum())
            self.mae_count += int(abs_error.size)
            self.saturation_count += int(
                np.count_nonzero(ghat[valid] >= self.saturation_threshold)
            )
            self.saturation_total += int(valid.sum())

        spearman = _safe_spearman(ghat, gtrain)
        self.image_spearman.append(spearman)

        if clear_mask is not None:
            clear = np.asarray(clear_mask, dtype=bool).reshape(-1)
            clear_values = ghat[valid & clear]
            if clear_values.size:
                clear_values = np.clip(clear_values, 0.0, self.gamma_max)
                hist, _ = np.histogram(clear_values, bins=self.hist_edges)
                self.hist += hist
        return spearman

    def _hist_quantile(self, q):
        total = int(self.hist.sum())
        if total == 0:
            return float("nan")
        target = max(1, int(math.ceil(float(q) * total)))
        idx = int(np.searchsorted(np.cumsum(self.hist), target, side="left"))
        idx = min(idx, len(self.hist_edges) - 2)
        return float(0.5 * (self.hist_edges[idx] + self.hist_edges[idx + 1]))

    def summary(self):
        spearman = np.asarray(self.image_spearman, dtype=np.float64)
        valid_spearman = spearman[np.isfinite(spearman)]
        return {
            "gamma_mae": (
                self.mae_sum / self.mae_count
                if self.mae_count
                else float("nan")
            ),
            "low_degradation_gamma_hat_p95": self._hist_quantile(0.95),
            "low_degradation_gamma_hat_p99": self._hist_quantile(0.99),
            "saturation_threshold": self.saturation_threshold,
            "saturation_fraction": (
                self.saturation_count / self.saturation_total
                if self.saturation_total
                else float("nan")
            ),
            "image_spearman_mean": (
                float(valid_spearman.mean())
                if valid_spearman.size
                else float("nan")
            ),
            "image_spearman_median": (
                float(np.median(valid_spearman))
                if valid_spearman.size
                else float("nan")
            ),
            "image_spearman_valid_count": int(valid_spearman.size),
            "image_count": int(len(self.image_spearman)),
        }


def _sample_name(batch, batch_idx, sample_idx):
    paths = batch.get("image_path", None)
    if isinstance(paths, (list, tuple)) and sample_idx < len(paths):
        return str(paths[sample_idx])
    if isinstance(paths, str):
        return paths
    return f"b{batch_idx}_{sample_idx}"


def _prefixed(metrics, prefix):
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# ---------------------------------------------------------------------------
# Core diagnostic
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gamma oracle / visualization diagnostic")
    parser.add_argument("--base", required=True, help="config yaml (same as training)")
    parser.add_argument("--ckpt", default=None, help="checkpoint path (overrides yaml)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--max-batches", type=int, default=0, help="0 = all batches")
    parser.add_argument("--num-vis", type=int, default=8, help="thickest-cloud samples to plot")
    parser.add_argument("--cloud-thr", type=float, default=0.1,
                        help="M threshold for the cloud region (matches evaluator)")
    parser.add_argument(
        "--saturation-thr",
        type=float,
        default=0.095,
        help="gamma_hat value counted as saturated (default: 0.095)",
    )
    parser.add_argument("--out", default="logs/gamma_diag")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gamma-clip", default="",
        help="comma-separated per-pixel gamma caps to sweep at inference, "
             "e.g. '0.02,0.03,0.05'. Each runs an extra 1-step pass with "
             "gamma_used = min(gamma_hat, c). No retraining needed.",
    )
    args, unknown = parser.parse_known_args()

    clip_values = []
    for tok in args.gamma_clip.split(","):
        tok = tok.strip()
        if tok:
            clip_values.append(float(tok))

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- config + model ---------------------------------------------------
    config = OmegaConf.load(args.base)
    cli = OmegaConf.from_dotlist(unknown) if unknown else OmegaConf.create()
    config = OmegaConf.merge(config, cli)
    if args.ckpt is not None:
        OmegaConf.update(config, "model.params.ckpt_path", args.ckpt, force_add=True)

    print("[diag] instantiating model ...")
    model = instantiate_from_config(config.model)
    model = model.to(device).eval()

    loss_fn = model.loss_fn
    sampler = model.sampler

    # ---- sanity: this diagnostic targets the degradation-aware bridge -----
    missing = [
        name for name in ("_get_degradation_prob", "_noise_scale_from_prob")
        if not hasattr(loss_fn, name)
    ] + [
        name for name in ("_noise_start", "_predict_noise_scale", "_denoise", "_finalize")
        if not hasattr(sampler, name)
    ]
    if missing:
        raise SystemExit(
            "[diag] this config does not use the noise-bridge gamma machinery "
            f"(missing: {missing}). Point --base at a *_noise_bridge_cfm.yaml."
        )
    if not getattr(sampler, "spatial_noise", False):
        print("[diag][warn] sampler.spatial_noise is False: gamma_hat will be a "
              "scalar; the oracle vs real comparison is only meaningful when "
              "spatial_noise=true.")

    noise_sigma = float(getattr(loss_fn, "noise_sigma", getattr(sampler, "noise_sigma", 0.1)))
    floor = float(getattr(sampler, "noise_sigma_floor", 0.08))
    print(f"[diag] noise_sigma(sigma_max)={noise_sigma}  noise_sigma_floor={floor}  "
          f"gamma_delta_tau={getattr(loss_fn, 'gamma_delta_tau', None)}  "
          f"reduction={getattr(loss_fn, 'gamma_delta_reduction', None)}")

    # ---- data -------------------------------------------------------------
    print(f"[diag] building '{args.split}' dataloader ...")
    data = instantiate_from_config(config.data)
    data.setup()
    if args.split == "test":
        loader = data.test_dataloader()
    else:
        loader = data.val_dataloader()

    real_acc = MetricAccumulator()
    oracle_acc = MetricAccumulator()
    raw_acc = MetricAccumulator()  # cloudy input vs clean (baseline reference)
    clip_accs = {c: MetricAccumulator() for c in clip_values}
    if clip_values:
        print(f"[diag] gamma-clip sweep: {clip_values}")

    stats = {
        ("clear", "hat"): RegionGammaStats(),
        ("cloud", "hat"): RegionGammaStats(),
        ("clear", "train"): RegionGammaStats(),
        ("cloud", "train"): RegionGammaStats(),
    }
    calibration_acc = CalibrationAccumulator(
        noise_sigma,
        saturation_threshold=args.saturation_thr,
    )
    per_image_rows = []

    vis_pool = []  # list of (mean_M, dict) kept to top num-vis

    use_evaluator = getattr(model, "image_metrics_type", None) == "evaluator"
    if not use_evaluator:
        print("[diag][warn] image_metrics is not 'evaluator'; region (cloudy/"
              "cloudfree) metrics will be unavailable.")

    torch.manual_seed(args.seed)
    n_batches = len(loader)
    limit = args.max_batches if args.max_batches > 0 else n_batches

    with torch.no_grad(), model.ema_scope("gamma-diag"):
        for b_idx, batch in enumerate(loader):
            if b_idx >= limit:
                break
            batch = _to_device(batch, device)
            try:
                batch["global_step"] = int(model.global_step)
            except Exception:
                batch["global_step"] = 0

            target = model.get_input(batch, model.input_key)      # clean [-1,1]
            mu_raw = model.get_input(batch, model.mean_key)        # cloudy [-1,1]
            z_x = model.encode_first_stage(target)
            z_mu = model.encode_first_stage(mu_raw)

            c, uc = model.conditioner.get_unconditional_conditioning(
                batch, force_uc_zero_embeddings=[]
            )

            # denoiser closure identical to ResidualDiffusionEngine.sample()
            denoiser = lambda inp, sigma, cond, st, **extra: model.denoiser(
                model.model, inp, sigma, cond, st, **extra
            )

            B = z_mu.shape[0]
            s_in = z_mu.new_ones([B])
            sigma_max = float(getattr(sampler.discretization, "sigma_max", 1.0))
            sigma = s_in * sigma_max
            st = sampler.sigma2st(sigma)

            # ---- gamma_hat (predicted; what inference actually uses) -------
            gamma_hat = sampler._predict_noise_scale(denoiser, z_mu, sigma, c, st, uc)
            if gamma_hat is None:  # scalar fallback when spatial_noise=False
                gamma_hat = torch.full_like(z_mu[:, :1], noise_sigma)

            # ---- gamma_train (oracle; from |cloudy - clean|) ---------------
            prob = loss_fn._get_degradation_prob(z_x, z_mu)
            gamma_train = loss_fn._noise_scale_from_prob(prob, noise_sigma)

            # Shared noise so REAL vs ORACLE differ ONLY by the gamma scale.
            randn = torch.randn_like(z_mu)

            # ---- REAL one-step (gamma_hat path) ----------------------------
            x_real, _ = sampler._one_step(denoiser, randn, z_mu, c, uc)
            x_real = model.decode_first_stage(x_real)

            # ---- ORACLE one-step (gamma_train injected) --------------------
            x_init_o = sampler._noise_start(randn, z_mu, gamma_train)
            v_o = sampler._denoise(x_init_o, denoiser, sigma, c, st, uc)
            x_oracle = x_init_o + append_dims(sigma, x_init_o.ndim) * v_o
            x_oracle = sampler._finalize(x_oracle, z_mu)
            x_oracle = model.decode_first_stage(x_oracle)

            # ---- CLIPPED one-step (gamma_used = min(gamma_hat, c)) ----------
            # No retraining: cap the head's over-prediction at inference. The
            # same shared noise is used so differences vs REAL are only gamma.
            x_clips = {}
            for c_val in clip_values:
                gamma_clip = gamma_hat.clamp(max=c_val)
                x_init_c = sampler._noise_start(randn, z_mu, gamma_clip)
                v_c = sampler._denoise(x_init_c, denoiser, sigma, c, st, uc)
                x_c = x_init_c + append_dims(sigma, x_init_c.ndim) * v_c
                x_c = sampler._finalize(x_c, z_mu)
                x_clips[c_val] = model.decode_first_stage(x_c)

            # ---- region gamma stats ----------------------------------------
            M = batch.get("M", None)
            m = None
            cloud = None
            clear = None
            if M is not None:
                m = M.float()
                if m.ndim == 3:
                    m = m.unsqueeze(1)
                if m.shape[-2:] != gamma_hat.shape[-2:]:
                    m = torch.nn.functional.interpolate(
                        m, size=gamma_hat.shape[-2:], mode="nearest"
                    )
                cloud = m > args.cloud_thr
                clear = ~cloud
                stats[("clear", "hat")].update(gamma_hat[:, :1][clear])
                stats[("cloud", "hat")].update(gamma_hat[:, :1][cloud])
                stats[("clear", "train")].update(gamma_train[:, :1][clear])
                stats[("cloud", "train")].update(gamma_train[:, :1][cloud])

            # ---- per-sample metrics (REAL / ORACLE / raw cloudy) -----------
            for i in range(B):
                t01 = model.scale_01(target[i])
                r01 = model.scale_01(x_real[i])
                o01 = model.scale_01(x_oracle[i])
                mu01 = model.scale_01(mu_raw[i])
                mkw = {}
                if use_evaluator and M is not None:
                    mkw["mask"] = batch["M"][i]
                real_metrics = _metric_dict(
                    model.img_metrics(t01.unsqueeze(0), r01.unsqueeze(0), **mkw)
                )
                oracle_metrics = _metric_dict(
                    model.img_metrics(t01.unsqueeze(0), o01.unsqueeze(0), **mkw)
                )
                raw_metrics = _metric_dict(
                    model.img_metrics(t01.unsqueeze(0), mu01.unsqueeze(0), **mkw)
                )
                real_acc.add(real_metrics)
                oracle_acc.add(oracle_metrics)
                raw_acc.add(raw_metrics)

                clip_metrics = {}
                for c_val in clip_values:
                    c01 = model.scale_01(x_clips[c_val][i])
                    c_metrics = _metric_dict(
                        model.img_metrics(
                            t01.unsqueeze(0), c01.unsqueeze(0), **mkw
                        )
                    )
                    clip_accs[c_val].add(c_metrics)
                    clip_metrics[c_val] = c_metrics

                ghat_i = gamma_hat[i, 0].detach().float().cpu().numpy()
                gtrain_i = gamma_train[i, 0].detach().float().cpu().numpy()
                clear_i = (
                    clear[i, 0].detach().cpu().numpy()
                    if clear is not None
                    else None
                )
                cloud_i = (
                    cloud[i, 0].detach().cpu().numpy()
                    if cloud is not None
                    else None
                )
                spearman = calibration_acc.update(
                    ghat_i,
                    gtrain_i,
                    clear_i,
                )
                all_stats = _safe_array_stats(ghat_i)
                train_stats = _safe_array_stats(gtrain_i)
                clear_hat_stats = _safe_array_stats(
                    ghat_i[clear_i] if clear_i is not None else []
                )
                clear_train_stats = _safe_array_stats(
                    gtrain_i[clear_i] if clear_i is not None else []
                )
                cloud_hat_stats = _safe_array_stats(
                    ghat_i[cloud_i] if cloud_i is not None else []
                )
                cloud_train_stats = _safe_array_stats(
                    gtrain_i[cloud_i] if cloud_i is not None else []
                )

                row = {
                    "image_path": _sample_name(batch, b_idx, i),
                    "seed": int(args.seed),
                    "batch_index": int(b_idx),
                    "sample_index": int(i),
                    "mask_mean": (
                        float(M[i].float().mean().item())
                        if M is not None
                        else float("nan")
                    ),
                    "gamma_mae": float(np.mean(np.abs(ghat_i - gtrain_i))),
                    "gamma_spearman": spearman,
                    "gamma_hat_mean": all_stats["mean"],
                    "gamma_train_mean": train_stats["mean"],
                    "gamma_saturation_fraction": float(
                        np.mean(ghat_i >= args.saturation_thr)
                    ),
                    "clear_gamma_hat_mean": clear_hat_stats["mean"],
                    "clear_gamma_hat_p95": clear_hat_stats["p95"],
                    "clear_gamma_hat_p99": clear_hat_stats["p99"],
                    "clear_gamma_train_mean": clear_train_stats["mean"],
                    "cloud_gamma_hat_mean": cloud_hat_stats["mean"],
                    "cloud_gamma_hat_p95": cloud_hat_stats["p95"],
                    "cloud_gamma_hat_p99": cloud_hat_stats["p99"],
                    "cloud_gamma_train_mean": cloud_train_stats["mean"],
                }
                row.update(_prefixed(raw_metrics, "raw_"))
                row.update(_prefixed(real_metrics, "real_"))
                row.update(_prefixed(oracle_metrics, "gt_gamma_"))
                for key in sorted(set(real_metrics) & set(oracle_metrics)):
                    row[f"gap_gt_minus_real_{key}"] = (
                        oracle_metrics[key] - real_metrics[key]
                    )
                for c_val, c_metrics in clip_metrics.items():
                    tag = f"clip_{c_val:g}_"
                    row.update(_prefixed(c_metrics, tag))
                per_image_rows.append(row)

                # collect for visualization (thickest cloud first)
                mean_m = float(M[i].mean()) if M is not None else 0.0
                vis_pool.append((
                    mean_m,
                    {
                        "name": _sample_name(batch, b_idx, i),
                        "cloudy": _rgb01(model, mu_raw[i]),
                        "clean": _rgb01(model, target[i]),
                        "real": _rgb01(model, x_real[i]),
                        "oracle": _rgb01(model, x_oracle[i]),
                        "ghat": gamma_hat[i, 0].detach().float().cpu().numpy(),
                        "gtrain": gamma_train[i, 0].detach().float().cpu().numpy(),
                        "M": (M[i].detach().float().cpu().numpy() if M is not None else None),
                    },
                ))
            vis_pool.sort(key=lambda x: x[0], reverse=True)
            vis_pool = vis_pool[: max(args.num_vis, 0)]

            print(f"[diag] batch {b_idx + 1}/{limit} done", flush=True)

    # ---- report -----------------------------------------------------------
    _print_metric_table(raw_acc.value(), real_acc.value(), oracle_acc.value())
    if clip_values:
        _print_clip_table(
            real_acc.value(),
            {c: clip_accs[c].value() for c in clip_values},
            oracle_acc.value(),
        )
    _print_gamma_table(stats, noise_sigma, floor)
    calibration_summary = _build_calibration_summary(
        calibration_acc.summary(), stats
    )
    _print_calibration_table(calibration_summary)
    _save_visualizations(vis_pool, args.out, noise_sigma)
    _write_summary(args.out, raw_acc.value(), real_acc.value(), oracle_acc.value(),
                   stats, noise_sigma, floor, calibration_summary)
    _write_per_image_csv(args.out, per_image_rows)
    _write_json_summary(
        args,
        raw_acc.value(),
        real_acc.value(),
        oracle_acc.value(),
        {c: clip_accs[c].value() for c in clip_values},
        calibration_summary,
        per_image_rows,
        noise_sigma,
        floor,
    )
    print(
        f"\n[diag] figures + summary.md + summary.json + per_image.csv "
        f"written to: {args.out}"
    )


def _fmt(v):
    return f"{v:8.3f}" if isinstance(v, (int, float)) and not math.isnan(v) else "     nan"


def _print_metric_table(raw, real, oracle):
    keys = [
        "PSNR", "PSNR_cloudy", "PSNR_cloudfree",
        "SSIM", "SSIM_cloudy", "SSIM_cloudfree",
        "RMSE", "RMSE_cloudy", "RMSE_cloudfree",
        "NDVI_MAE", "NDVI_RMSE",
    ]
    print("\n" + "=" * 72)
    print("Diagnostic A — oracle upper bound (1-step)")
    print("=" * 72)
    print(f"{'metric':<16}{'raw cloudy':>14}{'REAL (g_hat)':>16}{'ORACLE (g_tr)':>16}")
    print("-" * 72)
    for k in keys:
        if k in raw or k in real or k in oracle:
            print(f"{k:<16}{_fmt(raw.get(k, float('nan'))):>14}"
                  f"{_fmt(real.get(k, float('nan'))):>16}"
                  f"{_fmt(oracle.get(k, float('nan'))):>16}")
    print("-" * 72)
    print("Read: if ORACLE clear(PSNR_cloudfree)~32 & cloud(PSNR_cloudy)~24.6 ->")
    print("      bridge design is right, the gap is the gamma-head (case 1).")
    print("      if ORACLE PSNR_cloudfree also low -> gamma magnitude/precond (case 2).")


def _print_clip_table(real, clip_vals: dict, oracle):
    """Sweep of inference-time gamma caps: gamma_used = min(gamma_hat, c)."""
    print("\n" + "=" * 72)
    print("Diagnostic A' — inference-time gamma clip sweep (NO retrain)")
    print("=" * 72)
    print(f"{'setting':<16}{'PSNR':>12}{'PSNR_clear':>14}{'PSNR_cloudy':>14}")
    print("-" * 72)
    print(f"{'REAL (no clip)':<16}{_fmt(real.get('PSNR', float('nan'))):>12}"
          f"{_fmt(real.get('PSNR_cloudfree', float('nan'))):>14}"
          f"{_fmt(real.get('PSNR_cloudy', float('nan'))):>14}")
    for c in sorted(clip_vals):
        v = clip_vals[c]
        print(f"{'clip@' + f'{c:g}':<16}{_fmt(v.get('PSNR', float('nan'))):>12}"
              f"{_fmt(v.get('PSNR_cloudfree', float('nan'))):>14}"
              f"{_fmt(v.get('PSNR_cloudy', float('nan'))):>14}")
    print(f"{'ORACLE':<16}{_fmt(oracle.get('PSNR', float('nan'))):>12}"
          f"{_fmt(oracle.get('PSNR_cloudfree', float('nan'))):>14}"
          f"{_fmt(oracle.get('PSNR_cloudy', float('nan'))):>14}")
    print("-" * 72)
    print("Read: if a clip lifts PSNR_clear toward ORACLE without tanking")
    print("      PSNR_cloudy, that cap is a no-retrain inference fix.")


def _build_calibration_summary(calibration, stats):
    out = dict(calibration)
    clear_hat = stats[("clear", "hat")].summary()
    clear_train = stats[("clear", "train")].summary()
    cloud_hat = stats[("cloud", "hat")].summary()
    cloud_train = stats[("cloud", "train")].summary()
    out.update(
        {
            "low_degradation_gamma_hat_mean": clear_hat["mean"],
            "low_degradation_gamma_train_mean": clear_train["mean"],
            "high_degradation_gamma_hat_mean": cloud_hat["mean"],
            "high_degradation_gamma_train_mean": cloud_train["mean"],
        }
    )
    return out


def _print_calibration_table(summary):
    print("\n" + "=" * 72)
    print("Diagnostic C — formal gamma-head calibration")
    print("=" * 72)
    rows = [
        ("gamma MAE", "gamma_mae", False),
        ("low gamma_hat mean", "low_degradation_gamma_hat_mean", False),
        ("low gamma_hat p95", "low_degradation_gamma_hat_p95", False),
        ("low gamma_hat p99", "low_degradation_gamma_hat_p99", False),
        ("high gamma_hat mean", "high_degradation_gamma_hat_mean", False),
        ("saturation fraction", "saturation_fraction", True),
        ("image Spearman mean", "image_spearman_mean", False),
        ("image Spearman median", "image_spearman_median", False),
    ]
    for label, key, as_percent in rows:
        value = summary.get(key, float("nan"))
        if as_percent and isinstance(value, (int, float)) and math.isfinite(value):
            formatted = f"{100.0 * value:.4f}%"
        else:
            formatted = _fmt(value).strip()
        print(f"{label:<28}{formatted:>16}")
    print("-" * 72)
    print(
        f"saturation threshold={summary.get('saturation_threshold', float('nan')):.4f}; "
        f"valid Spearman images={summary.get('image_spearman_valid_count', 0)}/"
        f"{summary.get('image_count', 0)}"
    )


def _print_gamma_table(stats, noise_sigma, floor):
    ideal_clear = noise_sigma * floor
    ideal_cloud = noise_sigma
    print("\n" + "=" * 72)
    print("Diagnostic B — gamma ranges (per-pixel)")
    print("=" * 72)
    print(f"ideal:  clear ~ {ideal_clear:.4f} (= sigma_max*floor)   "
          f"cloud ~ {ideal_cloud:.4f} (= sigma_max)")
    print("-" * 72)
    print(f"{'region/source':<18}{'min':>10}{'mean':>10}{'max':>10}")
    print("-" * 72)
    for region in ("clear", "cloud"):
        for src in ("hat", "train"):
            s = stats[(region, src)].summary()
            tag = f"{region}/{src}"
            print(f"{tag:<18}{_fmt(s['min']):>10}{_fmt(s['mean']):>10}{_fmt(s['max']):>10}")
    print("-" * 72)
    print("Want: clear/hat ~ clear/train (small); cloud/hat ~ cloud/train (large).")
    print("If clear/hat >> clear/train -> head over-predicts gamma on clear sky.")


def _save_visualizations(vis_pool, out_dir, noise_sigma):
    if not vis_pool:
        return
    for rank, (mean_m, d) in enumerate(vis_pool):
        panels = [
            ("cloudy", d["cloudy"], None),
            ("clean", d["clean"], None),
            ("M mask", d["M"], "gray"),
            ("gamma_hat", d["ghat"], "viridis"),
            ("gamma_train", d["gtrain"], "viridis"),
            ("REAL (g_hat)", d["real"], None),
            ("ORACLE (g_tr)", d["oracle"], None),
        ]
        fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.4))
        for ax, (title, img, cmap) in zip(axes, panels):
            if img is None:
                ax.axis("off")
                ax.set_title(title + "\n(n/a)")
                continue
            if cmap in ("viridis",):
                im = ax.imshow(img, cmap=cmap, vmin=0.0, vmax=noise_sigma)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            elif cmap == "gray":
                ax.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0)
            else:
                ax.imshow(np.clip(img, 0, 1))
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"#{rank}  {d['name']}   mean(M)={mean_m:.3f}", y=1.02)
        fig.tight_layout()
        path = os.path.join(out_dir, f"vis_{rank:02d}_{d['name'].replace('/', '_')}.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)


def _write_summary(
    out_dir,
    raw,
    real,
    oracle,
    stats,
    noise_sigma,
    floor,
    calibration,
):
    lines = []
    lines.append("# Gamma diagnostic summary\n")
    lines.append(f"noise_sigma(sigma_max)={noise_sigma}  noise_sigma_floor={floor}\n")
    lines.append("\n## Diagnostic A — metrics (raw / REAL gamma_hat / ORACLE gamma_train)\n")
    keys = ["PSNR", "PSNR_cloudy", "PSNR_cloudfree", "SSIM", "SSIM_cloudy",
            "SSIM_cloudfree", "RMSE", "RMSE_cloudy", "RMSE_cloudfree",
            "NDVI_MAE", "NDVI_RMSE"]
    lines.append("| metric | raw cloudy | REAL (g_hat) | ORACLE (g_train) |\n")
    lines.append("|---|---|---|---|\n")
    for k in keys:
        if k in raw or k in real or k in oracle:
            lines.append(
                f"| {k} | {raw.get(k, float('nan')):.4f} | "
                f"{real.get(k, float('nan')):.4f} | "
                f"{oracle.get(k, float('nan')):.4f} |\n"
            )
    lines.append("\n## Diagnostic B — gamma ranges\n")
    lines.append(f"ideal clear ~ {noise_sigma * floor:.4f}, ideal cloud ~ {noise_sigma:.4f}\n\n")
    lines.append("| region/source | min | mean | max |\n")
    lines.append("|---|---|---|---|\n")
    for region in ("clear", "cloud"):
        for src in ("hat", "train"):
            s = stats[(region, src)].summary()
            lines.append(f"| {region}/{src} | {s['min']:.4f} | {s['mean']:.4f} | {s['max']:.4f} |\n")

    lines.append("\n## Diagnostic C — formal gamma-head calibration\n")
    lines.append("| metric | value |\n")
    lines.append("|---|---:|\n")
    for key in (
        "gamma_mae",
        "low_degradation_gamma_hat_mean",
        "low_degradation_gamma_train_mean",
        "low_degradation_gamma_hat_p95",
        "low_degradation_gamma_hat_p99",
        "high_degradation_gamma_hat_mean",
        "high_degradation_gamma_train_mean",
        "saturation_threshold",
        "saturation_fraction",
        "image_spearman_mean",
        "image_spearman_median",
        "image_spearman_valid_count",
        "image_count",
    ):
        value = calibration.get(key, float("nan"))
        if isinstance(value, float):
            value = f"{value:.6f}"
        lines.append(f"| {key} | {value} |\n")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.writelines(lines)


def _write_per_image_csv(out_dir, rows):
    path = os.path.join(out_dir, "per_image.csv")
    if not rows:
        with open(path, "w", newline="", encoding="utf-8"):
            pass
        return

    preferred = [
        "image_path",
        "seed",
        "batch_index",
        "sample_index",
        "mask_mean",
        "gamma_mae",
        "gamma_spearman",
        "gamma_hat_mean",
        "gamma_train_mean",
        "gamma_saturation_fraction",
        "clear_gamma_hat_mean",
        "clear_gamma_hat_p95",
        "clear_gamma_hat_p99",
        "clear_gamma_train_mean",
        "cloud_gamma_hat_mean",
        "cloud_gamma_hat_p95",
        "cloud_gamma_hat_p99",
        "cloud_gamma_train_mean",
    ]
    all_fields = set().union(*(row.keys() for row in rows))
    fieldnames = [key for key in preferred if key in all_fields]
    fieldnames.extend(sorted(all_fields - set(fieldnames)))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json_summary(
    args,
    raw,
    real,
    gt_derived,
    clip_metrics,
    calibration,
    per_image_rows,
    noise_sigma,
    floor,
):
    common = sorted(set(real) & set(gt_derived))
    metric_gap = {
        key: float(gt_derived[key] - real[key])
        for key in common
    }
    payload = {
        "meta": {
            "base": os.path.abspath(args.base),
            "checkpoint": (
                os.path.abspath(args.ckpt) if args.ckpt is not None else None
            ),
            "split": args.split,
            "seed": int(args.seed),
            "cloud_threshold": float(args.cloud_thr),
            "noise_sigma": float(noise_sigma),
            "noise_sigma_floor": float(floor),
            "num_images": int(len(per_image_rows)),
        },
        "calibration": calibration,
        "metrics": {
            "raw": raw,
            "real_predicted_gamma": real,
            "gt_derived_gamma_diagnostic": gt_derived,
            "gap_gt_minus_real": metric_gap,
        },
        "gamma_clip_metrics": {
            f"{clip:g}": metrics for clip, metrics in sorted(clip_metrics.items())
        },
    }
    with open(
        os.path.join(args.out, "summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
