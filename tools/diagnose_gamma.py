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
    parser.add_argument("--out", default="logs/gamma_diag")
    parser.add_argument("--seed", type=int, default=0)
    args, unknown = parser.parse_known_args()

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

    stats = {
        ("clear", "hat"): RegionGammaStats(),
        ("cloud", "hat"): RegionGammaStats(),
        ("clear", "train"): RegionGammaStats(),
        ("cloud", "train"): RegionGammaStats(),
    }

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

            # ---- region gamma stats ----------------------------------------
            M = batch.get("M", None)
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
                real_acc.add(model.img_metrics(t01.unsqueeze(0), r01.unsqueeze(0), **mkw))
                oracle_acc.add(model.img_metrics(t01.unsqueeze(0), o01.unsqueeze(0), **mkw))
                raw_acc.add(model.img_metrics(t01.unsqueeze(0), mu01.unsqueeze(0), **mkw))

                # collect for visualization (thickest cloud first)
                mean_m = float(M[i].mean()) if M is not None else 0.0
                vis_pool.append((
                    mean_m,
                    {
                        "name": str(batch.get("image_path", [f"b{b_idx}_{i}"])[i])
                        if isinstance(batch.get("image_path", None), (list, tuple))
                        else f"b{b_idx}_{i}",
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
    _print_gamma_table(stats, noise_sigma, floor)
    _save_visualizations(vis_pool, args.out, noise_sigma)
    _write_summary(args.out, raw_acc.value(), real_acc.value(), oracle_acc.value(),
                   stats, noise_sigma, floor)
    print(f"\n[diag] figures + summary written to: {args.out}")


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
    print(f"{'metric':<16}{'raw cloudy':>12}{'REAL (g_hat)':>14}{'ORACLE (g_tr)':>16}")
    print("-" * 72)
    for k in keys:
        if k in raw or k in real or k in oracle:
            print(f"{k:<16}{_fmt(raw.get(k, float('nan')))[4:]:>12}"
                  f"{_fmt(real.get(k, float('nan')))[2:]:>14}"
                  f"{_fmt(oracle.get(k, float('nan'))):>16}")
    print("-" * 72)
    print("Read: if ORACLE clear(PSNR_cloudfree)~32 & cloud(PSNR_cloudy)~24.6 ->")
    print("      bridge design is right, the gap is the gamma-head (case 1).")
    print("      if ORACLE PSNR_cloudfree also low -> gamma magnitude/precond (case 2).")


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


def _write_summary(out_dir, raw, real, oracle, stats, noise_sigma, floor):
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
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
