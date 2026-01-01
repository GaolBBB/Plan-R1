#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
'''
可视化RFT阶段的指标
python /home/gaosunxiang/Plan-R1/analysis/plot_plan_rft_metrics.py \
  --csv  /home/gaosunxiang/Plan-R1/lightning_logs/plan/version_1/metrics.csv \
  --outdir /home/gaosunxiang/Plan-R1/lightning_logs/plan/version_1/plots_rft \
  --smooth 50
'''


def rolling_mean(y, w: int):
    if w is None or w <= 1:
        return y
    return pd.Series(y).rolling(window=w, min_periods=max(1, w // 5)).mean().to_numpy()


def save_line_plot(df, xcol, ycol, out_path, title=None, y_log=False, smooth=0):
    if ycol not in df.columns:
        print(f"[WARN] missing column: {ycol}")
        return

    sub = df[[xcol, ycol]].dropna()
    if sub.empty:
        print(f"[WARN] no data for: {ycol}")
        return

    x = sub[xcol].to_numpy()
    y = sub[ycol].to_numpy()

    ys = rolling_mean(y, smooth) if smooth and smooth > 1 else y

    plt.figure(figsize=(10, 5))
    plt.plot(x, ys, linewidth=1.2)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title or f"{ycol} vs {xcol}")
    plt.grid(True, alpha=0.3)
    if y_log:
        plt.yscale("log")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] saved: {out_path}")


def save_two_lines(df, xcol, ycols, labels, out_path, title=None, smooth=0):
    # ycols: list of columns
    cols = [xcol] + ycols
    for c in ycols:
        if c not in df.columns:
            print(f"[WARN] missing column: {c} (skip {out_path})")
            return
    sub = df[cols].dropna()
    if sub.empty:
        print(f"[WARN] no data for: {ycols} (skip {out_path})")
        return

    x = sub[xcol].to_numpy()

    plt.figure(figsize=(10, 5))
    for c, lab in zip(ycols, labels):
        y = sub[c].to_numpy()
        ys = rolling_mean(y, smooth) if smooth and smooth > 1 else y
        plt.plot(x, ys, linewidth=1.2, label=lab)
    plt.xlabel(xcol)
    plt.ylabel("value")
    plt.title(title or " / ".join(ycols))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to metrics.csv")
    ap.add_argument("--outdir", default="plots_plan_rft", help="output directory for figures")
    ap.add_argument("--smooth", type=int, default=0, help="rolling mean window, e.g. 50; 0 disables")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Some loggers use 'step' column, some use 'global_step'—PlanR1 uses 'step' in your metrics.csv.
    xcol = "step" if "step" in df.columns else ("global_step" if "global_step" in df.columns else None)
    if xcol is None:
        raise ValueError("Cannot find step/global_step column in metrics.csv")

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # -------------------------
    # Core RFT losses & signals
    # -------------------------
    core_plots = [
        ("train_policy_loss_step", "train_policy_loss_step.png", False),
        ("train_kl_loss_step", "train_kl_loss_step.png", False),
        ("train_loss_step", "train_loss_step.png", False),
        ("train_reward_step", "train_reward_step.png", False),
    ]
    for col, fn, ylog in core_plots:
        save_line_plot(df, xcol, col, os.path.join(outdir, fn),
                       title=f"{col} vs step", y_log=ylog, smooth=args.smooth)

    # -------------------------
    # RFT UL diagnostics (the new ones you added)
    # -------------------------
    rft_ul_plots = [
        ("train_rft_ul_loss_step", "train_rft_ul_loss_step_log.png", True),
        ("train_rft_ul_loss_weighted_step", "train_rft_ul_loss_weighted_step_log.png", True),
        ("train_rft_neg_step_rate_step", "train_rft_neg_step_rate_step.png", False),
        ("train_rft_near_miss_rate_step", "train_rft_near_miss_rate_step.png", False),
        ("train_rft_done_traj_rate_step", "train_rft_done_traj_rate_step.png", False),
        ("train_rft_mean_p_neg_step", "train_rft_mean_p_neg_step.png", False),
    ]
    for col, fn, ylog in rft_ul_plots:
        save_line_plot(df, xcol, col, os.path.join(outdir, fn),
                       title=f"{col} vs step", y_log=ylog, smooth=args.smooth)

    # Optional: overlay hard-unsafe rates together
    save_two_lines(
        df, xcol,
        ["train_rft_neg_step_rate_step", "train_rft_done_traj_rate_step", "train_rft_near_miss_rate_step"],
        ["neg_step_rate", "done_traj_rate", "near_miss_rate"],
        os.path.join(outdir, "rft_rates_overlay.png"),
        title="RFT unsafe / done / near-miss rates (overlay)",
        smooth=args.smooth
    )

    # -------------------------
    # Entropy (collapse check)
    # -------------------------
    save_two_lines(
        df, xcol,
        ["train_entropy_mean_step", "train_entropy_min_step"],
        ["entropy_mean", "entropy_min"],
        os.path.join(outdir, "entropy_mean_min.png"),
        title="Plan distribution entropy (mean/min)",
        smooth=args.smooth
    )

    # -------------------------
    # GRPO group statistics
    # -------------------------
    group_plots = [
        ("train_group_safe_rate_step", "train_group_safe_rate_step.png", False),
        ("train_group_frac_all_safe_step", "train_group_frac_all_safe_step.png", False),
        ("train_group_frac_all_unsafe_step", "train_group_frac_all_unsafe_step.png", False),
    ]
    for col, fn, ylog in group_plots:
        save_line_plot(df, xcol, col, os.path.join(outdir, fn),
                       title=f"{col} vs step", y_log=ylog, smooth=args.smooth)

    save_two_lines(
        df, xcol,
        ["train_group_frac_all_safe_step", "train_group_frac_all_unsafe_step"],
        ["frac_all_safe", "frac_all_unsafe"],
        os.path.join(outdir, "group_all_safe_vs_all_unsafe.png"),
        title="GRPO groups: frac all-safe vs all-unsafe",
        smooth=args.smooth
    )

    # -------------------------
    # Masked reward/adv statistics (stability / scaling check)
    # -------------------------
    masked_plots = [
        ("train_reward_mean_masked_step", "train_reward_mean_masked_step.png", False),
        ("train_reward_std_masked_step", "train_reward_std_masked_step.png", False),
        ("train_adv_mean_masked_step", "train_adv_mean_masked_step.png", False),
        ("train_adv_std_masked_step", "train_adv_std_masked_step.png", False),
    ]
    for col, fn, ylog in masked_plots:
        save_line_plot(df, xcol, col, os.path.join(outdir, fn),
                       title=f"{col} vs step", y_log=ylog, smooth=args.smooth)

    # Optional: relative strength of weighted UL vs policy loss magnitude (rough diagnostic)
    if "train_rft_ul_loss_weighted_step" in df.columns and "train_policy_loss_step" in df.columns:
        sub = df[[xcol, "train_rft_ul_loss_weighted_step", "train_policy_loss_step"]].dropna()
        if not sub.empty:
            ratio = sub["train_rft_ul_loss_weighted_step"] / (sub["train_policy_loss_step"].abs() + 1e-12)
            tmp = pd.DataFrame({xcol: sub[xcol].to_numpy(), "rft_ul_over_abs_policy": ratio.to_numpy()})
            save_line_plot(tmp, xcol, "rft_ul_over_abs_policy",
                           os.path.join(outdir, "ratio_rft_ul_over_abs_policy.png"),
                           title="(weighted RFT UL) / |policy_loss| vs step",
                           y_log=False, smooth=args.smooth)

    print(f"\n[DONE] all figures saved to: {outdir}")


if __name__ == "__main__":
    main()