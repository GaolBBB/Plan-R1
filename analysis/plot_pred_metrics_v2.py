#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# python /home/gaosunxiang/Plan-R1/analysis/plot_pred_metrics_v2.py \
#   --csv  /home/gaosunxiang/Plan-R1/lightning_logs/pred/version_1/metrics.csv \
#   --outdir /home/gaosunxiang/Plan-R1/lightning_logs/pred/version_1/plots_ul_weight100 \
#   --smooth 50 \
#   --sft_ul_weight 100
def _to_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def load_metrics(csv_path: str) -> pd.DataFrame:
    # metrics.csv 里很多空字段，pandas 会读成 NaN；另外有些行会在 lr 列写数值但其余为空
    df = pd.read_csv(csv_path)

    # 统一把关键列转成数值
    for c in df.columns:
        if c in [
            "step", "epoch",
            "train_cls_loss_step", "train_ul_loss_step",
            "train_ul_hit_rate_step", "train_ul_unsafe_frac_step",
            "train_top1_unsafe_rate_step", "train_unsafe_prob_mass_step",
            "train_cls_loss_epoch", "train_ul_loss_epoch",
            "train_ul_hit_rate_epoch", "train_ul_unsafe_frac_epoch",
            "train_top1_unsafe_rate_epoch", "train_unsafe_prob_mass_epoch",
            "val_token_cls_acc", "val_cls_loss",
            "val_min_joint_ade", "val_min_joint_fde",
        ]:
            df[c] = _to_numeric(df[c])

    # 只保留 step 有效的行，并按 step 排序
    df = df[df["step"].notna()].copy()
    df = df.sort_values("step").reset_index(drop=True)

    return df


def rolling_mean(x: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return x
    return x.rolling(window=window, min_periods=max(1, window // 3)).mean()


def save_line_plot(df: pd.DataFrame, x_col: str, y_col: str, out_path: str,
                   title: str = None, y_log: bool = False, smooth: int = 0):
    # 只画有 y 的点
    sub = df[[x_col, y_col]].dropna()
    if sub.empty:
        print(f"[WARN] No data for {y_col}, skip.")
        return

    x = sub[x_col]
    y = sub[y_col]
    if smooth and smooth > 1:
        y = rolling_mean(y, smooth)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, linewidth=1.2)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if title:
        plt.title(title)
    if y_log:
        # UL loss 通常很小，log scale 更容易看趋势
        plt.yscale("log")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to metrics.csv")
    ap.add_argument("--outdir", default="plots_metrics", help="output dir for figures")
    ap.add_argument("--smooth", type=int, default=0, help="rolling mean window (e.g., 50). 0 disables.")
    ap.add_argument("--sft_ul_weight", type=float, default=1.0, help="SFT UL weight used in training (for weighted plots)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_metrics(args.csv)

    # 你关心的 step-level 指标
    save_line_plot(df, "step", "train_cls_loss_step",
                   os.path.join(args.outdir, "train_cls_loss_step.png"),
                   title="train_cls_loss_step vs step", smooth=args.smooth)

    save_line_plot(df, "step", "train_ul_loss_step",
                   os.path.join(args.outdir, "train_ul_loss_raw_step_log.png"),
                   title="train_ul_loss_step (raw, unweighted) vs step (log y)", y_log=True, smooth=args.smooth)

    # weighted UL contribution (sft_ul_weight * ul_loss)
    if "train_ul_loss_step" in df.columns:
        tmpw = df[["step", "train_ul_loss_step"]].dropna().copy()
        if not tmpw.empty:
            tmpw["train_ul_loss_weighted_step"] = tmpw["train_ul_loss_step"] * float(args.sft_ul_weight)
            save_line_plot(tmpw, "step", "train_ul_loss_weighted_step",
                           os.path.join(args.outdir, "train_ul_loss_weighted_step_log.png"),
                           title=f"train_ul_loss_step weighted (x{args.sft_ul_weight:g}) vs step (log y)", y_log=True, smooth=args.smooth)

    save_line_plot(df, "step", "train_ul_hit_rate_step",
                   os.path.join(args.outdir, "train_ul_hit_rate_step.png"),
                   title="train_ul_hit_rate_step vs step", smooth=args.smooth)

    save_line_plot(df, "step", "train_ul_unsafe_frac_step",
                   os.path.join(args.outdir, "train_ul_unsafe_frac_step.png"),
                   title="train_ul_unsafe_frac_step vs step", smooth=args.smooth)

    save_line_plot(df, "step", "train_top1_unsafe_rate_step",
                   os.path.join(args.outdir, "train_top1_unsafe_rate_step.png"),
                   title="train_top1_unsafe_rate_step vs step", smooth=args.smooth)

    save_line_plot(df, "step", "train_unsafe_prob_mass_step",
                   os.path.join(args.outdir, "train_unsafe_prob_mass_step_log.png"),
                   title="train_unsafe_prob_mass_step vs step (log y)", y_log=True, smooth=args.smooth)

    # 验证集指标（如果你的 metrics.csv 里已经开始写了，就能画）
    save_line_plot(df, "step", "val_token_cls_acc",
                   os.path.join(args.outdir, "val_token_cls_acc.png"),
                   title="val_token_cls_acc vs step", smooth=args.smooth)

    save_line_plot(df, "step", "val_cls_loss",
                   os.path.join(args.outdir, "val_cls_loss.png"),
                   title="val_cls_loss vs step", smooth=args.smooth)

    save_line_plot(df, "step", "val_min_joint_ade",
                   os.path.join(args.outdir, "val_min_joint_ade.png"),
                   title="val_min_joint_ade vs step", smooth=args.smooth)

    save_line_plot(df, "step", "val_min_joint_fde",
                   os.path.join(args.outdir, "val_min_joint_fde.png"),
                   title="val_min_joint_fde vs step", smooth=args.smooth)

    # 额外：UL 相对强度（对你后续调 weight 很有用）
    # ratio = ul_loss / cls_loss（都是 step-level）
    # 注意：这不是梯度比例，但作为“量级直觉”非常好用
    sub = df[["step", "train_ul_loss_step", "train_cls_loss_step"]].dropna()
    if not sub.empty:
        ratio = (sub["train_ul_loss_step"] / (sub["train_cls_loss_step"] + 1e-12)).replace([np.inf, -np.inf], np.nan)
        tmp = pd.DataFrame({"step": sub["step"], "ul_over_cls": ratio}).dropna()

        plt.figure(figsize=(10, 5))
        plt.plot(tmp["step"], rolling_mean(tmp["ul_over_cls"], args.smooth) if args.smooth else tmp["ul_over_cls"], linewidth=1.2)
        plt.xlabel("step")
        plt.ylabel("train_ul_loss_step (raw) / train_cls_loss_step")
        plt.title("Raw UL/CLS ratio vs step")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(args.outdir, "ratio_ul_over_cls.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[OK] saved: {out_path}")

        # Weighted UL/CLS ratio (reflects actual loss contribution scale during training)
        w = float(args.sft_ul_weight)
        plt.figure(figsize=(10, 5))
        y_w = (tmp["ul_over_cls"] * w)
        plt.plot(tmp["step"], rolling_mean(y_w, args.smooth) if args.smooth else y_w, linewidth=1.2)
        plt.xlabel("step")
        plt.ylabel(f"(train_ul_loss_step * {w:g}) / train_cls_loss_step")
        plt.title("Weighted UL/CLS ratio vs step")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path_w = os.path.join(args.outdir, "ratio_weighted_ul_over_cls.png")
        plt.savefig(out_path_w, dpi=200)
        plt.close()
        print(f"[OK] saved: {out_path_w}")

        # Extra ratio: unsafe probability mass vs CLS loss (helps interpret safety shift)
        sub2 = df[["step", "train_unsafe_prob_mass_step", "train_cls_loss_step"]].dropna()
        if not sub2.empty:
            ratio2 = (sub2["train_unsafe_prob_mass_step"] / (sub2["train_cls_loss_step"] + 1e-12)).replace([np.inf, -np.inf], np.nan)
            tmp2 = pd.DataFrame({"step": sub2["step"], "unsafe_mass_over_cls": ratio2}).dropna()

            plt.figure(figsize=(10, 5))
            plt.plot(tmp2["step"], rolling_mean(tmp2["unsafe_mass_over_cls"], args.smooth) if args.smooth else tmp2["unsafe_mass_over_cls"], linewidth=1.2)
            plt.xlabel("step")
            plt.ylabel("train_unsafe_prob_mass_step / train_cls_loss_step")
            plt.title("UnsafeMass/CLS ratio vs step")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_path2 = os.path.join(args.outdir, "ratio_unsafe_mass_over_cls.png")
            plt.savefig(out_path2, dpi=200)
            plt.close()
            print(f"[OK] saved: {out_path2}")

        # 给一个“建议 weight”范围：让 alpha*ul 大约占 cls 的 5%~20%
        median_ratio = float(tmp["ul_over_cls"].median())
        if median_ratio > 0:
            alpha_5 = 0.05 / median_ratio
            alpha_20 = 0.20 / median_ratio
            w = float(args.sft_ul_weight)
            print(f"[INFO] median(raw UL/CLS) = {median_ratio:.6g}")
            print(f"[INFO] with current sft_ul_weight={w:g}, median(weighted UL/CLS) ≈ {(median_ratio*w):.6g}")
            print(f"[SUGGEST] to make weighted UL about 5%~20% of CLS, try sft_ul_weight ~ {alpha_5:.1f} ~ {alpha_20:.1f}")
        else:
            print("[WARN] UL/CLS ratio median is 0 or invalid; cannot suggest weight.")
    else:
        print("[WARN] not enough data to compute UL/CLS ratio.")

    print(f"\nDone. Figures are in: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()