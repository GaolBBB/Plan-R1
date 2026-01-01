#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

# v3: 为了查看原始SFT的train_top1_unsafe_rate_step和train_unsafe_prob_mass_step这两个指标的可视化，在v2的基础上去掉了其他指标的可视化逻辑
# python /home/gaosunxiang/Plan-R1/analysis/plot_pred_metrics_v3.py \
#   --csv  /home/gaosunxiang/Plan-R1/lightning_logs/pred/version_2/metrics.csv \
#   --outdir /home/gaosunxiang/Plan-R1/lightning_logs/pred/version_2/plots \
#   --smooth 50
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
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_metrics(args.csv)

    # v3: current metrics.csv only contains these three step-level metrics
    save_line_plot(df, "step", "train_cls_loss_step",
                   os.path.join(args.outdir, "train_cls_loss_step.png"),
                   title="train_cls_loss_step vs step", smooth=args.smooth)

    save_line_plot(df, "step", "train_top1_unsafe_rate_step",
                   os.path.join(args.outdir, "train_top1_unsafe_rate_step.png"),
                   title="train_top1_unsafe_rate_step vs step", smooth=args.smooth)

    save_line_plot(df, "step", "train_unsafe_prob_mass_step",
                   os.path.join(args.outdir, "train_unsafe_prob_mass_step_log.png"),
                   title="train_unsafe_prob_mass_step vs step (log y)", y_log=True, smooth=args.smooth)

    print(f"\nDone. Figures are in: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()