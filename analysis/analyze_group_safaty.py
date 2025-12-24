import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def pick_col(df, names):
    return next((n for n in names if n in df.columns), None)

def pick_step_col(df):
    for c in ["step", "global_step"]:
        if c in df.columns:
            return c
    df = df.copy()
    df["__index__"] = np.arange(len(df))
    return "__index__"

def smooth(y, window=200):
    return pd.Series(y).rolling(window=window, min_periods=max(10, window//10)).mean().to_numpy()

def main(csv_path: str, tail_frac=0.2, smooth_window=200):
    df = pd.read_csv(csv_path)

    step_col = pick_step_col(df)

    # 兼容 Lightning 的 *_step / *_epoch 命名
    all_safe_col = pick_col(df, [
        "train_group_frac_all_safe_step",
        "train_group_frac_all_safe",
        "train_group_frac_all_safe_epoch",
    ])
    all_unsafe_col = pick_col(df, [
        "train_group_frac_all_unsafe_step",
        "train_group_frac_all_unsafe",
        "train_group_frac_all_unsafe_epoch",
    ])
    safe_rate_col = pick_col(df, [
        "train_group_safe_rate_step",
        "train_group_safe_rate",
        "train_group_safe_rate_epoch",
    ])

    missing = [("all_safe", all_safe_col), ("all_unsafe", all_unsafe_col), ("safe_rate", safe_rate_col)]
    missing = [k for k,v in missing if v is None]
    if missing:
        print("❌ metrics.csv 里缺少这些列：", missing)
        print("你需要确保 PlanR1.py 里 self.log 了：")
        print("  train_group_frac_all_safe / train_group_frac_all_unsafe / train_group_safe_rate")
        print("\n当前 columns：")
        print(df.columns.tolist())
        sys.exit(1)

    # 取子表并按 step 排序（Lightning 写入可能无序）
    d = df[[step_col, all_safe_col, all_unsafe_col, safe_rate_col]].dropna(how="all").copy()
    d = d.dropna(subset=[all_safe_col, all_unsafe_col, safe_rate_col], how="all")
    d = d.sort_values(step_col)

    step = d[step_col].to_numpy()
    all_safe = d[all_safe_col].to_numpy(dtype=float)
    all_unsafe = d[all_unsafe_col].to_numpy(dtype=float)
    safe_rate = d[safe_rate_col].to_numpy(dtype=float)

    # 平滑
    all_safe_s = smooth(all_safe, window=smooth_window)
    all_unsafe_s = smooth(all_unsafe, window=smooth_window)
    safe_rate_s = smooth(safe_rate, window=smooth_window)

    # “训练后期”定义：最后 tail_frac 的点
    n = len(d)
    t = max(10, int(n * tail_frac))
    tail = d.iloc[-t:]

    def summarize_tail(name, arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return {}
        return {
            f"{name}_mean": float(np.mean(arr)),
            f"{name}_p50": float(np.median(arr)),
            f"{name}_p90": float(np.quantile(arr, 0.9)),
            f"{name}_p99": float(np.quantile(arr, 0.99)),
            f"{name}_max": float(np.max(arr)),
        }

    tail_all_safe = tail[all_safe_col].to_numpy(dtype=float)
    tail_all_unsafe = tail[all_unsafe_col].to_numpy(dtype=float)
    tail_safe_rate = tail[safe_rate_col].to_numpy(dtype=float)

    # 你关心的“坏情况”：all_safe 或 all_unsafe 高
    # 这里给个可调阈值：>=0.8 表示“很多step都几乎是全安全/全不安全”
    thr = 0.8
    frac_steps_all_safe_high = float(np.mean(tail_all_safe >= thr))
    frac_steps_all_unsafe_high = float(np.mean(tail_all_unsafe >= thr))

    print("\n===== Group Safety (Tail Stage) =====")
    print(f"Tail window: last {tail_frac*100:.0f}% points (n={t})")
    for k,v in summarize_tail("all_safe", tail_all_safe).items():
        print(k, ":", v)
    for k,v in summarize_tail("all_unsafe", tail_all_unsafe).items():
        print(k, ":", v)
    for k,v in summarize_tail("safe_rate", tail_safe_rate).items():
        print(k, ":", v)

    print(f"\nIn tail, fraction of steps with all_safe >= {thr}: {frac_steps_all_safe_high:.3f}")
    print(f"In tail, fraction of steps with all_unsafe >= {thr}: {frac_steps_all_unsafe_high:.3f}")

    # 可视化
    plt.figure()
    plt.plot(step, all_safe, alpha=0.25, label=all_safe_col)
    plt.plot(step, all_safe_s, label=f"{all_safe_col} (rolling mean)")
    plt.plot(step, all_unsafe, alpha=0.25, label=all_unsafe_col)
    plt.plot(step, all_unsafe_s, label=f"{all_unsafe_col} (rolling mean)")
    plt.xlabel("step")
    plt.ylabel("fraction")
    plt.title("Group-level extremes: all-safe vs all-unsafe")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out1 = "group_all_safe_all_unsafe.png"
    plt.savefig(out1, dpi=200)
    print(f"\n📈 Saved plot to: {out1}")

    plt.figure()
    plt.plot(step, safe_rate, alpha=0.25, label=safe_rate_col)
    plt.plot(step, safe_rate_s, label=f"{safe_rate_col} (rolling mean)")
    plt.xlabel("step")
    plt.ylabel("safe rate")
    plt.title("Group safe rate over training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out2 = "group_safe_rate.png"
    plt.savefig(out2, dpi=200)
    print(f"📈 Saved plot to: {out2}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_group_safety.py /path/to/metrics.csv")
        sys.exit(1)
    main(sys.argv[1])

# (planr1) gaosunxiang@admin123-ESC8000A-E12:~/Plan-R1/analysis$ python analyze_group_safety.py /home/gaosunxiang/Plan-R1/lightning_logs/plan/version_0/metrics.csv

# ===== Group Safety (Tail Stage) =====
# Tail window: last 20% points (n=56)
# all_safe_mean : 0.9358258928571429
# all_safe_p50 : 0.9375
# all_safe_p90 : 1.0
# all_safe_p99 : 1.0
# all_safe_max : 1.0

# all_unsafe_mean : 0.009486607142857142
# all_unsafe_p50 : 0.0
# all_unsafe_p90 : 0.03125
# all_unsafe_p99 : 0.0625
# all_unsafe_max : 0.0625

# safe_rate_mean : 0.9716796875
# safe_rate_p50 : 0.98046875
# safe_rate_p90 : 1.0
# safe_rate_p99 : 1.0
# safe_rate_max : 1.0

# In tail, fraction of steps with all_safe >= 0.8: 0.964
# In tail, fraction of steps with all_unsafe >= 0.8: 0.000

# 📈 Saved plot to: group_all_safe_all_unsafe.png
# 📈 Saved plot to: group_safe_rate.png