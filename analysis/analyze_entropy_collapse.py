import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def pick_entropy_series(df: pd.DataFrame):
    """
    兼容 Lightning：
    - train_entropy_mean_step / train_entropy_mean_epoch
    - train_entropy_min_step / train_entropy_min_epoch
    也兼容你只记录 train_entropy_mean 这种情况。
    """
    # 候选列（按优先级）
    candidates = [
        "train_entropy_mean_step",
        "train_entropy_mean",
        "train_entropy_mean_epoch",
    ]
    candidates_min = [
        "train_entropy_min_step",
        "train_entropy_min",
        "train_entropy_min_epoch",
    ]

    ent_col = next((c for c in candidates if c in df.columns), None)
    entmin_col = next((c for c in candidates_min if c in df.columns), None)

    # step 列优先用 global step（Lightning 通常叫 step）
    step_col = "step" if "step" in df.columns else None
    if step_col is None:
        # 有些版本会叫 global_step
        step_col = "global_step" if "global_step" in df.columns else None
    if step_col is None:
        # 最差退化成行号
        step_col = "__index__"
        df = df.copy()
        df["__index__"] = np.arange(len(df))

    return df, step_col, ent_col, entmin_col


def smooth_series(x, y, window=200):
    """简单 rolling 平滑（window 用 step 数量，不是 token 数量）。"""
    s = pd.Series(y).rolling(window=window, min_periods=max(10, window // 10)).mean().to_numpy()
    return s


def detect_entropy_collapse(step, ent, ent_min=None, warmup_frac=0.1, tail_frac=0.2):
    """
    一个实用的“坍塌判据”：
    - 用训练前 10% 的熵均值作为 baseline（避免最初极不稳定点）
    - 看后 20% 是否长期低于 baseline 的某个比例（比如 20%）
    - 并且后段熵的波动很小（说明分布稳定地变尖，而不是暂时抖动）
    同时如果 ent_min 长期接近 0，会更强烈支持“坍塌”。

    返回：是否坍塌 + 解释字典
    """
    step = np.asarray(step)
    ent = np.asarray(ent)

    # 去掉 NaN
    mask = np.isfinite(step) & np.isfinite(ent)
    step, ent = step[mask], ent[mask]
    if len(ent) < 50:
        return False, {"reason": "有效点太少(<50)，无法判断"}

    n = len(ent)
    w = max(5, int(n * warmup_frac))
    t = max(10, int(n * tail_frac))

    baseline = np.nanmean(ent[:w])
    tail = ent[-t:]

    tail_mean = np.nanmean(tail)
    tail_std = np.nanstd(tail)

    # 相对坍塌阈值：后段均值低于 baseline 的 20%
    rel_drop = tail_mean / (baseline + 1e-12)

    # 绝对阈值（可按你的 token space 调）：熵接近 0.1 往往已经很尖了
    abs_low = tail_mean < 0.1

    # “稳定贴地”：波动也很小（比如 std < 0.05 * baseline）
    stable_low = tail_std < 0.05 * (baseline + 1e-12)

    # 加强证据：ent_min 贴近 0
    min_evidence = None
    if ent_min is not None:
        ent_min = np.asarray(ent_min)
        ent_min = ent_min[np.isfinite(ent_min)]
        if len(ent_min) > 0:
            tail_min_mean = np.nanmean(ent_min[-t:])
            min_evidence = tail_min_mean
        else:
            min_evidence = None

    # 判定：相对跌幅很大 + (绝对低 或 稳定低)
    collapse = (rel_drop < 0.2) and (abs_low or stable_low)

    details = {
        "baseline_entropy_mean(first_10%)": float(baseline),
        "tail_entropy_mean(last_20%)": float(tail_mean),
        "tail_entropy_std(last_20%)": float(tail_std),
        "tail_vs_baseline_ratio": float(rel_drop),
        "abs_low(tail_mean<0.1)": bool(abs_low),
        "stable_low(tail_std<0.05*baseline)": bool(stable_low),
    }
    if min_evidence is not None:
        details["tail_entropy_min_mean(last_20%)"] = float(min_evidence)
    return collapse, details


def main(csv_path: str):
    df = pd.read_csv(csv_path)

    df, step_col, ent_col, entmin_col = pick_entropy_series(df)

    if ent_col is None:
        print("❌ 没找到 entropy 列。你需要确保 PlanR1.py 里 log 了 train_entropy_mean。")
        print("当前 columns：")
        print(df.columns.tolist())
        sys.exit(1)

    # 取序列并排序（Lightning 有时会乱序写入）
    sub_cols = [step_col, ent_col]
    if entmin_col is not None:
        sub_cols.append(entmin_col)

    d = df[sub_cols].dropna(subset=[ent_col]).copy()
    d = d.sort_values(step_col)

    step = d[step_col].to_numpy()
    ent = d[ent_col].to_numpy()
    ent_s = smooth_series(step, ent, window=200)

    ent_min = None
    entmin_s = None
    if entmin_col is not None:
        ent_min = d[entmin_col].to_numpy()
        entmin_s = smooth_series(step, ent_min, window=200)

    collapse, details = detect_entropy_collapse(step, ent, ent_min=ent_min)

    # 打印结论
    print("\n===== Entropy Collapse Check =====")
    for k, v in details.items():
        print(f"{k}: {v}")
    print("Result:", "⚠️ 可能发生熵坍塌" if collapse else "✅ 暂无明显熵坍塌证据")

    # 画图
    plt.figure()
    plt.plot(step, ent, alpha=0.25, label=ent_col)
    plt.plot(step, ent_s, label=f"{ent_col} (rolling mean)")

    if entmin_col is not None:
        plt.plot(step, ent_min, alpha=0.25, label=entmin_col)
        plt.plot(step, entmin_s, label=f"{entmin_col} (rolling mean)")

    plt.xlabel("step")
    plt.ylabel("entropy")
    plt.title("Entropy over Training Steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out = "entropy_curve.png"
    plt.savefig(out, dpi=200)
    print(f"\n📈 Saved plot to: {out}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_entropy_collapse.py /path/to/metrics.csv")
        sys.exit(1)
    main(sys.argv[1])

# (planr1) gaosunxiang@admin123-ESC8000A-E12:~/Plan-R1/analysis$ python analyze_entropy_collapse.py /home/gaosunxiang/Plan-R1/lightning_logs/plan/version_0/metrics.csv

# ===== Entropy Collapse Check =====
# baseline_entropy_mean(first_10%): 1.9967834183147974
# tail_entropy_mean(last_20%): 1.5925783493689127
# tail_entropy_std(last_20%): 0.17816682439155548
# tail_vs_baseline_ratio: 0.7975719022707958
# abs_low(tail_mean<0.1): False
# stable_low(tail_std<0.05*baseline): False
# tail_entropy_min_mean(last_20%): 0.3561877544437136
# Result: ✅ 暂无明显熵坍塌证据

# 📈 Saved plot to: entropy_curve.png