#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

EMOTIONS = ['angry','disgusted','fearful','happy','neutral','sad','surprised']

# ----------------------------
# Utilities
# ----------------------------
def fisher_ci(r: float, n: int, alpha: float = 0.05):
    """95% CI of Pearson r via Fisher's z."""
    r = max(min(float(r), 0.999999), -0.999999)
    if n <= 3:
        return np.nan, np.nan
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1.0 / np.sqrt(n - 3)
    zcrit = 1.959963984540054  # ~97.5th for 95% CI
    lo, hi = z - zcrit * se, z + zcrit * se
    r_lo = (np.exp(2*lo) - 1) / (np.exp(2*lo) + 1)
    r_hi = (np.exp(2*hi) - 1) / (np.exp(2*hi) + 1)
    return float(r_lo), float(r_hi)

def corr_metrics(x, y):
    """Pearson & Spearman + CI (x: EAS, y: rating)."""
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = len(x)
    if n < 3:
        return dict(n=int(n), pearson_r=np.nan, pearson_ci_lo=np.nan, pearson_ci_hi=np.nan,
                    pearson_p=np.nan, spearman_rho=np.nan, spearman_p=np.nan)
    pr, pp = pearsonr(x, y)
    lo, hi = fisher_ci(pr, n)
    sr, sp = spearmanr(x, y)
    return dict(
        n=int(n),
        pearson_r=float(pr),
        pearson_ci_lo=float(lo),
        pearson_ci_hi=float(hi),
        pearson_p=float(pp),
        spearman_rho=float(sr),
        spearman_p=float(sp),
    )

def fisher_macro_from_per_emotion(per_emo_df: pd.DataFrame):
    """
    Fisher-z ağırlıklı makro ortalama ve %95 CI.
    Ağırlıklar: w_i = n_i - 3 (Fisher se = 1/sqrt(n-3)).
    Girdi: per_emo_df, kolonlar: ['emotion','n','pearson_r',...]
    Çıktı: dict
    """
    df = per_emo_df.copy()
    # Geçerli (finite) r ve n>=4 olanlar
    mask = np.isfinite(df["pearson_r"].to_numpy()) & np.isfinite(df["n"].to_numpy())
    df = df[mask]
    df = df[df["n"] >= 4]
    if df.empty:
        return {
            "r_macro": np.nan,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "sum_weights": 0.0,
            "k_used": 0,
        }

    r = df["pearson_r"].astype(float).to_numpy()
    n = df["n"].astype(int).to_numpy()

    # Fisher z ve ağırlıklar
    r = np.clip(r, -0.999999, 0.999999)
    z = 0.5 * np.log((1 + r) / (1 - r))
    w = (n - 3).astype(float)
    w = np.where(w > 0, w, 0.0)

    if w.sum() <= 0:
        return {
            "r_macro": np.nan,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "sum_weights": 0.0,
            "k_used": int(len(df)),
        }

    z_bar = np.sum(w * z) / np.sum(w)
    se = 1.0 / np.sqrt(np.sum(w))
    zcrit = 1.959963984540054

    z_lo, z_hi = z_bar - zcrit * se, z_bar + zcrit * se
    r_macro = np.tanh(z_bar)
    r_lo = np.tanh(z_lo)
    r_hi = np.tanh(z_hi)

    return {
        "r_macro": float(r_macro),
        "ci_lo": float(r_lo),
        "ci_hi": float(r_hi),
        "sum_weights": float(w.sum()),
        "k_used": int(len(df)),
    }

def make_scatter(df, out_png, x_label="EAS"):
    x = pd.to_numeric(df["eas"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["rating"], errors="coerce").to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]

    plt.figure(figsize=(6,4), dpi=400)
    plt.scatter(x, y, s=18, alpha=0.6)
    if len(x) >= 2:
        b1, b0 = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 200)
        plt.plot(xs, b1*xs + b0, linewidth=2)
    # Başlık yok—sadece eksen etiketleri
    plt.xlabel(x_label)
    plt.ylabel("Human Rating")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings_csv", default="ratings_aggregated/ratings_average.csv",
                    help="Columns: emotion,wav_path,rating,naturalness")
    ap.add_argument("--eas_csv", default="results/eas_scores.csv",
                    help="Columns: tts,prompt_id,wav_path,emotion,emotion_pred,top_score,eas,eas_x7")
    ap.add_argument("--out_dir", default="results/eas_human_corr_out",
                    help="Output directory")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load
    df_r = pd.read_csv(args.ratings_csv)
    df_e = pd.read_csv(args.eas_csv)

    # --- Column checks
    for c in ["emotion", "wav_path", "rating"]:
        if c not in df_r.columns:
            raise ValueError(f"ratings CSV missing column: {c}")
    if "wav_path" not in df_e.columns:
        raise ValueError("EAS CSV missing column: wav_path")

    # Prefer eas_x7 if present (1..7), else 'eas' (0..1)
    use_eas_col = "eas_x7" if "eas_x7" in df_e.columns else "eas"
    if use_eas_col not in df_e.columns:
        raise ValueError("EAS CSV must contain 'eas' or 'eas_x7'")

    # Keep only needed columns
    df_r = df_r[["emotion", "wav_path", "rating"]].copy()
    df_e = (
        df_e[["wav_path", use_eas_col, "emotion"]].copy()
        if "emotion" in df_e.columns else
        df_e[["wav_path", use_eas_col]].copy()
    )

    # --- Merge on wav_path
    merged = pd.merge(df_r, df_e, on="wav_path", how="inner")
    merged = merged.rename(columns={use_eas_col: "eas"})

    # Normalize emotion text (handle possible emotion_x / emotion_y from merge)
    if "emotion" not in merged.columns:
        if "emotion_x" in merged.columns:
            merged = merged.rename(columns={"emotion_x": "emotion"})
        elif "emotion_y" in merged.columns:
            merged = merged.rename(columns={"emotion_y": "emotion"})
    if "emotion_x" in merged.columns and "emotion_y" in merged.columns:
        merged["emotion"] = merged["emotion_x"]
        merged = merged.drop(columns=["emotion_x", "emotion_y"])
    if "emotion" in merged.columns:
        merged["emotion"] = merged["emotion"].astype(str).str.lower()
    else:
        raise KeyError("emotion column missing after merge; check inputs (ratings CSV must have 'emotion').")

    # Drop NaNs
    merged["rating"] = pd.to_numeric(merged["rating"], errors="coerce")
    merged["eas"]    = pd.to_numeric(merged["eas"], errors="coerce")
    merged = merged[np.isfinite(merged["rating"]) & np.isfinite(merged["eas"])]

    # Save merged for reproducibility
    merged_out = os.path.join(args.out_dir, "merged.csv")
    merged.to_csv(merged_out, index=False)

    # --- Overall correlations
    overall = corr_metrics(merged["eas"], merged["rating"])
    with open(os.path.join(args.out_dir, "overall.json"), "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    # --- Agreement at fixed midpoint (≥ 4)
    if use_eas_col == "eas_x7":
        thr_eas = 4.0       # EAS is on 1..7
        eas_label = "EAS ×7"
    else:
        thr_eas = 4.0 / 7.0 # EAS is on 0..1
        eas_label = "EAS"
    thr_hum = 4.0          # Human rating midpoint (1..7)

    agreement = (
        ((merged["eas"] >= thr_eas) & (merged["rating"] >= thr_hum)) |
        ((merged["eas"]  < thr_eas) & (merged["rating"]  < thr_hum))
    ).mean()

    with open(os.path.join(args.out_dir, "agreement.json"), "w", encoding="utf-8") as f:
        json.dump({
            "agreement_accuracy_midpoint": float(agreement),
            "thresholds": {
                "eas_threshold": float(thr_eas),
                "human_threshold": 4.0,
                "eas_scale": "1..7" if use_eas_col == "eas_x7" else "0..1"
            }
        }, f, indent=2)

    print("\n=== AGREEMENT ACCURACY ===")
    print(f"Fixed-midpoint (≥4): {agreement:.2%} (thr {eas_label}={thr_eas:.3f}, thr Human=4.000)")

    # --- Per emotion correlations
    rows = []
    for emo in EMOTIONS:
        sub = merged[merged["emotion"] == emo]
        met = corr_metrics(sub["eas"], sub["rating"])
        met["emotion"] = emo
        rows.append(met)
    per_emo = pd.DataFrame(rows)[[
        "emotion","n","pearson_r","pearson_ci_lo","pearson_ci_hi","pearson_p","spearman_rho","spearman_p"
    ]]
    per_emo_csv = os.path.join(args.out_dir, "per_emotion.csv")
    per_emo.to_csv(per_emo_csv, index=False)

    # --- Fisher-z macro across emotions
    macro = fisher_macro_from_per_emotion(per_emo)
    macro_json = os.path.join(args.out_dir, "per_emotion_macro_fisher.json")
    with open(macro_json, "w", encoding="utf-8") as f:
        json.dump(macro, f, indent=2, ensure_ascii=False)

    # --- Scatter (overall)
    xlab = "EAS ×7" if use_eas_col == "eas_x7" else "EAS"
    make_scatter(merged, os.path.join(args.out_dir, "scatter_overall.png"), x_label=xlab)

    # --- Console prints
    print("\n=== OVERALL ===")
    print(json.dumps(overall, indent=2))
    print("\n=== PER EMOTION ===")
    print(per_emo.to_string(index=False))
    print("\n=== PER-EMOTION MACRO (Fisher-z) ===")
    print(f"r_macro={macro['r_macro']:.3f}  95% CI [{macro['ci_lo']:.3f}, {macro['ci_hi']:.3f}] "
          f"(k={macro['k_used']}, sum_w={macro['sum_weights']:.1f})")

    print(f"\n[OK] Saved merged -> {merged_out}")
    print(f"[OK] Saved per-emotion -> {per_emo_csv}")
    print(f"[OK] Saved macro -> {macro_json}")
    print(f"[OK] Scatter -> {os.path.join(args.out_dir, 'scatter_overall.png')}")

if __name__ == "__main__":
    main()