#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

EMOTIONS = ['angry','disgusted','fearful','happy','neutral','sad','surprised']

def load_and_merge(baseline_csv, proposed_csv):
    db = pd.read_csv(baseline_csv)
    dp = pd.read_csv(proposed_csv)
    # normalize
    for df in (db, dp):
        df["emotion"] = df["emotion"].str.strip().str.lower()
        df["emotion_pred"] = df["emotion_pred"].str.strip().str.lower()
    # drop dups if any
    db = db.drop_duplicates(subset=["wav_path"])
    dp = dp.drop_duplicates(subset=["wav_path"])
    # join on wav_path
    merged = pd.merge(
        db[["wav_path","emotion","emotion_pred","tts","prompt_id"]].rename(
            columns={"emotion_pred":"emotion_pred_baseline",
                     "tts":"tts_base","prompt_id":"pid_base"}),
        dp[["wav_path","emotion_pred","tts","prompt_id","eas","eas_x7","top_score"]].rename(
            columns={"emotion_pred":"emotion_pred_prop",
                     "tts":"tts_prop","prompt_id":"pid_prop",
                     "top_score":"top_score_prop"}),
        on="wav_path", how="inner"
    )
    # trust baseline labels to equal proposed labels' ground truth (sanity)
    # but keep 'emotion' from baseline as y_true
    merged = merged.dropna(subset=["emotion","emotion_pred_baseline","emotion_pred_prop"])
    merged = merged[merged["emotion"].isin(EMOTIONS)]
    return merged

def mcnemar_exact(base_ok, prop_ok):
    base_ok = np.asarray(base_ok, dtype=bool)
    prop_ok = np.asarray(prop_ok, dtype=bool)
    b = int(np.sum(base_ok & ~prop_ok))
    c = int(np.sum(~base_ok & prop_ok))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0}
    p = binomtest(min(b,c), n=n, p=0.5, alternative='two-sided').pvalue
    return {"b": b, "c": c, "p": float(p)}

def acc_f1(y_true, y_pred):
    acc = np.mean(np.asarray(y_true)==np.asarray(y_pred))
    f1m = f1_score(y_true, y_pred, labels=EMOTIONS, average="macro", zero_division=0)
    return acc, f1m

def bootstrap_ci(y_true, yb, yp, B=10000, seed=1337):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    yb = np.asarray(yb)
    yp = np.asarray(yp)
    n = len(y_true)
    d_acc, d_f1 = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        acc_b, f1_b = acc_f1(y_true[idx], yb[idx])
        acc_p, f1_p = acc_f1(y_true[idx], yp[idx])
        d_acc.append(acc_p - acc_b)
        d_f1.append(f1_p - f1_b)
    d_acc = np.sort(d_acc)
    d_f1  = np.sort(d_f1)
    ci_acc = (float(d_acc[int(0.025*B)]), float(d_acc[int(0.975*B)-1]))
    ci_f1  = (float(d_f1[int(0.025*B)]),  float(d_f1[int(0.975*B)-1]))
    # one-sided p approx: P(Delta<=0)
    p_acc = float(np.mean(np.array(d_acc) <= 0.0))
    p_f1  = float(np.mean(np.array(d_f1) <= 0.0))
    return {"delta_acc_ci": ci_acc, "delta_f1_ci": ci_f1,
            "bootstrap_p_acc": p_acc, "bootstrap_p_f1": p_f1}

def per_group_mcnemar(df, by="emotion"):
    out = []
    for g, sub in df.groupby(by):
        y = sub["emotion"].tolist()
        yb = sub["emotion_pred_baseline"].tolist()
        yp = sub["emotion_pred_prop"].tolist()
        base_ok = (np.array(yb)==np.array(y))
        prop_ok = (np.array(yp)==np.array(y))
        mc = mcnemar_exact(base_ok, prop_ok)
        acc_b, f1_b = acc_f1(y, yb)
        acc_p, f1_p = acc_f1(y, yp)
        out.append({
            by: g, "n": len(sub),
            "acc_base": acc_b, "acc_prop": acc_p, "acc_delta": acc_p-acc_b,
            "f1_base": f1_b, "f1_prop": f1_p, "f1_delta": f1_p-f1_b,
            "b": mc["b"], "c": mc["c"], "mcnemar_p": mc["p"]
        })
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_csv", required=True,
                    help="baseline CSV: tts,prompt_id,wav_path,emotion,emotion_pred,top_score")
    ap.add_argument("--proposed_csv", required=True,
                    help="proposed CSV: tts,prompt_id,wav_path,emotion,emotion_pred,top_score,eas,eas_x7")
    ap.add_argument("--out_dir", default="results/sigtest_out")
    ap.add_argument("--bootstrap", type=int, default=10000, help="#resamples")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_and_merge(args.baseline_csv, args.proposed_csv)

    y  = df["emotion"].tolist()
    yb = df["emotion_pred_baseline"].tolist()
    yp = df["emotion_pred_prop"].tolist()

    acc_b, f1_b = acc_f1(y, yb)
    acc_p, f1_p = acc_f1(y, yp)

    base_ok = (np.array(yb)==np.array(y))
    prop_ok = (np.array(yp)==np.array(y))
    mc = mcnemar_exact(base_ok, prop_ok)

    boot = bootstrap_ci(y, yb, yp, B=args.bootstrap)

    summary = {
        "N": int(len(df)),
        "overall": {
            "accuracy_base": acc_b, "accuracy_prop": acc_p, "delta_acc": acc_p-acc_b,
            "macroF1_base": f1_b, "macroF1_prop": f1_p, "delta_macroF1": f1_p-f1_b,
            "mcnemar_b": mc["b"], "mcnemar_c": mc["c"], "mcnemar_p": mc["p"],
            **boot
        }
    }

    # per-emotion
    per_emo = per_group_mcnemar(df, by="emotion")
    per_emo.to_csv(os.path.join(args.out_dir, "per_emotion_mcnemar.csv"), index=False)
    summary["per_emotion"] = per_emo.to_dict(orient="records")

    # per-TTS (opsiyonel)
    if "tts_base" in df.columns:
        per_tts = per_group_mcnemar(df.assign(tts=df["tts_base"]), by="tts")
        per_tts.to_csv(os.path.join(args.out_dir, "per_tts_mcnemar.csv"), index=False)
        summary["per_tts"] = per_tts.to_dict(orient="records")

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # pretty print
    print("\n=== OVERALL ===")
    print(f"N={summary['N']}")
    print(f"Acc: base={acc_b*100:.2f}%  prop={acc_p*100:.2f}%  Δ={ (acc_p-acc_b)*100:.2f} pp")
    print(f"Macro-F1: base={f1_b:.4f}  prop={f1_p:.4f}  Δ={ (f1_p-f1_b):.4f}")
    print(f"McNemar exact: b={mc['b']}, c={mc['c']}, p={mc['p']:.4g}")
    print(f"Bootstrap 95% CI Δacc: [{boot['delta_acc_ci'][0]*100:.2f}, {boot['delta_acc_ci'][1]*100:.2f}] pp"
          f" (p≈{boot['bootstrap_p_acc']:.4f})")
    print(f"Bootstrap 95% CI ΔmacroF1: [{boot['delta_f1_ci'][0]:.4f}, {boot['delta_f1_ci'][1]:.4f}]"
          f" (p≈{boot['bootstrap_p_f1']:.4f})")

    print("\n=== PER-EMOTION (McNemar exact) ===")
    print(per_emo.to_string(index=False))

if __name__ == "__main__":
    main()