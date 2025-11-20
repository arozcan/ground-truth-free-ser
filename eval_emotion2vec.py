#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from funasr import AutoModel

ALL_LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised', '<unk>']

def load_csv(csv_path, tts_filter=None):
    """Load val.csv; validate required columns and optionally apply a TTS filter."""
    df = pd.read_csv(csv_path)
    required = {"wav_path", "emotion", "tts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Cleanup and normalization
    df = df.copy()
    df["emotion"] = df["emotion"].str.lower().str.strip()
    df["tts"] = df["tts"].str.lower().str.strip()
    if "prompt_id" in df.columns:
        df["prompt_id"] = df["prompt_id"].astype(str)

    if tts_filter:
        keep = {t.strip().lower() for t in tts_filter.split(",")}
        df = df[df["tts"].isin(keep)]

    # Drop rows with missing or non-existent wav files
    exists_mask = df["wav_path"].apply(lambda p: isinstance(p, str) and os.path.exists(p))
    missing_count = (~exists_mask).sum()
    if missing_count:
        print(f"[WARN] Missing audio files skipped: {missing_count}")
        df = df[exists_mask]

    if df.empty:
        raise ValueError("After filtering, no rows remain. Check CSV path / tts_filter / file paths.")
    return df

def predict_emotion(model, wav_path):
    """Use FunASR to return the top-scoring label and its score for a single file."""
    try:
        result = model.generate(
            wav_path,
            output_dir="./outputs",
            granularity="utterance",
            extract_embedding=False
        )
        labels = result[0]["labels"]          # e.g. ["emotion/angry", ...]
        scores = result[0]["scores"]
        cleaned = [lab.split("/")[-1] for lab in labels]
        # top-1
        if isinstance(scores, (list, tuple)) and len(scores) == len(cleaned) and len(scores) > 0:
            max_idx = int(scores.index(max(scores)))
            top = cleaned[max_idx]
            top_score = float(scores[max_idx])
        else:
            # Unexpected format; still try to return a label
            top = cleaned[0] if cleaned else "<unk>"
            top_score = None
        top = top if top in ALL_LABELS else "<unk>"
        return top, top_score
    except Exception as e:
        print(f"[ERR] Prediction failed for {wav_path}: {e}")
        return "<unk>", None

def evaluate_overall(y_true, y_pred, title="Overall"):
    """Genel metrikleri yazdır."""
    print(f"\n=== {title} ===")
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=ALL_LABELS, digits=4, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=ALL_LABELS)
    cm_df = pd.DataFrame(cm, index=ALL_LABELS, columns=ALL_LABELS)
    print("\nConfusion Matrix (Predicted ↓ / True →):")
    print(cm_df)
    return acc, cm_df

def main():
    parser = argparse.ArgumentParser(description="Evaluate Emotion2Vec+ on val.csv (wav_path,emotion,prompt_id,tts,...)")
    parser.add_argument("--csv_path", type=str, default="csv_with_emb_dual/val.csv",
                        help="Path to CSV with columns: wav_path,emotion,prompt_id,tts,...")
    parser.add_argument("--tts_filter", type=str, default=None,
                        help="Comma-separated TTS subset to evaluate (e.g., 'azure,openai'). If omitted, use all.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of rows for quick runs.")
    parser.add_argument("--mode", type=str, choices=["eval", "dump"], default="eval",
                        help="Run mode: 'eval' prints metrics, 'dump' saves per-sample predictions to CSV and exits.")
    parser.add_argument("--out_csv", type=str, default="results/emotion2vec_scores.csv",
                        help="When --mode dump, path to save per-sample predictions CSV.")
    args = parser.parse_args()

    # 1) Load CSV
    df = load_csv(args.csv_path, args.tts_filter)
    if args.limit is not None:
        df = df.iloc[:args.limit].reset_index(drop=True)
        print(f"[INFO] Using only first {len(df)} rows due to --limit.")

    # 2) FunASR model
    print("[INFO] Loading FunASR model: iic/emotion2vec_plus_large")
    model = AutoModel(model="iic/emotion2vec_plus_large", hub="hf", disable_update=True)

    # 3) Predictions
    y_true, y_pred, tts_list = [], [], []
    per_tts_true = defaultdict(list)
    per_tts_pred = defaultdict(list)
    rows = []  # dump modu için satırlar

    for i, row in df.iterrows():
        wav = row["wav_path"]
        gt = row["emotion"]
        tts = row["tts"]
        pid = str(row["prompt_id"]) if "prompt_id" in df.columns else None

        pred_label, pred_score = predict_emotion(model, wav)

        y_true.append(gt)
        y_pred.append(pred_label)
        tts_list.append(tts)

        per_tts_true[tts].append(gt)
        per_tts_pred[tts].append(pred_label)

        # Keep one row for dump mode
        rows.append({
            "tts": tts,
            "prompt_id": pid,
            "wav_path": wav,
            "emotion": gt,
            "emotion_pred": pred_label,
            "top_score": pred_score,
        })

        if (i + 1) % 50 == 0:
            print(f"[INFO] Processed {i+1}/{len(df)}")

    # 3.5) Dump mode: save to CSV and exit
    if args.mode == "dump":
        out_dir = os.path.dirname(args.out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_df = pd.DataFrame(rows)
        out_df.to_csv(args.out_csv, index=False)
        print(f"[OK] Saved per-sample predictions to: {args.out_csv} (N={len(out_df)})")
        return

    # 4) Overall results
    overall_acc, overall_cm = evaluate_overall(y_true, y_pred, title="Overall")

    # 5) Per-TTS results
    print("\n=== Per-TTS Breakdown ===")
    summary_rows = []
    for tts in sorted(per_tts_true.keys()):
        yt = per_tts_true[tts]
        yp = per_tts_pred[tts]
        acc, _ = evaluate_overall(yt, yp, title=f"TTS: {tts}")
        summary_rows.append((tts, len(yt), acc))

    if summary_rows:
        print("\nPer-TTS Summary:")
        print(f"{'TTS':<16}{'N':>8}{'Acc':>10}")
        for tts, n, acc in summary_rows:
            print(f"{tts:<16}{n:>8}{acc*100:>9.2f}%")

if __name__ == "__main__":
    main()