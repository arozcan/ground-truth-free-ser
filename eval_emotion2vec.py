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
    """val.csv dosyasını oku; gerekli kolonları doğrula ve opsiyonel TTS filtresi uygula."""
    df = pd.read_csv(csv_path)
    required = {"wav_path", "emotion", "tts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Temizlik
    df = df.copy()
    df["emotion"] = df["emotion"].str.lower().str.strip()
    df["tts"] = df["tts"].str.lower().str.strip()
    if "prompt_id" in df.columns:
        df["prompt_id"] = df["prompt_id"].astype(str)

    if tts_filter:
        keep = {t.strip().lower() for t in tts_filter.split(",")}
        df = df[df["tts"].isin(keep)]

    # Var olmayan wav dosyalarını düş
    exists_mask = df["wav_path"].apply(lambda p: isinstance(p, str) and os.path.exists(p))
    missing_count = (~exists_mask).sum()
    if missing_count:
        print(f"[WARN] Missing audio files skipped: {missing_count}")
        df = df[exists_mask]

    if df.empty:
        raise ValueError("After filtering, no rows remain. Check CSV path / tts_filter / file paths.")
    return df

def predict_emotion(model, wav_path):
    """FunASR ile tek dosya için en yüksek skorlu etiketi döndür."""
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
        top = cleaned[scores.index(max(scores))]
        return top if top in ALL_LABELS else "<unk>"
    except Exception as e:
        print(f"[ERR] Prediction failed for {wav_path}: {e}")
        return "<unk>"

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
    args = parser.parse_args()

    # 1) CSV'yi yükle
    df = load_csv(args.csv_path, args.tts_filter)
    if args.limit is not None:
        df = df.iloc[:args.limit].reset_index(drop=True)
        print(f"[INFO] Using only first {len(df)} rows due to --limit.")

    # 2) FunASR model
    print("[INFO] Loading FunASR model: iic/emotion2vec_plus_large")
    model = AutoModel(model="iic/emotion2vec_plus_large", hub="hf", disable_update=True)

    # 3) Tahminler
    y_true, y_pred, tts_list = [], [], []
    per_tts_true = defaultdict(list)
    per_tts_pred = defaultdict(list)

    for i, row in df.iterrows():
        wav = row["wav_path"]
        gt = row["emotion"]
        tts = row["tts"]
        pred = predict_emotion(model, wav)

        y_true.append(gt)
        y_pred.append(pred)
        tts_list.append(tts)

        per_tts_true[tts].append(gt)
        per_tts_pred[tts].append(pred)

        if (i + 1) % 50 == 0:
            print(f"[INFO] Processed {i+1}/{len(df)}")

    # 4) Genel sonuçlar
    overall_acc, overall_cm = evaluate_overall(y_true, y_pred, title="Overall")

    # 5) TTS-bazlı sonuçlar
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