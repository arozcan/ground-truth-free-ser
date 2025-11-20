# prepare_embeddings.py
# Purpose: run extract_embeddings.py sequentially for E2V+ and WavLM,
# then merge the resulting CSVs by wav_path (emb_e2v + emb_wavlm).

from pathlib import Path
import subprocess
import sys
import pandas as pd

# ==== FIXED SETTINGS (internally defined) ====
EXTRACT_SCRIPT = "extract_embeddings.py"

IN_CSVS = ["csv/train.csv", "csv/val.csv"]

# E2V+
E2V_OUT_DIR = "csv_with_emb_e2vplus"
E2V_EMB_ROOT = "embeddings/e2vplus"

# WavLM
WAVLM_OUT_DIR = "csv_with_emb_wavlm"
WAVLM_EMB_ROOT = "embeddings/wavlm_large"
WAVLM_MODEL_ID = "microsoft/wavlm-large"
WAVLM_POOLING = "mean"   # "mean" | "cls"
WAVLM_LAYER = None      # int or None (use all layers)
WAVLM_LAYER_RANGE = [6, 12]  # [start, end) 0=embeddings, 1..N=transformer layers
DEMEAN = False            # True | False

L2NORM = True
OVERWRITE = False

# Merged outputs
MERGED_OUT_DIR = "csv_with_emb_dual"
# ===========================================


def run(cmd: list):
    print("[CMD]", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout.rstrip())
    if res.returncode != 0:
        if res.stderr:
            print(res.stderr.rstrip(), file=sys.stderr)
        raise SystemExit(f"[ERROR] command failed with code {res.returncode}")
    return res


def merge_one_pair(e2v_csv: str, wavlm_csv: str, out_csv: str):
    df_e2v = pd.read_csv(e2v_csv)
    df_wav = pd.read_csv(wavlm_csv)

    # expected columns
    if "wav_path" not in df_e2v or "emb_path" not in df_e2v:
        raise ValueError(f"{e2v_csv} does not contain the required 'wav_path' and 'emb_path' columns.")
    if "wav_path" not in df_wav or "emb_path" not in df_wav:
        raise ValueError(f"{wavlm_csv} does not contain the required 'wav_path' and 'emb_path' columns.")

    df_e2v = df_e2v.rename(columns={"emb_path": "emb_e2v"})
    df_wav = df_wav.rename(columns={"emb_path": "emb_wavlm"})
    df_wav = df_wav[["wav_path", "emb_wavlm"]]

    merged = pd.merge(df_e2v, df_wav, on="wav_path", how="inner")

    # report
    lost_e2v = len(df_e2v) - len(merged)
    lost_wav = len(df_wav) - len(merged)
    if lost_e2v or lost_wav:
        print(f"[WARN] Unmatched rows during merge: e2vplus={lost_e2v}, wavlm={lost_wav}")

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"[OK] merged -> {out_csv}  (rows={len(merged)})")


def main():
    # prepare directories
    for p in [E2V_OUT_DIR, E2V_EMB_ROOT, WAVLM_OUT_DIR, WAVLM_EMB_ROOT, MERGED_OUT_DIR]:
        Path(p).mkdir(parents=True, exist_ok=True)

    # 1) extract e2vplus embeddings
    cmd_e2v = [
        sys.executable, EXTRACT_SCRIPT,
        "--backend", "e2vplus",
        "--out_dir", E2V_OUT_DIR,
        "--emb_root", E2V_EMB_ROOT,
        "--in_csvs", *IN_CSVS,
    ]
    if L2NORM:
        cmd_e2v.append("--l2norm")
    if OVERWRITE:
        cmd_e2v.append("--overwrite")
    run(cmd_e2v)

    # 2) extract wavlm embeddings
    cmd_wav = [
        sys.executable, EXTRACT_SCRIPT,
        "--backend", "wavlm",
        "--model_id", WAVLM_MODEL_ID,
        "--out_dir", WAVLM_OUT_DIR,
        "--emb_root", WAVLM_EMB_ROOT,
        "--pooling", WAVLM_POOLING,
        "--in_csvs", *IN_CSVS,
    ]
    # single layer or range: do not provide both
    if WAVLM_LAYER is not None:
        cmd_wav.extend(["--wavlm_layer", str(WAVLM_LAYER)])
    elif WAVLM_LAYER_RANGE is not None:
        # nargs=2 requires passing two separate values
        cmd_wav.extend(["--wavlm_layer_range", *map(str, WAVLM_LAYER_RANGE)])

    if L2NORM:
        cmd_wav.append("--l2norm")
    if OVERWRITE:
        cmd_wav.append("--overwrite")
    if DEMEAN:
        cmd_wav.append("--demean")

    run(cmd_wav)

    # 3) merge (train/val)
    for in_csv in IN_CSVS:
        name = Path(in_csv).name
        e2v_csv = str(Path(E2V_OUT_DIR) / name)
        wav_csv = str(Path(WAVLM_OUT_DIR) / name)
        out_csv = str(Path(MERGED_OUT_DIR) / name)
        merge_one_pair(e2v_csv, wav_csv, out_csv)

    print("[DONE] All embeddings prepared and merged.")


if __name__ == "__main__":
    main()