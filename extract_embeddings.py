# extract_embeddings.py
import os
import csv
import argparse
import hashlib
from pathlib import Path
from typing import Optional, List

import numpy as np

# ---- Common helpers ----------------------------------------------------------
def sha1_of_path(p: str) -> str:
    return hashlib.sha1(os.path.abspath(p).encode("utf-8")).hexdigest()

def ensure_dir(p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

# ---- Backend: FunASR / emotion2vec+ -----------------------------------------
_FUNASR_MODEL = None
def get_funasr_model(model_id: str = "iic/emotion2vec_plus_large"):
    global _FUNASR_MODEL
    if _FUNASR_MODEL is None:
        from funasr import AutoModel
        _FUNASR_MODEL = AutoModel(model=model_id, hub="hf", disable_update=True)
    return _FUNASR_MODEL

def extract_e2vplus_embedding(
    wav_path: str,
    model_id: str = "iic/emotion2vec_plus_large",
    keys_try: Optional[List[str]] = None,
    pooling: str = "mean",  # "mean" only matters when the output is multi-dimensional
) -> np.ndarray:
    """
    Returns a single float32 embedding vector from the FunASR emotion2vec+ output.
    Tries multiple possible keys because different versions may use different field names.
    """
    if keys_try is None:
        keys_try = ["embedding", "embeddings", "feats", "vector", "vectors", "spk_embedding"]

    model = get_funasr_model(model_id)
    result = model.generate(
        wav_path,
        output_dir=None,
        granularity="utterance",
        extract_embedding=True
    )
    rec = result[0]

    for key in keys_try:
        if key in rec and rec[key] is not None:
            emb = np.asarray(rec[key], dtype=np.float32)
            if emb.ndim > 1:
                if pooling == "mean":
                    emb = emb.mean(axis=0)
                else:
                    # başka pooling tipi istenirse genişletilebilir
                    emb = emb.mean(axis=0)
            return emb.squeeze().astype(np.float32)

    raise RuntimeError(f"No embedding field in FunASR output. keys={list(rec.keys())}")

# ---- Backend: HF / WavLM -----------------------------------------------------
_WAVLM = None
_WAVLM_PROC = None
_DEVICE = None

def setup_device():
    import torch
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE

def get_wavlm(model_id: str = "microsoft/wavlm-large"):
    """
    Load the WavLM model and a suitable processor/feature extractor independently of transformers version.
    Tries AutoProcessor first, then WavLMFeatureExtractor, and finally Wav2Vec2FeatureExtractor as fallbacks.
    """
    global _WAVLM, _WAVLM_PROC
    if _WAVLM is not None and _WAVLM_PROC is not None:
        return _WAVLM, _WAVLM_PROC

    from transformers import WavLMModel
    # Processor/FeatureExtractor için çoklu fallback
    proc = None
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model_id)
    except Exception:
        # This class exists in some older versions
        try:
            from transformers import WavLMFeatureExtractor
            proc = WavLMFeatureExtractor.from_pretrained(model_id)
        except Exception:
            from transformers import Wav2Vec2FeatureExtractor
            proc = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

    import torch
    _WAVLM = WavLMModel.from_pretrained(model_id).to(setup_device())
    _WAVLM.eval()
    _WAVLM_PROC = proc
    return _WAVLM, _WAVLM_PROC

def extract_wavlm_embedding(
    wav_path: str,
    model_id: str = "microsoft/wavlm-large",
    target_sr: int = 16000,
    pooling: str = "mean",   # "mean" | "cls" | "max"
    wavlm_layer: Optional[int] = None,
    wavlm_layer_range: Optional[List[int]] = None,   # [start, end)
    demean: bool = False,
) -> np.ndarray:
    """
    WavLM hidden_states -> (layer selection/range) -> (optional demean) -> pooling -> single float32 vector.
    """
    import librosa
    import torch

    model, proc = get_wavlm(model_id)
    wav, sr = librosa.load(wav_path, sr=target_sr, mono=True)

    # AutoProcessor / FeatureExtractor
    inputs = proc(wav, sampling_rate=target_sr, return_tensors="pt")
    input_values = inputs["input_values"].to(setup_device())
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(setup_device())

    with torch.no_grad():
        # Request hidden_states
        out = model(input_values, attention_mask=attention_mask, output_hidden_states=True)

        # Layer selection
        if wavlm_layer is not None:
            hs = out.hidden_states[wavlm_layer]         # (1, T, D)
        else:
            # Make the layer range safe
            if not wavlm_layer_range:
                wavlm_layer_range = [6, 10]             # default: average of layers 6..9
            s, e = wavlm_layer_range
            s = max(0, s); e = min(len(out.hidden_states), e)
            assert e > s, "wavlm_layer_range invalid (end must be greater than start)"
            hs_stack = torch.stack(out.hidden_states[s:e], dim=0)  # (K, 1, T, D)
            hs = hs_stack.mean(0)                                  # (1, T, D)

        # (Optional) per-dimension demean: slightly reduces lexical bias
        if demean:
            hs = hs - hs.mean(dim=1, keepdim=True)

        # Temporal pooling
        if pooling == "cls" and hs.shape[1] > 0:
            emb = hs[:, 0].squeeze(0)                # (D,)
        elif pooling == "max":
            emb = hs.max(dim=1).values.squeeze(0)    # (D,)
        else:
            emb = hs.mean(dim=1).squeeze(0)          # (D,)

    return emb.detach().cpu().numpy().astype(np.float32)

# ---- Main workflow -----------------------------------------------------------
def process_csv(
    in_csv: str,
    out_csv: str,
    emb_root: str,
    backend: str,
    model_id: str,
    overwrite: bool = False,
    l2norm: bool = False,
    pooling: str = "mean",
    wavlm_layer: Optional[int] = None,
    wavlm_layer_range: Optional[List[int]] = None,
    demean: bool = False,
):
    """
    Input CSV: expects at least 'wav_path' and 'emotion' (keeps 'prompt_id' if present).
    Output CSV: writes the same rows with an added 'emb_path' column.
    """
    rows = []
    with open(in_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        out_fields = list(dict.fromkeys(fieldnames + ["emb_path"]))
        for r in reader:
            wav_path = r["wav_path"]
            h = sha1_of_path(wav_path)
            emb_path = os.path.join(emb_root, f"{h}.npy")
            r["emb_path"] = emb_path

            need_write = overwrite or (not os.path.exists(emb_path))
            if need_write:
                try:
                    if backend == "e2vplus":
                        emb = extract_e2vplus_embedding(wav_path, model_id=model_id, pooling=pooling)
                    elif backend == "wavlm":
                        emb = extract_wavlm_embedding(
                            wav_path,
                            model_id=model_id,
                            pooling=pooling,
                            wavlm_layer=wavlm_layer,
                            wavlm_layer_range=wavlm_layer_range,
                            demean=demean,
                        )
                    else:
                        raise ValueError(f"Unknown backend: {backend}")

                    if l2norm:
                        n = np.linalg.norm(emb) + 1e-12
                        emb = (emb / n).astype(np.float32)

                    ensure_dir(emb_path)
                    np.save(emb_path, emb)
                except Exception as e:
                    print(f"[WARN] embedding failed: {wav_path} -> {e}")
                    r["emb_path"] = ""  # leave empty if embedding failed

            rows.append(r)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[OK] Wrote {out_csv}  (embeddings at: {emb_root})")

def main():
    ap = argparse.ArgumentParser(description="Extract embeddings and add emb_path to CSVs.")
    ap.add_argument("--backend", type=str, default="wavlm",
                    choices=["e2vplus", "wavlm"], help="Embedding backend")
    ap.add_argument("--model_id", type=str,
                    help="Model ID (HuggingFace or FunASR)")
    ap.add_argument("--in_csvs", nargs="+", default=["csv/train.csv", "csv/val.csv"],
                    help="Input CSV paths (e.g., train.csv val.csv)")
    ap.add_argument("--out_dir", type=str, default="csv_with_emb_wavlm",
                    help="Where to write updated CSVs (with emb_path column)")
    ap.add_argument("--emb_root", type=str, default="embeddings/wavlm_large",
                    help="Directory to store .npy embeddings")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute embeddings even if .npy exists")
    ap.add_argument("--l2norm", action="store_true", default=True,
                    help="Apply L2-normalization to embeddings")
    ap.add_argument("--pooling", type=str, default="mean",
                    choices=["mean", "cls","max"],
                    help="Temporal pooling strategy")
    ap.add_argument("--wavlm_layer", type=int, default=None,
                help="Use a single layer (0=embeddings, 1..N).")
    ap.add_argument("--wavlm_layer_range", nargs=2, type=int, default=[6, 10],
                    help="Average layers in the [start end) range, e.g., 6 10 -> layers 6,7,8,9.")
    ap.add_argument("--demean", action="store_true",
                    help="Subtract per-dimension mean over time from frame vectors before temporal pooling.")
    args = ap.parse_args()

    # Default model_id based on backend
    if not args.model_id:
        args.model_id = "iic/emotion2vec_plus_large" if args.backend == "e2vplus" else "microsoft/wavlm-large"

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.emb_root).mkdir(parents=True, exist_ok=True)

    # Load models once in advance (reduces latency on first call)
    if args.backend == "e2vplus":
        get_funasr_model(args.model_id)
    else:
        setup_device()
        get_wavlm(args.model_id)

    for in_csv in args.in_csvs:
        out_csv = os.path.join(args.out_dir, Path(in_csv).name)
        process_csv(
            in_csv, out_csv, args.emb_root,
            backend=args.backend,
            model_id=args.model_id,
            overwrite=args.overwrite,
            l2norm=args.l2norm,
            pooling=args.pooling,
            wavlm_layer=args.wavlm_layer,
            wavlm_layer_range=args.wavlm_layer_range,
            demean=args.demean,
        )

if __name__ == "__main__":
    main()