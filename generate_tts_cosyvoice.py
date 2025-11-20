# cosyvoice_batch_5fold_allrows.py
import os, csv, sys
from pathlib import Path
import torchaudio

# CUDA (if needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_HOME"] = "/usr/local/cuda-12.3"

# --- CosyVoice2 imports ---
COSYVOICE_DIR = "/home/arms/Workspace/emotion/CosyVoice"  # <<== your own path
sys.path.append(COSYVOICE_DIR)
sys.path.append(str(Path(COSYVOICE_DIR) / "third_party" / "Matcha-TTS"))
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# =========================
# USER SETTINGS
# =========================
PRETRAINED_DIR       = str(Path(COSYVOICE_DIR) / "pretrained_models" / "CosyVoice2-0.5B")
CSV_PATH             = "emotion_sentences.csv"   # columns: index,sentence,emotion
OUT_ROOT             = Path(__file__).resolve().parent / "output" / "sentences" / "cosyvoice2"
N_FOLDS              = 5

# Zero-shot reference speaker WAVs for each fold (same sentence spoken)
REF_SPK_WAVS = [
    "cosyvoice/nova.wav",
    "cosyvoice/coral.wav",
    "cosyvoice/echo.wav",
    "cosyvoice/onyx.wav",
    "cosyvoice/verse.wav",
]
# The text spoken in the reference wavs
REF_TEXT = "The meeting starts at 10 a.m. sharp tomorrow."  # The text spoken in the reference wavs

# Emotion → English instruction sentence
EMOTION_TO_STYLE_EN = {
    "happy":     "Say in a happy tone.",
    "sad":       "Say in a sad tone.",
    "angry":     "Say in an angry tone.",
    "fearful":   "Say in a fearful tone.",
    "disgusted": "Say in a disgusted tone.",
    "surprised": "Say in a surprised tone.",
    "neutral":   "Say in a neutral tone.",
}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# 1) register_zero_shot_speakers returns both spk_id and the 16k prompt audio
def register_zero_shot_speakers(cosyvoice, ref_spk_wavs, ref_text):
    spk_ids, prompt_list = [], []
    for i, wav_path in enumerate(ref_spk_wavs):
        spk_id = f"fold_spk_{i+1}"
        prompt_speech_16k = load_wav(wav_path, 16000)  # 16k mono
        ok = cosyvoice.add_zero_shot_spk(ref_text, prompt_speech_16k, spk_id)
        assert ok, f"Zero-shot speaker could not be added: {wav_path}"
        spk_ids.append(spk_id)
        prompt_list.append(prompt_speech_16k)
        print(f"[INFO] Registered zero-shot speaker {spk_id} from {wav_path}")
    cosyvoice.save_spkinfo()
    return spk_ids, prompt_list

def read_rows(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            idx  = str(r["index"]).strip()
            text = r["sentence"].strip()
            emo  = (r["emotion"] or "").strip().lower()
            if emo not in EMOTION_TO_STYLE_EN:
                emo = "neutral"
            instruct = EMOTION_TO_STYLE_EN[emo]
            rows.append({"id": idx, "text": text, "emotion": emo, "instruct": instruct})
    return rows

def main():
    ensure_dir(OUT_ROOT)

    # Single model instance
    cosyvoice = CosyVoice2(PRETRAINED_DIR, load_jit=False, load_trt=False, load_vllm=False, fp16=False)


    # All rows (each fold will generate all of them)
    all_rows = read_rows(CSV_PATH)
    print(f"[INFO] Total rows: {len(all_rows)} (each fold will render ALL rows)")

    # Register zero-shot speakers and retrieve their IDs
    assert len(REF_SPK_WAVS) >= N_FOLDS, "REF_SPK_WAVS count cannot be less than N_FOLDS."
    spk_ids, prompt_list = register_zero_shot_speakers(cosyvoice, REF_SPK_WAVS, REF_TEXT)

    for fold_idx in range(1,N_FOLDS):
        fold_no = fold_idx + 1
        out_dir = OUT_ROOT / f"fold_{fold_no}"
        ensure_dir(out_dir)

        spk_id = spk_ids[fold_idx]
        fold_prompt = prompt_list[fold_idx]     # <<< CRITICAL

        print(f"[INFO] FOLD {fold_no}/{N_FOLDS} -> out={out_dir}, zero_shot_spk_id={spk_id}")

        for r in all_rows:
            utt_id   = r["id"]
            text     = r["text"]
            instruct = r["instruct"]
            out_path = out_dir / f"{utt_id}.wav"

            gen = cosyvoice.inference_instruct2(
                text,
                instruct,
                fold_prompt,                 # <<< Provide the reference wav alongside the ID
                stream=False,
            )

            wrote_any = False
            for piece_idx, j in enumerate(gen):
                tmp = out_dir / f"{utt_id}_{piece_idx}.wav"
                torchaudio.save(str(tmp), j["tts_speech"].cpu(), cosyvoice.sample_rate)
                wrote_any = True

            p0 = out_dir / f"{utt_id}_0.wav"
            if wrote_any and p0.is_file():
                if out_path.exists():
                    out_path.unlink()
                p0.rename(out_path)
                print(f"[OK] fold={fold_no} -> {out_path.name}")
            else:
                print(f"[WARN] fold={fold_no} -> no output for {utt_id}")

    print("[DONE] CosyVoice2 5-fold (ALL rows per fold) completed.")

if __name__ == "__main__":
    main()