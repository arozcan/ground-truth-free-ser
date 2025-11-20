import os
from pathlib import Path
import torchaudio
import torch

# Fold klasörleri
REF_SPK_DIRS = [
    "output/sentences/genai/fold_1_zephyr",
    "output/sentences/genai/fold_2_leda",
    "output/sentences/genai/fold_3_fenrir",
    "output/sentences/genai/fold_4_kore",
    "output/sentences/genai/fold_5_charon",
]

# Birleştirilecek dosyalar (varsa “1.wav” doğru; bazı setlerde “01.wav” olabilir)
TARGET_FILES = ["1.wav", "31.wav", "41.wav", "51.wav"]

# Çıkış klasörü
OUT_DIR = Path("cosyvoice/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def concat_wavs(file_list, out_path):
    """Verilen wav dosyalarını sırayla birleştirir ve out_path'e kaydeder."""
    all_audio = []
    sample_rate = None

    for wav_path in file_list:
        if not Path(wav_path).is_file():
            print(f"[WARN] Skipping missing file: {wav_path}")
            continue
        wav, sr = torchaudio.load(str(wav_path))
        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            raise ValueError(f"Sample rate mismatch in {wav_path}: {sr} vs {sample_rate}")
        all_audio.append(wav)

    if not all_audio:
        raise RuntimeError("No valid wav files found to concatenate!")

    merged = torch.cat(all_audio, dim=1)  # time axis concat
    torchaudio.save(str(out_path), merged, sample_rate)
    print(f"[OK] Saved concatenated wav -> {out_path}")

def main():
    for fold_dir in REF_SPK_DIRS:
        fold_dir = Path(fold_dir)
        # Çıktı dosya adı: fold_*_{NAME} → {name}.wav
        # örn: fold_1_nova -> nova.wav
        tail = fold_dir.name.split("_", maxsplit=2)[-1]  # "nova", "coral", ...
        out_path = OUT_DIR / f"{tail}.wav"

        files = [fold_dir / fname for fname in TARGET_FILES]
        print(f"[INFO] Processing fold: {fold_dir}")
        concat_wavs(files, out_path)

if __name__ == "__main__":
    main()