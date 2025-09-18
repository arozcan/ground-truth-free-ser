import os
import csv
from pathlib import Path
from openai import OpenAI
from scipy.io import wavfile
from scipy.signal import resample
import numpy as np

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
# OpenAI istemcisi (API Key ortam değişkeninden alır)
client = OpenAI()

CSV_PATH = "emotion_sentences.csv"
OUTPUT_DIR = "output/sentences/openai"

# Kullanılacak konuşmacılar
# fold_1: nova
# fold_2: coral
# fold_3: echo
# fold_4: onyx
# fold_5: verse
SPEAKERS = ["nova", "coral", "echo", "onyx", "verse"]

# Prompta duygu düzeyine göre şablonlar
PROMPT_TEMPLATES = {
    "basic": lambda emotion: f"Speak in a {emotion} tone.",
    "moderate": lambda emotion: (
        f"Speak the sentence with a clear {emotion} tone. "
        f"Sound like you're naturally feeling {emotion}."
    ),
    "rich": lambda emotion: (
        f"You're a professional voice actor. Read the sentence with the emotional tone of someone who feels {emotion}. "
        f"Speak like you're performing in a dramatic audiobook. Emphasize the emotional style strongly."
    )
}

def resample_to_16k(input_path, output_path):
    """24kHz WAV dosyasını 16kHz int16 formatında yeniden örnekler"""
    rate, data = wavfile.read(input_path)
    if data.ndim > 1:
        data = data[:, 0]
    target_len = int(len(data) * 16000 / rate)
    resampled = resample(data, target_len)
    resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
    wavfile.write(output_path, 16000, resampled)

def generate_all(prompt_level="rich", selected_speakers=None):
    if selected_speakers is None:
        selected_speakers = SPEAKERS[:1]  # İlk 5'ini varsayılan al

    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence_id = int(row["index"])
            sentence = row["sentence"]
            emotion = row["emotion"].lower()
            instruction = PROMPT_TEMPLATES[prompt_level](emotion)

            for speaker in selected_speakers:
                speaker_dir = Path(OUTPUT_DIR) / prompt_level / speaker
                speaker_dir.mkdir(parents=True, exist_ok=True)

                wav_24k = speaker_dir / f"{sentence_id}_24k.wav"
                wav_16k = speaker_dir / f"{sentence_id}.wav"

                print(f"\n🔊 [{speaker}] {sentence_id} | {emotion} | prompt: {prompt_level}")

                try:
                    with client.audio.speech.with_streaming_response.create(
                        model="gpt-4o-mini-tts",
                        voice=speaker,
                        input=sentence,
                        instructions=instruction,
                        response_format="wav"
                    ) as response:
                        response.stream_to_file(wav_24k)
                    resample_to_16k(wav_24k, wav_16k)
                    print(f"✅ Saved: {wav_16k}")
                except Exception as e:
                    print(f"❌ Error [{speaker} {sentence_id}]: {e}")

if __name__ == "__main__":
    # Değiştirilebilir parametreler
    generate_all(prompt_level="basic", selected_speakers=["nova"])
