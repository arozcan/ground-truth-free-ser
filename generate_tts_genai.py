import os
import csv
import wave
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
from google import genai
from google.genai import types
import time

# Google API Key
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
client = genai.Client()

# Parameters
CSV_PATH = "emotion_sentences.csv"
FOLD = 5
OUTPUT_DIR = f"output/sentences/genai"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prompt template: includes the target emotion in the instruction
def build_prompt(sentence, emotion):
    return f"Say the following sentence in a {emotion} tone: '{sentence}'"

# Save 24kHz PCM as WAV
def save_pcm_as_wav_24khz(filename, pcm_bytes):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(24000)
        wf.writeframes(pcm_bytes)

# Resample WAV to 16kHz
def resample_wav_to_16khz(in_path, out_path):
    rate, data = wavfile.read(in_path)
    if data.ndim > 1:
        data = data[:, 0]
    target_len = int(len(data) * 16000 / rate)
    resampled = resample(data, target_len)
    resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
    wavfile.write(out_path, 16000, resampled)

# Main loop: Read CSV and synthesize
for fold_idx in range(5, FOLD + 1):
    output_dir = f"{OUTPUT_DIR}/fold_{fold_idx}"
    os.makedirs(output_dir, exist_ok=True)

    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["index"]) != 26:
                continue

            sentence_id = row["index"]
            sentence = row["sentence"]
            emotion = row["emotion"]

            prompt = build_prompt(sentence, emotion)
            print(f"🔊 Generating ID {sentence_id} (Fold {fold_idx})...")

            try:
                # Generate speech
                response = client.models.generate_content(
                    model="gemini-2.5-flash-preview-tts",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name='Charon'
                                )
                            )
                        )
                    )
                )

                # Get raw PCM
                raw_pcm = response.candidates[0].content.parts[0].inline_data.data

                # File paths
                wav_24_path = os.path.join(output_dir, f"{sentence_id}_24k.wav")
                wav_16_path = os.path.join(output_dir, f"{sentence_id}.wav")

                # Save and resample
                save_pcm_as_wav_24khz(wav_24_path, raw_pcm)
                resample_wav_to_16khz(wav_24_path, wav_16_path)

                print(f"✅ Saved: {wav_16_path}")
                #time.sleep(10)

            except Exception as e:
                print(f"❌ Error generating ID {sentence_id}: {e}")