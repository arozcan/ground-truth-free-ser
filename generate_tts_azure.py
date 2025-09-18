import os
import csv
from pathlib import Path
import azure.cognitiveservices.speech as speechsdk
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

# Azure API ayarları
os.environ["AZURE_SPEECH_KEY"] = "YOUR_AZURE_SPEECH_KEY"
os.environ["AZURE_REGION"] = "eastus"

CSV_PATH = "emotion_sentences.csv"
OUTPUT_DIR = "output/sentences/azure"
AZURE_REGION = os.environ["AZURE_REGION"]
AZURE_KEY = os.environ["AZURE_SPEECH_KEY"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

AZURE_SPEAKERS = {
    "en-US-JennyNeural": ["angry", "cheerful", "excited", "friendly", "hopeful", "sad", "shouting", "terrified", "unfriendly", "whispering"],
    "en-US-DavisNeural": ["angry", "cheerful", "excited", "friendly", "hopeful", "sad", "shouting", "terrified", "unfriendly", "whispering"],
    "en-US-GuyNeural":   ["angry", "cheerful", "excited", "friendly", "hopeful", "sad", "shouting", "terrified", "unfriendly", "whispering"],
    "en-US-SaraNeural":  ["angry", "cheerful", "excited", "friendly", "hopeful", "sad", "shouting", "terrified", "unfriendly", "whispering"],
    "en-US-AriaNeural":  ["angry", "cheerful", "excited", "friendly", "hopeful", "sad", "shouting", "terrified", "unfriendly", "whispering"],
}

EMOTION_MAP = {
    "angry": "angry",
    "happy": "cheerful",
    "sad": "sad",
    "neutral": None,
    "fearful": "terrified",
    "disgusted": "unfriendly",
    "surprised": "excited"
}

def build_ssml(text, emotion, speaker, degree="2.0"):
    style = EMOTION_MAP.get(emotion.lower())
    supported = AZURE_SPEAKERS[speaker]

    if style and style in supported:
        return f"""
        <speak version="1.0" xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-US">
            <voice name="{speaker}">
                <mstts:express-as style="{style}" styledegree="{degree}">
                    {text}
                </mstts:express-as>
            </voice>
        </speak>
        """.strip()
    else:
        return f"""
        <speak version="1.0" xml:lang="en-US">
            <voice name="{speaker}">
                {text}
            </voice>
        </speak>
        """.strip()

def resample_to_16k(input_path, output_path):
    rate, data = wavfile.read(input_path)
    if data.ndim > 1:
        data = data[:, 0]
    target_len = int(len(data) * 16000 / rate)
    resampled = resample(data, target_len)
    resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
    wavfile.write(output_path, 16000, resampled)

def synthesize_and_save(ssml, out_dir, idx, speaker):
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_KEY, region=AZURE_REGION)
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
    )
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    result = synthesizer.speak_ssml_async(ssml).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        # 24kHz kaydet
        wav_24k = out_dir / f"{idx}_24k.wav"
        with open(wav_24k, "wb") as f:
            f.write(result.audio_data)
        print(f"✅ Saved 24kHz: {wav_24k}")

        # 16kHz'e örnekle
        wav_16k = out_dir / f"{idx}.wav"
        resample_to_16k(wav_24k, wav_16k)
        print(f"🎯 Resampled to 16kHz: {wav_16k}")

    else:
        print(f"❌ Synthesis failed for {idx}. Reason: {result.reason}")
        if result.reason == speechsdk.ResultReason.Canceled:
            print(result.cancellation_details.reason, result.cancellation_details.error_details)

def generate_all(selected_speakers=None, degree="2.0"):
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["index"])
            text = row["sentence"]
            emotion = row["emotion"]

            if int(row["index"]) != 27:
                continue

            for speaker in (selected_speakers or list(AZURE_SPEAKERS.keys())):
                out_dir = Path(OUTPUT_DIR) / speaker
                out_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n🔊 [{speaker}] ID {idx} | Emotion: {emotion}")
                ssml = build_ssml(text, emotion, speaker, degree)
                synthesize_and_save(ssml, out_dir, idx, speaker)

if __name__ == "__main__":
    generate_all(
        selected_speakers=[
            "en-US-JennyNeural"#,"en-US-DavisNeural", "en-US-GuyNeural", "en-US-AriaNeural", "en-US-SaraNeural"
        ],
        degree="2.0"
    )