# build_ser_csv_v2_strat.py
import os
import csv
import argparse
from pathlib import Path
from glob import glob
from collections import defaultdict, Counter
import random
import math

EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def load_index_to_emotion(emotion_csv):
    idx2emo = {}
    with open(emotion_csv, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            idx = int(row["index"])
            emo = row["emotion"].strip().lower()
            assert emo in EMOTIONS, f"Unknown emotion '{emo}' at index {idx}"
            idx2emo[idx] = emo
    return idx2emo

def iter_wavs(tts_roots, num_per_fold=70, fold_prefix="fold_",
              min_fold=None, max_fold=None):
    """
    Yields tuples: (wav_path, index:int, fold_num:int or None, tts:str)
    fold_* adını parse ederek fold_num çıkarır (fold_1, fold_1_nova -> 1).
    min_fold/max_fold verilirse o aralık dışındakileri atar.
    tts: kök klasörün adı (ör. 'azure', 'cosyvoice2', vb.)
    """
    for root in tts_roots:
        tts_name = Path(root).name  # <-- TTS adı buradan geliyor
        for fold_dir in sorted(glob(os.path.join(root, f"{fold_prefix}*"))):
            fold_name = Path(fold_dir).name
            # fold numarasını okumayı dene
            fold_num = None
            parts = fold_name.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                fold_num = int(parts[1])
                if (min_fold is not None and fold_num < min_fold) or \
                   (max_fold is not None and fold_num > max_fold):
                    continue
            for i in range(1, num_per_fold + 1):
                wav_path = os.path.join(fold_dir, f"{i}.wav")
                if os.path.exists(wav_path):
                    yield wav_path, i, fold_num, tts_name  # <-- tts_name eklendi

def stratified_group_split(groups, val_ratio, seed=1337, ensure_at_least_one_per_class=True):
    """
    groups: dict[prompt_id] -> list[rows]; rows[0]['emotion'] etiket olarak alınır.
    Dönüş: (train_keys, val_keys) set'leri
    """
    rng = random.Random(seed)
    label_to_groups = defaultdict(list)
    for pid, rows in groups.items():
        label = rows[0]['emotion']
        label_to_groups[label].append(pid)

    # Karıştır
    for pids in label_to_groups.values():
        rng.shuffle(pids)

    all_pids = [pid for pids in label_to_groups.values() for pid in pids]
    total_groups = len(all_pids)
    target_val = max(1, int(round(total_groups * val_ratio)))

    # Sınıf bazında taban seçim: floor(len * ratio)
    per_label_take = {}
    remainders = []
    current_val = 0
    for label, pids in label_to_groups.items():
        n = len(pids)
        if n == 0:
            per_label_take[label] = 0
            continue
        raw = n * val_ratio
        take = math.floor(raw)
        rem = raw - take
        per_label_take[label] = take
        remainders.append((rem, label))
        current_val += take

    # En az 1 kuralı (opsiyonel)
    if ensure_at_least_one_per_class:
        for label, pids in label_to_groups.items():
            if len(pids) > 0 and per_label_take[label] == 0:
                per_label_take[label] = 1
                current_val += 1

    # Hedefe ulaşmak için ince ayar:
    # Fazla aldıysak azalt; az aldıysak arttır.
    def labels_sorted_for_add():
        # fractional remainder büyük olanları önce ekle
        return [lbl for _, lbl in sorted(remainders, key=lambda x: x[0], reverse=True)]

    def labels_sorted_for_drop():
        # fractional remainder küçük olanları önce düş
        return [lbl for _, lbl in sorted(remainders, key=lambda x: x[0])]

    # Üst sınıra saygı: per_label_take[label] <= len(pids)
    for label, pids in label_to_groups.items():
        per_label_take[label] = min(per_label_take[label], len(pids))

    if current_val < target_val:
        order = labels_sorted_for_add()
        i = 0
        while current_val < target_val and i < 100000:
            for lbl in order:
                if current_val >= target_val:
                    break
                if per_label_take[lbl] < len(label_to_groups[lbl]):
                    per_label_take[lbl] += 1
                    current_val += 1
            i += 1
    elif current_val > target_val:
        order = labels_sorted_for_drop()
        i = 0
        while current_val > target_val and i < 100000:
            for lbl in order:
                if current_val <= target_val:
                    break
                # Düşürürken sınıfı tamamen sıfıra indirmemeye çalış (eğer ensure aktifse)
                min_allowed = 1 if (ensure_at_least_one_per_class and len(label_to_groups[lbl]) > 0) else 0
                if per_label_take[lbl] > min_allowed:
                    per_label_take[lbl] -= 1
                    current_val -= 1
            i += 1

    # Son seçim
    val_keys = set()
    for label, pids in label_to_groups.items():
        take = per_label_take[label]
        val_keys.update(pids[:take])

    train_keys = set(all_pids) - val_keys
    return train_keys, val_keys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emotion_csv", required=True,
                    help="emotion_sentences.csv (index,emotion,sentence)")
    ap.add_argument("--tts_roots", nargs="+", required=True,
                    help="TTS root folders containing fold_* subfolders")
    ap.add_argument("--train_out", default="train.csv")
    ap.add_argument("--val_out", default="val.csv")
    ap.add_argument("--num_per_fold", type=int, default=70)
    ap.add_argument("--val_ratio", type=float, default=0.2,
                    help="Validation oranı (0-1)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--group_by", choices=["text", "text_speaker"], default="text",
                    help="prompt gruplama: text=idx{index}, text_speaker=idx{index}_f{fold}")
    ap.add_argument("--min_fold", type=int, default=None,
                    help="Yalnızca bu alt sınırdaki fold ve üstünü al (opsiyonel)")
    ap.add_argument("--max_fold", type=int, default=None,
                    help="Yalnızca bu üst sınırdaki fold ve altını al (opsiyonel)")
    ap.add_argument("--ensure_at_least_one_per_class", action="store_true",
                    help="Val setinde her sınıftan en az 1 grup bulundurmaya çalış")
    args = ap.parse_args()

    random.seed(args.seed)

    idx2emo = load_index_to_emotion(args.emotion_csv)

    # Grupları topla
    groups = defaultdict(list)  # key -> list of rows
    for wav_path, index, fold_num, tts_name in iter_wavs(
        args.tts_roots,
        num_per_fold=args.num_per_fold,
        min_fold=args.min_fold,
        max_fold=args.max_fold
    ):
        if index not in idx2emo:
            continue
        emo = idx2emo[index]
        if args.group_by == "text_speaker" and fold_num is not None:
            prompt_id = f"idx{index}_f{fold_num}"
        else:
            prompt_id = f"idx{index}"

        # tts alanını da satıra ekliyoruz
        row = {"wav_path": wav_path, "emotion": emo, "prompt_id": prompt_id, "tts": tts_name}
        groups[prompt_id].append(row)

    # Stratified split (grup bazlı)
    train_keys, val_keys = stratified_group_split(
        groups,
        val_ratio=args.val_ratio,
        seed=args.seed,
        ensure_at_least_one_per_class=args.ensure_at_least_one_per_class
    )

    train_rows, val_rows = [], []
    for k in train_keys:
        train_rows.extend(groups[k])
    for k in val_keys:
        val_rows.extend(groups[k])

    # Yazıcı
    def write_csv(path, rows):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            # tts sütununu ekliyoruz
            w = csv.DictWriter(f, fieldnames=["wav_path", "emotion", "prompt_id", "tts"])
            w.writeheader()
            w.writerows(rows)

    write_csv(args.train_out, train_rows)
    write_csv(args.val_out, val_rows)

    # Özetler
    def class_count(rows):
        c = Counter([r["emotion"] for r in rows])
        return {e: c.get(e, 0) for e in EMOTIONS}

    def group_count(keys):
        gc = Counter([groups[k][0]["emotion"] for k in keys])
        return {e: gc.get(e, 0) for e in EMOTIONS}

    print(f"[OK] groups total={len(groups)}  train={len(train_keys)}  val={len(val_keys)}  (val_ratio={args.val_ratio})")
    print("[groups/train by class]:", group_count(train_keys))
    print("[groups/val   by class]:", group_count(val_keys))
    print(f"[rows/train by class]  :", class_count(train_rows))
    print(f"[rows/val   by class]  :", class_count(val_rows))

    # Örnek satırlar
    if train_rows:
        print("[sample][train]:", train_rows[0])
    if val_rows:
        print("[sample][val]  :", val_rows[0])

if __name__ == "__main__":
    main()