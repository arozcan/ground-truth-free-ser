# proto_emotion_classifier.py
import os
import csv
import math
import json
import time
import argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

# -----------------------------
# Config
# -----------------------------
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
EMO2IDX = {e: i for i, e in enumerate(EMOTIONS)}

# -----------------------------
# Dataset (precomputed embeddings ONLY)
# -----------------------------
class SERProtoDataset(Dataset):
    def __init__(self, csv_path: str, emb_mode: str = "both", input_l2norm: bool = False):
        assert emb_mode in ("e2v", "wavlm", "both")
        self.emb_mode = emb_mode
        self.input_l2norm = input_l2norm
        self.items = []

        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                emo = r["emotion"].strip().lower()
                if emo not in EMO2IDX:
                    continue
                self.items.append({
                    "wav_path": r.get("wav_path"),
                    "emotion": emo,
                    "y": EMO2IDX[emo],
                    "prompt_id": r.get("prompt_id"),
                    "intensity": r.get("intensity"),
                    "emb_path": r.get("emb_path"),     # eski e2v kolon adı olabilir
                    "emb_e2v": r.get("emb_e2v"),
                    "emb_wavlm": r.get("emb_wavlm"),
                    "tts": r.get("tts")
                })

        # --- Boyutları otomatik çıkar ---
        self.dim_e2v = self._infer_dim(stream="e2v")
        self.dim_wavlm = self._infer_dim(stream="wavlm")

    def _infer_dim(self, stream: str) -> Optional[int]:
        key = "emb_e2v" if stream == "e2v" else "emb_wavlm"
        fallback_key = "emb_path" if stream == "e2v" else None
        for it in self.items:
            p = it.get(key) or (it.get(fallback_key) if fallback_key else None)
            if p and os.path.exists(p):
                try:
                    v = np.load(p, mmap_mode="r")
                    return int(v.shape[-1])
                except Exception:
                    continue
        # Eğer hiç bulunamazsa: akış kullanılmayacaksa None; kullanılacaksa makul varsayılan
        if self.emb_mode in ("e2v", "both") and stream == "e2v":
            return 1024  # emotion2vec+ tipik
        if self.emb_mode in ("wavlm", "both") and stream == "wavlm":
            return 768   # WavLM-Base için makul varsayılan (Large ise 1024)
        return None

    def _load_vec(self, p):
        if not p or not os.path.exists(p):
            return None
        v = np.load(p).astype(np.float32).squeeze()
        if self.input_l2norm:
            n = np.linalg.norm(v) + 1e-12
            v = (v / n).astype(np.float32)
        return v

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        # E2V yolu: yeni kolon -> yoksa geriye dönük 'emb_path'
        x_e2v = self._load_vec(it.get("emb_e2v") or it.get("emb_path")) if self.emb_mode in ("e2v", "both") else None
        x_wavlm = self._load_vec(it.get("emb_wavlm")) if self.emb_mode in ("wavlm", "both") else None

        # Eksikse boyuta uygun dummy
        if x_e2v is None and (self.emb_mode in ("e2v", "both")):
            x_e2v = np.zeros((self.dim_e2v or 1024,), dtype=np.float32)
        if x_wavlm is None and (self.emb_mode in ("wavlm", "both")):
            x_wavlm = np.zeros((self.dim_wavlm or 768,), dtype=np.float32)

        sample = {
            "x_e2v": torch.from_numpy(x_e2v) if x_e2v is not None else None,
            "x_wavlm": torch.from_numpy(x_wavlm) if x_wavlm is not None else None,
        }
        y = torch.tensor(it["y"], dtype=torch.long)
        meta = {
            "wav_path": it["wav_path"],
            "prompt_id": it["prompt_id"],
            "intensity": it["intensity"],
            "tts": it.get("tts")
        }
        return sample, y, meta

    # Modelin giriş boyutlarını okuyabilmek için yardımcı
    def inferred_dims(self) -> Tuple[Optional[int], Optional[int]]:
        return self.dim_e2v, self.dim_wavlm
# -----------------------------
# Model: D->proj_dim + 7 learnable prototypes
# -----------------------------
class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach

    def forward(self, x):
        if self.training and self.sigma > 0:
            scale = self.sigma * (x.detach() if self.is_relative_detach else x)
            sampled_noise = torch.randn_like(x) * scale
            return x + sampled_noise
        return x
    
class FusionProtoClassifier(nn.Module):
    def __init__(self,
                 emb_mode="both",
                 fusion="concat",
                 proj_dim=256,
                 num_classes=7,
                 scale_init=20.0,
                 dropout=0.0,
                 d_e2v: int = 1024,
                 d_wavlm: int = 1024,
                 fusion_alpha: float = 0.5,
                 learn_fusion_alpha: bool = False):
        super().__init__()
        assert emb_mode in ("e2v", "wavlm", "both")
        assert fusion in ("avg", "concat", "projcat") or emb_mode != "both"

        self.emb_mode = emb_mode
        self.fusion = fusion

        if emb_mode in ("e2v", "wavlm"):
            D = d_e2v if emb_mode == "e2v" else d_wavlm
            self.proj_single = nn.Sequential(
                nn.Linear(D, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            fused_dim = proj_dim

        else:
            # iki akış
            if fusion == "avg":
                self.proj_e2v = nn.Sequential(
                    nn.Linear(d_e2v, proj_dim), nn.LayerNorm(proj_dim),
                    nn.GELU(), nn.Dropout(dropout)
                )
                self.proj_wav = nn.Sequential(
                    nn.Linear(d_wavlm, proj_dim), nn.LayerNorm(proj_dim),
                    nn.GELU(), nn.Dropout(dropout)
                )
                if learn_fusion_alpha:
                    # öğrenilebilir parametre
                    self.fusion_alpha = nn.Parameter(torch.tensor(float(fusion_alpha)))
                else:
                    # sabit parametre (buffer olarak tutulur)
                    self.register_buffer("fusion_alpha", torch.tensor(float(fusion_alpha)))
                fused_dim = proj_dim
            elif fusion == "concat":
                half = max(1, proj_dim // 2)
                self.proj_e2v = nn.Sequential(nn.Linear(d_e2v, half), nn.LayerNorm(half), nn.GELU(), nn.Dropout(dropout))
                self.proj_wav = nn.Sequential(nn.Linear(d_wavlm, half), nn.LayerNorm(half), nn.GELU(), GaussianNoise(0.9), nn.Dropout(dropout))
                fused_dim = half * 2
            else:  # projcat
                self.proj_e2v = nn.Sequential(nn.Linear(d_e2v, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(), nn.Dropout(dropout))
                self.proj_wav = nn.Sequential(nn.Linear(d_wavlm, proj_dim), GaussianNoise(0.5), nn.LayerNorm(proj_dim), nn.GELU(), nn.Dropout(dropout))
                self.fuse = nn.Sequential(
                    nn.Linear(2 * proj_dim, proj_dim),
                    nn.LayerNorm(proj_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                fused_dim = proj_dim

        self.prototypes = nn.Parameter(torch.randn(num_classes, fused_dim))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale_init), dtype=torch.float32))

    def forward(self, x_e2v=None, x_wavlm=None):
        # z: (B, fused_dim)
        if self.emb_mode in ("e2v", "wavlm"):
            x = x_e2v if self.emb_mode == "e2v" else x_wavlm
            z = self.proj_single(x)
        else:
            z_e2v = self.proj_e2v(x_e2v)
            z_wav = self.proj_wav(x_wavlm)
            if self.fusion == "avg":
                z_e2v = self.proj_e2v(x_e2v)
                z_wav = self.proj_wav(x_wavlm)
                alpha = torch.clamp(self.fusion_alpha, 0.0, 1.0)  # güvenlik
                z = alpha * z_e2v + (1.0 - alpha) * z_wav
            elif self.fusion == "concat":
                z = torch.cat([z_e2v, z_wav], dim=-1)
            else:  # projcat
                z = torch.cat([z_e2v, z_wav], dim=-1)
                z = self.fuse(z)

        z = F.normalize(z, dim=-1)
        E = F.normalize(self.prototypes, dim=-1)
        cos = torch.matmul(z, E.t())
        logits = cos * torch.exp(self.log_scale)
        return logits, z, E

    @torch.no_grad()
    def compute_eas(self, z, target_indices):
        E = F.normalize(self.prototypes, dim=-1)
        e_t = E[target_indices]
        eas = ((z * e_t).sum(dim=-1) + 1.0) / 2.0
        return eas.clamp(0.0, 1.0)

# -----------------------------
# Training / Evaluation
# -----------------------------
def set_seed(sd: int = 1337):
    import random
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)

def collate(batch):
    xs, ys, metas = zip(*batch)
    # x_e2v
    if xs[0]["x_e2v"] is not None:
        x_e2v = torch.stack([d["x_e2v"] for d in xs], dim=0).float()
    else:
        x_e2v = None
    # x_wavlm
    if xs[0]["x_wavlm"] is not None:
        x_wavlm = torch.stack([d["x_wavlm"] for d in xs], dim=0).float()
    else:
        x_wavlm = None
    y = torch.stack(ys, dim=0).long()
    return {"x_e2v": x_e2v, "x_wavlm": x_wavlm}, y, metas

def train_one_epoch(model, loader, opt, device, label_smoothing=0.0, proto_ema=0.9):
    model.train()
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    tot_loss = tot_corr = tot_n = 0
    C = model.prototypes.shape[0]

    for batch_x, y, _ in loader:
        x_e2v = batch_x["x_e2v"].to(device) if batch_x["x_e2v"] is not None else None
        x_wavlm = batch_x["x_wavlm"].to(device) if batch_x["x_wavlm"] is not None else None
        y = y.to(device)

        logits, z, _ = model(x_e2v=x_e2v, x_wavlm=x_wavlm)  # <-- z'yi al
        loss = ce(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # ----- EMA prototype update (grad yok) -----
        with torch.no_grad():
            # z zaten L2-normalize; prototipleri de normalize tutmak faydalı
            for c in range(C):
                m = (y == c)
                if m.any():
                    cls_mean = z[m].mean(dim=0)  # (D,)
                    # new = ema * old + (1-ema) * batch_mean
                    model.prototypes.data[c].mul_(proto_ema).add_(cls_mean, alpha=(1.0 - proto_ema))
                    # İsteğe bağlı: prototipi birim-normda tut
                    model.prototypes.data[c] = F.normalize(model.prototypes.data[c], dim=0)

        # ----- metrikler -----
        with torch.no_grad():
            pred = logits.argmax(-1)
            n = y.numel()
            tot_loss += loss.item() * n
            tot_corr += (pred == y).sum().item()
            tot_n += n

    return tot_loss / max(1, tot_n), tot_corr / max(1, tot_n)

@torch.no_grad()
def evaluate(model, loader, device, report_path: Optional[str] = None,
             compute_ims: bool = True, verbose: bool = True):
    model.eval()
    all_preds, all_targets, all_eas, metas_accum = [], [], [], []

    for batch_x, y, metas in loader:
        x_e2v = batch_x["x_e2v"].to(device) if batch_x["x_e2v"] is not None else None
        x_wavlm = batch_x["x_wavlm"].to(device) if batch_x["x_wavlm"] is not None else None
        y = y.to(device)

        logits, z, _ = model(x_e2v=x_e2v, x_wavlm=x_wavlm)
        pred = logits.argmax(-1)
        eas = model.compute_eas(z, y)

        all_preds.extend(pred.cpu().tolist())
        all_targets.extend(y.cpu().tolist())
        all_eas.extend(eas.cpu().tolist())
        metas_accum.extend(metas)

    idx2emo = {i: e for e, i in EMO2IDX.items()}
    y_true = [idx2emo[i] for i in all_targets]
    y_pred = [idx2emo[i] for i in all_preds]

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=EMOTIONS, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=EMOTIONS)
    cm_df = pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS)

    results = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm_df.to_dict(),
        "mean_eas": float(np.mean(all_eas)) if len(all_eas) > 0 else None
    }
    if compute_ims:
        ims = compute_ims_score(all_eas, metas_accum, all_targets)
        results["ims"] = ims

    if verbose:
        print("\n📊 Classification Report")
        print(report)
        print(f"🎯 Accuracy: {acc*100:.2f}%")
        if results["mean_eas"] is not None:
            print(f"💛 Mean EAS: {results['mean_eas']:.4f}")
        print("\n🧩 Confusion Matrix (Pred ↓ / True →)")
        print(cm_df)

    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(results, indent=2, ensure_ascii=False))
    return results

@torch.no_grad()
def evaluate_tts(model, loader, device, report_path: Optional[str] = None, verbose: bool = True):
    model.eval()
    all_preds, all_targets, metas_accum = [], [], []
    all_eas = []  # <-- EAS birikimi

    for batch_x, y, metas in loader:
        x_e2v = batch_x["x_e2v"].to(device) if batch_x["x_e2v"] is not None else None
        x_wavlm = batch_x["x_wavlm"].to(device) if batch_x["x_wavlm"] is not None else None
        y = y.to(device)

        # z'yi al ki EAS hesaplayabilelim
        logits, z, _ = model(x_e2v=x_e2v, x_wavlm=x_wavlm)
        pred = logits.argmax(-1)

        # batch EAS
        eas = model.compute_eas(z, y)  # (B,)
        all_eas.extend(eas.detach().cpu().tolist())

        all_preds.extend(pred.cpu().tolist())
        all_targets.extend(y.cpu().tolist())
        metas_accum.extend(metas)

    # label -> name
    idx2emo = {i: e for e, i in EMO2IDX.items()}
    y_true = [idx2emo[i] for i in all_targets]
    y_pred = [idx2emo[i] for i in all_preds]

    # ---- Genel özet
    overall_acc = accuracy_score(y_true, y_pred)
    overall_report = classification_report(
        y_true, y_pred, labels=EMOTIONS, digits=4, zero_division=0, output_dict=True
    )
    overall_cm = confusion_matrix(y_true, y_pred, labels=EMOTIONS)
    overall_cm_df = pd.DataFrame(overall_cm, index=EMOTIONS, columns=EMOTIONS)
    overall_mean_eas = float(np.mean(all_eas)) if len(all_eas) > 0 else None

    # ---- TTS bazında
    tts_set = sorted({(m.get("tts") or "unknown") for m in metas_accum})
    per_tts = {}
    table_rows = []
    for tts_name in tts_set:
        idxs = [i for i, m in enumerate(metas_accum) if (m.get("tts") or "unknown") == tts_name]
        if not idxs:
            continue
        y_true_tts = [y_true[i] for i in idxs]
        y_pred_tts = [y_pred[i] for i in idxs]
        eas_tts    = [all_eas[i] for i in idxs]

        acc = accuracy_score(y_true_tts, y_pred_tts)
        rep = classification_report(
            y_true_tts, y_pred_tts, labels=EMOTIONS, digits=4, zero_division=0, output_dict=True
        )
        cm = confusion_matrix(y_true_tts, y_pred_tts, labels=EMOTIONS)
        cm_df = pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS)
        mean_eas_tts = float(np.mean(eas_tts)) if len(eas_tts) > 0 else None

        per_tts[tts_name] = {
            "N": len(idxs),
            "accuracy": acc,
            "mean_eas": mean_eas_tts,
            "classification_report": rep,
            "confusion_matrix": cm_df.to_dict(),
        }
        table_rows.append((tts_name, len(idxs), acc, mean_eas_tts))

    # Konsol çıktısı (özet tablo + detaylar)
    if verbose:
        print("\n🔍 TTS-based Accuracy")
        print(f"{'TTS':<24}{'N':>6}{'Acc':>10}{'Mean EAS':>12}")
        for tts_name, N, acc, me in sorted(table_rows, key=lambda x: x[0].lower()):
            acc_s = f"{acc*100:.2f}%"
            me_s  = f"{me:.4f}" if me is not None else "-"
            print(f"{tts_name:<24}{N:>6}{acc_s:>10}{me_s:>12}")

        for tts_name in sorted(per_tts.keys()):
            block = per_tts[tts_name]
            print(f"\n=== {tts_name.upper()} ===")
            print(f"Accuracy: {block['accuracy']*100:.2f}%  (N={block['N']})")
            if block["mean_eas"] is not None:
                print(f"Mean EAS: {block['mean_eas']:.4f}")
            # classification report
            df_rep = pd.DataFrame(block["classification_report"]).transpose()
            print("\nClassification Report:")
            print(df_rep)
            # confusion matrix
            df_cm = pd.DataFrame(block["confusion_matrix"])
            df_cm.index = EMOTIONS
            df_cm.columns = EMOTIONS
            print("\nConfusion Matrix (Pred ↓ / True →):")
            print(df_cm)

        print("\n📊 Overall")
        print(f"Accuracy: {overall_acc*100:.2f}%")
        if overall_mean_eas is not None:
            print(f"Mean EAS: {overall_mean_eas:.4f}")
        print(pd.DataFrame(overall_report).transpose())
        print("\nOverall Confusion Matrix (Pred ↓ / True →):")
        print(overall_cm_df)

    results = {
        "overall": {
            "accuracy": overall_acc,
            "mean_eas": overall_mean_eas,
            "classification_report": overall_report,
            "confusion_matrix": overall_cm_df.to_dict(),
        },
        "per_tts": per_tts,
    }

    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(results, indent=2, ensure_ascii=False))

    return results

def compute_ims_score(eas_list: List[float], metas: List[dict], y_true_idx: List[int]) -> Optional[dict]:
    buckets: Dict[Tuple[str,int], List[Tuple[str, float]]] = {}
    for eas, meta, y in zip(eas_list, metas, y_true_idx):
        pid = meta.get("prompt_id", None)
        inten = (meta.get("intensity") or "").lower()
        if pid is None or inten not in ("mild", "base", "strong"):
            continue
        key = (pid, y)
        buckets.setdefault(key, []).append((inten, float(eas)))

    good, total = 0, 0
    for _, lst in buckets.items():
        inten_map = {k: v for k, v in lst}
        if all(k in inten_map for k in ("mild", "base", "strong")):
            total += 1
            if inten_map["mild"] <= inten_map["base"] <= inten_map["strong"]:
                good += 1

    if total == 0:
        return None
    return {"monotonic_fraction": good / total, "count": total}

# -----------------------------
# Reranking (per prompt_id)
# -----------------------------
@torch.no_grad()
def rerank_by_eas(model, csv_path: str, device, topk: int = 1,
                  emb_mode="both", fusion="concat", input_l2norm=False, out_csv="reranked.csv"):
    ds = SERProtoDataset(csv_path, emb_mode=emb_mode, input_l2norm=input_l2norm)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, collate_fn=collate)

    all_rows = []
    for x, y, metas in loader:
        x = x.to(device)
        y = y.to(device)
        logits, z, _ = model(x)
        eas = model.compute_eas(z, y).cpu().tolist()
        preds = logits.argmax(dim=-1).cpu().tolist()

        for meta, yi, pi, si in zip(metas, y.cpu().tolist(), preds, eas):
            row = {
                "wav_path": meta["wav_path"],
                "prompt_id": meta["prompt_id"],
                "true_emotion": EMOTIONS[yi],
                "pred_emotion": EMOTIONS[pi],
                "eas": si
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    if "prompt_id" not in df.columns or df["prompt_id"].isna().all():
        print("[INFO] No prompt_id present; skipping reranking.")
        return None

    df = df.sort_values(["prompt_id", "true_emotion", "eas"], ascending=[True, True, False])
    top_df = df.groupby(["prompt_id", "true_emotion"]).head(topk).reset_index(drop=True)
    top_df.to_csv(out_csv, index=False)
    print(f"[OK] Reranked results saved to {out_csv}")
    return top_df

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train",
        choices=["train", "eval", "rerank", "eval_tts"],
        help="Çalışma modu")
    parser.add_argument("--train_csv", type=str, default="csv_with_emb_dual/train.csv",
                        help="CSV with precomputed paths (emb_e2v/emb_wavlm)")
    parser.add_argument("--val_csv", type=str, default="csv_with_emb_dual/val.csv",
                        help="CSV for validation/eval")
    # embedding okuma seçenekleri
    parser.add_argument("--emb_mode", type=str, default="both", choices=["e2v", "wavlm", "both"])
    parser.add_argument("--fusion", type=str, default="concat", choices=["avg","concat","projcat"])
    parser.add_argument("--input_l2norm", action="store_true",
                    help="Girdi embeddinglerini L2-normalize et")
    # eğitim hiperparametreleri
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path", type=str, default="proto_ckpt.pt")
    parser.add_argument("--load_path", type=str, default=None, help="Load an existing checkpoint")
    parser.add_argument("--report_path", type=str, default=None, help="JSON file to dump metrics")
    parser.add_argument("--rerank_topk", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--proto_ema", type=float, default=0.9,
                    help="Prototip EMA momentumu (0.0=hepsi batch, 1.0=neredeyse sabit)")
    parser.add_argument("--freeze_prototypes", action="store_true",
                    help="Prototipleri opt. dışı bırak (grad yok), yalnızca EMA ile güncelle")
    parser.add_argument("--fusion_alpha", type=float, default=0.5,
                    help="Fusion=avg iken e2v için ağırlık (0..1). wavlm için 1-alpha kullanılır.")
    parser.add_argument("--learn_fusion_alpha", action="store_true",
                    help="True olursa fusion_alpha parametresi öğrenilebilir hale gelir.")

    args = parser.parse_args()
    set_seed(args.seed)

    # Dataset(ler)
    train_ds = None
    if args.mode == "train":
        train_ds = SERProtoDataset(args.train_csv, emb_mode=args.emb_mode, input_l2norm=args.input_l2norm)

    val_ds = SERProtoDataset(args.val_csv, emb_mode=args.emb_mode, input_l2norm=args.input_l2norm)

    # Boyutlar (train varsa ondan, yoksa val'den)
    d_e2v, d_wavlm = (train_ds.inferred_dims() if train_ds is not None else val_ds.inferred_dims())

    # Model
    model = FusionProtoClassifier(
        emb_mode=args.emb_mode,
        fusion=args.fusion,
        proj_dim=args.proj_dim,
        num_classes=len(EMOTIONS),
        dropout=args.dropout,
        d_e2v=d_e2v or 1024,
        d_wavlm=d_wavlm or 768,   # base için doğal varsayılan
    )
    device = torch.device(args.device)
    model = model.to(device)

    # Load if provided
    if args.load_path and os.path.exists(args.load_path):
        ckpt = torch.load(args.load_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[OK] Loaded checkpoint from {args.load_path}")

    if args.mode == "train":
        # prototipleri veriden başlat (opsiyonel ama faydalı)
        init_prototypes_from_data(model, args.train_csv, emb_mode=args.emb_mode, input_l2norm=args.input_l2norm)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=2, collate_fn=collate, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=2, collate_fn=collate)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

        best_acc, bad_eps = -1.0, 0
        for ep in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, opt, device,
                label_smoothing=args.label_smoothing,
                proto_ema=args.proto_ema
            )
            val_res = evaluate(model, val_loader, device, report_path=None, verbose=False)
            acc = val_res["accuracy"]
            dt = time.time() - t0
            print(f"\n[EPOCH {ep}/{args.epochs}] train_loss={tr_loss:.4f} "
                  f"train_acc={tr_acc*100:.2f}%  val_acc={acc*100:.2f}%  ({dt:.1f}s)")

            if acc > best_acc:
                best_acc = acc; bad_eps = 0
                torch.save({"model": model.state_dict(), "config": vars(args)}, args.save_path)
                print(f"[OK] Saved best checkpoint to {args.save_path}")
            else:
                bad_eps += 1
                if bad_eps >= args.early_stop_patience:
                    print(f"[EARLY STOP] no val improvement for {bad_eps} epochs.")
                    break

        # final eval dump (best checkpoint ile)
        best_path = args.save_path
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            print(f"[OK] Re-loaded best checkpoint from {best_path} for final report")
        evaluate(model, val_loader, device, report_path=args.report_path, verbose=True)

    elif args.mode == "eval":
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=2, collate_fn=collate)
        evaluate(model, val_loader, device, report_path=args.report_path)
    
    elif args.mode == "eval_tts":
        val_ds = SERProtoDataset(args.val_csv, emb_mode=args.emb_mode, input_l2norm=args.input_l2norm)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=2, collate_fn=collate)

        # Modeli (eval ile aynı şekilde) hazırla
        d_e2v, d_wavlm = val_ds.inferred_dims()
        model = FusionProtoClassifier(
            emb_mode=args.emb_mode,
            fusion=args.fusion,
            proj_dim=args.proj_dim,
            num_classes=len(EMOTIONS),
            dropout=args.dropout,
            d_e2v=d_e2v or 1024,
            d_wavlm=d_wavlm or 768,
        ).to(torch.device(args.device))

        if args.load_path and os.path.exists(args.load_path):
            ckpt = torch.load(args.load_path, map_location=torch.device(args.device))
            model.load_state_dict(ckpt["model"])
            print(f"[OK] Loaded checkpoint from {args.load_path}")

        evaluate_tts(model, val_loader, torch.device(args.device), report_path=args.report_path, verbose=True)

    elif args.mode == "rerank":
        rerank_by_eas(model, args.val_csv, device, topk=args.rerank_topk,
                      emb_mode=args.emb_mode, fusion=args.fusion,
                      input_l2norm=args.input_l2norm, out_csv="reranked.csv")

# ----- yardımcı: prototip init (dataset sürümü) -----
@torch.no_grad()
def init_prototypes_from_data(model, csv_path, emb_mode="both", input_l2norm=False):
    ds = SERProtoDataset(csv_path, emb_mode=emb_mode, input_l2norm=input_l2norm)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate)
    device = next(model.parameters()).device
    C = len(EMOTIONS)
    D = model.prototypes.shape[1]
    sums = torch.zeros(C, D, device=device)
    counts = torch.zeros(C, device=device)
    model.eval()
    for batch_x, y, _ in loader:
        x_e2v = batch_x["x_e2v"].to(device) if batch_x["x_e2v"] is not None else None
        x_wavlm = batch_x["x_wavlm"].to(device) if batch_x["x_wavlm"] is not None else None
        # Sadece z’yi al
        _, z, _ = model(x_e2v=x_e2v, x_wavlm=x_wavlm)
        y = y.to(device)
        for c in range(C):
            m = (y == c)
            if m.any():
                sums[c] += z[m].sum(0)
                counts[c] += m.sum()
    centroids = torch.where(counts[:, None] > 0, sums / counts[:, None], sums)
    model.prototypes.data = centroids

if __name__ == "__main__":
    main()