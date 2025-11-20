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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
                    "emb_path": r.get("emb_path"),     # legacy e2v column name for backward compatibility
                    "emb_e2v": r.get("emb_e2v"),
                    "emb_wavlm": r.get("emb_wavlm"),
                    "tts": r.get("tts")
                })

        # --- Automatically infer embedding dimensions ---
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
        # If nothing is found: return None if the stream will not be used, otherwise fall back to a reasonable default
        if self.emb_mode in ("e2v", "both") and stream == "e2v":
            return 1024  # emotion2vec+ tipik
        if self.emb_mode in ("wavlm", "both") and stream == "wavlm":
            return 1024   # WavLM-Base için makul varsayılan (Large ise 1024)
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
        # E2V path: prefer the new column, fall back to legacy 'emb_path' if needed
        x_e2v = self._load_vec(it.get("emb_e2v") or it.get("emb_path")) if self.emb_mode in ("e2v", "both") else None
        x_wavlm = self._load_vec(it.get("emb_wavlm")) if self.emb_mode in ("wavlm", "both") else None

        # If missing, create a dummy vector with the appropriate dimension
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

    # Helper to read the model's input dimensions
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
            # two-stream fusion
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
                    # learnable fusion weight parameter
                    self.fusion_alpha = nn.Parameter(torch.tensor(float(fusion_alpha)))
                else:
                    # fixed fusion weight stored as a buffer
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
                alpha = torch.clamp(self.fusion_alpha, 0.0, 1.0)  # safety clamp
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

        # ----- EMA prototype update (no gradients) -----
        with torch.no_grad():
            # z is already L2-normalized; keeping prototypes normalized is beneficial
            for c in range(C):
                m = (y == c)
                if m.any():
                    cls_mean = z[m].mean(dim=0)  # (D,)
                    # new = ema * old + (1 - ema) * batch_mean
                    model.prototypes.data[c].mul_(proto_ema).add_(cls_mean, alpha=(1.0 - proto_ema))
                    # Optionally keep each prototype at unit norm
                    model.prototypes.data[c] = F.normalize(model.prototypes.data[c], dim=0)

        # ----- metrics accumulation -----
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
    all_eas = []  # <-- accumulate EAS values

    for batch_x, y, metas in loader:
        x_e2v = batch_x["x_e2v"].to(device) if batch_x["x_e2v"] is not None else None
        x_wavlm = batch_x["x_wavlm"].to(device) if batch_x["x_wavlm"] is not None else None
        y = y.to(device)

        # obtain z so we can compute EAS
        logits, z, _ = model(x_e2v=x_e2v, x_wavlm=x_wavlm)
        pred = logits.argmax(-1)

        # batch-level EAS
        eas = model.compute_eas(z, y)  # (B,)
        all_eas.extend(eas.detach().cpu().tolist())

        all_preds.extend(pred.cpu().tolist())
        all_targets.extend(y.cpu().tolist())
        metas_accum.extend(metas)

    # label -> name
    idx2emo = {i: e for e, i in EMO2IDX.items()}
    y_true = [idx2emo[i] for i in all_targets]
    y_pred = [idx2emo[i] for i in all_preds]

    # ---- Overall summary
    overall_acc = accuracy_score(y_true, y_pred)
    overall_report = classification_report(
        y_true, y_pred, labels=EMOTIONS, digits=4, zero_division=0, output_dict=True
    )
    overall_cm = confusion_matrix(y_true, y_pred, labels=EMOTIONS)
    overall_cm_df = pd.DataFrame(overall_cm, index=EMOTIONS, columns=EMOTIONS)
    overall_mean_eas = float(np.mean(all_eas)) if len(all_eas) > 0 else None

    # ---- Per-TTS breakdown
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

    # Console output (summary table + detailed per-TTS reports)
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
# Feature export & t-SNE utilities
# -----------------------------
@torch.no_grad()
def export_embeddings(model, loader, device, out_npz: str = "results/val_embeddings.npz"):
    """
    Export fused (pre-logit) embeddings used for cosine classification and EAS.
    For emb_mode='both':
      - z_e2v:  projected Emotion2Vec+ features (after linear+LN+GELU+dropout)
      - z_wav:  projected WavLM features     (after linear+LN+GELU+dropout or GaussianNoise if defined)
      - z:      fused feature after fusion layer (for projcat: fuse( concat(z_e2v, z_wav) ))
                then L2-normalized (this is the representation used against prototypes)
    Saved fields:
      - z (N,D), optionally z_e2v (N,d), z_wav (N,d)
      - emotion (list[str]), y (int labels), tts (list[str]), prompt_id (list), wav_path (list[str])
      - pred (int), top_score (float softmax prob)
      - prototypes (C,D) [L2-normalized], classes (list[str])
    """
    model.eval()
    Z_fused, Z_e2v, Z_wav = [], [], []
    emos, yidx, tts_list, pids, wavs = [], [], [], [], []
    preds, tops = [], []

    for batch_x, y, metas in loader:
        x_e2v = batch_x["x_e2v"].to(device) if batch_x["x_e2v"] is not None else None
        x_wav = batch_x["x_wavlm"].to(device) if batch_x["x_wavlm"] is not None else None
        y = y.to(device)

        # Manually mirror forward pass to access per-stream projections
        if model.emb_mode in ("e2v", "wavlm"):
            z_fused = model.proj_single(x_e2v if model.emb_mode == "e2v" else x_wav)
            z_e2v = None
            z_wav = None
        else:
            z_e2v = model.proj_e2v(x_e2v)
            z_wav = model.proj_wav(x_wav)
            if model.fusion == "avg":
                alpha = torch.clamp(getattr(model, "fusion_alpha"), 0.0, 1.0)
                z_fused = alpha * z_e2v + (1.0 - alpha) * z_wav
            elif model.fusion == "concat":
                z_fused = torch.cat([z_e2v, z_wav], dim=-1)
            else:  # projcat
                z_fused = model.fuse(torch.cat([z_e2v, z_wav], dim=-1))

        # Normalize (this is the exact z used for cosine/prototypes)
        z = F.normalize(z_fused, dim=-1)

        # logits/pred/top prob for context
        E = F.normalize(model.prototypes, dim=-1)
        logits = torch.matmul(z, E.t()) * torch.exp(model.log_scale)
        prob = torch.softmax(logits, dim=-1)
        pred = logits.argmax(-1)
        top = prob[torch.arange(prob.size(0)), pred]

        # Accumulate
        Z_fused.append(z.detach().cpu().float().numpy())
        if z_e2v is not None:
            Z_e2v.append(z_e2v.detach().cpu().float().numpy())
            Z_wav.append(z_wav.detach().cpu().float().numpy())
        preds.extend(pred.cpu().tolist())
        tops.extend(top.cpu().tolist())

        idx2emo = {i: e for e, i in EMO2IDX.items()}
        yidx.extend(y.cpu().tolist())
        emos.extend([idx2emo[i] for i in y.cpu().tolist()])
        for m in metas:
            tts_list.append(m.get("tts"))
            pids.append(m.get("prompt_id"))
            wavs.append(m.get("wav_path"))

    pack = {
        "z": np.vstack(Z_fused),
        "emotion": np.array(emos),
        "y": np.array(yidx, dtype=np.int64),
        "tts": np.array(tts_list),
        "prompt_id": np.array(pids),
        "wav_path": np.array(wavs),
        "pred": np.array(preds, dtype=np.int64),
        "top_score": np.array(tops, dtype=np.float32),
        "prototypes": F.normalize(model.prototypes, dim=-1).detach().cpu().numpy(),
        "classes": np.array(EMOTIONS),
    }
    if len(Z_e2v) > 0:
        pack["z_e2v"] = np.vstack(Z_e2v)
        pack["z_wav"] = np.vstack(Z_wav)

    out_dir = os.path.dirname(out_npz)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    np.savez(out_npz, **pack)
    print(f"[OK] Exported embeddings to {out_npz} "
          f"(z shape={pack['z'].shape}, prototypes={pack['prototypes'].shape})")



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

@torch.no_grad()
def dump_eas_csv(model, loader, device, out_csv: str = "eas_scores.csv", scale_eas_to_7: bool = False):
    """
    For each sample, compute the EAS score with respect to the true label and write it to a CSV file.
    Columns:
      - emotion         (true emotion string)
      - wav_path
      - eas             (0..1)
      - eas_x7          (optional; linearly scaled to 1..7)
      - pred_emotion
      - tts
      - prompt_id
      - intensity (currently commented out in the output)
    """
    model.eval()
    rows = []
    idx2emo = {i: e for e, i in EMO2IDX.items()}

    for batch_x, y, metas in loader:
        x_e2v = batch_x["x_e2v"].to(device) if batch_x["x_e2v"] is not None else None
        x_wavlm = batch_x["x_wavlm"].to(device) if batch_x["x_wavlm"] is not None else None
        y = y.to(device)

        logits, z, _ = model(x_e2v=x_e2v, x_wavlm=x_wavlm)
        pred = logits.argmax(-1)
        eas = model.compute_eas(z, y)  # true label'a göre EAS
        probs = torch.softmax(logits, dim=-1)

        for i in range(y.size(0)):
            emo_true = idx2emo[y[i].item()]
            emo_pred = idx2emo[pred[i].item()]
            meta = metas[i]
            wav_path = meta.get("wav_path")
            pid = meta.get("prompt_id")
            intensity = meta.get("intensity")
            tts = meta.get("tts")
            eas_val = float(eas[i].item())
            # top-1 confidence (softmax over logits at predicted class)
            top_conf = float(probs[i, pred[i]].item())

            row = {
                "emotion": emo_true,
                "wav_path": wav_path,
                "eas": eas_val,
                "emotion_pred": emo_pred,
                "top_score": top_conf,
                "tts": tts,
                "prompt_id": pid,
                #"intensity": intensity,
            }
            if scale_eas_to_7:
                row["eas_x7"] = eas_val * 7.0
            rows.append(row)

    df = pd.DataFrame(rows)
    # Column order harmonized with dump_eas_csv style
    ordered_cols = [ "tts", "prompt_id", "wav_path", "emotion","emotion_pred", "top_score","eas"]
    if "eas_x7" in df.columns:
        ordered_cols.append("eas_x7")
    #ordered_cols += [ "intensity"]
    df = df.reindex(columns=ordered_cols)
    # Write CSV to disk
    df.to_csv(out_csv, index=False)
    print(f"[OK] EAS per-sample CSV written to: {out_csv} (N={len(df)})")
    return df

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train",
        choices=["train", "eval", "rerank", "eval_tts", "dump_eas", "export_emb"],
        help="Run mode: train / eval / rerank / eval_tts / dump_eas / export_emb")
    parser.add_argument("--emb_npz", type=str, default="results/val_embeddings.npz",
                        help="Path to write/read embeddings when using export_emb / plot_tsne")
    parser.add_argument("--train_csv", type=str, default="csv_with_emb_dual/train.csv",
                        help="CSV with precomputed paths (emb_e2v/emb_wavlm)")
    parser.add_argument("--val_csv", type=str, default="csv_with_emb_dual/val.csv",
                        help="CSV for validation/eval")
    # embedding read options
    parser.add_argument("--emb_mode", type=str, default="both", choices=["e2v", "wavlm", "both"])
    parser.add_argument("--fusion", type=str, default="concat", choices=["avg","concat","projcat"])
    parser.add_argument("--input_l2norm", action="store_true",
                    help="L2-normalize input embeddings before feeding them to the model")
    # training hyperparameters
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
                    help="Prototype EMA momentum (0.0 = use only batch mean, 1.0 = almost fixed prototypes)")
    parser.add_argument("--freeze_prototypes", action="store_true",
                    help="Exclude prototypes from optimization (no gradients), update them only via EMA")
    parser.add_argument("--fusion_alpha", type=float, default=0.5,
                    help="Fusion=avg: weight assigned to Emotion2Vec (0..1); WavLM uses 1 - alpha.")
    parser.add_argument("--learn_fusion_alpha", action="store_true",
                    help="If set, make fusion_alpha a learnable parameter.")
    # validation scores dump
    parser.add_argument("--out_csv", type=str, default="eas_scores.csv",
                        help="Output CSV path when running in dump_eas mode")
    parser.add_argument("--scale_eas_to_7", action="store_true",
                        help="Scale EAS from 0..1 to 1..7 and add it as an 'eas_x7' column")

    args = parser.parse_args()
    set_seed(args.seed)

    # Datasets
    train_ds = None
    if args.mode == "train":
        train_ds = SERProtoDataset(args.train_csv, emb_mode=args.emb_mode, input_l2norm=args.input_l2norm)

    val_ds = SERProtoDataset(args.val_csv, emb_mode=args.emb_mode, input_l2norm=args.input_l2norm)

    # Dimensions (from train set if available, otherwise from validation set)
    d_e2v, d_wavlm = (train_ds.inferred_dims() if train_ds is not None else val_ds.inferred_dims())

    # Model
    model = FusionProtoClassifier(
        emb_mode=args.emb_mode,
        fusion=args.fusion,
        proj_dim=args.proj_dim,
        num_classes=len(EMOTIONS),
        dropout=args.dropout,
        d_e2v=d_e2v or 1024,
        d_wavlm=d_wavlm or 768,   # natural default for WavLM-Base
    )
    device = torch.device(args.device)
    model = model.to(device)

    # Load if provided
    if args.load_path and os.path.exists(args.load_path):
        ckpt = torch.load(args.load_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[OK] Loaded checkpoint from {args.load_path}")

    if args.mode == "train":
        # initialize prototypes from data (optional but often helpful)
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

        # final evaluation dump using the best checkpoint
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
        evaluate_tts(model, val_loader, device, report_path=args.report_path, verbose=True)

    elif args.mode == "rerank":
        rerank_by_eas(model, args.val_csv, device, topk=args.rerank_topk,
                      emb_mode=args.emb_mode, fusion=args.fusion,
                      input_l2norm=args.input_l2norm, out_csv="reranked.csv")
    elif args.mode == "dump_eas":
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=2, collate_fn=collate)
        model.eval()
        if not (args.load_path and os.path.exists(args.load_path)):
            print("[WARN] --load_path not provided; EAS will be computed with randomly initialized prototypes.")
        dump_eas_csv(model, val_loader, device, out_csv=args.out_csv, scale_eas_to_7=args.scale_eas_to_7)

    elif args.mode == "export_emb":
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=2, collate_fn=collate)
        export_embeddings(model, val_loader, device, out_npz=args.emb_npz)
        return

# ----- helper: prototype initialization from dataset -----
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
        # Only use z (projected embeddings) for computing class centroids
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