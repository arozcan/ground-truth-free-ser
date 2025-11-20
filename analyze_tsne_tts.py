#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

EMOTIONS = ['angry','disgusted','fearful','happy','neutral','sad','surprised']
EMO_COLORS = {e:c for e,c in zip(
    EMOTIONS, ["C0","C1","C2","C3","C4","C5","C6"]
)}
MARKERS = ["o","s","D","^","v","P","*"]

def knn_alt_stats(df_sub: pd.DataFrame, k: int = 10):
    """k-NN (2D t-SNE uzayında) ile her örnek için:
       - nn_impurity: komşuların kaçta kaçı farklı duygu
       - nn_alt: en sık görülen alternatif duygu
       Döndürür: zenginleştirilmiş DataFrame ve (gerçek, alternatif) sayım matrisi.
    """
    if len(df_sub) < 3:
        df_sub = df_sub.copy()
        df_sub["nn_impurity"] = np.nan
        df_sub["nn_alt"] = None
        return df_sub, pd.DataFrame()

    X = df_sub[["tsne_x","tsne_y"]].to_numpy()
    emos = df_sub["emotion"].to_numpy()
    nnb = min(k+1, len(df_sub))  # +1 kendisi
    nbrs = NearestNeighbors(n_neighbors=nnb, metric="euclidean").fit(X)
    _, idx = nbrs.kneighbors(X)
    idx = idx[:,1:]  # kendisini at

    nn_impurity, nn_alt = [], []
    for i, neigh in enumerate(idx):
        neigh_emos = emos[neigh]
        diff = neigh_emos[neigh_emos != emos[i]]
        imp = len(diff) / len(neigh)
        if len(diff) > 0:
            vals, cnts = np.unique(diff, return_counts=True)
            alt = vals[np.argmax(cnts)]
        else:
            alt = None
        nn_impurity.append(imp); nn_alt.append(alt)

    out = df_sub.copy()
    out["nn_impurity"] = nn_impurity
    out["nn_alt"] = nn_alt
    pair_counts = out.groupby(["emotion","nn_alt"]).size().unstack(fill_value=0)
    return out, pair_counts

def plot_tts_subset(df_sub: pd.DataFrame, protos: pd.DataFrame, out_png: str):
    plt.figure(figsize=(7,5), dpi=300)
    for i, emo in enumerate(EMOTIONS):
        m = (df_sub["emotion"] == emo)
        if not np.any(m): 
            continue
        plt.scatter(df_sub.loc[m,"tsne_x"], df_sub.loc[m,"tsne_y"],
                    s=20, alpha=0.75, marker=MARKERS[i%len(MARKERS)],
                    color=EMO_COLORS[emo], label=emo)

    # Prototipleri aynı renkle çiz
    if protos is not None and not protos.empty:
        for emo in EMOTIONS:
            mp = (protos["emotion"] == emo)
            if not np.any(mp): 
                continue
            plt.scatter(protos.loc[mp,"tsne_x"], protos.loc[mp,"tsne_y"],
                        marker="*", s=220, edgecolor="k", linewidths=0.8,
                        color=EMO_COLORS[emo], label=f"{emo} (proto)")

    # Legendi tekilleştir
    h, l = plt.gca().get_legend_handles_labels()
    seen, H, L = set(), [], []
    for hh, ll in zip(h, l):
        if ll not in seen:
            H.append(hh); L.append(ll); seen.add(ll)
    plt.legend(H, L, loc="best", fontsize=8.5, ncol=2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsne_points", required=True, help="tsne_points.csv (örnekler)")
    ap.add_argument("--tsne_protos", required=False, default=None, help="tsne_points_protos.csv (prototipler)")
    ap.add_argument("--out_dir", required=True, help="çıktı klasörü")
    ap.add_argument("--k", type=int, default=10, help="k-NN k (varsayılan 10)")
    ap.add_argument("--proto_from_centroid", action="store_true",
                    help="If set (or if --tsne_protos is missing/invalid), compute prototype positions as per-emotion centroids in the same t-SNE space.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.tsne_points)
    need_cols = {"wav_path","emotion","tts","tsne_x","tsne_y"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"tsne_points.csv eksik kolonlar: {missing}")

    protos = None
    if args.tsne_protos and os.path.exists(args.tsne_protos):
        protos = pd.read_csv(args.tsne_protos)
        if not {"emotion","tsne_x","tsne_y"} <= set(protos.columns):
            protos = None

    # Fallback: per-emotion centroids in t-SNE if prototypes file is absent or user requested centroid mode
    if protos is None and args.proto_from_centroid:
        cent = df.groupby("emotion", as_index=False)[["tsne_x","tsne_y"]].mean()
        cent["source"] = "centroid"
        protos = cent

    # TTS bazlı özet
    summary_rows = []
    for tts, g in df.groupby("tts"):
        g = g.copy()
        # Silhouette (aynı 2D uzayda, etiket: emotion)
        try:
            sil = silhouette_score(g[["tsne_x","tsne_y"]], g["emotion"]) if g["emotion"].nunique() > 1 and len(g) > 3 else np.nan
        except Exception:
            sil = np.nan
        # kNN impurity ve alternatif çiftleri
        g2, pair_counts = knn_alt_stats(g, k=args.k)
        impurity_mean = float(np.nanmean(g2["nn_impurity"])) if len(g2) else np.nan

        # Her duygu için en sık alternatif (kısa özet)
        alt_summ = []
        for emo in EMOTIONS:
            sub = g2[g2["emotion"] == emo]
            if len(sub) == 0 or sub["nn_alt"].dropna().empty:
                continue
            vals, cnts = np.unique(sub["nn_alt"].dropna(), return_counts=True)
            j = int(np.argmax(cnts))
            alt_summ.append(f"{emo}->{vals[j]} ({int(cnts[j])})")
        alt_summ_str = "; ".join(alt_summ) if alt_summ else "-"

        summary_rows.append({
            "tts": tts, "n": len(g),
            "silhouette": round(float(sil), 3) if not np.isnan(sil) else np.nan,
            "knn_impurity_mean": round(float(impurity_mean), 3) if not np.isnan(impurity_mean) else np.nan,
            "top_alt_pairs": alt_summ_str
        })

        # Kaydet: çift sayım matrisi
        pair_out = os.path.join(args.out_dir, f"pair_counts_{tts}.csv")
        pair_counts.to_csv(pair_out)

        # Görsel: TTS alt kümesi
        proto_df = protos if protos is not None else None
        out_png = os.path.join(args.out_dir, f"tsne_by_tts_{tts}.png")
        plot_tts_subset(g, proto_df, out_png)

        # Ayrıntılı zenginleştirilmiş CSV (opsiyonel)
        det_out = os.path.join(args.out_dir, f"detailed_{tts}.csv")
        g2.to_csv(det_out, index=False)

    pd.DataFrame(summary_rows).to_csv(os.path.join(args.out_dir, "tts_summary.csv"), index=False)
    print(f"[OK] Yazıldı -> {args.out_dir}")
    print(" - tts_summary.csv  (silhouette, kNN impurity, en sık alternatif çiftler)")
    print(" - pair_counts_<tts>.csv  (gerçek x alternatif sayımları)")
    print(" - tsne_by_tts_<tts>.png  (TTS bazlı t-SNE dağılımları)")
    print(" - detailed_<tts>.csv  (örnek başına kNN metrikleri)")
    if protos is None:
        print("Note: Prototype markers were not drawn (no --tsne_protos provided and --proto_from_centroid not set).")
    elif "source" in protos.columns and (protos["source"] == "centroid").any():
        print("Note: Prototype markers use per-emotion centroids in t-SNE space (centroid fallback).")
    else:
        print("Note: Prototype markers were drawn from --tsne_protos.")
if __name__ == "__main__":
    main()