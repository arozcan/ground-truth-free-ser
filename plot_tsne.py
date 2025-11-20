# plot_tsne.py
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
import math
import csv as _csv

EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def plot_tsne_from_npz(npz_path: str,
                       out_png: str = "img/tsne_fused.png",
                       perplexity: float = 15.0,
                       n_iter: int = 3000,
                       include_prototypes: bool = True,
                       metric: str = "cosine",
                       init: str = "pca",
                       auto_select: bool = False,
                       select_by: str = "f1",
                       perp_grid=None,
                       eval_out: str = None,
                       color_by: str = "emotion",
                       dump_csv: str = None,
                       protos_out: str = None,
                       proto_from_centroid: bool = False,
                       nn_k: int = 10,
                       nn_space: str = "tsne"):
    """
    Render a t-SNE scatter from an embeddings NPZ exported by export_embeddings().
    If prototypes are present and include_prototypes=True, fit is performed on
    [samples; prototypes] jointly so prototype points appear in the same space.
    When auto_select=True, a small sweep over 'perp_grid' is performed and the
    perplexity that maximizes trustworthiness (then silhouette, then 3-NN macro-F1)
    is chosen. Selection metrics can be written to CSV via eval_out.
    Selection criterion is controlled by 'select_by' (one of: 'f1', 'silhouette', 'trust'); default is 'f1'.

    If dump_csv is provided, a per-sample CSV is written with t-SNE coordinates, EAS, prototype margin,
    correctness, and local k-NN confusion stats. The nn_space parameter controls the space for nearest-neighbor
    statistics ('tsne' for t-SNE space, 'emb' for original embedding space).
    If protos_out is provided, a prototype CSV is written with header 'class,tsne_x,tsne_y'. If model prototypes are unavailable or excluded, setting proto_from_centroid=True computes class centroids (from t-SNE coordinates) and writes those instead.
    """
    data = np.load(npz_path, allow_pickle=True)
    X = data["z"]
    emos = data["emotion"]
    classes = list(data["classes"])
    y_idx = data["y"] if "y" in data.files else None
    preds = data["pred"] if "pred" in data.files else None
    tts = data["tts"] if "tts" in data.files else None
    top_score = data["top_score"] if "top_score" in data.files else None
    P = data["prototypes"] if ("prototypes" in data.files) else None

    wav_path = data["wav_path"] if ("wav_path" in data.files) else None
    prompt_id = data["prompt_id"] if ("prompt_id" in data.files) else None
    intensity = data["intensity"] if ("intensity" in data.files) else None

    # coloring
    if color_by == "emotion":
        labels = emos
        legend_labels = classes
    elif color_by == "tts" and tts is not None:
        labels = tts
        legend_labels = sorted(list({t for t in labels}))
    elif color_by == "pred" and preds is not None:
        idx2emo = {i: e for i, e in enumerate(classes)}
        labels = np.array([idx2emo[int(i)] for i in preds])
        legend_labels = classes
    elif color_by == "correct" and (preds is not None) and (y_idx is not None):
        corr = (preds.astype(int) == y_idx.astype(int))
        labels = np.where(corr, "correct", "incorrect")
        legend_labels = ["correct", "incorrect"]
    else:
        labels = emos
        legend_labels = classes

    prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
    base_colors = (prop_cycle.by_key()["color"] if prop_cycle is not None
                   else ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"])
    color_map = {lab: base_colors[i % len(base_colors)] for i, lab in enumerate(legend_labels)}

    def _fit_once(perp: float):
        try:
            tsne = TSNE(n_components=2, perplexity=float(perp),
                        random_state=1337, init=init, learning_rate="auto",
                        metric=metric, n_iter=n_iter)
        except TypeError:
            tsne = TSNE(n_components=2, perplexity=float(perp),
                        random_state=1337, init=init, learning_rate="auto")
        if include_prototypes and P is not None:
            X_fit = np.concatenate([X, P], axis=0)
        else:
            X_fit = X
        Y_fit = tsne.fit_transform(X_fit)
        if include_prototypes and P is not None:
            Y = Y_fit[:X.shape[0]]
            Yp = Y_fit[X.shape[0]:]
        else:
            Y, Yp = Y_fit, None

        # selection metrics
        k_tw = int(min(max(5, float(perp)), max(1, X.shape[0]-1)))
        try:
            tw = trustworthiness(X, Y, n_neighbors=k_tw, metric=metric)
        except TypeError:
            tw = trustworthiness(X, Y, n_neighbors=k_tw)
        sil = np.nan
        if len(set(labels)) > 1:
            lab2id = {lab:i for i,lab in enumerate(sorted(set(labels)))}
            y_int = np.array([lab2id[l] for l in labels])
            sil = silhouette_score(Y, y_int)
        f1m = np.nan
        try:
            lab2id = {lab:i for i,lab in enumerate(sorted(set(labels)))}
            y_int = np.array([lab2id[l] for l in labels])
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(Y, y_int)
            f1m = f1_score(y_int, knn.predict(Y), average="macro")
        except Exception:
            pass

        return Y, Yp, dict(perplexity=float(perp), trust=tw,
                           silhouette=float(sil) if not np.isnan(sil) else np.nan,
                           f1_3nn=float(f1m) if not np.isnan(f1m) else np.nan,
                           kl=getattr(tsne, "kl_divergence_", np.nan))

    chosen_perp = perplexity
    metrics_rows = []
    if auto_select:
        grid = perp_grid if (perp_grid and len(perp_grid)>0) else [10,15,20,25,30]
        best = None
        best_Y = best_Yp = None
        for p in grid:
            Y_tmp, Yp_tmp, m = _fit_once(p)
            metrics_rows.append(m)
            # Define comparison key based on selection criterion
            # Use -inf for NaNs to avoid selecting invalid metrics
            def _safe(v):
                return -float("inf") if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)
            if select_by == "f1":
                key = (round(_safe(m["f1_3nn"]), 6), round(_safe(m["silhouette"]), 6), round(_safe(m["trust"]), 6))
            elif select_by == "silhouette":
                key = (round(_safe(m["silhouette"]), 6), round(_safe(m["f1_3nn"]), 6), round(_safe(m["trust"]), 6))
            else:  # "trust"
                key = (round(_safe(m["trust"]), 6), round(_safe(m["silhouette"]), 6), round(_safe(m["f1_3nn"]), 6))
            if (best is None) or (key > best[0]):
                best = (key, p, m)
                best_Y, best_Yp = Y_tmp, Yp_tmp
        chosen_perp = float(best[1])
        Y, Yp = best_Y, best_Yp
    else:
        Y, Yp, m = _fit_once(perplexity)
        metrics_rows.append(m)

    # Derive prototype coordinates from class centroids if requested and missing
    if (Yp is None or (include_prototypes is False)) and proto_from_centroid and (y_idx is not None):
        # Compute per-class centroids in t-SNE space for consistency with the scatter
        Yp = np.zeros((len(classes), 2), dtype=float)
        for ci in range(len(classes)):
            mask = (y_idx.astype(int) == ci)
            if np.any(mask):
                Yp[ci] = Y[mask].mean(axis=0)
            else:
                Yp[ci] = np.array([np.nan, np.nan])

    if eval_out:
        os.makedirs(os.path.dirname(eval_out), exist_ok=True)
        with open(eval_out, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            w.writeheader()
            for r in metrics_rows:
                w.writerow(r)

    # Compute normalized embeddings and prototype margin
    Z = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    Ep = None
    if P is not None:
        Ep = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-8)
    eas = None
    margin = None
    if (y_idx is not None) and (P is not None):
        Cos = Z @ Ep.T
        t_idx = y_idx.astype(int)
        t_cos = Cos[np.arange(Cos.shape[0]), t_idx]
        # EAS in [0,1]
        eas = (t_cos + 1.0) / 2.0
        # margin vs. best competing prototype
        Cos[np.arange(Cos.shape[0]), t_idx] = -np.inf
        best_other = Cos.max(axis=1)
        margin = t_cos - best_other

    # Compute local k-NN confusion stats if dump_csv is set
    nn_impurity = None
    nn_alt = None
    nn_alt_frac = None
    knn3_match = None
    if dump_csv:
        if nn_space == "tsne":
            Y_space = Y
            metric_nn = "euclidean"
        else:
            Y_space = Z
            metric_nn = "cosine"
        k = min(nn_k+1, len(Y_space))
        nbrs = NearestNeighbors(n_neighbors=k, metric=metric_nn)
        indices = nbrs.fit(Y_space).kneighbors(return_distance=False)
        nn_impurity = np.zeros(len(Y_space))
        nn_alt = []
        nn_alt_frac = np.zeros(len(Y_space))
        knn3_match = np.zeros(len(Y_space), dtype=bool)

        if y_idx is not None:
            y_true = y_idx.astype(int)
        else:
            y_true = None
        if preds is not None:
            idx2emo = {i: e for i, e in enumerate(classes)}
            y_pred = preds.astype(int)
        else:
            y_pred = None

        for i in range(len(Y_space)):
            neigh_idx = indices[i, 1:k]  # exclude self at index 0
            if y_true is not None:
                neigh_labels = y_true[neigh_idx]
                same_count = np.sum(neigh_labels == y_true[i])
                frac_same = same_count / (k-1)
                nn_impurity[i] = 1.0 - frac_same
                # find most frequent different label
                diff_labels = neigh_labels[neigh_labels != y_true[i]]
                if len(diff_labels) == 0:
                    nn_alt.append("")
                    nn_alt_frac[i] = 0.0
                else:
                    vals, counts = np.unique(diff_labels, return_counts=True)
                    max_idx = np.argmax(counts)
                    nn_alt.append(classes[vals[max_idx]])
                    nn_alt_frac[i] = counts[max_idx] / len(diff_labels)
            else:
                nn_alt.append("")
                nn_impurity[i] = np.nan
                nn_alt_frac[i] = np.nan

        # 3-NN majority label in chosen space
        if y_true is not None:
            knn3 = KNeighborsClassifier(n_neighbors=3)
            knn3.fit(Y_space, y_true)
            knn3_pred = knn3.predict(Y_space)
            knn3_match = (knn3_pred == y_true)
        else:
            knn3_match = np.array([False]*len(Y_space))

    # Build rows for CSV output if dump_csv is set
    if dump_csv:
        rows = []
        for i in range(len(X)):
            row = {
                "idx": i,
                "wav_path": wav_path[i] if (wav_path is not None and len(wav_path)>i) else "",
                "prompt_id": prompt_id[i] if (prompt_id is not None and len(prompt_id)>i) else "",
                "tts": tts[i] if (tts is not None and len(tts)>i) else "",
                "emotion": emos[i] if (emos is not None and len(emos)>i) else "",
                "emotion_pred": classes[int(preds[i])] if (preds is not None and len(preds)>i) else "",
                "correct": "",
                "top_score": top_score[i] if (top_score is not None and len(top_score)>i) else "",
                "eas": eas[i] if (eas is not None and len(eas)>i) else "",
                "proto_margin": margin[i] if (margin is not None and len(margin)>i) else "",
                "tsne_x": Y[i,0] if (Y is not None and len(Y)>i) else "",
                "tsne_y": Y[i,1] if (Y is not None and len(Y)>i) else "",
                "nn_space": nn_space,
                "nn_k": nn_k,
                "nn_impurity": nn_impurity[i] if (nn_impurity is not None and len(nn_impurity)>i) else "",
                "nn_alt": nn_alt[i] if (nn_alt is not None and len(nn_alt)>i) else "",
                "nn_alt_frac": nn_alt_frac[i] if (nn_alt_frac is not None and len(nn_alt_frac)>i) else "",
                "knn3_match": knn3_match[i] if (knn3_match is not None and len(knn3_match)>i) else "",
            }
            if preds is not None and y_idx is not None and len(preds)>i and len(y_idx)>i:
                row["correct"] = int(preds[i]) == int(y_idx[i])
            rows.append(row)

        os.makedirs(os.path.dirname(dump_csv), exist_ok=True)
        with open(dump_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["idx", "wav_path", "prompt_id", "tts", "emotion", "emotion_pred", "correct",
                          "top_score", "eas", "proto_margin", "tsne_x", "tsne_y",
                          "nn_space", "nn_k", "nn_impurity", "nn_alt", "nn_alt_frac", "knn3_match"]
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        print(f"[OK] Wrote per-sample t-SNE CSV -> {dump_csv}")

    # Decide prototype CSV path (explicit overrides derived)
    proto_csv_path = None
    if protos_out:
        proto_csv_path = protos_out
    elif dump_csv:
        proto_csv_path = dump_csv.replace(".csv", "_protos.csv")

    # Write prototype CSV if coordinates exist
    if (Yp is not None) and (proto_csv_path is not None):
        os.makedirs(os.path.dirname(proto_csv_path), exist_ok=True)
        with open(proto_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = _csv.writer(f)
            writer.writerow(["emotion", "tsne_x", "tsne_y"])
            for ci, cls in enumerate(classes):
                writer.writerow([cls, float(Yp[ci, 0]), float(Yp[ci, 1])])
        print(f"[OK] Wrote prototype t-SNE CSV -> {proto_csv_path}")

    plt.figure(figsize=(7, 5), dpi=400)
    markers = ["o", "s", "D", "^", "v", "P", "*"]
    if top_score is not None:
        ts_norm = (top_score - np.min(top_score)) / (np.ptp(top_score) + 1e-8)
        sizes = 10 + 30 * ts_norm
    else:
        sizes = 18

    # samples
    for i, lab in enumerate(legend_labels):
        m = (labels == lab)
        if np.sum(m) == 0:
            continue
        plt.scatter(
            Y[m, 0], Y[m, 1],
            s=(sizes[m] if isinstance(sizes, np.ndarray) else sizes),
            alpha=0.7,
            marker=markers[i % len(markers)],
            color=color_map[lab],
            label=lab
        )

    # prototypes (aynı renk)
    if Yp is not None:
        for ci, cls in enumerate(classes):
            proto_color = color_map[cls] if color_by == "emotion" else "k"
            plt.scatter(
                Yp[ci, 0], Yp[ci, 1],
                marker="*", s=220,
                edgecolor="k", linewidths=0.8,
                color=proto_color,
                label=(f"{cls} (proto)" if color_by == "emotion" else "prototype")
            )

    # Build a clean legend that **keeps** prototype entries (e.g., "angry (proto)")
    # and removes only exact duplicate strings.
    handles, labels_legend = plt.gca().get_legend_handles_labels()
    uniq_handles, uniq_labels, seen = [], [], set()
    for h, l in zip(handles, labels_legend):
        if l not in seen:
            uniq_handles.append(h)
            uniq_labels.append(l)
            seen.add(l)
    plt.legend(uniq_handles, uniq_labels, loc="best", fontsize=8, ncol=2)

    plt.tight_layout()
    out_dir = os.path.dirname(out_png)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_png)
    plt.close()
    if auto_select:
        msg = f"[OK] Saved t-SNE plot to {out_png} (perplexity={chosen_perp}, metric={metric}, init={init}, selected_by={select_by})"
    else:
        msg = f"[OK] Saved t-SNE plot to {out_png} (perplexity={chosen_perp}, metric={metric}, init={init})"
    if dump_csv:
        msg += f"\n[OK] Per-sample CSV saved to {dump_csv}"
    if (Yp is not None):
        if protos_out:
            msg += f"\n[OK] Prototype CSV saved to {protos_out}"
        elif dump_csv:
            msg += f"\n[OK] Prototype CSV saved to {dump_csv.replace('.csv', '_protos.csv')}"
    print(msg)

def main():
    ap = argparse.ArgumentParser(description="t-SNE visualization for fused embeddings NPZ")
    ap.add_argument("--emb_npz", type=str, required=True, help="Path to embeddings npz (from export_embeddings)")
    ap.add_argument("--tsne_out", type=str, default="img/tsne_fused.png", help="Output PNG path")
    ap.add_argument("--tsne_perplexity", type=float, default=15.0, help="Perplexity (typ. 5–50)")
    ap.add_argument("--tsne_n_iter", type=int, default=5000, help="Max iterations")
    ap.add_argument("--no_proto_in_tsne", action="store_true", help="Exclude prototypes from t-SNE fit/plot")
    ap.add_argument("--tsne_metric", type=str, default="cosine", choices=["cosine","euclidean"], help="Distance metric")
    ap.add_argument("--tsne_init", type=str, default="pca", choices=["pca","random"], help="Initialization")
    ap.add_argument("--tsne_auto_perp", action="store_true", help="Grid-search perplexity and pick best by metrics")
    ap.add_argument("--tsne_select_by", type=str, default="f1",
                    choices=["f1","silhouette","trust"],
                    help="Selection criterion for auto-perplexity: maximize f1 (default), silhouette, or trust")
    ap.add_argument("--tsne_perp_grid", type=str, default="10,15,20,25,30,35,40", help="Comma-separated perplexities when auto")
    ap.add_argument("--tsne_eval_out", type=str, default=None, help="CSV to write selection metrics")
    ap.add_argument("--tsne_color_by", type=str, default="emotion", choices=["emotion","tts","pred","correct"],
                    help="Color points by this metadata; prototypes use class color when color_by='emotion'")
    ap.add_argument("--tsne_dump_csv", type=str, default=None, help="Write per-sample t-SNE coordinates and diagnostics to CSV")
    ap.add_argument("--tsne_nn_k", type=int, default=10, help="k for local neighbor metrics in dump CSV")
    ap.add_argument("--tsne_nn_space", type=str, default="tsne", choices=["tsne","emb"], help="Space for neighbor metrics: t-SNE (2D) or original embedding (cosine)")
    ap.add_argument("--tsne_protos_out", type=str, default=None, help="Write prototype coordinates CSV (class,tsne_x,tsne_y)")
    ap.add_argument("--proto_from_centroid", action="store_true", help="If set, compute per-class centroids as prototypes when model prototypes are unavailable or excluded")
    args = ap.parse_args()

    perp_grid = [float(p.strip()) for p in args.tsne_perp_grid.split(",") if p.strip()]
    plot_tsne_from_npz(
        args.emb_npz,
        out_png=args.tsne_out,
        perplexity=args.tsne_perplexity,
        n_iter=args.tsne_n_iter,
        include_prototypes=(not args.no_proto_in_tsne),
        metric=args.tsne_metric,
        init=args.tsne_init,
        auto_select=args.tsne_auto_perp,
        select_by=args.tsne_select_by,
        perp_grid=perp_grid,
        eval_out=args.tsne_eval_out,
        color_by=args.tsne_color_by,
        dump_csv=args.tsne_dump_csv,
        protos_out=args.tsne_protos_out,
        proto_from_centroid=args.proto_from_centroid,
        nn_k=args.tsne_nn_k,
        nn_space=args.tsne_nn_space,
    )

if __name__ == "__main__":
    main()