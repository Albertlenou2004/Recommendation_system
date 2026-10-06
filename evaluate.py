"""
evaluate.py - STEP 4 Evaluation, baselines, interpretation and figures (no PyTorch needed:
it receives the predicted-rating matrix produced by the trained model).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (ARTIFACT_DIR, CONTENT_DIM, LIKE_THRESHOLD, MMR_DIVERSITY_EVAL, POP_WEIGHT_GRID,
                    SEED, TOP_K)
from metrics import classification_metrics, mask_from, ranking_metrics, rating_metrics
from ranking import blend_scores


# ------------------------------------------------------------------ baselines
def svd_baseline(ds, user_mean, k=20):
    """Classical matrix-factorisation baseline (truncated SVD of the mean-centred rating matrix)."""
    tr = ds.train
    R = np.zeros((ds.n_users, ds.n_items))
    R[tr["user_idx"], tr["item_idx"]] = tr["rating"].values - user_mean[tr["user_idx"].values]
    Uu, S, Vt = np.linalg.svd(R, full_matrices=False)
    return np.clip(user_mean[:, None] + (Uu[:, :k] * S[:k]) @ Vt[:k], 1, 5)


def rating_baselines(ds):
    U, I, tr = ds.n_users, ds.n_items, ds.train
    mu = tr["rating"].mean()
    user_mean = tr.groupby("user_idx")["rating"].mean().reindex(range(U)).fillna(mu).values
    item_mean = tr.groupby("item_idx")["rating"].mean().reindex(range(I)).fillna(mu).values
    return {"Global mean": np.full((U, I), mu),
            "User+Item bias": np.clip(mu + (user_mean[:, None] - mu) + (item_mean[None, :] - mu), 1, 5),
            "SVD matrix factorisation": svd_baseline(ds, user_mean)}


# ------------------------------------------------------------------ tuning of the ranking layer
def tune_pop_weight(ds, fb, pred):
    """Select the popularity weight on the VALIDATION set (NDCG@10); test set stays untouched."""
    U, I = ds.n_users, ds.n_items
    seen_train = mask_from(ds.train, U, I)
    rel_val = mask_from(ds.val[ds.val["rating"] >= LIKE_THRESHOLD], U, I)
    results = {}
    for w in POP_WEIGHT_GRID:
        m = ranking_metrics(blend_scores(pred, fb.pop_norm, w), seen_train, rel_val, fb.content_norm, fb.pop_norm)
        results[w] = m["ndcg@10"]
        print(f"[tune] pop_weight={w:<4} validation NDCG@10={m['ndcg@10']:.4f}")
    best = max(results, key=results.get)
    return best, results


# ------------------------------------------------------------------ main evaluation
def evaluate_all(ds, fb, ncf_pred, pop_weight, history, out_dir=ARTIFACT_DIR, model_name="Hybrid NCF"):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    U, I = ds.n_users, ds.n_items
    test = ds.test
    y = test["rating"].values.astype(float)
    idx = (test["user_idx"].values, test["item_idx"].values)
    report = {}

    # ---- A. rating prediction + classification on the held-out TEST set
    preds = {**rating_baselines(ds), model_name: ncf_pred}
    report["rating"] = {n: rating_metrics(y, m[idx]) for n, m in preds.items()}
    report["classification"] = {n: classification_metrics(y, m[idx]) for n, m in preds.items()}

    # ---- B. top-K ranking on the TEST set (candidates = items not seen in train/val)
    seen = mask_from(ds.train, U, I) | mask_from(ds.val, U, I)
    rel = mask_from(test[test["rating"] >= LIKE_THRESHOLD], U, I)
    cn, pop = fb.content_norm, fb.pop_norm
    profile = fb.user_features[:, :CONTENT_DIM]
    profile_n = profile / np.maximum(np.linalg.norm(profile, axis=1, keepdims=True), 1e-9)
    rng = np.random.default_rng(SEED)
    blended = blend_scores(ncf_pred, pop, pop_weight)
    candidates = {
        "Random": rng.random((U, I)),
        "Popularity": np.tile(pop, (U, 1)),
        "Content-based (cosine)": profile_n @ cn.T,
        "SVD matrix factorisation": blend_scores(preds["SVD matrix factorisation"], pop, 0.0),
        model_name: blend_scores(ncf_pred, pop, 0.0),
        f"{model_name} + popularity prior": blended,
    }
    ranking = {n: ranking_metrics(s, seen, rel, cn, pop) for n, s in candidates.items()}
    ranking[f"{model_name} + pop + MMR"] = ranking_metrics(blended, seen, rel, cn, pop, mmr_diversity=MMR_DIVERSITY_EVAL)
    report["ranking"] = ranking
    report["settings"] = {"pop_weight": pop_weight, "mmr_diversity": MMR_DIVERSITY_EVAL,
                          "like_threshold": LIKE_THRESHOLD, "top_k": TOP_K}
    report["history"] = history

    # ---- save artefacts
    (out_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2))
    comp = pd.DataFrame(ranking).T
    comp.to_csv(out_dir / "model_comparison.csv")
    _plot_history(history, fig_dir / "training_curve.png")
    _plot_confusion(report["classification"][model_name]["confusion_matrix"], fig_dir / "confusion_matrix.png")
    _plot_comparison(comp, fig_dir / "ranking_comparison.png")
    _plot_eda(ds, fig_dir / "eda.png")
    return report


# ------------------------------------------------------------------ plots
def _plot_history(history, path):
    if not history:
        return
    h = pd.DataFrame(history)
    plt.figure(figsize=(6, 4))
    plt.plot(h["epoch"], h["train_rmse"], label="train RMSE")
    plt.plot(h["epoch"], h["val_rmse"], label="validation RMSE")
    plt.xlabel("epoch"); plt.ylabel("RMSE"); plt.title("Training curve"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def _plot_confusion(cm, path):
    cm = np.array(cm)
    plt.figure(figsize=(4.2, 4))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["not liked", "liked"]); plt.yticks([0, 1], ["not liked", "liked"])
    plt.xlabel("predicted"); plt.ylabel("actual"); plt.title("Confusion matrix (test set)")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def _plot_comparison(comp, path):
    ax = comp[["ndcg@10", "recall@10", "precision@10"]].plot(kind="bar", figsize=(9, 4.5))
    ax.set_title("Top-10 ranking quality on the test set"); ax.set_ylabel("score"); ax.grid(axis="y", alpha=.3)
    plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def _plot_eda(ds, path):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ds.ratings_all["rating"].value_counts().sort_index().plot(kind="bar", ax=ax[0], color="#4C78A8")
    ax[0].set_title("Rating distribution"); ax[0].set_xlabel("rating")
    ds.ratings_all.groupby("user_idx").size().plot(kind="hist", bins=30, ax=ax[1], color="#59A14F")
    ax[1].set_title("Ratings per user"); ax[1].set_xlabel("# ratings")
    counts = ds.ratings_all.groupby("item_idx").size().sort_values(ascending=False).values
    ax[2].plot(counts); ax[2].set_title("Item popularity (long tail)"); ax[2].set_xlabel("item rank")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


# ------------------------------------------------------------------ results write-up
def _md_table(df, floatfmt="{:.4f}"):
    cols = ["Model"] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for name in df.index:
        cells = []
        for c in df.columns:
            v = df.loc[name, c]                                  # per-cell access keeps int columns as ints
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join([str(name)] + cells) + " |")
    return "\n".join(lines)


def write_results_md(report, ds, best_hp, tuning_df, pop_results, out_path, model_name="Hybrid NCF"):
    """Auto-generates RESULTS.md with the REAL numbers of this run (tables + interpretation)."""
    r, rk, cl = report["rating"], report["ranking"], report["classification"]
    ours = r[model_name]
    best_base = min((k for k in r if k != model_name), key=lambda k: r[k]["RMSE"])
    gain = 100 * (r[best_base]["RMSE"] - ours["RMSE"]) / r[best_base]["RMSE"]
    pop_k, ncf_k, blend_k = "Popularity", model_name, f"{model_name} + popularity prior"
    mmr_k = f"{model_name} + pop + MMR"

    txt = ["# Evaluation Results (auto-generated by run_pipeline.py)\n",
           "## 1. Data-quality report\n", "```", *[f"{k}: {v}" for k, v in ds.report.items()], "```\n",
           "## 2. Selected hyper-parameters (validation RMSE grid search)\n",
           f"Best configuration: `{ {k: best_hp[k] for k in ['emb_dim','lr','dropout','batch_size','weight_decay']} }`\n",
           _md_table(tuning_df.set_index(tuning_df.index.map(lambda i: f"config {i+1}")), "{:.4f}"), "",
           f"Popularity-weight search (validation NDCG@10): { {k: round(v, 4) for k, v in pop_results.items()} } "
           f"-> selected **{report['settings']['pop_weight']}**\n",
           "## 3. Rating prediction (test set)\n", _md_table(pd.DataFrame(r).T), "",
           "## 4. Classification view: liked (>=4) vs not liked (predicted >= 3.5)\n",
           _md_table(pd.DataFrame(cl).T.drop(columns=["confusion_matrix"])), "",
           "Confusion matrix of the hybrid model: `" + str(cl[model_name]["confusion_matrix"]) + "` "
           "(rows = actual [not liked, liked], columns = predicted).\n",
           "## 5. Top-K ranking (test set)\n",
           _md_table(pd.DataFrame(rk).T[["precision@10", "recall@10", "hit_rate@10", "ndcg@10",
                                          "coverage@10", "diversity@10", "avg_popularity@10"]]), "",
           "## 6. Interpretation\n"]
    txt.append(f"* **Rating accuracy** - the hybrid model reaches RMSE {ours['RMSE']:.3f} / MAE {ours['MAE']:.3f} / "
               f"R2 {ours['R2']:.3f}. Against the strongest baseline ({best_base}, RMSE {r[best_base]['RMSE']:.3f}) "
               f"this is a {'reduction' if gain > 0 else 'change'} of {abs(gain):.1f}% in RMSE"
               f"{'' if gain > 0 else ' (the baseline is competitive on this data)'}.")
    svd_n = rk["SVD matrix factorisation"]["ndcg@10"]
    ours_n = rk[blend_k]["ndcg@10"]
    txt.append(f"* **Ranking** - NDCG@10 is {rk[ncf_k]['ndcg@10']:.3f} for the hybrid model versus "
               f"{rk['Random']['ndcg@10']:.3f} for random, {rk[pop_k]['ndcg@10']:.3f} for popularity and "
               f"{rk['Content-based (cosine)']['ndcg@10']:.3f} for pure content-based filtering; the tuned popularity prior "
               f"gives {ours_n:.3f}. Compared with the classical SVD baseline ({svd_n:.3f}) the final system is "
               f"{'better' if ours_n > svd_n else 'not better'} ({100 * (ours_n - svd_n) / max(svd_n, 1e-9):+.1f}%).")
    maj = cl["Global mean"]["accuracy"]
    txt.append(f"* **Classification view** - accuracy of the hybrid model is {cl[model_name]['accuracy']:.3f} against "
               f"{maj:.3f} for the trivial 'everything is liked' predictor, because the classes are imbalanced; "
               f"therefore precision/recall/F1 and especially the ranking metrics are the more informative indicators.")
    txt.append(f"* **Diversity trade-off** - MMR re-ranking changes list diversity from {rk[blend_k]['diversity@10']:.3f} "
               f"to {rk[mmr_k]['diversity@10']:.3f} and catalogue coverage from {rk[blend_k]['coverage@10']:.3f} to "
               f"{rk[mmr_k]['coverage@10']:.3f}, while NDCG@10 moves from {rk[blend_k]['ndcg@10']:.3f} to "
               f"{rk[mmr_k]['ndcg@10']:.3f}: this illustrates the accuracy-versus-diversity trade-off that the 'Diversity' slider in the app exposes.")
    txt.append("* **Parameter adjustment** - the learning rate, embedding size and dropout were chosen by validation "
               "RMSE; early stopping and weight decay limited over-fitting (see `figures/training_curve.png`); the "
               "popularity weight was chosen by validation NDCG@10.")
    out_path.write_text("\n".join(txt) + "\n")
