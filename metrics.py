"""
metrics.py - STEP 4.1/4.2 Evaluation metrics for recommender systems

Rating prediction : RMSE, MAE, R^2
Classification    : accuracy, precision, recall, F1, confusion matrix  (liked vs not liked)
Ranking (Top-K)   : Precision@K, Recall@K, HitRate@K, NDCG@K
Beyond accuracy   : catalogue coverage, average popularity, intra-list diversity
"""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score)

from config import KS, LIKE_THRESHOLD, PRED_THRESHOLD, TOP_K
from ranking import mmr_rerank


def rating_metrics(y_true, y_pred):
    return {"RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred))}


def classification_metrics(y_true, y_pred, true_thr=LIKE_THRESHOLD, pred_thr=PRED_THRESHOLD):
    yt = (np.asarray(y_true) >= true_thr).astype(int)
    yp = (np.asarray(y_pred) >= pred_thr).astype(int)
    return {"accuracy": float(accuracy_score(yt, yp)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "confusion_matrix": confusion_matrix(yt, yp, labels=[0, 1]).tolist()}


def mask_from(df, n_users, n_items):
    """Boolean (U, I) matrix, True where the (user,item) pair occurs in df."""
    m = np.zeros((n_users, n_items), dtype=bool)
    m[df["user_idx"].values, df["item_idx"].values] = True
    return m


def ranking_metrics(scores, exclude_mask, relevant_mask, content_norm, pop_norm,
                    ks=KS, mmr_diversity=0.0, pool=50):
    """
    scores        (U, I) ranking scores (higher = better)
    exclude_mask  (U, I) items the user already interacted with in train/val -> never recommended
    relevant_mask (U, I) held-out items the user liked  (ground truth)
    """
    s = np.where(exclude_mask, -np.inf, scores)
    users = np.where(relevant_mask.sum(1) > 0)[0]                  # evaluate users that have >=1 relevant item
    kmax = max(ks)
    if mmr_diversity > 0:
        top = np.array([mmr_rerank(s[u], content_norm, kmax, mmr_diversity, pool) for u in users])
    else:
        top = np.argsort(-s[users], axis=1)[:, :kmax]
    hits = relevant_mask[users[:, None], top]                      # (n_users, kmax) booleans
    n_rel = relevant_mask[users].sum(1)
    disc = 1.0 / np.log2(np.arange(2, kmax + 2))                   # DCG discounts 1/log2(rank+1)

    out = {}
    for k in ks:
        h = hits[:, :k]
        out[f"precision@{k}"] = float((h.sum(1) / k).mean())
        out[f"recall@{k}"] = float((h.sum(1) / n_rel).mean())
        out[f"hit_rate@{k}"] = float(h.any(1).mean())
        dcg = (h * disc[:k]).sum(1)
        ideal = np.cumsum(disc)[np.minimum(n_rel, k) - 1]
        out[f"ndcg@{k}"] = float((dcg / ideal).mean())

    top10 = top[:, :TOP_K]
    out["coverage@10"] = float(len(np.unique(top10)) / scores.shape[1])
    out["avg_popularity@10"] = float(pop_norm[top10].mean())
    sims = np.einsum("nkd,nld->nkl", content_norm[top10], content_norm[top10])
    off_diag = (sims.sum((1, 2)) - np.trace(sims, axis1=1, axis2=2)) / (TOP_K * (TOP_K - 1))
    out["diversity@10"] = float((1.0 - off_diag).mean())           # 1 - mean pairwise cosine
    out["users_evaluated"] = int(len(users))
    return out
