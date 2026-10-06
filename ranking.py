"""
ranking.py - Scoring / re-ranking layer (pure NumPy, no deep-learning dependency)

final_score = (predicted_rating - 1)/4  +  pop_weight * popularity_prior
then (optional) MMR re-ranking: trades a little relevance for a more DIVERSE top-K list
(mitigates the 'filter bubble' risk discussed in the Responsible-AI section).
"""
import numpy as np


def blend_scores(pred, pop_norm, pop_weight=0.0):
    """pred: (...,I) predicted ratings in [1,5]; pop_norm: (I,) in [0,1]."""
    return (np.clip(pred, 1, 5) - 1) / 4.0 + pop_weight * pop_norm


def mmr_rerank(scores, content_norm, k=10, diversity=0.3, pool=50):
    """Maximal Marginal Relevance.  scores: 1-D array with -inf for excluded items.
    diversity=0 -> plain top-k; diversity=1 -> maximum novelty relative to items already selected."""
    cand = np.argsort(-scores)[:pool]
    cand = cand[np.isfinite(scores[cand])]
    if len(cand) == 0:
        return []
    s = scores[cand]
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)               # relevance scaled to [0, 1]
    lam = 1.0 - diversity
    sims = content_norm[cand] @ content_norm[cand].T             # pairwise cosine similarity
    selected = [0]                                               # best-scored candidate first
    while len(selected) < min(k, len(cand)):
        rest = [j for j in range(len(cand)) if j not in selected]
        max_sim = sims[np.ix_(rest, selected)].max(axis=1)       # similarity to what is already chosen
        mmr = lam * s[rest] - (1.0 - lam) * max_sim
        selected.append(rest[int(np.argmax(mmr))])
    return cand[selected].tolist()


def top_k(scores, k=10):
    idx = np.argsort(-scores)[:k]
    return idx[np.isfinite(scores[idx])].tolist()
