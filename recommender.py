"""
recommender.py - STEP 1.2 Output pipeline: Top-K recommendations + scores + rationales.

Loads the saved artefacts (model.pt, features.npz, clean CSVs).  If PyTorch is not installed the
pre-computed score matrix (scores.npy) is used, so the demo still runs.
"""
import json

import numpy as np
import pandas as pd

from config import ARTIFACT_DIR, GENRES
from ranking import blend_scores, mmr_rerank, top_k


class Recommender:
    def __init__(self, artifact_dir=ARTIFACT_DIR):
        d = artifact_dir
        self.meta = json.loads((d / "meta.json").read_text())
        self.items = pd.read_csv(d / "items_clean.csv")
        self.users = pd.read_csv(d / "users_clean.csv")
        self.ratings = pd.read_csv(d / "ratings_clean.csv")
        arr = np.load(d / "features.npz")
        self.user_features, self.item_features = arr["user_features"], arr["item_features"]
        self.content_norm, self.pop_norm = arr["content_norm"], arr["pop_norm"]
        self.pop_weight = float(self.meta["best_pop_weight"])
        self.item_genres = [s.split("|") for s in self.items["genres"]]
        self.seen = self.ratings.groupby("user_idx")["item_idx"].apply(set).to_dict()
        self.model, self.scores = None, None
        self._load_model(d)

    # ------------------------------------------------------------ model loading
    def _load_model(self, d):
        try:
            import torch
            from model import HybridNCF
            hp = self.meta["hyperparameters"]
            self.model = HybridNCF(self.user_features, self.item_features, hp["emb_dim"],
                                   tuple(hp["hidden"]), hp["dropout"], self.meta["global_mean"])
            self.model.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
            self.model.eval()
            self._torch = torch
        except Exception as exc:                                    # torch missing / model.pt missing
            print(f"[recommender] using pre-computed scores ({exc.__class__.__name__})")
            self.scores = np.load(d / "scores.npy")

    def predict_user(self, user_idx):
        """Predicted rating (1..5) of every item for one existing user."""
        if self.model is None:
            return self.scores[user_idx]
        torch = self._torch
        with torch.no_grad():
            n = len(self.items)
            u = torch.full((n,), int(user_idx), dtype=torch.long)
            i = torch.arange(n)
            return self.model(u, i).clamp(1, 5).numpy()

    # ------------------------------------------------------------ helpers
    def user_idx_from_id(self, user_id):
        return int(self.users.loc[self.users["user_id"] == user_id, "user_idx"].iloc[0])

    def user_genre_profile(self, user_idx):
        prof = np.clip(self.user_features[user_idx, :len(GENRES)], 0, None)
        return pd.Series(prof / max(prof.sum(), 1e-9), index=GENRES, name="taste share")

    def user_history(self, user_idx, n=8):
        h = self.ratings[self.ratings["user_idx"] == user_idx].sort_values("rating", ascending=False).head(n)
        h = h.merge(self.items[["item_idx", "title", "genres"]], on="item_idx")
        return h[["title", "genres", "rating"]]

    def explain(self, user_idx, item_idx):
        """Rule-based, faithful rationale (no hallucination: only uses real data)."""
        clauses = []
        prof = self.user_features[user_idx, :len(GENRES)]
        top_genres = [GENRES[j] for j in np.argsort(-prof)[:3] if prof[j] > 0]
        matched = [g for g in self.item_genres[item_idx] if g in top_genres]
        if matched:
            clauses.append(f"matches your favourite genres ({', '.join(matched)})")
        liked = self.ratings[(self.ratings["user_idx"] == user_idx) & (self.ratings["rating"] >= 4)]
        if len(liked):
            sims = self.content_norm[liked["item_idx"].values] @ self.content_norm[item_idx]
            j = int(np.argmax(sims))
            if sims[j] > 0.5:
                title = self.items.loc[liked["item_idx"].values[j], "title"]
                clauses.append(f"is similar to '{title}', which you rated {int(liked['rating'].values[j])}/5")
        if not clauses:
            clauses.append("is predicted by the neural model from users with similar rating patterns")
        return "Recommended because it " + " and ".join(clauses) + "."

    def _format(self, idxs, pred, scores, explain_fn):
        rows = []
        for rank, i in enumerate(idxs, 1):
            it = self.items.iloc[i]
            rows.append({"rank": rank, "item_idx": int(i), "title": it["title"], "genres": it["genres"],
                         "year": int(it["year"]), "description": it["description"],
                         "predicted_rating": float(pred[i]) if pred is not None else np.nan,
                         "score": float(scores[i]), "explanation": explain_fn(i)})
        return pd.DataFrame(rows)

    def _genre_mask(self, genres):
        if not genres:
            return np.ones(len(self.items), dtype=bool)
        return np.array([any(g in gs for g in genres) for gs in self.item_genres])

    # ------------------------------------------------------------ public API
    def recommend(self, user_idx, k=10, diversity=0.0, pop_weight=None, genres=None):
        """Top-K for an EXISTING user. diversity in [0,1] activates MMR re-ranking."""
        pw = self.pop_weight if pop_weight is None else pop_weight
        pred = self.predict_user(user_idx)
        scores = blend_scores(pred, self.pop_norm, pw)
        mask = self._genre_mask(genres)
        mask[list(self.seen.get(user_idx, []))] = False           # never recommend what was already rated
        scores = np.where(mask, scores, -np.inf)
        idxs = mmr_rerank(scores, self.content_norm, k, diversity) if diversity > 0 else top_k(scores, k)
        return self._format(idxs, pred, scores, lambda i: self.explain(user_idx, i))

    def recommend_new_user(self, liked_genres, liked_item_idx=None, k=10, diversity=0.0):
        """COLD-START: no history -> content-based profile from chosen genres / liked items."""
        g = np.array([float(x in liked_genres) for x in GENRES])
        g = g / max(np.linalg.norm(g), 1e-9)
        prof = np.concatenate([g, np.zeros(self.content_norm.shape[1] - len(GENRES))])
        if liked_item_idx:
            prof = 0.5 * prof + 0.5 * self.content_norm[list(liked_item_idx)].mean(axis=0)
        prof = prof / max(np.linalg.norm(prof), 1e-9)
        scores = self.content_norm @ prof + 0.15 * self.pop_norm
        scores[list(liked_item_idx or [])] = -np.inf
        idxs = mmr_rerank(scores, self.content_norm, k, diversity) if diversity > 0 else top_k(scores, k)

        def why(i):
            m = [x for x in self.item_genres[i] if x in liked_genres]
            return ("Recommended because it matches your selected genres (" + ", ".join(m) + ")."
                    if m else "Recommended because its description is similar to the titles you picked.")
        return self._format(idxs, None, scores, why)
