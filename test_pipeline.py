"""
test_pipeline.py - automated tests (run: python -m pytest -q  or  python test_pipeline.py)
Covers data cleaning, split integrity (no leakage), metrics maths, MMR and (if torch exists) the model.
"""
import tempfile
from pathlib import Path

import numpy as np

from config import CONTENT_DIM
from data_generation import generate_all
from features import build_features
from metrics import ranking_metrics
from preprocessing import build_dataset
from ranking import mmr_rerank

_DIR = Path(tempfile.mkdtemp())
generate_all(_DIR)
DS = build_dataset(_DIR, verbose=False)
FB = build_features(DS)


def test_cleaning_removes_problems():
    r = DS.ratings_all
    assert r["rating"].between(1, 5).all()
    assert not r.duplicated(["user_idx", "item_idx"]).any()
    assert DS.report["ratings_dropped_duplicates"] > 0 and DS.report["ratings_dropped_out_of_range"] > 0
    assert not DS.users["age"].isna().any() and not DS.items["description"].isna().any()


def test_split_has_no_leakage():
    key = lambda d: set(zip(d["user_idx"], d["item_idx"]))
    assert not key(DS.train) & key(DS.test)
    assert not key(DS.train) & key(DS.val)
    assert not key(DS.val) & key(DS.test)
    assert set(DS.test["user_idx"]) <= set(DS.train["user_idx"]) | set(DS.val["user_idx"])


def test_feature_shapes_and_normalisation():
    assert FB.user_features.shape[0] == DS.n_users and FB.item_features.shape[0] == DS.n_items
    assert FB.content_norm.shape[1] == CONTENT_DIM
    assert np.allclose(np.linalg.norm(FB.content_norm, axis=1), 1, atol=1e-4)
    assert np.isfinite(FB.user_features).all() and np.isfinite(FB.item_features).all()


def test_ranking_metrics_toy_example():
    # 1 user, 12 items; item 2 is the only relevant item and is ranked FIRST
    scores = np.linspace(0, 1, 12)[None, :].copy(); scores[0, 2] = 5.0
    rel = np.zeros((1, 12), bool); rel[0, 2] = True
    m = ranking_metrics(scores, np.zeros((1, 12), bool), rel, np.eye(12, dtype=np.float32),
                        np.zeros(12), ks=(5, 10))
    assert m["precision@5"] == 0.2 and m["recall@5"] == 1.0 and m["ndcg@5"] == 1.0
    # same item ranked SECOND -> NDCG = 1/log2(3)
    scores[0, 11] = 9.0
    m2 = ranking_metrics(scores, np.zeros((1, 12), bool), rel, np.eye(12, dtype=np.float32),
                         np.zeros(12), ks=(5, 10))
    assert abs(m2["ndcg@5"] - 1 / np.log2(3)) < 1e-9


def test_ranking_metrics_on_real_data_perfect_oracle():
    from metrics import mask_from
    U, I = DS.n_users, DS.n_items
    seen = mask_from(DS.train, U, I)
    rel = mask_from(DS.val[DS.val["rating"] >= 4], U, I)
    oracle = rel.astype(float)                       # scores = ground truth -> NDCG must be 1
    m = ranking_metrics(oracle, seen, rel, FB.content_norm, FB.pop_norm)
    assert abs(m["ndcg@10"] - 1.0) < 1e-9 and m["hit_rate@10"] == 1.0


def test_mmr_increases_diversity_and_excludes_masked():
    s = np.random.default_rng(0).random(DS.n_items)
    s[:10] = -np.inf
    plain = mmr_rerank(s, FB.content_norm, 10, 0.0)
    diverse = mmr_rerank(s, FB.content_norm, 10, 0.8)
    assert len(set(plain)) == 10 and len(set(diverse)) == 10
    assert not set(range(10)) & set(diverse)
    sim = lambda ids: (FB.content_norm[ids] @ FB.content_norm[ids].T).mean()
    assert sim(diverse) <= sim(plain)


def test_model_uses_lstm_and_forward_shape_if_torch_available():
    try:
        import torch
    except ImportError:
        return
    from model import HybridNCF
    m = HybridNCF(FB.user_features, FB.item_features, emb_dim=8, hidden=(16, 8))
    assert hasattr(m, "lstm") and isinstance(m.lstm, torch.nn.LSTM)
    out = m(torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]))
    assert out.shape == (3,) and torch.isfinite(out).all()


def test_ui_html_is_escaped_and_styled():
    import ui
    h = ui.recommendation_card(1, "<Evil>", 2020, "Sci-Fi|Drama", "<script>", "why", predicted_rating=4.2, top=True)
    assert "<Evil>" not in h and "<script>" not in h and "g-Sci-Fi" in h and "rec-card top" in h
    css = ui.load_css()
    assert css.count("{") == css.count("}") and ".rec-card" in css


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name)
