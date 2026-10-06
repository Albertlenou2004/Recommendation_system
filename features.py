"""
features.py - STEP 3.1 Feature Engineering (all statistics computed from TRAIN data only -> no leakage)

Item vector  (42-d) = [ genre multi-hot (8) | LSA text embedding (32) | year (1) | log-popularity (1) ]
User vector  (n-d)  = [ taste profile (40) | age, mean rating, log #ratings (3) | gender/occupation one-hot ]

The 'taste profile' is the average content vector of the items the user LIKED (rating >= 4),
which places users in the same space as items -> enables cosine similarity and explanations.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize

from config import CONTENT_DIM, GENRES, LIKE_THRESHOLD, SEED, TEXT_DIM


@dataclass
class FeatureBundle:
    user_features: np.ndarray      # (U, du)
    item_features: np.ndarray      # (I, di)
    content_norm: np.ndarray       # (I, 40) L2-normalised content vectors (for cosine similarity)
    pop_norm: np.ndarray           # (I,) log-popularity scaled to [0, 1]
    tfidf: object
    svd: object


def genre_matrix(items):
    """Multi-hot genre encoding (categorical encoding for a multi-valued attribute)."""
    return np.array([[g in s.split("|") for g in GENRES] for s in items["genres"]], dtype=np.float32)


def build_item_text_embeddings(items):
    """TF-IDF over title + genres + description, compressed with truncated SVD (LSA).
    Lightweight, offline and deterministic; can be swapped for a sentence-transformer."""
    corpus = (items["title"] + " " + items["genres"].str.replace("|", " ", regex=False)
              + " " + items["description"]).tolist()
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words="english",
                            sublinear_tf=True, min_df=2)
    X = tfidf.fit_transform(corpus)
    svd = TruncatedSVD(n_components=TEXT_DIM, random_state=SEED)
    Z = normalize(svd.fit_transform(X))
    return Z.astype(np.float32), tfidf, svd


def build_features(ds):
    U, I = ds.n_users, ds.n_items
    train = ds.train

    # ------------------------------------------------ ITEM FEATURES
    G = genre_matrix(ds.items)
    Gn = normalize(G)
    Z, tfidf, svd = build_item_text_embeddings(ds.items)
    content = np.hstack([Gn, Z]).astype(np.float32)                    # (I, 40)
    assert content.shape[1] == CONTENT_DIM

    counts = np.bincount(train["item_idx"], minlength=I)
    pop_norm = (np.log1p(counts) / np.log1p(counts).max()).astype(np.float32)
    year = ds.items["year"].values.astype(np.float32)
    year_scaled = (year - year.mean()) / (year.std() + 1e-9)
    item_features = np.hstack([content, year_scaled[:, None], pop_norm[:, None]]).astype(np.float32)

    # ------------------------------------------------ USER FEATURES
    def mean_profile(df):
        W = sparse.csr_matrix((np.ones(len(df)), (df["user_idx"], df["item_idx"])), shape=(U, I))
        n = np.asarray(W.sum(1)).ravel()
        return (W @ content) / np.maximum(n, 1)[:, None], n

    liked_profile, n_liked = mean_profile(train[train["rating"] >= LIKE_THRESHOLD])
    all_profile, _ = mean_profile(train)
    profile = np.where(n_liked[:, None] > 0, liked_profile, all_profile)   # fallback if user liked nothing

    stats = train.groupby("user_idx")["rating"].agg(["mean", "count"]).reindex(range(U))
    stats["mean"] = stats["mean"].fillna(train["rating"].mean())
    stats["count"] = stats["count"].fillna(0)
    numeric = np.column_stack([ds.users["age"].values, stats["mean"].values, np.log1p(stats["count"].values)])
    numeric = StandardScaler().fit_transform(numeric)                  # zero mean / unit variance
    demo = pd.get_dummies(ds.users[["gender", "occupation"]]).astype(np.float32).values
    user_features = np.hstack([profile, numeric, demo]).astype(np.float32)

    content_norm = normalize(content).astype(np.float32)
    print(f"[features] user_features={user_features.shape} item_features={item_features.shape}")
    return FeatureBundle(user_features, item_features, content_norm, pop_norm, tfidf, svd)
