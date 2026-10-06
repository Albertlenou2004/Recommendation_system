"""
preprocessing.py - STEP 2.3 Data cleaning, encoding and splitting.

Pipeline:  load raw CSV -> clean (missing / invalid / duplicate) -> k-core filter
           -> contiguous integer ids (needed by nn.Embedding) -> per-user train/val/test split
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (DATA_DIR, MIN_RATINGS_PER_USER, RATING_MAX, RATING_MIN, SEED,
                    TEST_FRACTION, VAL_FRACTION)


@dataclass
class Dataset:
    users: pd.DataFrame
    items: pd.DataFrame
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    ratings_all: pd.DataFrame      # every clean rating with a 'split' column
    report: dict                   # data-quality report (printed + saved)

    @property
    def n_users(self):
        return len(self.users)

    @property
    def n_items(self):
        return len(self.items)


def load_raw(data_dir=DATA_DIR):
    return (pd.read_csv(data_dir / "users.csv"),
            pd.read_csv(data_dir / "items.csv"),
            pd.read_csv(data_dir / "ratings.csv"))


def clean_data(users, items, ratings):
    """Handle missing values, invalid ratings, duplicates and sparse users. Returns a report."""
    rep = {}
    users, items, ratings = users.copy(), items.copy(), ratings.copy()

    # ---- users: impute numeric with median, categorical with 'unknown'
    rep["users_missing_age"] = int(users["age"].isna().sum())
    users["age"] = users["age"].fillna(users["age"].median()).clip(13, 90)
    for col in ["gender", "occupation"]:
        users[col] = users[col].fillna("unknown").astype(str).str.strip().str.lower()

    # ---- items: missing text -> empty string (title+genres still give a text signal)
    rep["items_missing_description"] = int(items["description"].isna().sum())
    rep["items_missing_year"] = int(items["year"].isna().sum())
    items["description"] = items["description"].fillna("").astype(str)
    items["year"] = items["year"].fillna(items["year"].median())
    items["genres"] = items["genres"].fillna("unknown").astype(str)
    items["title"] = items["title"].astype(str).str.strip()

    # ---- ratings
    rep["ratings_raw"] = len(ratings)
    ratings = ratings.dropna(subset=["user_id", "item_id", "rating"])
    rep["ratings_dropped_nan"] = rep["ratings_raw"] - len(ratings)
    n = len(ratings)
    ratings = ratings[ratings["rating"].between(RATING_MIN, RATING_MAX)]
    rep["ratings_dropped_out_of_range"] = n - len(ratings)
    n = len(ratings)
    ratings = ratings.drop_duplicates(["user_id", "item_id"], keep="last")
    rep["ratings_dropped_duplicates"] = n - len(ratings)
    ratings = ratings[ratings["user_id"].isin(users["user_id"]) & ratings["item_id"].isin(items["item_id"])]

    # ---- k-core: remove users with too little history to learn from
    counts = ratings.groupby("user_id").size()
    keep = counts[counts >= MIN_RATINGS_PER_USER].index
    rep["users_dropped_sparse"] = int(len(users) - users["user_id"].isin(keep).sum())
    ratings = ratings[ratings["user_id"].isin(keep)]
    users = users[users["user_id"].isin(keep)]
    ratings["rating"] = ratings["rating"].astype(int)
    rep["ratings_clean"] = len(ratings)
    return users, items, ratings, rep


def encode_ids(users, items, ratings):
    """Map raw ids to contiguous indices 0..N-1 (rows of the embedding tables)."""
    users = users.sort_values("user_id").reset_index(drop=True)
    items = items.sort_values("item_id").reset_index(drop=True)
    users["user_idx"] = users.index
    items["item_idx"] = items.index
    ratings = ratings.copy()
    ratings["user_idx"] = ratings["user_id"].map(dict(zip(users["user_id"], users["user_idx"]))).astype(int)
    ratings["item_idx"] = ratings["item_id"].map(dict(zip(items["item_id"], items["item_idx"]))).astype(int)
    return users, items, ratings[["user_idx", "item_idx", "rating"]].reset_index(drop=True)


def split_ratings(ratings, seed=SEED):
    """Per-user hold-out: every user keeps ~80% in train and ~20% in test, so the test set
    measures how well we predict the *unseen* items of users we know."""
    shuffled = ratings.sample(frac=1, random_state=seed).reset_index(drop=True)
    shuffled["rank"] = shuffled.groupby("user_idx").cumcount()
    shuffled["n"] = shuffled.groupby("user_idx")["rank"].transform("size")
    n_test = np.maximum(1, np.round(shuffled["n"] * TEST_FRACTION)).astype(int)
    is_test = shuffled["rank"] >= (shuffled["n"] - n_test)
    cols = ["user_idx", "item_idx", "rating"]
    test = shuffled.loc[is_test, cols].reset_index(drop=True)
    trainval = shuffled.loc[~is_test, cols]
    train, val = train_test_split(trainval, test_size=VAL_FRACTION, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test


def build_dataset(data_dir=DATA_DIR, verbose=True):
    users, items, ratings = load_raw(data_dir)
    users, items, ratings, report = clean_data(users, items, ratings)
    users, items, ratings = encode_ids(users, items, ratings)
    train, val, test = split_ratings(ratings)
    all_df = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")],
                       ignore_index=True)
    report.update(n_users=len(users), n_items=len(items), n_train=len(train), n_val=len(val), n_test=len(test),
                  density_pct=round(100 * len(ratings) / (len(users) * len(items)), 2))
    if verbose:
        print("[preprocess] data-quality report:")
        for k, v in report.items():
            print(f"    {k:32s} {v}")
    return Dataset(users, items, train, val, test, all_df, report)


if __name__ == "__main__":
    build_dataset()
