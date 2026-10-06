"""
data_generation.py - STEP 2.2 Data Acquisition
Programmatically builds a realistic MovieLens-style dataset:
    users.csv    (demographics)
    items.csv    (title, genres, year, text description)
    ratings.csv  (explicit 1-5 feedback; ratings >= 4 are treated as implicit 'likes')

Why synthetic?  The exam is performed offline in an ICT room; a generated dataset is fully
reproducible (fixed seed) and has KNOWN structure (user taste = genre preferences + latent
factors + biases + noise), so evaluation results are meaningful.
Deliberate data-quality problems (missing values, duplicates, invalid ratings) are injected so
that the cleaning pipeline in preprocessing.py has real work to do.
"""
import numpy as np
import pandas as pd

from config import (DATA_DIR, GENRES, N_ITEMS, N_USERS, OCCUPATIONS, SEED)

GENRE_WORDS = {
    "Action": ["explosive", "chase", "fight", "mercenary", "heist", "battle", "rescue", "revenge", "weapon", "soldier", "fast", "hero"],
    "Comedy": ["hilarious", "prank", "awkward", "wedding", "roommates", "mishap", "silly", "banter", "chaos", "laugh", "quirky", "party"],
    "Drama": ["emotional", "family", "loss", "courtroom", "struggle", "redemption", "choices", "sacrifice", "grief", "ambition", "betrayal", "truth"],
    "Sci-Fi": ["galaxy", "robot", "alien", "future", "spaceship", "quantum", "android", "colony", "time", "experiment", "cyber", "planet"],
    "Romance": ["love", "heartbreak", "kiss", "couple", "destiny", "letters", "passion", "reunion", "soulmate", "dance", "proposal", "summer"],
    "Horror": ["haunted", "ghost", "scream", "dark", "curse", "monster", "nightmare", "cabin", "demon", "blood", "fear", "ritual"],
    "Documentary": ["real", "interview", "history", "nature", "investigation", "archive", "wildlife", "culture", "science", "expedition", "facts", "society"],
    "Animation": ["animated", "adventure", "magical", "talking", "kingdom", "colorful", "friends", "dragon", "toy", "fairy", "musical", "playful"],
}
GENERIC_WORDS = ["story", "journey", "city", "secret", "friendship", "night", "world", "past",
                 "mystery", "small", "town", "unexpected", "young", "island", "road"]
ADJ = ["Silent", "Last", "Golden", "Broken", "Hidden", "Crimson", "Lost", "Eternal", "Midnight", "Rising",
       "Frozen", "Wild", "Distant", "Burning", "Electric", "Forgotten", "Hollow", "Bright", "Final", "Secret"]
NOUN = ["River", "Empire", "Garden", "Signal", "Horizon", "Kingdom", "Harbor", "Echo", "Mirror", "Voyage",
        "Fortress", "Orchard", "Frontier", "Legacy", "Shadow", "Promise", "Machine", "Summit", "Letter", "Storm"]


def _make_titles(rng, n):
    """Unique, readable titles such as 'The Silent River' / 'The Silent River 2'."""
    titles, seen = [], {}
    for _ in range(n):
        base = f"The {rng.choice(ADJ)} {rng.choice(NOUN)}"
        seen[base] = seen.get(base, 0) + 1
        titles.append(base if seen[base] == 1 else f"{base} {seen[base]}")
    return titles


def generate_items(rng, n_items=N_ITEMS):
    titles = _make_titles(rng, n_items)
    rows = []
    for idx in range(n_items):
        k = int(rng.choice([1, 2, 3], p=[0.45, 0.40, 0.15]))          # number of genres
        genres = list(rng.choice(GENRES, size=k, replace=False))
        pool = [w for g in genres for w in GENRE_WORDS[g]]
        n_words = int(rng.integers(10, 15))
        # 70% genre-specific vocabulary, 30% generic -> text carries real (but noisy) signal
        words = [rng.choice(pool) if rng.random() < 0.7 else rng.choice(GENERIC_WORDS) for _ in range(n_words)]
        rows.append({"item_id": idx + 1, "title": titles[idx], "genres": "|".join(genres),
                     "year": int(rng.integers(1980, 2026)), "description": " ".join(words)})
    return pd.DataFrame(rows)


def generate_users(rng, n_users=N_USERS):
    ages = np.clip(rng.normal(32, 11, n_users), 16, 70).astype(int)
    users = pd.DataFrame({
        "user_id": np.arange(1, n_users + 1),
        "age": ages,
        "gender": rng.choice(["F", "M", "Other"], n_users, p=[0.48, 0.48, 0.04]),
        "occupation": rng.choice(OCCUPATIONS, n_users),
    })
    # Hidden (ground-truth) genre preference of each user; demographics influence it weakly
    pref = np.zeros((n_users, len(GENRES)))
    gi = {g: i for i, g in enumerate(GENRES)}
    for u in range(n_users):
        alpha = np.full(len(GENRES), 0.35)
        if ages[u] > 50:
            alpha[[gi["Drama"], gi["Documentary"]]] += 0.6
        if ages[u] < 25:
            alpha[[gi["Animation"], gi["Action"], gi["Comedy"]]] += 0.6
        pref[u] = rng.dirichlet(alpha)
    return users, pref


def generate_ratings(rng, users, items, pref):
    U, I = len(users), len(items)
    G = np.array([[g in s.split("|") for g in GENRES] for s in items["genres"]], dtype=float)
    Gn = G / G.sum(1, keepdims=True)
    affinity = pref @ Gn.T                                             # (U, I) genre match
    z = (affinity - affinity.mean(1, keepdims=True)) / (affinity.std(1, keepdims=True) + 1e-9)

    k = 4                                                              # latent taste factors (what pure CF can learn)
    latent = (rng.normal(size=(U, k)) @ rng.normal(size=(I, k)).T) / np.sqrt(k)
    user_bias = rng.normal(0, 0.30, U)                                 # generous / harsh raters
    item_bias = rng.normal(0, 0.35, I)                                 # globally good / bad items
    cont = (3.4 + 0.75 * z + 0.4 * latent + user_bias[:, None] + item_bias[None, :]
            + rng.normal(0, 0.45, (U, I)))
    true_rating = np.clip(np.rint(cont), 1, 5).astype(int)

    popularity = rng.lognormal(0, 1.0, I)                              # long-tail popularity
    n_per_user = np.clip(rng.lognormal(3.4, 0.6, U).astype(int), 12, 150)
    frames = []
    for u in range(U):
        # users mostly consume popular items that match their taste (realistic selection bias)
        p = popularity * np.exp(1.0 * np.clip(z[u], -3, 3))
        p = p / p.sum()
        chosen = rng.choice(I, size=n_per_user[u], replace=False, p=p)
        frames.append(pd.DataFrame({"user_id": u + 1, "item_id": chosen + 1, "rating": true_rating[u, chosen]}))
    return pd.concat(frames, ignore_index=True)


def inject_quality_problems(rng, users, items, ratings):
    """Add missing values, duplicates and invalid ratings so cleaning is meaningful."""
    users = users.copy()
    items = items.copy()
    users["age"] = users["age"].astype(float).mask(rng.random(len(users)) < 0.03)
    items["description"] = items["description"].mask(rng.random(len(items)) < 0.04)
    items["year"] = items["year"].astype(float).mask(rng.random(len(items)) < 0.02)

    ratings = ratings.copy()
    ratings["rating"] = ratings["rating"].astype(float)
    bad_pos = rng.choice(len(ratings), size=25, replace=False)
    ratings.iloc[bad_pos, ratings.columns.get_loc("rating")] = rng.choice([0.0, 6.0, np.nan], size=25)
    dup = ratings.iloc[rng.choice(len(ratings), size=150, replace=False)]
    ratings = pd.concat([ratings, dup], ignore_index=True)
    ratings = ratings.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return users, items, ratings


def generate_all(out_dir=DATA_DIR, seed=SEED):
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = generate_items(rng)
    users, pref = generate_users(rng)
    ratings = generate_ratings(rng, users, items, pref)
    users, items, ratings = inject_quality_problems(rng, users, items, ratings)
    users.to_csv(out_dir / "users.csv", index=False)
    items.to_csv(out_dir / "items.csv", index=False)
    ratings.to_csv(out_dir / "ratings.csv", index=False)
    print(f"[data] users={len(users)} items={len(items)} ratings={len(ratings)} -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    generate_all()
