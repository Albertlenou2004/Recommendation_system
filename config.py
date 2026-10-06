"""
config.py - Central configuration for the Hybrid Recommendation System.
Every tunable value lives here so the experiment is fully reproducible.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"
FIG_DIR = ARTIFACT_DIR / "figures"

# ---------------------------------------------------------------- data
SEED = 42
GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Horror", "Documentary", "Animation"]
OCCUPATIONS = ["student", "engineer", "teacher", "artist", "manager", "retired", "other"]
N_USERS = 1000
N_ITEMS = 600
RATING_MIN, RATING_MAX = 1, 5
MIN_RATINGS_PER_USER = 5          # users with fewer ratings are removed (cold users)
TEST_FRACTION = 0.20              # per-user hold-out for testing
VAL_FRACTION = 0.10               # share of the remaining train data used for validation

# ---------------------------------------------------------------- features
TEXT_DIM = 32                     # TF-IDF -> SVD (LSA) text embedding size
CONTENT_DIM = len(GENRES) + TEXT_DIM

# ---------------------------------------------------------------- evaluation
LIKE_THRESHOLD = 4.0              # true rating >= 4  => "relevant / liked"
PRED_THRESHOLD = 3.5              # predicted rating >= 3.5 => "predicted like" (classification view)
TOP_K = 10
KS = (5, 10, 20)

# ---------------------------------------------------------------- model / training
DEFAULT_HP = dict(
    emb_dim=32,
    hidden=(128, 64),
    dropout=0.3,
    lr=1e-3,
    weight_decay=1e-5,
    batch_size=512,
    epochs=40,
    patience=5,
)
# Grid searched on the VALIDATION set (never on the test set)
TUNING_GRID = dict(emb_dim=[16, 32, 64], lr=[1e-3, 3e-3], dropout=[0.2, 0.4])
TUNING_EPOCHS = 25

# ---------------------------------------------------------------- ranking layer
POP_WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.5]   # popularity prior blended with predicted rating
MMR_DIVERSITY_EVAL = 0.3                       # diversity strength used in the evaluation table
