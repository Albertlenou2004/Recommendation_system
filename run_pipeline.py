"""
run_pipeline.py - Runs the COMPLETE project end-to-end:

    python run_pipeline.py                # full run (generate data, tune, train, evaluate, save)
    python run_pipeline.py --skip-tuning  # faster: use DEFAULT_HP from config.py
"""
import argparse
import json
import time

import joblib
import numpy as np

from config import ARTIFACT_DIR, DATA_DIR, DEFAULT_HP, SEED
from data_generation import generate_all
from evaluate import evaluate_all, tune_pop_weight, write_results_md
from features import build_features
from preprocessing import build_dataset


def save_artifacts(ds, fb, hp, pred, pop_weight, model=None):
    """Persist everything needed to reproduce / run the demo without retraining."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ds.items.to_csv(ARTIFACT_DIR / "items_clean.csv", index=False)
    ds.users.to_csv(ARTIFACT_DIR / "users_clean.csv", index=False)
    ds.ratings_all.to_csv(ARTIFACT_DIR / "ratings_clean.csv", index=False)
    np.savez_compressed(ARTIFACT_DIR / "features.npz", user_features=fb.user_features,
                        item_features=fb.item_features, content_norm=fb.content_norm, pop_norm=fb.pop_norm)
    np.save(ARTIFACT_DIR / "scores.npy", pred)
    joblib.dump({"tfidf": fb.tfidf, "svd": fb.svd}, ARTIFACT_DIR / "text_encoder.joblib")
    hp_json = {k: (list(v) if isinstance(v, tuple) else v) for k, v in hp.items()}
    (ARTIFACT_DIR / "hyperparameters.json").write_text(json.dumps(hp_json, indent=2))
    (ARTIFACT_DIR / "meta.json").write_text(json.dumps({
        "n_users": ds.n_users, "n_items": ds.n_items, "global_mean": float(ds.train["rating"].mean()),
        "best_pop_weight": pop_weight, "hyperparameters": hp_json, "seed": SEED}, indent=2))
    if model is not None:
        import torch
        torch.save(model.state_dict(), ARTIFACT_DIR / "model.pt")
    print(f"[save] artefacts written to {ARTIFACT_DIR}")


def main(skip_tuning=False, regenerate=False):
    t0 = time.time()
    if regenerate or not (DATA_DIR / "ratings.csv").exists():
        generate_all()
    ds = build_dataset()                                  # STEP 2: cleaning, encoding, splitting
    fb = build_features(ds)                               # STEP 3.1: feature engineering

    from train import get_device, predict_matrix, train_model, tune_hyperparameters
    device = get_device()
    print(f"[train] device = {device}")
    if skip_tuning:
        hp, tuning_df = dict(DEFAULT_HP), None
    else:
        hp, tuning_df = tune_hyperparameters(ds, fb, device)          # STEP 3.3: tuning on validation set
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        tuning_df.to_csv(ARTIFACT_DIR / "tuning_results.csv", index=False)
    print(f"[train] final hyper-parameters: {hp}")
    model, history, best_val = train_model(ds, fb, hp, device)
    print(f"[train] best validation RMSE = {best_val:.4f}")

    pred = predict_matrix(model, ds.n_users, ds.n_items, device)
    pop_weight, pop_results = tune_pop_weight(ds, fb, pred)           # tune ranking layer on validation
    report = evaluate_all(ds, fb, pred, pop_weight, history)          # STEP 4: test-set evaluation
    save_artifacts(ds, fb, hp, pred, pop_weight, model)               # STEP 5: reproducibility

    import pandas as pd
    tdf = tuning_df if tuning_df is not None else pd.DataFrame([{"note": "tuning skipped", "val_rmse": best_val}])
    write_results_md(report, ds, hp, tdf, pop_results, ARTIFACT_DIR / "RESULTS.md")
    print(f"[done] finished in {time.time() - t0:.0f}s -> see artifacts/RESULTS.md, then run: streamlit run app.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tuning", action="store_true")
    ap.add_argument("--regenerate-data", action="store_true")
    a = ap.parse_args()
    main(a.skip_tuning, a.regenerate_data)
