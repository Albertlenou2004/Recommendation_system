# Hybrid Neural Recommendation System
**Module ITLPA701 - Python and Fundamentals of AI | Rwanda Polytechnic, Ngoma College**
**Selected AI approach: Deep Learning (feed-forward neural network) - Hybrid Neural Collaborative Filtering in PyTorch**

## 1. Setup (Step 2.1 - environment)
```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```
Works in PyCharm, Jupyter or Spyder (resources listed in the assessment); the Streamlit app runs from any terminal.

## 2. Run the whole project
```bash
python run_pipeline.py                 # generate data -> clean -> features -> tune -> train -> evaluate -> save
python run_pipeline.py --skip-tuning   # faster run using DEFAULT_HP from config.py
python -m pytest -q                    # automated tests (or: python test_pipeline.py)
streamlit run app.py                   # interactive demo (needs the pipeline to have run once)
```
Everything is seeded (`SEED = 42`), so results are reproducible.

## 3. Files
| File | Purpose | Rubric |
|---|---|---|
| `config.py` | all paths, hyper-parameters, thresholds | environment configured |
| `data_generation.py` | builds users / items / ratings CSV with injected data-quality problems | data acquired |
| `preprocessing.py` | cleaning, k-core filter, id encoding, per-user train/val/test split | data pre-processed |
| `features.py` | genre multi-hot, TF-IDF+SVD text embeddings, user taste profile, scaling | data features engineered |
| `model.py` | `HybridNCF` (GMF + MLP + content branches) | model features engineered |
| `train.py` | training loop, early stopping, grid-search tuning | parameters selected / implemented |
| `ranking.py` | popularity blend, MMR diversity re-ranking | parameters adjusted |
| `metrics.py`, `evaluate.py` | RMSE/MAE/R2, accuracy/P/R/F1/confusion, P@K/R@K/NDCG@K, coverage, diversity, baselines, plots, `RESULTS.md` | evaluation |
| `recommender.py` | inference API: Top-K + scores + rationale, cold-start | application |
| `app.py` | Streamlit web interface (existing user, cold-start, performance dashboard) | demonstration |
| `ui.py` | HTML component builders (hero, cards, chips, stat tiles) - escaped and unit-tested | demonstration |
| `assets/style.css` | custom CSS theme (gradient hero, recommendation cards, genre chips, score bars, sidebar, buttons) | demonstration |
| `.streamlit/config.toml` | Streamlit theme colours and server options | demonstration |
| `test_pipeline.py` | unit tests (cleaning, no leakage, metric maths, MMR, model shape) | quality |
| `PROJECT_REPORT.md` | full written documentation (architecture, justification, risks) | all |

## 3b. Folder structure (everything lives in ONE folder)
```
recsys_project/
|-- app.py  ui.py  recommender.py          # interface + inference
|-- run_pipeline.py  train.py  model.py    # training
|-- data_generation.py  preprocessing.py  features.py
|-- metrics.py  evaluate.py  ranking.py  config.py  test_pipeline.py
|-- assets/style.css                       # CSS theme
|-- .streamlit/config.toml                 # Streamlit theme
|-- requirements.txt  README.md  PROJECT_REPORT.md
|-- data/        (created by the pipeline)
`-- artifacts/   (created by the pipeline: model, features, results, figures)
```
Customise the look by editing the colour variables at the top of `assets/style.css` (`--primary`, `--accent`, ...).

## 4. Saved artefacts (`artifacts/`, created by the pipeline - reproducibility)
`model.pt` (weights + feature tables), `hyperparameters.json`, `meta.json`, `features.npz`, `scores.npy`,
`text_encoder.joblib` (TF-IDF + SVD), clean CSVs, `tuning_results.csv`, `evaluation_report.json`,
`model_comparison.csv`, **`RESULTS.md` (all real numbers + interpretation of YOUR run)** and `figures/*.png`.

## 5. Using a real dataset (optional)
Replace the three CSV files in `data/` by MovieLens converted to the same columns
(`users: user_id,age,gender,occupation` / `items: item_id,title,genres(|-separated),year,description` /
`ratings: user_id,item_id,rating`) and run `python run_pipeline.py`. No code change is needed.
