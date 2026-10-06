# Hybrid Neural Recommendation System
### Continuous Assessment Test - Practical | ITLPA701 Python and Fundamentals of AI
**Institution:** Rwanda Polytechnic - Ngoma College | **Department:** ICT | **Trade:** IT | **Year 3, Semester 1** | **Academic year 2026-2027**
**Module team:** Eng. UWAMAHORO Leopord | **Candidate:** ____________ | **Reg. number:** ____________

---
## Step 1 - Problem domain and architecture

### 1.1 Real-world problem
Streaming and e-commerce users face thousands of items and cannot find the ones they will enjoy. A recommender system filters the catalogue and ranks the items that best match each person. The chosen domain is **movie recommendation** (the pipeline is domain-independent: books, products or music need only a different items table). It belongs to the assessment's *document/item retrieval* family and uses **user-behaviour (ratings) plus textual metadata (genres, descriptions)**.

### 1.2 Selected AI approach (one approach, as required)
**Deep Learning with a feed-forward neural network (FNN)**: a *Hybrid Neural Collaborative Filtering* model (NeuMF-style) implemented in PyTorch.

**Justification**
| Alternative | Why it was not selected |
|---|---|
| LLM-based | An LLM can talk about movies but has no knowledge of *our* users' rating history; it is slow, costly and can hallucinate titles that do not exist in the catalogue. |
| RAG | Excellent for question answering over documents; here the task is predicting personal preference from interaction data, not answering questions. |
| Agentic AI | Needs tools and controlled actions; recommendation is a scoring problem, not a multi-step action problem. |
| **Neural CF (chosen)** | Learns non-linear user-item interactions from behaviour data, scales to large catalogues, predicts a score for every item in milliseconds, and the evaluation (RMSE, Precision@K, NDCG...) is standard and objective. Content features remove the weakness of pure collaborative filtering on sparse data (cold-start). |

*Explanations ("why was this recommended") are produced by a transparent rule-based module that reads real data, so there is no hallucination risk.*

### 1.3 Architecture
```
INPUT PIPELINE                         CORE ENGINE                                OUTPUT PIPELINE
------------------                     ---------------------------------------    -------------------------------
users.csv  (age, gender, job) --+      user_id -> Embedding(d) ----+
items.csv  (title, genres,      |      user features -> Linear+ReLU +-> up
           year, description)   +--> clean/encode --> features:               GMF  = (ue+up) * (ie+ip)
ratings.csv (user,item,1-5)  ---+      * genre multi-hot (8)                  MLP  = FFN[128->64] on concat(ue,ie,up,ip)
                                       * TF-IDF -> SVD text embedding (32)    fuse = Linear(concat(GMF, MLP))
                                       * user taste profile (mean of liked    score = fuse + user_bias + item_bias + mu
                                         items' content vectors)                      |
                                       * scaled numeric + one-hot demographics       v
                                       item_id -> Embedding(d) ----+          blend with popularity prior
                                       item features -> Linear+ReLU -> ip     -> optional MMR diversity re-ranking
                                                                              -> Top-K list + predicted rating
                                                                                 + rule-based rationale
                                                                              -> Streamlit web interface / CSV
```

---
## Step 2 - Resources and data preprocessing (30%)

### 2.1 Environment (functionalities specified and configured)
Python 3.10+, PyCharm/Jupyter/Spyder. Libraries (`requirements.txt`): `numpy`, `pandas`, `scipy`, `scikit-learn` (TF-IDF, SVD, scaling, metrics), `torch` (neural network), `matplotlib` (figures), `joblib` (saving encoders), `streamlit` (interface), `pytest` (tests). Functionalities: data generation -> cleaning -> features -> training -> evaluation -> inference -> web demo, each in its own module. All settings are centralised in `config.py`; a fixed seed (42) makes the experiment reproducible.

### 2.2 Data acquisition
A MovieLens-style dataset is generated programmatically (`data_generation.py`): **1000 users, 600 items, about 35 000 explicit ratings (1-5)**. Ratings >= 4 are treated as implicit "likes" for ranking evaluation. The generator uses a known structure (genre preference + latent taste + user bias + item bias + noise, long-tail popularity, selection bias) so that a learning algorithm has real patterns to find. The same code can read a real dataset (MovieLens/Kaggle) converted to the same three CSV schemas.

### 2.3 Cleaning and preprocessing (`preprocessing.py`)
Deliberate quality problems are injected (about 3% missing ages, 4% missing descriptions, duplicated ratings, ratings of 0, 6 or NaN). The pipeline then:
1. imputes numeric gaps with the median and categorical gaps with `unknown`; missing text becomes an empty string (title and genres still describe the item);
2. removes ratings that are missing, outside 1-5 or duplicated; checks referential integrity (every rating refers to an existing user and item);
3. removes users with fewer than 5 ratings (k-core filter);
4. encodes ids to contiguous integers (rows of the embedding tables) and genres as multi-hot vectors; one-hot encodes gender and occupation;
5. **splits per user**: about 20% of each user's ratings -> test, 10% of the remainder -> validation, the rest -> train. Statistics are computed from the training split only, so there is **no data leakage** (verified by unit tests).
The exact number of rows affected by each step is printed and saved in `artifacts/RESULTS.md` (data-quality report).

---
## Step 3 - Feature engineering and model implementation (50%)

### 3.1 Feature engineering (`features.py`)
* **Item vector (42-d):** genre multi-hot (8) + LSA text embedding (32, TF-IDF over title, genres and description, compressed by truncated SVD and L2-normalised) + scaled year (1) + log-popularity from training counts (1).
* **User vector:** taste profile (40-d) = mean content vector of the items the user liked; age, mean rating and log(#ratings) standardised; one-hot gender and occupation. Placing user profiles and items in the same content space makes cosine similarity possible and gives the explanation module something truthful to say.
* **Interaction data:** the sparse user x item rating matrix is represented as (user_idx, item_idx, rating) triples feeding the embedding layers.

### 3.2 Model features (`model.py`)
`HybridNCF` combines (a) **ID embeddings** (collaborative signal), (b) **feature projections** (content signal), (c) a **GMF branch** (element-wise product, the neural generalisation of matrix factorisation), (d) an **MLP branch** (128 -> 64 with ReLU and dropout, learns non-linear interactions), (e) a fusion layer plus **user bias, item bias and global mean** (captures generous raters and universally popular items). Feature tables are stored as model buffers, so `model.pt` is self-contained.

### 3.3 Hyper-parameters (selected and explained)
| Parameter | Value / search | Reason |
|---|---|---|
| Embedding dimension | grid 16, 32, 64 | capacity vs. over-fitting on ~25 000 training ratings |
| Learning rate | grid 1e-3, 3e-3 | Adam-type optimisers are stable in this range; larger is faster but noisier |
| Dropout | grid 0.2, 0.4 | main regulariser of the MLP branch |
| Hidden layers | (128, 64) | funnel-shaped FFN: wide first, compressed later |
| Weight decay (AdamW) | 1e-5 | mild L2 regularisation on embeddings |
| Batch size | 512 | good gradient estimates, fast on CPU |
| Epochs / early stopping | max 40, patience 5 | stop when validation RMSE stops improving; best weights restored |
| LR scheduler | ReduceLROnPlateau x0.5 | fine-tunes convergence |
| Loss | MSE | explicit ratings are numeric targets |
| Top-K, popularity weight | K=10; weight grid 0-0.5 | ranking-layer parameters, selected on validation NDCG@10 |
| MMR diversity | 0-1 (user slider) | relevance/diversity trade-off |
The 12 grid combinations are evaluated on the **validation** set only (`tuning_results.csv`); the test set is used once for the final report.

### 3.4 Implementation and testing
`train.py` implements the training loop, validation, early stopping and grid search; `test_pipeline.py` contains automated tests (cleaning, no train/test overlap, feature shapes, metric mathematics on a hand-computed example, perfect-oracle NDCG = 1, MMR behaviour, model output shape).

---
## Step 4 - Evaluation (20%)

### 4.1 Metrics chosen (appropriate to the Deep-Learning approach and to recommendation)
| Family | Metrics | Why |
|---|---|---|
| Rating prediction | RMSE, MAE, R2 | how close predicted stars are to real stars |
| Classification (liked >= 4 vs. not) | accuracy, precision, recall, F1, confusion matrix | required by the assessment for deep learning |
| Top-K ranking | Precision@K, Recall@K, HitRate@K, NDCG@K (K = 5, 10, 20) | users only see a short list - ranking quality matters most |
| Beyond accuracy | catalogue coverage, average popularity, intra-list diversity | detects popularity bias and filter bubbles |

### 4.2 Evaluation protocol (`metrics.py`, `evaluate.py`)
Test items are never used for training or tuning. For ranking, every item a user has already rated in train/validation is removed from the candidate list, and the held-out liked items are the ground truth. The hybrid model is compared with **baselines**: random, popularity, content-based cosine, global mean, user+item bias, and classical SVD matrix factorisation. Comparing with baselines shows whether the neural network actually adds value.

### 4.3 Results and interpretation
Running `python run_pipeline.py` produces **`artifacts/RESULTS.md`**, containing the real tables of your run (data-quality report, tuning table, rating, classification and ranking metrics, confusion matrix) and an automatically written interpretation, plus the figures `training_curve.png`, `confusion_matrix.png`, `ranking_comparison.png`, `eda.png`. **Copy those tables and figures here before submitting** (numbers differ slightly between machines, so none are hard-coded in this document).

How to read them in your discussion:
* *RMSE/MAE:* an RMSE below the global-mean baseline proves the model learned personal taste; R2 > 0 means it explains part of the rating variance.
* *Training curve:* validation RMSE flattening while training RMSE keeps falling is over-fitting; early stopping and dropout limit it.
* *Classification:* the classes are imbalanced (most ratings are likes), so accuracy must be compared with the trivial "everything is liked" predictor; recall of the "liked" class matters less than precision in the top of the list.
* *Ranking:* NDCG@10 and Recall@10 well above random and popularity show useful personalisation; the comparison with SVD tells honestly whether the deep model is better than a classical method on this data.
* *Diversity:* the MMR row quantifies the price (change in NDCG) of a more varied list.

### 4.4 Parameters adjusted after evaluation
(1) learning rate, embedding size and dropout by validation RMSE; (2) epochs by early stopping; (3) popularity weight by validation NDCG@10 - a higher weight can raise accuracy but lowers coverage and diversity, which is why it is tuned and not maximised; (4) MMR strength left to the user. If you change any value in `config.py`, re-run the pipeline and record the new results to show the adjustment cycle.

---
## Step 5 - Demonstration and reproducibility
**Interface:** `streamlit run app.py`.
* *Existing user* - choose a user, see the profile, learned genre taste and history, then Top-K recommendations with predicted rating and a rationale ("matches your favourite genres... similar to 'X' which you rated 5/5"); sliders for K, diversity, popularity boost and a genre filter; CSV download.
* *New user (cold start)* - pick genres/titles; the system builds a content-based profile.
* *Model performance* - headline metric tiles, comparison tables and figures.

The interface is styled with a custom CSS theme (`assets/style.css`, injected through `ui.py`): gradient header, recommendation cards with rank badge, predicted-rating bar and colour-coded genre chips, explanation box, styled sidebar and buttons, plus Streamlit theme settings in `.streamlit/config.toml`. All user-visible text is HTML-escaped (unit-tested).

**Saved artefacts** so the application can be reproduced without retraining: `model.pt`, `hyperparameters.json`, `meta.json`, `features.npz`, `scores.npy`, `text_encoder.joblib`, clean CSV files. Reproduction steps: install requirements -> `python run_pipeline.py` (or copy the `artifacts` folder) -> `streamlit run app.py`.

---
## Step 6 - Responsible use: risks and mitigation
| Risk / limitation | Explanation | Mitigation in this project | Further steps |
|---|---|---|---|
| **Filter bubble** | the model keeps recommending what the user already likes, narrowing exposure | MMR diversity re-ranking; diversity and coverage measured; user-controlled slider | add exploration (e.g., epsilon-greedy / bandits), "surprise me" mode |
| **Cold-start** | new users/items have no history, so collaborative signal is missing | content-based fallback for new users from genres/titles; content features inside the model | onboarding questionnaire; text embeddings for brand-new items |
| **Popularity bias** | popular items get most ratings and are over-recommended; niche items starve | popularity weight is tuned, not maximised; popularity and coverage reported | re-weighting / inverse-propensity training |
| **Algorithmic bias / fairness** | demographic features (age, gender) could lead to stereotyped recommendations | demographics are only weak inputs; recommendations explained; evaluation can be sliced by group | audit metrics per demographic group; remove sensitive features if bias is found |
| **Data privacy** | ratings reveal personal preferences | synthetic data in this project; no personal identifiers | consent, anonymisation, data minimisation, retention limits |
| **Synthetic-data limitation** | real behaviour is messier than generated data | pipeline accepts real datasets unchanged | validate on MovieLens/Kaggle before any real use |
| **Offline metrics vs. real satisfaction** | high NDCG does not guarantee that users are happy | several complementary metrics + baselines | A/B testing, user surveys |
| **LLM hallucination** (if an LLM were added) | an LLM could invent titles or reasons | not used; explanations are rule-based and built only from real data | if added, restrict the LLM to the retrieved candidate list (RAG) and verify titles |

---
## Rubric self-check (30 marks)
| Assessable outcome | Criterion (marks) | Where evidenced |
|---|---|---|
| Data preprocessing (30%) | environment functionalities specified (1.5) / configured (1.5) | Section 2.1, `requirements.txt`, `config.py` |
| | data acquired from the data source (3) | 2.2, `data_generation.py` |
| | data pre-processed (3) | 2.3, `preprocessing.py`, data-quality report |
| Deep learning (50%) | data features engineered (3) | 3.1, `features.py` |
| | model features engineered (3) | 3.2, `model.py` |
| | approach and model selected and justified (1.5) | 1.2 |
| | parameters selected and explained (1.5) | 3.3 |
| | application implemented and tested (6) | `train.py`, `recommender.py`, `test_pipeline.py` |
| Model evaluation (20%) | appropriate metrics selected (1) | 4.1 |
| | evaluation implemented, results interpreted (2) | `metrics.py`, `evaluate.py`, `RESULTS.md`, 4.3 |
| | parameters adjusted from results (1) | 4.4, `tuning_results.csv` |
| | model/config/prompts saved for reproduction (1) | `artifacts/`, Section 5 |
| | deployed through a web application (1) | `app.py` |
| Task 8 | responsible use discussed | Step 6 |

## Likely oral-defence questions
1. *Why is a hybrid model better than pure collaborative filtering?* It uses content/profile features, so sparse users and new items still get sensible scores.
2. *What does the GMF branch do versus the MLP branch?* GMF is a learned, generalised dot product (linear interactions); the MLP learns non-linear interactions; the fusion layer combines both.
3. *How did you avoid data leakage?* Per-user split; every statistic (profiles, popularity, scaling of behavioural features) uses training data only; the test set is not used for tuning; unit tests check set overlap.
4. *Why NDCG and not only RMSE?* Users see a ranked list; RMSE does not measure ordering at the top.
5. *What is the cold-start problem and how is it handled?* No history for new users; the app builds a content-based profile from chosen genres/titles.
6. *How would you move this to production?* Real data, nightly retraining, approximate nearest-neighbour retrieval + neural re-ranking, monitoring and A/B tests.
