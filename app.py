"""
app.py - STEP 5 Streamlit demo.   Run with:   streamlit run app.py
Modes: existing user (personalised, neural model) | new user (cold start) | model performance.
Styling: assets/style.css (loaded via ui.py) + .streamlit/config.toml (theme colours).
"""
import json

import pandas as pd
import streamlit as st

import ui
from config import ARTIFACT_DIR, FIG_DIR, GENRES
from recommender import Recommender

st.set_page_config(page_title="Hybrid Movie Recommender", page_icon="🎬", layout="wide")
st.markdown(ui.load_css(), unsafe_allow_html=True)             # apply the custom CSS theme
st.markdown(ui.hero(
    "🎬 Hybrid Neural Recommendation System",
    "Neural Collaborative Filtering + LSTM sequence encoder + content features + diversity re-ranking. "
    "Python & Fundamentals of AI - Rwanda Polytechnic, Ngoma College.",
    ["PyTorch + LSTM", "Hybrid Neural Recommender (LSTM)", "Cold-start ready", "Explainable", "Diversity-aware"]), unsafe_allow_html=True)

if not (ARTIFACT_DIR / "meta.json").exists():
    st.error("No trained artefacts found. Run `python run_pipeline.py` first, then restart this app.")
    st.stop()


@st.cache_resource(show_spinner="Loading model and data ...")
def load_recommender():
    return Recommender()


rec = load_recommender()

# ------------------------------------------------------------------ sidebar controls
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Mode", ["Existing user", "New user (cold start)", "Model performance"])
k = st.sidebar.slider("Number of recommendations (K)", 3, 20, 10)
diversity = st.sidebar.slider("Diversity (MMR)", 0.0, 1.0, 0.0, 0.1,
                              help="0 = most relevant only; higher = more varied list (reduces filter-bubble effect)")
pop_w = st.sidebar.slider("Popularity boost", 0.0, 0.5, float(rec.pop_weight), 0.05,
                          help="Higher values favour widely-rated items; lower values favour niche items")
genre_filter = st.sidebar.multiselect("Restrict to genres (optional)", GENRES)
st.sidebar.info("Engine: " + ("PyTorch + LSTM Hybrid Recommender (live)" if rec.model is not None else "pre-computed scores"))


def show_recommendations(df):
    """Render the Top-K list as styled cards + CSV download."""
    if df.empty:
        st.markdown(ui.empty_state("No items match the current filters."), unsafe_allow_html=True)
        return
    html = "".join(ui.recommendation_card(
        r["rank"], r["title"], r["year"], r["genres"], r["description"], r["explanation"],
        predicted_rating=r["predicted_rating"], score=r["score"], top=(r["rank"] == 1))
        for _, r in df.iterrows())
    st.markdown(html, unsafe_allow_html=True)
    st.download_button("⬇ Download as CSV", df.drop(columns=["description"]).to_csv(index=False),
                       file_name="recommendations.csv", mime="text/csv")


# ------------------------------------------------------------------ mode 1: existing user
if mode == "Existing user":
    uid = st.selectbox("Select a user ID", rec.users["user_id"].tolist())
    u = rec.user_idx_from_id(uid)
    row = rec.users.iloc[u]
    st.markdown(ui.section("User profile"), unsafe_allow_html=True)
    st.markdown(ui.stat_tiles([("Age", int(row["age"])), ("Gender", str(row["gender"]).title()),
                               ("Occupation", str(row["occupation"]).title()),
                               ("Ratings given", len(rec.seen.get(u, [])))]), unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(ui.section("Learned taste (genre share)"), unsafe_allow_html=True)
        st.bar_chart(rec.user_genre_profile(u), color="#4F46E5")
    with c2:
        st.markdown(ui.section("Highest-rated history"), unsafe_allow_html=True)
        st.markdown("".join(ui.history_row(r.title, r.genres, r.rating)
                            for r in rec.user_history(u, 6).itertuples()), unsafe_allow_html=True)
    st.markdown(ui.section(f"Top-{k} recommendations"), unsafe_allow_html=True)
    show_recommendations(rec.recommend(u, k, diversity, pop_w, genre_filter or None))

# ------------------------------------------------------------------ mode 2: cold start
elif mode == "New user (cold start)":
    st.markdown(ui.section("Tell us what you like"), unsafe_allow_html=True)
    st.caption("A brand-new user has no history, so the system builds a content-based profile "
               "from the genres and titles you pick (cold-start strategy).")
    liked_genres = st.multiselect("Genres you enjoy", GENRES, default=["Sci-Fi"])
    titles = st.multiselect("Titles you already like (optional)", rec.items["title"].tolist())
    liked_idx = rec.items.index[rec.items["title"].isin(titles)].tolist()
    if liked_genres or liked_idx:
        st.markdown(ui.section(f"Top-{k} recommendations"), unsafe_allow_html=True)
        show_recommendations(rec.recommend_new_user(liked_genres, liked_idx, k, diversity))
    else:
        st.markdown(ui.empty_state("Pick at least one genre or title to get recommendations."), unsafe_allow_html=True)

# ------------------------------------------------------------------ mode 3: performance
else:
    rep_path = ARTIFACT_DIR / "evaluation_report.json"
    if not rep_path.exists():
        st.warning("evaluation_report.json not found - run the pipeline first.")
    else:
        rep = json.loads(rep_path.read_text())
        ours = [m for m in rep["rating"] if "NCF" in m][0]
        r, cl = rep["rating"][ours], rep["classification"][ours]
        rk = rep["ranking"][[m for m in rep["ranking"] if m.endswith("popularity prior")][0]]
        st.markdown(ui.section("Headline results (test set)"), unsafe_allow_html=True)
        st.markdown(ui.stat_tiles([("RMSE", f"{r['RMSE']:.3f}"), ("MAE", f"{r['MAE']:.3f}"),
                                   ("R²", f"{r['R2']:.3f}"), ("F1 (liked)", f"{cl['f1']:.3f}"),
                                   ("NDCG@10", f"{rk['ndcg@10']:.3f}"), ("Recall@10", f"{rk['recall@10']:.3f}")]),
                    unsafe_allow_html=True)
        st.markdown(ui.section("Rating prediction vs. baselines"), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rep["rating"]).T.round(4))
        st.markdown(ui.section("Top-K ranking vs. baselines"), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rep["ranking"]).T.round(4))
        st.markdown(ui.section("Figures"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        for col, name in zip([c1, c2], ["training_curve.png", "confusion_matrix.png"]):
            if (FIG_DIR / name).exists():
                col.image(str(FIG_DIR / name))
        for name in ["ranking_comparison.png", "eda.png"]:
            if (FIG_DIR / name).exists():
                st.image(str(FIG_DIR / name))
