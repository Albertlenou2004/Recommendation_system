"""
ui.py - HTML/CSS building blocks for the Streamlit interface (pure Python, no Streamlit import,
so it can be unit-tested).  All visual styling lives in assets/style.css.
"""
from html import escape
from pathlib import Path

CSS_PATH = Path(__file__).resolve().parent / "assets" / "style.css"


def load_css():
    """Return the stylesheet wrapped in <style> tags (inject with st.markdown(..., unsafe_allow_html=True))."""
    return f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>"


def hero(title, subtitle, tags):
    chips = "".join(f'<span class="tag">{escape(t)}</span>' for t in tags)
    return (f'<div class="hero"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p>'
            f'<div class="tags">{chips}</div></div>')


def section(title):
    return f'<div class="section-title">{escape(title)}</div>'


def stat_tiles(items):
    """items: list of (label, value)."""
    tiles = "".join(f'<div class="stat"><div class="label">{escape(str(l))}</div>'
                    f'<div class="value">{escape(str(v))}</div></div>' for l, v in items)
    return f'<div class="stat-grid">{tiles}</div>'


def genre_chips(genres):
    """genres: '|' separated string or list."""
    if isinstance(genres, str):
        genres = genres.split("|")
    return '<div class="chips">' + "".join(
        f'<span class="chip g-{escape(g)}">{escape(g)}</span>' for g in genres) + "</div>"


def stars(rating, max_stars=5):
    full = int(round(rating))
    return "★" * full + "☆" * (max_stars - full)


def recommendation_card(rank, title, year, genres, description, explanation,
                        predicted_rating=None, score=None, top=False):
    """One recommendation as an HTML card. Existing users show a predicted rating,
    cold-start users show a content-match score."""
    if predicted_rating is not None and predicted_rating == predicted_rating:       # not NaN
        pct = max(0.0, min(100.0, (predicted_rating - 1) / 4 * 100))
        head_right = f'<span class="rec-rating">{stars(predicted_rating)} {predicted_rating:.2f}</span>'
        bar_label = "predicted rating"
    else:
        pct = max(0.0, min(100.0, float(score or 0) * 100))
        head_right = f'<span class="rec-rating">match {pct:.0f}%</span>'
        bar_label = "content match"
    return (
        f'<div class="rec-card{" top" if top else ""}">'
        f'<div class="rank">{int(rank)}</div>'
        '<div class="rec-body">'
        f'<div class="rec-head"><div><span class="rec-title">{escape(str(title))}</span>'
        f'<span class="rec-year">{escape(str(year))}</span></div>{head_right}</div>'
        f'<div class="bar-wrap"><div class="bar"><span style="width:{pct:.0f}%"></span></div>'
        f'<span class="bar-label">{bar_label}</span></div>'
        f'<div class="rec-desc">{escape(str(description))}</div>'
        f'{genre_chips(genres)}'
        f'<div class="why">💡 {escape(str(explanation))}</div>'
        '</div></div>'
    )


def history_row(title, genres, rating):
    g = genres.replace("|", " · ")
    return (f'<div class="hist-row"><span><b>{escape(str(title))}</b> '
            f'<span style="color:#6B7280;font-size:.8rem">· {escape(g)}</span></span>'
            f'<span class="stars">{stars(rating)}</span></div>')


def empty_state(message):
    return f'<div class="empty">{escape(message)}</div>'
