import functools
import json
import logging
import re
import time
import unicodedata
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import requests
import sklearn
from django.conf import settings
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_DIR = PROJECT_ROOT / "ML-model" / "model_v2"
NEIGHBOR_TABLE_PATH = MODEL_DIR / "neighbor_table.npy"

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

_model_data = None
_title_to_index = None
_poster_cache: dict = {}  # tmdb_id → full URL or None
_neighbor_table = None  # (n_movies, n_neighbors) int32


class ModelArtifactError(RuntimeError):
    """The on-disk model artifacts are missing, stale or inconsistent."""


REBUILD_HINT = "Rebuild with: python ML-model/notebooks/build_model.py (from the repo root)."


def _load_artifacts():
    """Load model_v2 and validate that every artifact belongs to the same build.

    Raises ModelArtifactError (never silently falls back) on a missing file, an
    sklearn version mismatch or a neighbor table that does not match the model.
    """
    import joblib
    import pandas as pd
    import scipy.sparse

    meta_path = MODEL_DIR / "model_meta.json"
    if not meta_path.exists():
        raise ModelArtifactError(f"{meta_path} not found. {REBUILD_HINT}")
    meta = json.loads(meta_path.read_text())

    saved_ver = tuple(int(x) for x in meta["sklearn_version"].split(".")[:2])
    current_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if saved_ver != current_ver:
        raise ModelArtifactError(
            f"Model was built with scikit-learn {meta['sklearn_version']} but "
            f"{sklearn.__version__} is installed. {REBUILD_HINT}"
        )

    tfidf = joblib.load(MODEL_DIR / "vectorizer.joblib")
    matrix = scipy.sparse.load_npz(str(MODEL_DIR / "tfidf_matrix.npz"))
    df = pd.read_csv(MODEL_DIR / "movies.csv")

    n = int(meta["n_movies"])
    if not (matrix.shape[0] == len(df) == n):
        raise ModelArtifactError(
            f"Inconsistent artifacts: meta says {n} movies, matrix has {matrix.shape[0]} rows, "
            f"movies.csv has {len(df)} rows. {REBUILD_HINT}"
        )

    table = None
    if NEIGHBOR_TABLE_PATH.exists():
        table = np.load(NEIGHBOR_TABLE_PATH)
        if table.shape[0] != n or (table.size and int(table.max()) >= n):
            raise ModelArtifactError(
                f"neighbor_table.npy has shape {table.shape} but the model has {n} movies. "
                f"{REBUILD_HINT}"
            )
    return (tfidf, matrix, df), table


def _load_model():
    global _model_data, _title_to_index, _poster_cache, _neighbor_table
    if _model_data is None:
        _model_data, _neighbor_table = _load_artifacts()
        logger.info("Loaded model from %s", MODEL_DIR)

        movies_df = _model_data[2]
        _title_to_index = {title.lower(): i for i, title in enumerate(movies_df["title"])}

        # Build poster cache from precomputed column (zero network calls)
        if "poster_path" in movies_df.columns:
            for tmdb_id, path in zip(movies_df["id"], movies_df["poster_path"]):
                if tmdb_id != tmdb_id:  # NaN
                    continue
                url = f"{TMDB_IMAGE_BASE}{path}" if path and str(path) != "nan" else None
                _poster_cache[int(tmdb_id)] = url
            found = sum(v is not None for v in _poster_cache.values())
            logger.info("Poster cache loaded: %d/%d entries", found, len(_poster_cache))

    return _model_data, _title_to_index


# ---------------------------------------------------------------------------
# Title matching
#
# Catalog titles are MovieLens-style: "Matrix, The (1999)", "Seven (a.k.a. Se7en) (1995)".
# Users type "The Matrix" or "se7en". We therefore match on a normalized key
# (no year, article moved back to the front and dropped, alias titles indexed) and only
# fall back to fuzzy matching when nothing matches exactly.
# ---------------------------------------------------------------------------

FUZZY_CUTOFF = 0.75

_YEAR_RE = re.compile(r"\s*\((\d{4})\)\s*$")
_PAREN_RE = re.compile(r"\(([^)]*)\)")
_TRAILING_ARTICLE_RE = re.compile(
    r"^(?P<body>.+?),\s*(?P<art>the|a|an|le|la|les|l'|el|il|lo|der|die|das|un|une)$"
)
_LEADING_ARTICLE_RE = re.compile(r"^(the|a|an)\s+")
_AKA_RE = re.compile(r"^(a\.k\.a\.|aka)\s*")


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("&", " and ").replace("'", "")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _key(text: str) -> str:
    text = text.strip().lower()
    m = _TRAILING_ARTICLE_RE.match(text)
    if m:
        art, body = m.group("art"), m.group("body")
        text = f"{art}{body}" if art.endswith("'") else f"{art} {body}"
    return _LEADING_ARTICLE_RE.sub("", _clean(text))


def _variants(title: str) -> tuple[str | None, list[str]]:
    """Return (year, normalized keys) for a title: the main title plus any alias."""
    title = title.strip().lower()
    m = _YEAR_RE.search(title)
    year = m.group(1) if m else None
    base = title[: m.start()] if m else title
    names = [_PAREN_RE.sub(" ", base)] + [
        _AKA_RE.sub("", g.strip()) for g in _PAREN_RE.findall(base)
    ]
    return year, [k for k in dict.fromkeys(_key(n) for n in names) if k]


@functools.lru_cache(maxsize=4)
def _alias_index(titles: tuple) -> dict:
    """normalized key → [(year, full lowercase title), ...] in catalog order."""
    index: dict = {}
    for full in titles:
        year, keys = _variants(full)
        for k in keys:
            index.setdefault(k, []).append((year, full))
    return index


def _pick(candidates: list, year: str | None) -> str:
    if year:
        for cand_year, full in candidates:
            if cand_year == year:
                return full
    return candidates[0][1]


def resolve_title(query: str, title_to_index: dict) -> str | None:
    """Map a user query to a catalog title key (lowercase), or None if nothing is close.

    Order: exact title → normalized title/alias (year disambiguates remakes) → fuzzy.
    """
    q = query.strip().lower()
    if q in title_to_index:
        return q

    year, keys = _variants(q)
    if not keys:
        return None
    index = _alias_index(tuple(title_to_index))

    if keys[0] in index:
        return _pick(index[keys[0]], year)

    close = get_close_matches(keys[0], index.keys(), n=1, cutoff=FUZZY_CUTOFF)
    return _pick(index[close[0]], year) if close else None


def same_title(query: str, matched_title: str) -> bool:
    """True if `matched_title` is just a differently-formatted spelling of `query`
    (as opposed to a fuzzy correction the user should be told about)."""
    q_year, q_keys = _variants(query)
    m_year, m_keys = _variants(matched_title)
    return bool(q_keys) and q_keys[0] in m_keys and (q_year is None or q_year == m_year)


@functools.lru_cache(maxsize=512)
def _fetch_poster_live(tmdb_id: int) -> str | None:
    """Last-resort live fetch for cache misses. Called at most once per tmdb_id."""
    api_key = settings.TMDB_API_KEY
    if not api_key:
        return None
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            params={"api_key": api_key},
            timeout=5,
        )
        r.raise_for_status()
        path = r.json().get("poster_path")
        return f"{TMDB_IMAGE_BASE}{path}" if path else None
    except Exception:
        return None


def get_poster(tmdb_id: int | None) -> str | None:
    if tmdb_id is None:
        return None
    url = _poster_cache.get(int(tmdb_id))
    if url is None and int(tmdb_id) not in _poster_cache:
        # genuine cache miss (not seen at load time) — fetch live and remember
        url = _fetch_poster_live(int(tmdb_id))
        _poster_cache[int(tmdb_id)] = url
    return url


RERANK_BETA = 0.3  # exponent for vote_average in score = sim × rating^β


def _clean_value(row, column):
    value = row.get(column)
    return None if value is None or str(value) == "nan" else value


def recommend_movies(movie_title, top_n=10, rerank=True):
    t0 = time.perf_counter()
    (tfidf_vectorizer, tfidf_matrix, movies_df), title_to_index = _load_model()

    matched_key = resolve_title(movie_title, title_to_index)
    if matched_key is None:
        raise ValueError(f"Movie '{movie_title}' not found in dataset")

    idx = title_to_index[matched_key]
    matched_title = movies_df.iloc[idx]["title"]

    if _neighbor_table is not None:
        candidate_indices = _neighbor_table[idx]
        if rerank and "vote_average" in movies_df.columns:
            sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix[candidate_indices]).flatten()
            ratings = movies_df.iloc[candidate_indices]["vote_average"].fillna(0).to_numpy()
            scores = sims * np.power(np.maximum(ratings, 0), RERANK_BETA)
            order = scores.argsort()[-top_n:][::-1]
            similar_indices = candidate_indices[order]
        else:
            similar_indices = candidate_indices[:top_n]
    else:
        cosine_similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        if rerank and "vote_average" in movies_df.columns:
            ratings = movies_df["vote_average"].fillna(0).to_numpy().astype(float)
            scores = cosine_similarities * np.power(np.maximum(ratings, 0), RERANK_BETA)
        else:
            scores = cosine_similarities
        scores[idx] = -1
        similar_indices = scores.argsort()[-top_n:][::-1]

    recommendations = []
    for _, row in movies_df.iloc[similar_indices].iterrows():
        tmdb_id = _clean_value(row, "id")
        release_date = _clean_value(row, "release_date")
        vote_average = _clean_value(row, "vote_average")
        recommendations.append(
            {
                "title": row["title"],
                "poster": get_poster(int(tmdb_id) if tmdb_id is not None else None),
                "release_date": str(release_date) if release_date is not None else None,
                "vote_average": float(vote_average) if vote_average is not None else None,
                "tmdb_id": int(tmdb_id) if tmdb_id is not None else None,
            }
        )

    elapsed = time.perf_counter() - t0
    logger.info("recommend '%s' → '%s' in %.3fs", movie_title, matched_title, elapsed)
    return matched_title, recommendations
