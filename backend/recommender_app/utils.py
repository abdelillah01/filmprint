import ast
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

from .profile import profile_scores

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_DIR = PROJECT_ROOT / "ML-model" / "model_v2"
NEIGHBOR_TABLE_PATH = MODEL_DIR / "neighbor_table.npy"
# Optional enrichments, keyed by MovieLens movieId. Missing files are fine.
STATS_PATH = PROJECT_ROOT / "ML-model" / "data" / "processed" / "film_stats.csv"
# IMDb ratings are licensed for personal use and are NOT committed: see scripts/fetch_imdb.py
IMDB_PATH = PROJECT_ROOT / "ML-model" / "data" / "external" / "imdb_ratings.csv"

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

_model_data = None
_title_to_index = None
_poster_cache: dict = {}  # tmdb_id → full URL or None
_neighbor_table = None  # (n_movies, n_neighbors) int32
_extras: dict = {}  # movieId -> imdb_id / runtime / tmdb_votes / imdb_rating / imdb_votes


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
    global _model_data, _title_to_index, _poster_cache, _neighbor_table, _extras
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

        _extras = _load_extras()

    return _model_data, _title_to_index


def _num(value, cast):
    """`cast(value)`, or None for missing/NaN."""
    if value is None or value != value:
        return None
    return cast(value)


def _load_extras() -> dict:
    """Per-film facts that enrich each result: IMDb id, runtime, TMDB votes, IMDb rating.

    Everything here is optional: a missing or unreadable file is logged and skipped, and the
    affected fields are simply null in the API (the UI then shows nothing for them).
    """
    import pandas as pd

    extras: dict = {}
    if STATS_PATH.exists():
        try:
            for r in pd.read_csv(STATS_PATH).itertuples(index=False):
                extras[int(r.movieId)] = {
                    "imdb_id": _num(r.imdb_id, str),
                    "runtime": _num(r.runtime, int),
                    "tmdb_votes": _num(r.tmdb_votes, int),
                }
        except Exception:
            logger.warning("Could not read %s; film stats disabled", STATS_PATH, exc_info=True)
    else:
        logger.info(
            "%s not found; run scripts/build_film_stats.py for runtimes and links", STATS_PATH
        )

    if IMDB_PATH.exists() and extras:
        try:
            ratings = {
                r.imdb_id: (float(r.imdb_rating), int(r.imdb_votes))
                for r in pd.read_csv(IMDB_PATH).itertuples(index=False)
            }
            for info in extras.values():
                if info["imdb_id"] in ratings:
                    info["imdb_rating"], info["imdb_votes"] = ratings[info["imdb_id"]]
        except Exception:
            logger.warning("Could not read %s; IMDb ratings disabled", IMDB_PATH, exc_info=True)
    else:
        logger.info("No IMDb ratings loaded (run scripts/fetch_imdb.py to add them)")
    return extras


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
RERANK_POOL = 50  # candidates (by similarity) that the rating rerank may reorder


class TitleNotFound(ValueError):
    """One or more liked films could not be matched to the catalog."""

    def __init__(self, titles):
        self.titles = list(titles)
        super().__init__(f"Movie(s) not found in dataset: {', '.join(self.titles)}")


def _clean_value(row, column):
    value = row.get(column)
    return None if value is None or str(value) == "nan" else value


# ---------------------------------------------------------------------------
# Per-film metadata used by filters (genres, year, rating), parsed once per model
# ---------------------------------------------------------------------------

_meta_cache: dict = {}

# MovieLens tags "IMAX" like a genre, but it is a screening format.
_NOT_GENRES = frozenset({"imax"})


def _film_meta(df):
    """(years, ratings, genre sets) aligned with the rows of `df`; empty where a column is absent."""
    key = (id(df), len(df))
    meta = _meta_cache.get(key)
    if meta is None:
        n = len(df)
        years = np.zeros(n, dtype=int)
        if "release_date" in df.columns:
            for i, v in enumerate(df["release_date"]):
                head = str(v)[:4]
                years[i] = int(head) if head.isdigit() else 0
        ratings = (
            df["vote_average"].fillna(0).to_numpy(dtype=float)
            if "vote_average" in df.columns
            else np.zeros(n)
        )
        genres = [frozenset()] * n
        if "genres" in df.columns:
            parsed = []
            for v in df["genres"]:
                try:
                    parsed.append(
                        frozenset(
                            g
                            for g in ast.literal_eval(v)
                            if isinstance(g, str) and g not in _NOT_GENRES
                        )
                    )
                except (ValueError, SyntaxError):
                    parsed.append(frozenset())
            genres = parsed
        meta = _meta_cache[key] = (years, ratings, genres)
    return meta


def _effective_ratings(df, tmdb_ratings):
    """IMDb rating where we have one, otherwise the TMDB rating."""
    if "movieId" not in df.columns or not _extras:
        return tmdb_ratings
    effective = tmdb_ratings.copy()
    for i, movie_id in enumerate(df["movieId"]):
        imdb = _extras.get(_num(movie_id, int), {}).get("imdb_rating")
        if imdb is not None:
            effective[i] = imdb
    return effective


def _allowed_mask(df, filters):
    """Boolean mask of films that pass the filters. `genres` means 'has at least one of'."""
    n = len(df)
    ok = np.ones(n, dtype=bool)
    if not filters:
        return ok
    years, ratings, genres = _film_meta(df)

    include = {g.lower() for g in filters.get("genres") or ()}
    exclude = {g.lower() for g in filters.get("exclude_genres") or ()}
    if include:
        ok &= np.fromiter((bool(g & include) for g in genres), dtype=bool, count=n)
    if exclude:
        ok &= np.fromiter((not (g & exclude) for g in genres), dtype=bool, count=n)
    if filters.get("year_min") is not None:
        ok &= years >= filters["year_min"]
    if filters.get("year_max") is not None:
        ok &= (years <= filters["year_max"]) & (years > 0)
    if filters.get("min_rating") is not None:
        # The UI shows IMDb ratings, so filter on those (TMDB only where IMDb has none).
        ok &= _effective_ratings(df, ratings) >= filters["min_rating"]
    return ok


def _has_filters(filters):
    return bool(filters) and any(v not in (None, [], ()) for v in filters.values())


def get_meta():
    """Facts the UI needs to build its filters."""
    (_, _, df), _ = _load_model()
    years, _, genres = _film_meta(df)
    known = years[years > 0]
    names = sorted({g for gs in genres for g in gs if not g.startswith("(")})
    return {
        "genres": names,
        "year_min": int(known.min()) if known.size else None,
        "year_max": int(known.max()) if known.size else None,
        "n_movies": len(df),
    }


# ---------------------------------------------------------------------------
# Recommending
# ---------------------------------------------------------------------------


def _resolve_all(titles, title_to_index):
    """Resolve each title; returns (indices, unresolved titles), de-duplicated, order kept."""
    indices, unresolved = [], []
    for t in titles:
        key = resolve_title(t, title_to_index)
        if key is None:
            unresolved.append(t)
        elif title_to_index[key] not in indices:
            indices.append(title_to_index[key])
    return indices, unresolved


def _neighbor_rerank(idx, top_n, rerank, df, matrix):
    """Fast path for one liked film and no filters: precomputed neighbors, then rating rerank."""
    candidates = _neighbor_table[idx]
    if rerank and "vote_average" in df.columns:
        sims = cosine_similarity(matrix[idx], matrix[candidates]).flatten()
        ratings = df.iloc[candidates]["vote_average"].fillna(0).to_numpy()
        scores = sims * np.power(np.maximum(ratings, 0), RERANK_BETA)
        return candidates[scores.argsort()[-top_n:][::-1]]
    return candidates[:top_n]


def _profile_rank(like_idx, dislike_idx, allowed, top_n, rerank, df, matrix):
    """General path: score every film against the taste profile, keep allowed ones, rerank."""
    sims = profile_scores(matrix, like_idx, dislike_idx)
    cand = np.flatnonzero(allowed)
    if cand.size == 0:
        return cand
    order = cand[np.argsort(-sims[cand], kind="stable")]
    if rerank and "vote_average" in df.columns:
        pool = order[:RERANK_POOL]
        ratings = df["vote_average"].fillna(0).to_numpy(dtype=float)[pool]
        scores = np.maximum(sims[pool], 0) * np.power(np.maximum(ratings, 0), RERANK_BETA)
        return pool[np.argsort(-scores, kind="stable")[:top_n]]
    return order[:top_n]


EXTRA_FIELDS = ("imdb_id", "imdb_rating", "imdb_votes", "tmdb_votes", "runtime")


def _extras_for(row):
    movie_id = _num(row.get("movieId"), int)
    info = _extras.get(movie_id, {}) if movie_id is not None else {}
    return {field: info.get(field) for field in EXTRA_FIELDS}


def _movie_dict(row, genres):
    tmdb_id = _clean_value(row, "id")
    release_date = _clean_value(row, "release_date")
    vote_average = _clean_value(row, "vote_average")
    return {
        "title": row["title"],
        "poster": get_poster(int(tmdb_id) if tmdb_id is not None else None),
        "release_date": str(release_date) if release_date is not None else None,
        "vote_average": float(vote_average) if vote_average is not None else None,
        "tmdb_id": int(tmdb_id) if tmdb_id is not None else None,
        "genres": sorted(g for g in genres if not g.startswith("(")),
        **_extras_for(row),
    }


def recommend(likes, dislikes=(), exclude=(), filters=None, top_n=10, rerank=True):
    """Recommend films for a taste profile.

    likes:    titles the user likes (at least one). The profile is their average.
    dislikes: titles to steer away from.
    exclude:  titles already seen; never recommended.
    filters:  dict with genres / exclude_genres / year_min / year_max / min_rating.

    Raises TitleNotFound if a liked title is not in the catalog. Unknown dislikes/excludes are
    reported in `ignored`, not treated as errors.
    """
    t0 = time.perf_counter()
    (_, matrix, df), title_to_index = _load_model()

    like_idx, missing = _resolve_all(likes, title_to_index)
    if missing:
        raise TitleNotFound(missing)
    if not like_idx:
        raise TitleNotFound([])

    dislike_idx, ignored_d = _resolve_all(dislikes, title_to_index)
    exclude_idx, ignored_e = _resolve_all(exclude, title_to_index)
    dislike_idx = [i for i in dislike_idx if i not in like_idx]  # a like wins over a dislike

    corrections = []
    for query in likes:
        key = resolve_title(query, title_to_index)
        matched = df.iloc[title_to_index[key]]["title"]
        if not same_title(query, matched):
            corrections.append({"query": query.strip(), "matched": matched})

    simple = (
        len(like_idx) == 1
        and not dislike_idx
        and not exclude_idx
        and not _has_filters(filters)
        and _neighbor_table is not None
    )
    if simple:
        chosen = _neighbor_rerank(like_idx[0], top_n, rerank, df, matrix)
    else:
        allowed = _allowed_mask(df, filters)
        allowed[like_idx + dislike_idx + exclude_idx] = False
        chosen = _profile_rank(like_idx, dislike_idx, allowed, top_n, rerank, df, matrix)

    _, _, genres = _film_meta(df)
    recommendations = [_movie_dict(df.iloc[i], genres[i]) for i in chosen]

    def titles_of(indices):
        return [df.iloc[i]["title"] for i in indices]

    result = {
        "matched": df.iloc[like_idx[0]]["title"],
        "likes": titles_of(like_idx),
        "dislikes": titles_of(dislike_idx),
        "ignored": ignored_d + ignored_e,
        "corrections": corrections,
        "liked": [_movie_dict(df.iloc[i], genres[i]) for i in like_idx],
        "recommendations": recommendations,
    }
    logger.info(
        "recommend likes=%d dislikes=%d exclude=%d filtered=%s fast=%s -> %d in %.3fs",
        len(like_idx),
        len(dislike_idx),
        len(exclude_idx),
        _has_filters(filters),
        simple,
        len(recommendations),
        time.perf_counter() - t0,
    )
    return result


def recommend_movies(movie_title, top_n=10, rerank=True):
    """Single-film convenience wrapper: returns (matched title, recommendations)."""
    result = recommend([movie_title], top_n=top_n, rerank=rerank)
    return result["matched"], result["recommendations"]


def search_titles(query, limit=8):
    """Autocomplete: catalog titles matching what the user has typed so far.

    Prefix matches first (shortest first), then substring matches, then close fuzzy matches.
    """
    (_, _, df), title_to_index = _load_model()
    qkey = _key(_YEAR_RE.sub("", query.strip().lower()))
    if not qkey:
        return []
    index = _alias_index(tuple(title_to_index))

    starts = sorted((k for k in index if k.startswith(qkey)), key=len)
    contains = sorted((k for k in index if qkey in k and not k.startswith(qkey)), key=len)
    keys = starts + contains
    if len(keys) < limit and len(qkey) >= 4:
        keys += [
            k for k in get_close_matches(qkey, index.keys(), n=limit, cutoff=0.7) if k not in keys
        ]

    out, seen = [], set()
    for k in keys:
        for _year, full in index[k]:
            if full not in seen:
                seen.add(full)
                out.append(df.iloc[title_to_index[full]]["title"])
                if len(out) >= limit:
                    return out
    return out


def popular_movies(genre=None, n=18):
    """The best-known films (most TMDB votes), optionally within one genre.

    Used for the cold-start screen, before the user has told us anything. Only films with a
    poster are returned, so the picker never shows an empty tile.
    """
    (_, _, df), _ = _load_model()
    _, tmdb_ratings, genres = _film_meta(df)
    if "movieId" in df.columns:
        votes = np.array(
            [(_extras.get(_num(m, int)) or {}).get("tmdb_votes") or 0 for m in df["movieId"]]
        )
    else:
        votes = np.zeros(len(df))
    # Without vote counts fall back to the rating so the order is still meaningful.
    order = np.lexsort((-tmdb_ratings, -votes))

    wanted = genre.lower() if genre else None
    out = []
    for i in order:
        if wanted and wanted not in genres[i]:
            continue
        movie = _movie_dict(df.iloc[i], genres[i])
        if movie["poster"] is None:
            continue
        out.append(movie)
        if len(out) >= n:
            break
    return out
