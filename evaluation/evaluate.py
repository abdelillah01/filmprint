#!/usr/bin/env python3
"""
Leave-one-out evaluation of movie recommender models.

Usage (from anywhere):
    python scripts/download_data.py      # once: fetches ratings.csv
    python evaluation/evaluate.py [--no-embeddings]

Outputs:
    evaluation/results.md

Protocol (no leakage):
  * Users are split 50/50 into TRAIN and TEST users (seeded).
  * Models that learn from ratings (collaborative filtering, popularity) see ONLY the
    TRAIN users' ratings. Content models use no ratings at all.
  * Every model is scored on the same TEST users. For each test user, SEEDS_PER_USER
    highly-rated catalog movies are held out one at a time as the query; the user's
    other highly-rated catalog movies are the relevant set.
  * Metrics are averaged per user first, then across users, with a 95% CI over users.
"""

import argparse
import importlib.util
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "ML-model" / "data"
PROCESSED = DATA_DIR / "processed" / "merged_movies.csv"
RAW = DATA_DIR / "raw"
RESULTS_MD = Path(__file__).resolve().parent / "results.md"

sys.path.insert(0, str(REPO_ROOT / "ML-model"))
from recommender.data_loader import add_metadata_column  # noqa: E402

K = 10
MIN_RATINGS = 20
HIGH_RATING = 4.0
SEEDS_PER_USER = 5
TEST_FRACTION = 0.5
SEED = 42
RERANK_BETA = 0.3  # must match backend/recommender_app/utils.py
RERANK_POOL = 50  # must match the neighbor table width used by the API


# ---------------------------------------------------------------------------
# Data and protocol
# ---------------------------------------------------------------------------


def load_data():
    movies = pd.read_csv(PROCESSED)
    movies = movies.dropna(subset=["combined", "title"]).reset_index(drop=True)
    movies = add_metadata_column(movies)
    ratings_path = RAW / "ratings.csv"
    if not ratings_path.exists():
        sys.exit(f"{ratings_path} not found - run: python scripts/download_data.py")
    ratings = pd.read_csv(ratings_path)
    mid_to_idx = {int(m): i for i, m in enumerate(movies["movieId"])}
    ratings = ratings[ratings["movieId"].isin(mid_to_idx)].copy()
    ratings["idx"] = ratings["movieId"].map(mid_to_idx)
    return movies, ratings


def split_users(ratings):
    """Seeded user-level split. Returns (train_ratings, test_cases)."""
    rng = np.random.default_rng(SEED)
    users = np.array(sorted(ratings["userId"].unique()))
    rng.shuffle(users)
    n_test = int(len(users) * TEST_FRACTION)
    test_users, train_users = set(users[:n_test]), set(users[n_test:])

    train = ratings[ratings["userId"].isin(train_users)]

    cases = {}  # user -> [(seed_idx, set(relevant_idx)), ...]
    counts = ratings.groupby("userId").size()
    for uid in sorted(test_users):
        if counts[uid] < MIN_RATINGS:
            continue
        ur = ratings[ratings["userId"] == uid]
        liked = sorted(set(ur[ur["rating"] >= HIGH_RATING]["idx"]))
        if len(liked) < 2:
            continue
        seeds = rng.choice(liked, size=min(SEEDS_PER_USER, len(liked)), replace=False)
        cases[uid] = [(int(s), set(liked) - {int(s)}) for s in seeds]
    return train, cases


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _ap(hits, n_relevant):
    s, found = 0.0, 0
    for k, h in enumerate(hits, 1):
        if h:
            found += 1
            s += found / k
    return s / min(n_relevant, K)


def _ndcg(hits, n_relevant):
    dcg = sum(h / np.log2(k + 1) for k, h in enumerate(hits, 1))
    ideal = sum(1.0 / np.log2(k + 1) for k in range(1, min(n_relevant, K) + 1))
    return dcg / ideal


def top_k(scores, seed_idx):
    s = np.array(scores, dtype=float)
    s[seed_idx] = -np.inf
    return np.argsort(-s, kind="stable")[:K].tolist()


def mean_ci(values):
    v = np.asarray(values, dtype=float)
    half = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return float(v.mean()), float(half)


def evaluate(scores_fn, cases, n_movies):
    per_user = {"p": [], "r": [], "ap": [], "ndcg": []}
    recommended = set()
    for seeds in cases.values():
        acc = {"p": [], "r": [], "ap": [], "ndcg": []}
        for seed_idx, relevant in seeds:
            rec = top_k(scores_fn(seed_idx), seed_idx)
            recommended.update(rec)
            hits = [1 if i in relevant else 0 for i in rec]
            acc["p"].append(sum(hits) / K)
            acc["r"].append(sum(hits) / len(relevant))
            acc["ap"].append(_ap(hits, len(relevant)))
            acc["ndcg"].append(_ndcg(hits, len(relevant)))
        for key, vals in acc.items():
            per_user[key].append(np.mean(vals))
    ndcg, ndcg_ci = mean_ci(per_user["ndcg"])
    return {
        "P@10": float(np.mean(per_user["p"])),
        "R@10": float(np.mean(per_user["r"])),
        "MAP@10": float(np.mean(per_user["ap"])),
        "NDCG@10": ndcg,
        "NDCG_ci": ndcg_ci,
        "coverage": len(recommended) / n_movies,
        "n_users": len(per_user["ndcg"]),
    }


def beyond_accuracy(scores_fn, ref_matrix, movies, ratings, n_samples=300):
    """Diversity is always measured in ONE shared feature space so rows are comparable."""
    rng = np.random.default_rng(SEED)
    pop = ratings.groupby("idx").size() / ratings["userId"].nunique()
    divs, novs = [], []
    for seed_idx in rng.choice(len(movies), size=min(n_samples, len(movies)), replace=False):
        rec = top_k(scores_fn(int(seed_idx)), int(seed_idx))
        sims = cosine_similarity(ref_matrix[rec])
        divs.append(1 - sims[np.triu_indices(len(rec), k=1)].mean())
        novs.append(np.mean([-np.log2(pop.get(i, 1e-9)) for i in rec]))
    return {"diversity": float(np.mean(divs)), "novelty": float(np.mean(novs))}


# ---------------------------------------------------------------------------
# Models: each returns scores_fn(seed_idx) -> score per catalog movie (higher = better)
# ---------------------------------------------------------------------------


def popularity_most_liked(train, n_movies):
    scores = np.zeros(n_movies)
    liked = train[train["rating"] >= HIGH_RATING].groupby("idx").size()
    scores[liked.index.to_numpy()] = liked.to_numpy()
    return lambda seed_idx: scores


def popularity_top_rated(train, n_movies, min_votes=10):
    scores = np.zeros(n_movies)
    g = train.groupby("idx")["rating"].agg(["mean", "count"])
    g = g[g["count"] >= min_votes]
    scores[g.index.to_numpy()] = g["mean"].to_numpy()
    return lambda seed_idx: scores


def tfidf_model(movies, column, **kwargs):
    kwargs.setdefault("max_features", 10_000)
    mat = TfidfVectorizer(stop_words="english", **kwargs).fit_transform(movies[column].fillna(""))
    return (lambda seed_idx: linear_kernel(mat[seed_idx], mat).ravel()), mat


def rerank_model(movies, mat):
    """What the API actually serves: top-50 by cosine, re-ranked by sim x vote_average^beta."""
    rating = np.maximum(movies["vote_average"].fillna(0).to_numpy(float), 0) ** RERANK_BETA

    def scores(seed_idx):
        sims = linear_kernel(mat[seed_idx], mat).ravel()
        sims[seed_idx] = -1
        pool = np.argsort(-sims, kind="stable")[:RERANK_POOL]
        out = np.full(len(sims), -np.inf)
        out[pool] = sims[pool] * rating[pool]
        return out

    return scores


def dense_model(embeddings):
    return lambda seed_idx: embeddings @ embeddings[seed_idx]


def svd_model(tfidf_matrix, n_components=200):
    reduced = normalize(TruncatedSVD(n_components, random_state=SEED).fit_transform(tfidf_matrix))
    return dense_model(reduced)


def item_cf_model(train, n_movies):
    """Item-item cosine over binary 'liked' (>= 4.0) from TRAIN users only."""
    liked = train[train["rating"] >= HIGH_RATING]
    uid_map = {u: i for i, u in enumerate(sorted(liked["userId"].unique()))}
    rows = liked["userId"].map(uid_map).to_numpy()
    ui = csr_matrix(
        (np.ones(len(liked)), (rows, liked["idx"].to_numpy())), shape=(len(uid_map), n_movies)
    )
    ui_csc, iu = ui.tocsc(), ui.T.tocsr()
    norms = np.sqrt(np.asarray(ui.power(2).sum(axis=0)).ravel())

    def scores(seed_idx):
        raw = (iu @ ui_csc[:, seed_idx]).toarray().ravel()
        denom = norms * norms[seed_idx]
        return np.divide(raw, denom, out=np.zeros_like(raw), where=denom > 0)

    return scores


def hybrid_model(content_fn, cf_fn, alpha):
    """alpha * content + (1 - alpha) * CF, each min-max normalised per query."""

    def norm(v):
        v = np.asarray(v, dtype=float)
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)

    return lambda seed_idx: alpha * norm(content_fn(seed_idx)) + (1 - alpha) * norm(cf_fn(seed_idx))


def sentence_embedding_model(movies):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = (movies["title"].fillna("") + " " + movies["combined"].fillna("")).tolist()
    print("  Encoding with sentence-transformers...", flush=True)
    return dense_model(normalize(model.encode(texts, show_progress_bar=False, batch_size=64)))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def significant(a, b):
    """Crude check: do the 95% CIs of two NDCG values fail to overlap?"""
    return abs(a["NDCG@10"] - b["NDCG@10"]) > (a["NDCG_ci"] ** 2 + b["NDCG_ci"] ** 2) ** 0.5


def pct(new, old):
    return (new - old) / max(old, 1e-9) * 100


def beyond(a, b):
    return "beyond" if significant(a, b) else "within"


def interpretation(res):
    def ndcg(name):
        return res[name]["NDCG@10"]

    content = [n for n in res if n.startswith("M") and n[1] in "12345"]
    best = max(res, key=ndcg)
    best_content = max(content, key=ndcg)
    out = [f"**Best overall**: {best} (NDCG@10 = {ndcg(best):.4f})."]
    if best.startswith("Popularity"):
        out.append(
            "**Caveat - popularity wins this protocol.** A list that ignores the query and "
            f"always returns the same few well-liked films ({res[best]['coverage']:.2%} of the "
            "catalog) outscores every query-aware model. Held-out 'also liked' items are "
            "dominated by globally popular films, so this metric mostly rewards popularity. It "
            "does not measure what the product promises - films similar to the one you typed - "
            "and the NDCG numbers below should be read as 'agreement with what users liked', "
            "not 'quality of similarity'. Popularity-corrected metrics (e.g. NDCG with items "
            "weighted by inverse popularity) would be the next step."
        )

    m1, m2, m3 = "M1: TF-IDF metadata", "M2: TF-IDF + overview", "M3: TF-IDF tuned"
    pop = max((n for n in res if n.startswith("Popularity")), key=ndcg)
    beats = significant(res[m1], res[pop]) and ndcg(m1) > ndcg(pop)
    out.append(
        f"**Baseline check**: the strongest popularity baseline ({pop}) reaches NDCG@10 = "
        f"{ndcg(pop):.4f}; M1 reaches {ndcg(m1):.4f}. "
        + (
            "The content model beats the popularity baseline beyond noise."
            if beats
            else "The content model does NOT clearly beat popularity - treat its signal with caution."
        )
    )

    out.append(
        f"**Overview text (M1 -> M2)**: NDCG@10 {ndcg(m1):.4f} -> {ndcg(m2):.4f} "
        f"({pct(ndcg(m2), ndcg(m1)):+.1f}%), "
        + (
            "a difference larger than the confidence intervals."
            if significant(res[m1], res[m2])
            else "within the confidence intervals, i.e. not distinguishable from noise."
        )
        + " The production model uses metadata only (M1)."
    )
    out.append(
        f"**Tuned TF-IDF (M3)** vs M1: {ndcg(m3):.4f} vs {ndcg(m1):.4f} "
        f"({pct(ndcg(m3), ndcg(m1)):+.1f}%, {beyond(res[m3], res[m1])} noise)."
    )
    m4 = "M4: SVD/LSA"
    out.append(
        f"**SVD/LSA (M4)** built on M3: {ndcg(m4):.4f} vs {ndcg(m3):.4f} "
        f"({pct(ndcg(m4), ndcg(m3)):+.1f}%), diversity {res[m4]['diversity']:.3f} vs "
        f"{res[m3]['diversity']:.3f}."
    )
    m5 = "M5: Sentence embeddings"
    if m5 in res:
        out.append(
            f"**Sentence embeddings (M5)**: {ndcg(m5):.4f} vs {ndcg(m1):.4f} for M1 "
            f"({pct(ndcg(m5), ndcg(m1)):+.1f}%, {beyond(res[m5], res[m1])} noise)."
        )
    prod = "Production: M1 + rating rerank"
    out.append(
        f"**Production rerank**: sim x vote_average^{RERANK_BETA} over the top-{RERANK_POOL} "
        f"gives {ndcg(prod):.4f} vs {ndcg(m1):.4f} for plain M1 "
        f"({pct(ndcg(prod), ndcg(m1)):+.1f}%, {beyond(res[prod], res[m1])} noise)."
    )

    cf = "M6: Item-item CF"
    out.append(
        f"**Collaborative filtering (M6)**, trained on train users only: {ndcg(cf):.4f} vs "
        f"{ndcg(best_content):.4f} for the best content model ({best_content}), "
        f"{pct(ndcg(cf), ndcg(best_content)):+.0f}%. Because CF never sees the test users, "
        "this gap is an out-of-sample result, not leakage."
    )
    hybrids = [n for n in res if n.startswith("M7")]
    bh = max(hybrids, key=ndcg)
    adds_value = ndcg(bh) > max(ndcg(cf), ndcg(best_content)) and significant(res[bh], res[cf])
    out.append(
        f"**Hybrid (M7)**: best blend is {bh} at {ndcg(bh):.4f} vs CF alone {ndcg(cf):.4f} "
        f"({beyond(res[bh], res[cf])} noise). "
        + (
            "Blending adds value over either signal alone."
            if adds_value
            else "Blending does not measurably beat the better single signal."
        )
    )

    by_div = sorted(res, key=lambda n: res[n]["diversity"])
    by_cov = sorted(res, key=lambda n: res[n]["coverage"])
    out.append(
        f"**Diversity / coverage** (diversity measured in the shared M1 feature space): "
        f"lowest diversity {by_div[0]} ({res[by_div[0]]['diversity']:.3f}), "
        f"highest {by_div[-1]} ({res[by_div[-1]]['diversity']:.3f}); "
        f"lowest coverage {by_cov[0]} ({res[by_cov[0]]['coverage']:.1%}), "
        f"highest {by_cov[-1]} ({res[by_cov[-1]]['coverage']:.1%})."
    )
    return out


# ---------------------------------------------------------------------------
# Personalisation: does a richer taste profile beat a single query film?
# ---------------------------------------------------------------------------

PERSONALIZATION_DRAWS = 3  # independent draws per user
PROFILE_SEEDS = 5  # liked films revealed per draw (a row may use only the first k)
MAX_DISLIKES = 3
LOW_RATING = 2.5  # ratings at or below this count as dislikes


def load_profile_scores():
    """Load the production scoring function straight from the backend source file."""
    path = REPO_ROOT / "backend" / "recommender_app" / "profile.py"
    spec = importlib.util.spec_from_file_location("profile_scoring", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.profile_scores


def personalization_cases(ratings, users):
    """Per user: draws of (5 revealed likes, held-out relevant set, dislikes, watched films).

    The relevant set is the user's other liked films and is the same for every row, so rows
    with different numbers of likes are directly comparable. Revealed likes that a row does
    not use are simply never recommended, so they neither help nor hurt it.
    """
    rng = np.random.default_rng(SEED + 1)
    cases = {}
    for uid in sorted(users):
        ur = ratings[ratings["userId"] == uid]
        liked = sorted(set(ur[ur["rating"] >= HIGH_RATING]["idx"]))
        if len(liked) < PROFILE_SEEDS + 1:
            continue
        disliked = sorted(set(ur[ur["rating"] <= LOW_RATING]["idx"]))
        rated = set(ur["idx"])
        draws = []
        for _ in range(PERSONALIZATION_DRAWS):
            seeds = [int(x) for x in rng.choice(liked, PROFILE_SEEDS, replace=False)]
            relevant = set(liked) - set(seeds)
            dislikes = (
                [int(x) for x in rng.choice(disliked, min(MAX_DISLIKES, len(disliked)), False)]
                if disliked
                else []
            )
            draws.append(
                {
                    "seeds": seeds,
                    "relevant": relevant,
                    "dislikes": dislikes,
                    "watched": rated - relevant,  # everything they have rated except the held-out
                }
            )
        cases[uid] = draws
    return cases


def profile_ranker(matrix, profile_scores, k, use_dislikes=False, hide_watched=False):
    def rank(draw):
        likes = draw["seeds"][:k]
        dislikes = draw["dislikes"] if use_dislikes else []
        scores = profile_scores(matrix, likes, dislikes)
        banned = set(draw["seeds"]) | set(dislikes)
        if hide_watched:
            banned |= draw["watched"]
        scores[list(banned)] = -np.inf
        return np.argsort(-scores, kind="stable")[:K].tolist()

    return rank


def popularity_ranker(train, n_movies, hide_watched=False):
    base = popularity_most_liked(train, n_movies)(0)

    def rank(draw):
        scores = base.astype(float).copy()
        banned = set(draw["seeds"])
        if hide_watched:
            banned |= draw["watched"]
        scores[list(banned)] = -np.inf
        return np.argsort(-scores, kind="stable")[:K].tolist()

    return rank


def evaluate_cases(rank_fn, cases):
    per_user = {"p": [], "r": [], "ndcg": []}
    for draws in cases.values():
        acc = {"p": [], "r": [], "ndcg": []}
        for d in draws:
            hits = [1 if i in d["relevant"] else 0 for i in rank_fn(d)]
            acc["p"].append(sum(hits) / K)
            acc["r"].append(sum(hits) / len(d["relevant"]))
            acc["ndcg"].append(_ndcg(hits, len(d["relevant"])))
        for key, vals in acc.items():
            per_user[key].append(float(np.mean(vals)))
    ndcg, ci = mean_ci(per_user["ndcg"])
    return {
        "P@10": float(np.mean(per_user["p"])),
        "R@10": float(np.mean(per_user["r"])),
        "NDCG@10": ndcg,
        "NDCG_ci": ci,
        "ndcg_users": per_user["ndcg"],
        "n_users": len(per_user["ndcg"]),
    }


def personalization_section(ratings, train, test_cases, matrix, n_movies):
    profile_scores = load_profile_scores()
    cases = personalization_cases(ratings, test_cases.keys())

    setups = {
        "1 liked film (baseline)": profile_ranker(matrix, profile_scores, 1),
        "3 liked films": profile_ranker(matrix, profile_scores, 3),
        "5 liked films": profile_ranker(matrix, profile_scores, 5),
        "3 liked + dislikes": profile_ranker(matrix, profile_scores, 3, use_dislikes=True),
        "3 liked + dislikes + hide watched": profile_ranker(
            matrix, profile_scores, 3, use_dislikes=True, hide_watched=True
        ),
        "5 liked + dislikes + hide watched": profile_ranker(
            matrix, profile_scores, 5, use_dislikes=True, hide_watched=True
        ),
        "Popularity: most liked": popularity_ranker(train, n_movies),
        "Popularity: most liked + hide watched": popularity_ranker(
            train, n_movies, hide_watched=True
        ),
    }
    res = {name: evaluate_cases(fn, cases) for name, fn in setups.items()}
    base = res["1 liked film (baseline)"]
    base_users = np.asarray(base["ndcg_users"])

    lines = [
        "## Personalization lift",
        "",
        "Does telling the model more about the user help? Same held-out users and the same "
        "relevant films for every row (content model M1, no rating rerank). "
        f"{next(iter(res.values()))['n_users']} users x {PERSONALIZATION_DRAWS} draws.",
        "",
        "| Setup | P@10 | R@10 | NDCG@10 (95% CI) | Δ NDCG vs 1 film (paired 95% CI) |",
        "|-------|------|------|------------------|-----------------------------------|",
    ]
    for name, r in res.items():
        if name == "1 liked film (baseline)":
            delta = "n/a"
        else:
            d, half = mean_ci(np.asarray(r["ndcg_users"]) - base_users)
            delta = f"{d:+.4f} ± {half:.4f}" + (" *" if abs(d) > half else "")
        lines.append(
            f"| {name} | {r['P@10']:.4f} | {r['R@10']:.4f} "
            f"| **{r['NDCG@10']:.4f}** ± {r['NDCG_ci']:.4f} | {delta} |"
        )
    lines += ["", "\\* the paired 95% CI excludes zero.", ""]

    def step(name, versus):
        """Paired comparison of one setup against the one it builds on."""
        a, b = res[name], res[versus]
        d, half = mean_ci(np.asarray(a["ndcg_users"]) - np.asarray(b["ndcg_users"]))
        verdict = "a real gain" if d > half else ("a real loss" if d < -half else "within noise")
        return (
            f"- **{name}** vs {versus}: NDCG@10 {a['NDCG@10']:.4f} vs {b['NDCG@10']:.4f} "
            f"({pct(a['NDCG@10'], b['NDCG@10']):+.1f}%), {verdict}."
        )

    one = "1 liked film (baseline)"
    out = [
        step("3 liked films", one),
        step("5 liked films", "3 liked films"),
        step("3 liked + dislikes", "3 liked films"),
        step("3 liked + dislikes + hide watched", "3 liked + dislikes"),
    ]
    pop, pop_hide = (
        res["Popularity: most liked"]["NDCG@10"],
        res["Popularity: most liked + hide watched"]["NDCG@10"],
    )
    out.append(
        f"- Hiding watched films also lifts the popularity baseline ({pop:.4f} -> {pop_hide:.4f}), "
        "so part of that gain is not about taste."
    )
    best = res["5 liked + dislikes + hide watched"]["NDCG@10"]
    out.append(
        f"- Even the richest profile ({best:.4f}) stays far below query-ignoring popularity "
        f"({pop_hide:.4f}) under this protocol; see the caveat above about what the metric rewards."
    )

    lines += out + [
        "",
        "**Protocol.** For each test user, 5 liked films are revealed per draw and the user's "
        "*other* liked films are the relevant set (identical for every row). A row using fewer "
        "than 5 likes never recommends the unused revealed ones. *Dislikes* are up to 3 films the "
        f"user rated <= {LOW_RATING}. *Hide watched* removes every film the user rated except the "
        "held-out relevant ones, which is what a real user's own history would remove. The "
        "scoring function is `backend/recommender_app/profile.py`, loaded directly, so this "
        "measures the production code. Filters and in-session thumbs feedback have no offline "
        "ground truth and are not measured here.",
        "",
    ]
    return lines


def run(use_embeddings=True):
    print("Loading data...", flush=True)
    movies, ratings = load_data()
    n = len(movies)
    train, cases = split_users(ratings)
    print(
        f"  {n} catalog movies | {len(ratings)} ratings | "
        f"{train['userId'].nunique()} train users | {len(cases)} test users",
        flush=True,
    )

    models = {}
    models["Popularity: most liked"] = popularity_most_liked(train, n)
    models["Popularity: top rated (>=10 votes)"] = popularity_top_rated(train, n)

    fn_m1, mat_m1 = tfidf_model(movies, "metadata")
    models["M1: TF-IDF metadata"] = fn_m1
    models["Production: M1 + rating rerank"] = rerank_model(movies, mat_m1)
    fn_m2, _ = tfidf_model(movies, "combined")
    models["M2: TF-IDF + overview"] = fn_m2
    fn_m3, mat_m3 = tfidf_model(
        movies, "combined", ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20_000
    )
    models["M3: TF-IDF tuned"] = fn_m3
    models["M4: SVD/LSA"] = svd_model(mat_m3)
    if use_embeddings:
        try:
            models["M5: Sentence embeddings"] = sentence_embedding_model(movies)
        except ImportError:
            print("M5 skipped: sentence-transformers not installed", flush=True)
    fn_cf = item_cf_model(train, n)
    models["M6: Item-item CF"] = fn_cf
    for alpha in (0.2, 0.5, 0.8):
        models[f"M7: Hybrid M3+CF (alpha={alpha})"] = hybrid_model(fn_m3, fn_cf, alpha)

    res = {}
    for name, fn in models.items():
        print(f"  Evaluating {name}...", flush=True)
        res[name] = {
            **evaluate(fn, cases, n),
            **beyond_accuracy(fn, mat_m1, movies, ratings),
        }

    n_users = next(iter(res.values()))["n_users"]
    lines = [
        "# Evaluation Results",
        "",
        f"Generated: {date.today()}  |  Catalog: {n} movies  |  Test users: {n_users} "
        f"(x{SEEDS_PER_USER} held-out seeds each)  |  Train users: {train['userId'].nunique()}",
        "",
        "## Accuracy metrics",
        "",
        "| Model | P@10 | R@10 | MAP@10 | NDCG@10 (95% CI) | Coverage | Diversity | Novelty |",
        "|-------|------|------|--------|------------------|----------|-----------|---------|",
    ]
    for name, r in res.items():
        lines.append(
            f"| {name} | {r['P@10']:.4f} | {r['R@10']:.4f} | {r['MAP@10']:.4f} "
            f"| **{r['NDCG@10']:.4f}** ± {r['NDCG_ci']:.4f} | {r['coverage']:.2%} "
            f"| {r['diversity']:.4f} | {r['novelty']:.2f} |"
        )

    lines += ["", "## Interpretation", ""]
    for para in interpretation(res):
        lines += [para, ""]

    print("Evaluating personalisation...", flush=True)
    lines += personalization_section(ratings, train, cases, mat_m1, n)
    lines += [
        "## Evaluation methodology",
        "",
        f"- **Split**: users are shuffled (seed {SEED}) and split {int(TEST_FRACTION * 100)}/"
        f"{int((1 - TEST_FRACTION) * 100)} into test and train users. Collaborative filtering "
        "and the popularity baselines are built from **train users only**; content models use "
        "no ratings. All models are scored on the same test users, so no model sees the "
        "ratings it is graded against.",
        f"- **Protocol**: for each test user with >={MIN_RATINGS} ratings and >=2 liked catalog "
        f"movies, up to {SEEDS_PER_USER} liked movies (>={HIGH_RATING} stars) are held out one at "
        "a time as the query; the user's other liked catalog movies are the relevant set. "
        f"The model returns its top-{K}, excluding the query.",
        "- **Aggregation**: metrics are averaged over a user's seeds, then over users. "
        "The ± is a 95% normal-approximation CI over users; differences smaller than the "
        "combined CIs are reported as noise.",
        "- **Metrics**: Precision/Recall/MAP/NDCG@10; catalog coverage (fraction of the library "
        "ever recommended); intra-list diversity (1 - mean pairwise cosine similarity within a "
        "list, always in the M1 TF-IDF space so models are comparable, 300 sampled queries); "
        "novelty (mean -log2 popularity; higher = more obscure).",
        "- **Baselines**: *most liked* ranks by the number of train-user ratings >= 4.0; "
        "*top rated* ranks by mean rating among films with >=10 train-user votes. Both ignore "
        "the query.",
        "- **Hybrid**: alpha x content + (1 - alpha) x CF over min-max normalised full score "
        "vectors. All three alphas are reported; none was selected on the test users.",
        f"- **Production row**: replicates what the API serves (M1 top-{RERANK_POOL} "
        f"neighbors re-ranked by sim x vote_average^{RERANK_BETA}).",
        "",
        "## Known limitations",
        "",
        "- Small catalog (MovieLens small ∩ TMDB 5000) and only ~610 MovieLens users; CF is "
        "trained on half of them, so its absolute numbers would improve with more data.",
        "- Relevance is 'the user also liked it', which favours popular, well-connected films "
        "(a query-independent popularity list is a very strong baseline here) and rewards CF "
        "structurally; it does not measure how good a *content-similar* list feels (e.g. "
        "sequels are trivially similar but may not be what a user wants).",
        "- Seeds are drawn from each user's liked films, so queries skew towards popular titles.",
        "- Queries outside the catalog fall back to title matching, which the API now "
        "normalises and reports via the `query` field when it had to correct a typo.",
    ]
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nResults written to {RESULTS_MD}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embeddings", action="store_true", help="skip M5 (needs a model download)")
    run(use_embeddings=not ap.parse_args().no_embeddings)
