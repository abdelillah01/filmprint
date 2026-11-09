"""
Build per-film facts for the catalog from data already in the repo (no network).

Usage (from anywhere):
    python scripts/build_film_stats.py

Reads  : ML-model/data/processed/merged_movies.csv   (the catalog: movieId, TMDB id)
         ML-model/data/raw/links.csv                  (MovieLens movieId -> IMDb id)
         ML-model/data/raw/tmdb_5000_movies.csv       (runtime, vote_count)
Writes : ML-model/data/processed/film_stats.csv
         columns: movieId, imdb_id (e.g. tt0114709), runtime (minutes), tmdb_votes

The API merges this file in at startup. It holds identifiers and TMDB facts only, so it is
safe to commit; IMDb *ratings* come from scripts/fetch_imdb.py and are kept out of git.
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "ML-model" / "data"
CATALOG = DATA / "processed" / "merged_movies.csv"
LINKS = DATA / "raw" / "links.csv"
TMDB_MOVIES = DATA / "raw" / "tmdb_5000_movies.csv"
OUT = DATA / "processed" / "film_stats.csv"


def main() -> None:
    catalog = pd.read_csv(CATALOG, usecols=["movieId", "id"]).dropna()
    catalog["id"] = catalog["id"].astype(int)

    links = pd.read_csv(LINKS, usecols=["movieId", "imdbId"])
    links["imdb_id"] = links["imdbId"].apply(lambda v: f"tt{int(v):07d}" if pd.notna(v) else None)

    tmdb = pd.read_csv(TMDB_MOVIES, usecols=["id", "runtime", "vote_count"])
    tmdb = tmdb.rename(columns={"vote_count": "tmdb_votes"})

    stats = (
        catalog.merge(links[["movieId", "imdb_id"]], on="movieId", how="left")
        .merge(tmdb, on="id", how="left")
        .drop(columns=["id"])
    )
    # 0 minutes / 0 votes mean "unknown" in the TMDB dump
    stats["runtime"] = stats["runtime"].where(stats["runtime"] > 0).astype("Int64")
    stats["tmdb_votes"] = stats["tmdb_votes"].where(stats["tmdb_votes"] > 0).astype("Int64")

    stats.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(stats)} films)")
    print(f"  with IMDb id : {stats['imdb_id'].notna().sum()}")
    print(f"  with runtime : {stats['runtime'].notna().sum()}")
    print(f"  with votes   : {stats['tmdb_votes'].notna().sum()}")


if __name__ == "__main__":
    main()
