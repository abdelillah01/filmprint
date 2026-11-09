"""
Fetch IMDb ratings for the catalog from IMDb's official free dataset.

Usage (from anywhere):
    python scripts/build_film_stats.py     # once: creates the IMDb ids this needs
    python scripts/fetch_imdb.py

Downloads https://datasets.imdbws.com/title.ratings.tsv.gz (about 7 MB), keeps only the rows
for films in this catalog, and writes ML-model/data/external/imdb_ratings.csv
(columns: imdb_id, imdb_rating, imdb_votes). The download itself is deleted afterwards.

LICENSE: IMDb's datasets are free for personal and non-commercial use only
(https://developer.imdb.com/non-commercial-datasets/). That is why the output lives in a
gitignored folder and is not committed or redistributed: each person runs this script for
themselves. The app works without the file and simply shows no IMDb rating.
Information courtesy of IMDb (https://www.imdb.com).

The file is a snapshot, so ratings do not update until you run this again.
"""

import csv
import gzip
import tempfile
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "ML-model" / "data"
FILM_STATS = DATA / "processed" / "film_stats.csv"
OUT_DIR = DATA / "external"
OUT = OUT_DIR / "imdb_ratings.csv"

URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"


def catalog_imdb_ids() -> set[str]:
    if not FILM_STATS.exists():
        raise SystemExit(f"{FILM_STATS} not found - run: python scripts/build_film_stats.py")
    with open(FILM_STATS, newline="", encoding="utf-8") as f:
        return {row["imdb_id"] for row in csv.DictReader(f) if row["imdb_id"]}


def download(dest: Path) -> None:
    print(f"Downloading {URL} ...")
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    print(f"  {dest.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    wanted = catalog_imdb_ids()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        gz = Path(tmp) / "title.ratings.tsv.gz"
        download(gz)
        with gzip.open(gz, "rt", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row["tconst"] in wanted:
                    rows.append((row["tconst"], row["averageRating"], row["numVotes"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["imdb_id", "imdb_rating", "imdb_votes"])
        writer.writerows(sorted(rows))
    print(f"Wrote {OUT}: {len(rows)} of {len(wanted)} catalog films have an IMDb rating")


if __name__ == "__main__":
    main()
