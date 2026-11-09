import json
import platform
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse
import sklearn
from sklearn.metrics.pairwise import linear_kernel

TEXT_COLUMN = "metadata"
N_NEIGHBORS = 50
NEIGHBOR_BATCH = 256


def build_tfidf_model(df, text_column=TEXT_COLUMN):
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(stop_words="english", max_features=10000)
    tfidf_matrix = tfidf.fit_transform(df[text_column])
    print("TF-IDF matrix built with shape:", tfidf_matrix.shape)
    return tfidf, tfidf_matrix


def build_neighbor_table(matrix, top_n=N_NEIGHBORS, batch=NEIGHBOR_BATCH):
    """Top-`top_n` most similar rows (excluding self) for every row, best first.

    The catalog is static, so this is computed once at build time and the API does a lookup.
    """
    n = matrix.shape[0]
    table = np.empty((n, top_n), dtype=np.int32)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        sims = linear_kernel(matrix[start:end], matrix)
        for i, row in enumerate(sims):
            row[start + i] = -1  # exclude self
            table[start + i] = row.argsort()[-top_n:][::-1]
    print(f"Neighbor table built with shape: {table.shape}")
    return table


def save_model_v2(tfidf, matrix, df, neighbors, output_dir, text_column=TEXT_COLUMN):
    """Save every model artifact into one directory so they cannot drift apart.

    output_dir/
      vectorizer.joblib    — TF-IDF vectorizer
      tfidf_matrix.npz     — scipy sparse matrix
      movies.csv           — movie metadata DataFrame
      neighbor_table.npy   — precomputed top-N neighbors per movie
      model_meta.json      — build provenance; the API refuses to load on a mismatch
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(tfidf, out / "vectorizer.joblib")
    scipy.sparse.save_npz(str(out / "tfidf_matrix.npz"), matrix)
    df.to_csv(out / "movies.csv", index=False)
    np.save(out / "neighbor_table.npy", neighbors)

    meta = {
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "build_date": str(date.today()),
        "n_movies": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "n_neighbors": int(neighbors.shape[1]),
        "text_column": text_column,
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Model artifacts saved to {out.resolve()}/")
    print(f"   {meta}")


def load_model_v2(output_dir):
    out = Path(output_dir)
    import pandas as pd

    tfidf = joblib.load(out / "vectorizer.joblib")
    matrix = scipy.sparse.load_npz(str(out / "tfidf_matrix.npz"))
    df = pd.read_csv(out / "movies.csv")
    return tfidf, matrix, df
