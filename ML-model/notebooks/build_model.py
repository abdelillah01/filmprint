"""Build the production model artifacts (run from anywhere):

    python ML-model/notebooks/build_model.py

Writes ML-model/model_v2/ (vectorizer, matrix, movies, neighbor table, metadata).
"""

import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent
ML_MODEL_DIR = NOTEBOOKS_DIR.parent
sys.path.append(str(ML_MODEL_DIR))

from recommender.data_loader import load_clean_data  # noqa: E402
from recommender.model_builder import (  # noqa: E402
    build_neighbor_table,
    build_tfidf_model,
    save_model_v2,
)

df = load_clean_data()
tfidf, matrix = build_tfidf_model(df)
neighbors = build_neighbor_table(matrix)
save_model_v2(tfidf, matrix, df, neighbors, ML_MODEL_DIR / "model_v2")

# Smoke test: neighbors of a known title
sample = df["title"].iloc[50]
print("\nNeighbors of:", sample)
for i in neighbors[50][:5]:
    print("-", df.iloc[i]["title"])
