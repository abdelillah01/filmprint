import ast
from pathlib import Path

import pandas as pd

METADATA_COLUMNS = ("genres", "keywords", "cast", "crew")


def _parse_tokens(value):
    """Parse a stringified list column (e.g. "['tomhanks', 'timallen']") into a list."""
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return parsed if isinstance(parsed, list) else []


def add_metadata_column(df):
    """Add `metadata`: genres + keywords + cast + director tokens, WITHOUT the plot overview.

    evaluation/results.md shows this beats the overview-augmented `combined` text
    (NDCG@10 0.0657 vs 0.0586), so it is the text the production model is built on.
    """
    df = df.copy()
    df["metadata"] = df.apply(
        lambda row: " ".join(tok for col in METADATA_COLUMNS for tok in _parse_tokens(row[col])),
        axis=1,
    )
    return df


def load_clean_data(filepath=None):
    """
    Load and clean the merged movie dataset for model training.
    """
    if filepath is None:
        filepath = (
            Path(__file__).resolve().parent.parent / "data" / "processed" / "merged_movies.csv"
        )

    df = pd.read_csv(filepath)

    # Drop missing or duplicate values
    df = df.dropna(subset=["combined", "title"])
    df = df.drop_duplicates(subset=["title"])

    # Convert to string
    df["combined"] = df["combined"].astype(str)
    df["title"] = df["title"].astype(str)

    df = add_metadata_column(df).reset_index(drop=True)

    print(f"Loaded {len(df)} movies from {filepath}")
    return df
