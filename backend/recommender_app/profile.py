"""Taste-profile scoring. Pure numpy/scipy: no Django imports.

The evaluation harness (evaluation/evaluate.py) loads this exact file, so the production
scoring and the measured scoring cannot drift apart.
"""

import numpy as np

# How strongly disliked films push the profile away (Rocchio-style relevance feedback).
DISLIKE_WEIGHT = 0.5


def profile_scores(matrix, likes, dislikes=(), dislike_weight=DISLIKE_WEIGHT):
    """Cosine similarity of every row of `matrix` to a taste profile.

    The profile is the mean TF-IDF vector of the liked films minus `dislike_weight` times the
    mean vector of the disliked ones. `matrix` rows are L2-normalised (TfidfVectorizer default),
    so with a single like and no dislikes this equals the plain cosine to that film.

    Returns an array of length n_rows in [-1, 1]. Excluding the profile's own films is the
    caller's job.
    """
    likes = list(likes)
    if not likes:
        raise ValueError("at least one liked film is required")

    profile = np.asarray(matrix[likes].mean(axis=0)).ravel()
    dislikes = list(dislikes)
    if dislikes:
        profile = profile - dislike_weight * np.asarray(matrix[dislikes].mean(axis=0)).ravel()

    norm = np.linalg.norm(profile)
    if norm == 0:
        return np.zeros(matrix.shape[0])
    return np.asarray(matrix @ profile).ravel() / norm
