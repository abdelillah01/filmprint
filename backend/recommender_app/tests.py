import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from django.test import Client, TestCase
from scipy.sparse import csr_matrix

# ---------------------------------------------------------------------------
# Helpers — a tiny in-memory model so tests never touch the real pickle
# ---------------------------------------------------------------------------


def _make_fake_model():
    titles = [
        "Toy Story (1995)",
        "Toy Story 2 (1999)",
        "Toy Story 3 (2010)",
        "The Matrix (1999)",
        "Inception (2010)",
    ]
    combined = [
        "animation adventure comedy",
        "animation adventure comedy",
        "animation adventure comedy",
        "action scifi thriller",
        "action scifi thriller",
    ]
    df = pd.DataFrame({"title": titles, "combined": combined})

    matrix = csr_matrix(np.eye(5))

    vectorizer = MagicMock()
    return (vectorizer, matrix, df)


# ---------------------------------------------------------------------------
# P1-11 guard: importing utils must NOT load the model on import
# ---------------------------------------------------------------------------


class LazyModelLoadTest(TestCase):
    def test_import_does_not_load_model(self):
        import recommender_app.utils as utils

        self.assertIsNone(utils._model_data)


# ---------------------------------------------------------------------------
# P1-07: unit tests for recommend_movies
# ---------------------------------------------------------------------------

FAKE_MODEL = _make_fake_model()


class RecommendMoviesTest(TestCase):
    def setUp(self):
        patcher_model = patch(
            "recommender_app.utils._load_model",
            return_value=(FAKE_MODEL, {t.lower(): i for i, t in enumerate(FAKE_MODEL[2]["title"])}),
        )
        patcher_poster = patch("recommender_app.utils.get_poster", return_value=None)
        self.mock_model = patcher_model.start()
        self.mock_poster = patcher_poster.start()
        self.addCleanup(patcher_model.stop)
        self.addCleanup(patcher_poster.stop)

    def test_returns_n_results(self):
        from recommender_app.utils import recommend_movies

        _, results = recommend_movies("Toy Story (1995)", top_n=2)
        self.assertEqual(len(results), 2)

    def test_results_have_title_and_poster_keys(self):
        from recommender_app.utils import recommend_movies

        _, results = recommend_movies("Toy Story (1995)", top_n=1)
        self.assertIn("title", results[0])
        self.assertIn("poster", results[0])

    def test_excludes_seed_movie(self):
        from recommender_app.utils import recommend_movies

        _, results = recommend_movies("Toy Story (1995)", top_n=4)
        titles = [r["title"] for r in results]
        self.assertNotIn("Toy Story (1995)", titles)

    def test_matched_title_returned(self):
        from recommender_app.utils import recommend_movies

        matched, _ = recommend_movies("Toy Story (1995)", top_n=1)
        self.assertEqual(matched, "Toy Story (1995)")

    def test_unknown_title_raises_value_error(self):
        from recommender_app.utils import recommend_movies

        with self.assertRaises(ValueError):
            recommend_movies("xyzzy no such film", top_n=5)

    def test_empty_input_raises_value_error(self):
        from recommender_app.utils import recommend_movies

        with self.assertRaises(ValueError):
            recommend_movies("", top_n=5)

    def test_whitespace_only_raises_value_error(self):
        from recommender_app.utils import recommend_movies

        with self.assertRaises(ValueError):
            recommend_movies("   ", top_n=5)


# ---------------------------------------------------------------------------
# P1-09: fuzzy matching resolves a deliberate typo
# ---------------------------------------------------------------------------


class FuzzyMatchTest(TestCase):
    def setUp(self):
        patcher_model = patch(
            "recommender_app.utils._load_model",
            return_value=(FAKE_MODEL, {t.lower(): i for i, t in enumerate(FAKE_MODEL[2]["title"])}),
        )
        patcher_poster = patch("recommender_app.utils.get_poster", return_value=None)
        self.mock_model = patcher_model.start()
        self.mock_poster = patcher_poster.start()
        self.addCleanup(patcher_model.stop)
        self.addCleanup(patcher_poster.stop)

    def test_typo_resolves_to_correct_movie(self):
        from recommender_app.utils import recommend_movies

        # "toy stroy" is close enough to "toy story (1995)" at cutoff=0.5
        matched, results = recommend_movies("toy stroy", top_n=2)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIn("toy story", matched.lower())


# ---------------------------------------------------------------------------
# P1-08: endpoint HTTP codes
# ---------------------------------------------------------------------------


class RecommendViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        patcher_fn = patch(
            "recommender_app.views.recommend_movies",
            return_value=("Toy Story (1995)", [{"title": "Toy Story 2 (1999)", "poster": None}]),
        )
        self.mock_fn = patcher_fn.start()
        self.addCleanup(patcher_fn.stop)

    def test_post_returns_200(self):
        res = self.client.post(
            "/api/recommend/",
            data=json.dumps({"title": "Toy Story"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)

    def test_missing_title_returns_400(self):
        res = self.client.post(
            "/api/recommend/",
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)

    def test_get_returns_405(self):
        res = self.client.get("/api/recommend/")
        self.assertEqual(res.status_code, 405)

    def test_unknown_movie_returns_404(self):
        self.mock_fn.side_effect = ValueError("not found")
        res = self.client.post(
            "/api/recommend/",
            data=json.dumps({"title": "xyzzy"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 404)


# ---------------------------------------------------------------------------
# P1-10: fetch_poster never hits TMDB during the test suite
# ---------------------------------------------------------------------------


class PosterCacheTest(TestCase):
    def test_get_poster_uses_cache_no_network(self):
        """get_poster() must return from _poster_cache without any HTTP call."""
        import recommender_app.utils as utils

        original = utils._poster_cache.copy()
        try:
            utils._poster_cache[862] = "http://example.com/poster.jpg"
            with patch("requests.get") as mock_get:
                mock_get.side_effect = AssertionError("network must not be called")
                result = utils.get_poster(862)
            self.assertEqual(result, "http://example.com/poster.jpg")
            mock_get.assert_not_called()
        finally:
            utils._poster_cache.clear()
            utils._poster_cache.update(original)


# ---------------------------------------------------------------------------
# Title matching: MovieLens-style catalog titles vs. what users actually type
# ---------------------------------------------------------------------------

CATALOG_TITLES = [
    "Toy Story (1995)",
    "Seven (a.k.a. Se7en) (1995)",
    "Matrix, The (1999)",
    "Martian, The (2015)",
    "Up (2009)",
    "Heat (1995)",
    "Dark Knight, The (2008)",
    "Beautiful Mind, A (2001)",
    "Dracula (1931)",
    "Dracula (1992)",
    "Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001)",
]
CATALOG_INDEX = {t.lower(): i for i, t in enumerate(CATALOG_TITLES)}


class TitleMatchingTest(TestCase):
    def resolve(self, query):
        from recommender_app.utils import resolve_title

        return resolve_title(query, CATALOG_INDEX)

    def test_the_matrix_is_not_the_martian(self):
        self.assertEqual(self.resolve("The Matrix"), "matrix, the (1999)")

    def test_short_titles_resolve(self):
        self.assertEqual(self.resolve("Up"), "up (2009)")
        self.assertEqual(self.resolve("heat"), "heat (1995)")

    def test_alias_title_resolves(self):
        self.assertEqual(self.resolve("Se7en"), "seven (a.k.a. se7en) (1995)")
        self.assertEqual(self.resolve("Seven"), "seven (a.k.a. se7en) (1995)")

    def test_leading_article_is_optional(self):
        self.assertEqual(self.resolve("Dark Knight"), "dark knight, the (2008)")
        self.assertEqual(self.resolve("a beautiful mind"), "beautiful mind, a (2001)")

    def test_exact_catalog_spelling_still_works(self):
        self.assertEqual(self.resolve("Matrix, The (1999)"), "matrix, the (1999)")

    def test_year_disambiguates_remakes(self):
        self.assertEqual(self.resolve("Dracula (1992)"), "dracula (1992)")
        self.assertEqual(self.resolve("Dracula (1931)"), "dracula (1931)")

    def test_foreign_title_alias_and_accents(self):
        self.assertEqual(self.resolve("amelie"), CATALOG_TITLES[-1].lower())
        self.assertEqual(self.resolve("Amélie"), CATALOG_TITLES[-1].lower())

    def test_typo_falls_back_to_fuzzy(self):
        self.assertEqual(self.resolve("toy stroy"), "toy story (1995)")

    def test_unrelated_query_does_not_match(self):
        self.assertIsNone(self.resolve("xyzzy no such film"))
        self.assertIsNone(self.resolve("Parasite"))
        self.assertIsNone(self.resolve(""))
        self.assertIsNone(self.resolve("(1999)"))

    def test_same_title_ignores_formatting_but_not_corrections(self):
        from recommender_app.utils import same_title

        self.assertTrue(same_title("The Matrix", "Matrix, The (1999)"))
        self.assertTrue(same_title("Toy Story", "Toy Story (1995)"))
        self.assertTrue(same_title("se7en", "Seven (a.k.a. Se7en) (1995)"))
        self.assertFalse(same_title("toy stroy", "Toy Story (1995)"))


class RealCatalogTest(TestCase):
    """Runs against the committed model_v2 artifacts (no mocks)."""

    FEATURED = [
        ("Inception", "Inception (2010)"),
        ("The Dark Knight", "Dark Knight, The (2008)"),
        ("Toy Story", "Toy Story (1995)"),
        ("Interstellar", "Interstellar (2014)"),
        ("Pulp Fiction", "Pulp Fiction (1994)"),
        ("The Matrix", "Matrix, The (1999)"),
        ("Forrest Gump", "Forrest Gump (1994)"),
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from recommender_app.utils import _load_artifacts

        (_, matrix, df), table = _load_artifacts()
        cls.matrix, cls.df, cls.table = matrix, df, table
        cls.index = {t.lower(): i for i, t in enumerate(df["title"])}

    def test_featured_titles_resolve_to_the_right_movie(self):
        from recommender_app.utils import resolve_title

        for query, expected in self.FEATURED:
            with self.subTest(query=query):
                self.assertEqual(resolve_title(query, self.index), expected.lower())

    def test_artifacts_are_consistent(self):
        n = len(self.df)
        self.assertEqual(self.matrix.shape[0], n)
        self.assertEqual(self.table.shape[0], n)
        self.assertLess(int(self.table.max()), n)

    def test_neighbor_table_excludes_self(self):
        for i in (0, 100, 1000, len(self.df) - 1):
            self.assertNotIn(i, self.table[i])


class ArtifactGuardTest(TestCase):
    def test_neighbor_table_for_a_different_model_is_rejected(self):
        import tempfile
        from pathlib import Path

        from recommender_app import utils

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "neighbor_table.npy"
            np.save(bad, np.zeros((7, 50), dtype=np.int32))
            with patch.object(utils, "NEIGHBOR_TABLE_PATH", bad):
                with self.assertRaises(utils.ModelArtifactError):
                    utils._load_artifacts()

    def test_sklearn_version_mismatch_fails_loudly(self):
        from recommender_app import utils

        with patch.object(utils.sklearn, "__version__", "0.1.0"):
            with self.assertRaises(utils.ModelArtifactError):
                utils._load_artifacts()


class RecommendViewCorrectionTest(TestCase):
    def post(self, title, **mock_kwargs):
        with patch("recommender_app.views.recommend_movies", **mock_kwargs):
            return Client().post(
                "/api/v1/recommend/",
                data=json.dumps({"title": title}),
                content_type="application/json",
            )

    def test_no_query_field_for_formatting_only_difference(self):
        res = self.post("The Matrix", return_value=("Matrix, The (1999)", []))
        self.assertEqual(res.status_code, 200)
        self.assertNotIn("query", res.json())

    def test_query_field_returned_for_a_real_correction(self):
        res = self.post("toy stroy", return_value=("Toy Story (1995)", []))
        self.assertEqual(res.json()["query"], "toy stroy")

    def test_missing_model_is_503_without_leaking_details(self):
        from recommender_app.utils import ModelArtifactError

        res = self.post("x", side_effect=ModelArtifactError("/secret/path"))
        self.assertEqual(res.status_code, 503)
        self.assertNotIn("secret", res.content.decode())

    def test_unexpected_error_is_500_without_leaking_details(self):
        res = self.post("x", side_effect=RuntimeError("db password is hunter2"))
        self.assertEqual(res.status_code, 500)
        self.assertNotIn("hunter2", res.content.decode())
