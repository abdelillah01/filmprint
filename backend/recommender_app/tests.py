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
        patcher_fn = patch("recommender_app.views.recommend", return_value=_recommend_result())
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
        from recommender_app.utils import TitleNotFound

        self.mock_fn.side_effect = TitleNotFound(["xyzzy"])
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


def _recommend_result(**overrides):
    result = {
        "matched": "Toy Story (1995)",
        "likes": ["Toy Story (1995)"],
        "dislikes": [],
        "ignored": [],
        "corrections": [],
        "recommendations": [
            {
                "title": "Toy Story 2 (1999)",
                "poster": None,
                "release_date": "1999-11-24",
                "vote_average": 7.3,
                "tmdb_id": 863,
                "genres": ["animation"],
            }
        ],
    }
    result.update(overrides)
    return result


class RecommendViewCorrectionTest(TestCase):
    def post(self, body, **mock_kwargs):
        with patch("recommender_app.views.recommend", **mock_kwargs) as mock_fn:
            res = Client().post(
                "/api/v1/recommend/",
                data=json.dumps(body),
                content_type="application/json",
            )
        return res, mock_fn

    def test_no_query_field_for_formatting_only_difference(self):
        res, _ = self.post({"title": "The Matrix"}, return_value=_recommend_result())
        self.assertEqual(res.status_code, 200)
        self.assertNotIn("query", res.json())
        self.assertEqual(res.json()["corrections"], [])

    def test_query_field_returned_for_a_single_corrected_title(self):
        correction = [{"query": "toy stroy", "matched": "Toy Story (1995)"}]
        res, _ = self.post(
            {"title": "toy stroy"}, return_value=_recommend_result(corrections=correction)
        )
        body = res.json()
        self.assertEqual(body["query"], "toy stroy")
        self.assertEqual(body["corrections"], correction)

    def test_corrections_listed_for_multi_film_requests_without_query(self):
        correction = [{"query": "toy stroy", "matched": "Toy Story (1995)"}]
        res, _ = self.post(
            {"likes": ["toy stroy", "Inception"]},
            return_value=_recommend_result(corrections=correction),
        )
        self.assertNotIn("query", res.json())
        self.assertEqual(res.json()["corrections"], correction)

    def test_missing_model_is_503_without_leaking_details(self):
        from recommender_app.utils import ModelArtifactError

        res, _ = self.post({"title": "x"}, side_effect=ModelArtifactError("/secret/path"))
        self.assertEqual(res.status_code, 503)
        self.assertNotIn("secret", res.content.decode())

    def test_unexpected_error_is_500_without_leaking_details(self):
        res, _ = self.post({"title": "x"}, side_effect=RuntimeError("db password is hunter2"))
        self.assertEqual(res.status_code, 500)
        self.assertNotIn("hunter2", res.content.decode())

    def test_unresolved_like_is_404_and_names_the_titles(self):
        from recommender_app.utils import TitleNotFound

        res, _ = self.post({"likes": ["Parasite"]}, side_effect=TitleNotFound(["Parasite"]))
        self.assertEqual(res.status_code, 404)
        self.assertEqual(res.json()["unresolved"], ["Parasite"])


class RecommendRequestContractTest(TestCase):
    def post(self, body):
        with patch("recommender_app.views.recommend", return_value=_recommend_result()) as fn:
            res = Client().post(
                "/api/v1/recommend/",
                data=json.dumps(body),
                content_type="application/json",
            )
        return res, fn

    def test_profile_is_passed_through_to_recommend(self):
        res, fn = self.post(
            {
                "likes": ["Inception", "The Matrix"],
                "dislikes": ["Titanic"],
                "exclude": ["Interstellar"],
                "filters": {"genres": ["sci-fi"], "year_min": 2000, "min_rating": 6.5},
                "n": 5,
            }
        )
        self.assertEqual(res.status_code, 200)
        args, kwargs = fn.call_args
        self.assertEqual(args[0], ["Inception", "The Matrix"])
        self.assertEqual(kwargs["dislikes"], ["Titanic"])
        self.assertEqual(kwargs["exclude"], ["Interstellar"])
        self.assertEqual(kwargs["filters"]["genres"], ["sci-fi"])
        self.assertEqual(kwargs["filters"]["year_min"], 2000)
        self.assertEqual(kwargs["top_n"], 5)

    def test_title_is_shorthand_for_the_first_like(self):
        _, fn = self.post({"title": "Inception", "likes": ["The Matrix"]})
        self.assertEqual(fn.call_args[0][0], ["Inception", "The Matrix"])

    def test_rejects_a_request_with_no_films(self):
        for body in ({}, {"likes": []}, {"dislikes": ["Titanic"]}):
            with self.subTest(body=body):
                res, _ = self.post(body)
                self.assertEqual(res.status_code, 400)

    def test_rejects_invalid_fields(self):
        bad = [
            {"likes": ["x"] * 11},
            {"title": "x", "dislikes": ["y"] * 11},
            {"title": "x", "n": 0},
            {"title": "x", "n": 51},
            {"title": "x", "filters": {"year_min": 2010, "year_max": 2000}},
            {"title": "x", "filters": {"min_rating": 11}},
            {"likes": [""]},
        ]
        for body in bad:
            with self.subTest(body=body):
                res, _ = self.post(body)
                self.assertEqual(res.status_code, 400)

    def test_autocomplete_has_its_own_throttle_budget(self):
        from django.conf import settings

        rates = settings.REST_FRAMEWORK["DEFAULT_THROTTLE_RATES"]
        self.assertIn("autocomplete", rates)
        self.assertGreater(
            int(rates["autocomplete"].split("/")[0]), int(rates["anon"].split("/")[0])
        )


class TitlesAndMetaEndpointTest(TestCase):
    def test_titles_returns_suggestions(self):
        with patch(
            "recommender_app.views.search_titles", return_value=["Matrix, The (1999)"]
        ) as fn:
            res = Client().get("/api/v1/titles/", {"q": "mat"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"titles": ["Matrix, The (1999)"]})
        fn.assert_called_once_with("mat")

    def test_titles_without_query_is_empty_not_an_error(self):
        with patch("recommender_app.views.search_titles", return_value=[]):
            res = Client().get("/api/v1/titles/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"titles": []})

    def test_meta_returns_filter_facts(self):
        facts = {"genres": ["action"], "year_min": 1916, "year_max": 2016, "n_movies": 3536}
        with patch("recommender_app.views.get_meta", return_value=facts):
            res = Client().get("/api/v1/meta/")
        self.assertEqual(res.json(), facts)


# ---------------------------------------------------------------------------
# Taste profile maths (pure functions, tiny hand-checkable matrix)
# ---------------------------------------------------------------------------


class ProfileScoresTest(TestCase):
    def setUp(self):
        # Three orthogonal "themes" plus one film that mixes themes 0 and 1.
        self.matrix = csr_matrix(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [2**-0.5, 2**-0.5, 0.0],
                ]
            )
        )

    def test_single_like_equals_cosine_to_that_film(self):
        from recommender_app.profile import profile_scores

        scores = profile_scores(self.matrix, [3])
        self.assertAlmostEqual(scores[0], 2**-0.5)
        self.assertAlmostEqual(scores[2], 0.0)
        self.assertAlmostEqual(scores[3], 1.0)

    def test_two_likes_average_their_tastes(self):
        from recommender_app.profile import profile_scores

        scores = profile_scores(self.matrix, [0, 1])
        # The mixed film sits exactly between the two liked themes; the third theme is unrelated.
        self.assertAlmostEqual(scores[3], 1.0)
        self.assertAlmostEqual(scores[2], 0.0)

    def test_dislike_pushes_its_theme_down(self):
        from recommender_app.profile import profile_scores

        plain = profile_scores(self.matrix, [3])
        steered = profile_scores(self.matrix, [3], dislikes=[1])
        self.assertLess(steered[1], plain[1])
        self.assertGreater(steered[0], steered[1])

    def test_requires_at_least_one_like(self):
        from recommender_app.profile import profile_scores

        with self.assertRaises(ValueError):
            profile_scores(self.matrix, [])

    def test_cancelling_profile_scores_zero_everywhere(self):
        from recommender_app.profile import profile_scores

        scores = profile_scores(self.matrix, [0], dislikes=[0], dislike_weight=1.0)
        self.assertTrue(np.all(scores == 0))


# ---------------------------------------------------------------------------
# Personalised recommendations against the real committed model (no mocks of the maths)
# ---------------------------------------------------------------------------


class RealModelMixin:
    """Patches the real committed model into utils (no mocks of the maths)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from recommender_app import utils

        model, table = utils._load_artifacts()
        cls.model, cls.table = model, table
        cls.index = {t.lower(): i for i, t in enumerate(model[2]["title"])}

    def setUp(self):
        from recommender_app import utils

        for target, value in (
            (patch.object(utils, "_load_model", return_value=(self.model, self.index)), None),
            (patch.object(utils, "_neighbor_table", self.table), None),
            (patch.object(utils, "get_poster", return_value=None), None),
        ):
            target.start()
            self.addCleanup(target.stop)
        self.utils = utils

    def titles(self, result):
        return [r["title"] for r in result["recommendations"]]


class RealProfileTest(RealModelMixin, TestCase):
    def test_several_likes_return_a_full_list_without_the_likes(self):
        result = self.utils.recommend(["Inception", "The Matrix"])
        self.assertEqual(len(result["recommendations"]), 10)
        self.assertEqual(result["likes"], ["Inception (2010)", "Matrix, The (1999)"])
        for liked in result["likes"]:
            self.assertNotIn(liked, self.titles(result))

    def test_two_films_blend_both_tastes(self):
        both = set(self.titles(self.utils.recommend(["Toy Story", "The Dark Knight"], top_n=30)))
        only_a = set(self.titles(self.utils.recommend(["Toy Story"], top_n=30)))
        only_b = set(self.titles(self.utils.recommend(["The Dark Knight"], top_n=30)))
        self.assertTrue(both & only_a, "profile should keep some of film A's neighbours")
        self.assertTrue(both & only_b, "profile should keep some of film B's neighbours")

    def test_exclude_removes_seen_films(self):
        base = self.titles(self.utils.recommend(["Toy Story"]))
        result = self.utils.recommend(["Toy Story"], exclude=base[:3])
        for title in base[:3]:
            self.assertNotIn(title, self.titles(result))
        self.assertEqual(len(result["recommendations"]), 10)

    def test_dislike_moves_results_away_from_the_disliked_film(self):
        from sklearn.metrics.pairwise import cosine_similarity

        matrix = self.model[1]
        like, dislike = self.index["batman forever (1995)"], self.index["toy story (1995)"]

        def mean_sim(result):
            idx = [self.index[t.lower()] for t in self.titles(result)]
            return float(cosine_similarity(matrix[dislike], matrix[idx]).mean())

        plain = self.utils.recommend(["Batman Forever"], exclude=["Toy Story"], top_n=20)
        steered = self.utils.recommend(["Batman Forever"], dislikes=["Toy Story"], top_n=20)
        self.assertLessEqual(mean_sim(steered), mean_sim(plain))
        self.assertNotIn(like, [self.index[t.lower()] for t in self.titles(steered)])

    def test_genre_filter_keeps_only_that_genre(self):
        result = self.utils.recommend(["Inception"], filters={"genres": ["animation"]})
        self.assertEqual(len(result["recommendations"]), 10)
        for movie in result["recommendations"]:
            self.assertIn("animation", movie["genres"])

    def test_exclude_genres_filter(self):
        result = self.utils.recommend(["Toy Story"], filters={"exclude_genres": ["animation"]})
        for movie in result["recommendations"]:
            self.assertNotIn("animation", movie["genres"])

    def test_year_and_rating_filters(self):
        result = self.utils.recommend(
            ["Inception"], filters={"year_min": 2005, "year_max": 2012, "min_rating": 7.0}
        )
        self.assertTrue(result["recommendations"])
        for movie in result["recommendations"]:
            self.assertTrue("2005" <= movie["release_date"][:4] <= "2012", movie)
            self.assertGreaterEqual(movie["vote_average"], 7.0)

    def test_filters_that_match_nothing_return_an_empty_list(self):
        result = self.utils.recommend(["Inception"], filters={"year_min": 2100})
        self.assertEqual(result["recommendations"], [])

    def test_fast_path_and_general_path_agree_for_one_film(self):
        for query in ("Toy Story", "Inception", "The Matrix", "Heat"):
            with self.subTest(query=query):
                fast = self.titles(self.utils.recommend([query]))
                with patch.object(self.utils, "_neighbor_table", None):
                    general = self.titles(self.utils.recommend([query]))
                self.assertEqual(fast, general)

    def test_unknown_like_raises_but_unknown_dislike_is_ignored(self):
        with self.assertRaises(self.utils.TitleNotFound) as ctx:
            self.utils.recommend(["Inception", "Parasite"])
        self.assertEqual(ctx.exception.titles, ["Parasite"])

        result = self.utils.recommend(["Inception"], dislikes=["Parasite"], exclude=["Nonexistent"])
        self.assertEqual(result["ignored"], ["Parasite", "Nonexistent"])

    def test_a_like_wins_over_a_dislike_of_the_same_film(self):
        result = self.utils.recommend(["Inception"], dislikes=["Inception"])
        self.assertEqual(result["dislikes"], [])

    def test_corrections_reported_per_title(self):
        result = self.utils.recommend(["toy stroy", "The Matrix"])
        self.assertEqual(
            result["corrections"], [{"query": "toy stroy", "matched": "Toy Story (1995)"}]
        )

    def test_search_titles(self):
        self.assertIn("Matrix, The (1999)", self.utils.search_titles("mat"))
        self.assertEqual(self.utils.search_titles("toy sto")[0], "Toy Story (1995)")
        self.assertEqual(self.utils.search_titles(""), [])
        self.assertEqual(self.utils.search_titles("   "), [])
        self.assertLessEqual(len(self.utils.search_titles("the")), 8)

    def test_meta_lists_genres_and_year_range(self):
        meta = self.utils.get_meta()
        self.assertIn("animation", meta["genres"])
        self.assertFalse([g for g in meta["genres"] if g.startswith("(")])
        self.assertNotIn("imax", meta["genres"])
        self.assertEqual((meta["year_min"], meta["year_max"]), (1916, 2016))
        self.assertEqual(meta["n_movies"], len(self.model[2]))


# ---------------------------------------------------------------------------
# Per-film extras: IMDb id/rating, runtime, TMDB votes
# ---------------------------------------------------------------------------


class ExtrasLoadingTest(TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def write(self, name, text):
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def load(self, stats=None, imdb=None):
        from recommender_app import utils

        missing = self.dir / "missing.csv"
        with (
            patch.object(utils, "STATS_PATH", stats or missing),
            patch.object(utils, "IMDB_PATH", imdb or missing),
        ):
            return utils._load_extras()

    STATS = "movieId,imdb_id,runtime,tmdb_votes\n1,tt0114709,81,5269\n2,tt0113497,104,\n3,,,\n"
    IMDB = "imdb_id,imdb_rating,imdb_votes\ntt0114709,8.3,1100000\n"

    def test_stats_and_imdb_are_merged_by_movie_id(self):
        extras = self.load(self.write("s.csv", self.STATS), self.write("i.csv", self.IMDB))
        self.assertEqual(
            extras[1],
            {
                "imdb_id": "tt0114709",
                "runtime": 81,
                "tmdb_votes": 5269,
                "imdb_rating": 8.3,
                "imdb_votes": 1100000,
            },
        )

    def test_film_without_an_imdb_rating_has_no_rating_fields(self):
        extras = self.load(self.write("s.csv", self.STATS), self.write("i.csv", self.IMDB))
        self.assertNotIn("imdb_rating", extras[2])
        self.assertIsNone(extras[2]["tmdb_votes"])  # blank cell -> None, not NaN
        self.assertIsNone(extras[3]["imdb_id"])

    def test_works_without_the_imdb_file(self):
        extras = self.load(self.write("s.csv", self.STATS))
        self.assertEqual(extras[1]["runtime"], 81)
        self.assertNotIn("imdb_rating", extras[1])

    def test_works_without_any_files(self):
        self.assertEqual(self.load(), {})

    def test_unreadable_file_is_skipped_not_fatal(self):
        extras = self.load(self.write("s.csv", self.STATS), self.write("i.csv", "garbage\n\x00"))
        self.assertEqual(extras[1]["runtime"], 81)

    def test_committed_stats_cover_the_catalog(self):
        from recommender_app import utils

        extras = utils._load_extras()
        self.assertGreater(len(extras), 3000)
        self.assertEqual(extras[1]["imdb_id"], "tt0114709")  # Toy Story
        self.assertEqual(extras[1]["runtime"], 81)


class ExtrasInRecommendationsTest(RealModelMixin, TestCase):
    def with_extras(self, extras):
        patcher = patch.object(self.utils, "_extras", extras)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_results_carry_ratings_runtime_and_votes(self):
        first = self.utils.recommend(["Toy Story"], top_n=1)["recommendations"][0]
        movie_id = int(self.model[2][self.model[2]["title"] == first["title"]]["movieId"].iloc[0])
        self.with_extras(
            {
                movie_id: {
                    "imdb_id": "tt1",
                    "imdb_rating": 7.9,
                    "imdb_votes": 12345,
                    "tmdb_votes": 678,
                    "runtime": 99,
                }
            }
        )
        again = self.utils.recommend(["Toy Story"], top_n=1)["recommendations"][0]
        self.assertEqual(
            (again["imdb_id"], again["imdb_rating"], again["imdb_votes"]),
            ("tt1", 7.9, 12345),
        )
        self.assertEqual((again["tmdb_votes"], again["runtime"]), (678, 99))

    def test_missing_extras_are_null_not_errors(self):
        self.with_extras({})
        movie = self.utils.recommend(["Toy Story"], top_n=1)["recommendations"][0]
        for field in ("imdb_id", "imdb_rating", "imdb_votes", "tmdb_votes", "runtime"):
            self.assertIn(field, movie)
            self.assertIsNone(movie[field])

    def test_liked_films_are_returned_with_details(self):
        result = self.utils.recommend(["Inception", "The Matrix"])
        self.assertEqual(
            [m["title"] for m in result["liked"]], ["Inception (2010)", "Matrix, The (1999)"]
        )
        self.assertIn("sci-fi", result["liked"][0]["genres"])


# ---------------------------------------------------------------------------
# Cold start: popular films, and the IMDb-based rating filter
# ---------------------------------------------------------------------------


class PopularMoviesTest(RealModelMixin, TestCase):
    def setUp(self):
        super().setUp()
        # Real extras (committed film_stats.csv) so popularity is real; posters always present.
        extras = self.utils._load_extras()
        patcher = patch.object(self.utils, "_extras", extras)
        patcher.start()
        self.addCleanup(patcher.stop)
        poster = patch.object(self.utils, "get_poster", return_value="http://example.com/p.jpg")
        poster.start()
        self.addCleanup(poster.stop)

    def test_returns_n_well_known_films(self):
        films = self.utils.popular_movies(n=12)
        self.assertEqual(len(films), 12)
        votes = [m["tmdb_votes"] for m in films]
        self.assertEqual(votes, sorted(votes, reverse=True))
        self.assertGreater(votes[0], 5000)

    def test_genre_restricts_the_list(self):
        films = self.utils.popular_movies(genre="Animation", n=10)
        self.assertEqual(len(films), 10)
        for movie in films:
            self.assertIn("animation", movie["genres"])

    def test_unknown_genre_is_empty_not_an_error(self):
        self.assertEqual(self.utils.popular_movies(genre="nonsense"), [])

    def test_films_without_a_poster_are_skipped(self):
        with patch.object(self.utils, "get_poster", return_value=None):
            self.assertEqual(self.utils.popular_movies(n=5), [])


class PopularEndpointTest(TestCase):
    def get(self, query=""):
        with patch("recommender_app.views.popular_movies", return_value=[{"title": "x"}]) as fn:
            res = Client().get(f"/api/v1/popular/{query}")
        return res, fn

    def test_defaults(self):
        res, fn = self.get()
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"genre": None, "movies": [{"title": "x"}]})
        fn.assert_called_once_with(genre=None, n=18)

    def test_genre_and_n_are_passed_through(self):
        _, fn = self.get("?genre=sci-fi&n=6")
        fn.assert_called_once_with(genre="sci-fi", n=6)

    def test_rejects_bad_n(self):
        for q in ("?n=0", "?n=49", "?n=abc"):
            with self.subTest(q=q):
                res, _ = self.get(q)
                self.assertEqual(res.status_code, 400)


class MinRatingUsesImdbTest(RealModelMixin, TestCase):
    def test_filter_follows_the_imdb_rating_when_there_is_one(self):
        df = self.model[2]
        ids = df["movieId"].astype(int).tolist()
        tmdb = df["vote_average"].fillna(0).to_numpy(float)
        # Film A: low TMDB, high IMDb.  Film B: high TMDB, low IMDb.
        a = int(np.argmin(np.where(tmdb > 0, tmdb, 99)))
        b = int(np.argmax(tmdb))
        extras = {ids[a]: {"imdb_rating": 9.0}, ids[b]: {"imdb_rating": 3.0}}
        with patch.object(self.utils, "_extras", extras):
            mask = self.utils._allowed_mask(df, {"min_rating": 7.0})
        self.assertTrue(mask[a], "high IMDb rating should pass even though TMDB is low")
        self.assertFalse(mask[b], "low IMDb rating should fail even though TMDB is high")

    def test_falls_back_to_tmdb_without_imdb(self):
        df = self.model[2]
        tmdb = df["vote_average"].fillna(0).to_numpy(float)
        with patch.object(self.utils, "_extras", {}):
            mask = self.utils._allowed_mask(df, {"min_rating": 7.0})
        self.assertTrue(np.array_equal(mask, tmdb >= 7.0))
