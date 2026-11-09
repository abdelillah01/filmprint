import logging

from drf_spectacular.utils import OpenApiExample, OpenApiParameter, extend_schema
from rest_framework import serializers, status
from rest_framework.decorators import api_view, throttle_classes
from rest_framework.response import Response
from rest_framework.throttling import AnonRateThrottle

from .utils import (
    ModelArtifactError,
    TitleNotFound,
    get_meta,
    popular_movies,
    recommend,
    search_titles,
)

logger = logging.getLogger(__name__)

MAX_LIKES = 10
MAX_DISLIKES = 10
MAX_EXCLUDE = 200


class AutocompleteThrottle(AnonRateThrottle):
    """Typing fires many small requests, so autocomplete gets its own, higher budget."""

    scope = "autocomplete"


class FiltersSerializer(serializers.Serializer):
    genres = serializers.ListField(
        child=serializers.CharField(max_length=40),
        required=False,
        max_length=30,
        help_text="Keep films that have at least one of these genres.",
    )
    exclude_genres = serializers.ListField(
        child=serializers.CharField(max_length=40), required=False, max_length=30
    )
    year_min = serializers.IntegerField(required=False, min_value=1800, max_value=2100)
    year_max = serializers.IntegerField(required=False, min_value=1800, max_value=2100)
    min_rating = serializers.FloatField(required=False, min_value=0, max_value=10)

    def validate(self, attrs):
        lo, hi = attrs.get("year_min"), attrs.get("year_max")
        if lo is not None and hi is not None and lo > hi:
            raise serializers.ValidationError("year_min must not exceed year_max")
        return attrs


class RecommendRequestSerializer(serializers.Serializer):
    title = serializers.CharField(
        min_length=1,
        max_length=300,
        required=False,
        help_text="A single liked film. Shorthand for likes=[title].",
    )
    likes = serializers.ListField(
        child=serializers.CharField(min_length=1, max_length=300),
        required=False,
        max_length=MAX_LIKES,
        help_text="Films the user likes. The profile is their average.",
    )
    dislikes = serializers.ListField(
        child=serializers.CharField(min_length=1, max_length=300),
        required=False,
        max_length=MAX_DISLIKES,
        help_text="Films to steer away from.",
    )
    exclude = serializers.ListField(
        child=serializers.CharField(min_length=1, max_length=300),
        required=False,
        max_length=MAX_EXCLUDE,
        help_text="Films already seen; never recommended.",
    )
    filters = FiltersSerializer(required=False)
    n = serializers.IntegerField(min_value=1, max_value=50, default=10, required=False)

    def validate(self, attrs):
        likes = list(attrs.get("likes", []))
        if attrs.get("title"):
            likes.insert(0, attrs["title"])
        if not likes:
            raise serializers.ValidationError("Provide `title` or at least one film in `likes`.")
        if len(likes) > MAX_LIKES:
            raise serializers.ValidationError(f"At most {MAX_LIKES} liked films.")
        attrs["likes"] = likes
        return attrs


class MovieSerializer(serializers.Serializer):
    title = serializers.CharField()
    poster = serializers.URLField(allow_null=True)
    release_date = serializers.CharField(allow_null=True)
    vote_average = serializers.FloatField(allow_null=True)
    tmdb_id = serializers.IntegerField(allow_null=True)
    genres = serializers.ListField(child=serializers.CharField())
    imdb_id = serializers.CharField(allow_null=True)
    imdb_rating = serializers.FloatField(allow_null=True)
    imdb_votes = serializers.IntegerField(allow_null=True)
    tmdb_votes = serializers.IntegerField(allow_null=True)
    runtime = serializers.IntegerField(allow_null=True, help_text="Minutes")


class CorrectionSerializer(serializers.Serializer):
    query = serializers.CharField()
    matched = serializers.CharField()


class RecommendResponseSerializer(serializers.Serializer):
    matched = serializers.CharField(help_text="First liked film as matched in the catalog.")
    query = serializers.CharField(
        required=False, help_text="Set only for a single-film request that was fuzzily corrected."
    )
    likes = serializers.ListField(child=serializers.CharField())
    dislikes = serializers.ListField(child=serializers.CharField())
    ignored = serializers.ListField(
        child=serializers.CharField(), help_text="Dislikes/excludes that were not in the catalog."
    )
    corrections = CorrectionSerializer(many=True)
    liked = MovieSerializer(many=True, help_text="Details of the matched liked films.")
    recommendations = MovieSerializer(many=True)


class PopularQuerySerializer(serializers.Serializer):
    genre = serializers.CharField(required=False, allow_blank=True, max_length=40)
    n = serializers.IntegerField(required=False, min_value=1, max_value=48, default=18)


class PopularResponseSerializer(serializers.Serializer):
    genre = serializers.CharField(allow_null=True)
    movies = MovieSerializer(many=True)


class MetaSerializer(serializers.Serializer):
    genres = serializers.ListField(child=serializers.CharField())
    year_min = serializers.IntegerField(allow_null=True)
    year_max = serializers.IntegerField(allow_null=True)
    n_movies = serializers.IntegerField()


@extend_schema(
    request=RecommendRequestSerializer,
    responses={200: RecommendResponseSerializer},
    examples=[
        OpenApiExample("One film", value={"title": "Inception"}, request_only=True),
        OpenApiExample(
            "Taste profile with filters",
            value={
                "likes": ["Inception", "The Matrix"],
                "dislikes": ["Titanic"],
                "exclude": ["Interstellar"],
                "filters": {"genres": ["sci-fi"], "year_min": 2000, "min_rating": 6.5},
            },
            request_only=True,
        ),
    ],
    summary="Get movie recommendations",
    description=(
        "Returns up to *n* films for a taste profile: the average of the liked films, pushed away "
        "from the disliked ones, restricted by the filters, minus anything in `exclude`, with a "
        "rating-weighted rerank. Title matching is fuzzy: a near-miss resolves silently, and "
        "`corrections` lists every title that was changed."
    ),
)
@api_view(["POST"])
@throttle_classes([AnonRateThrottle])
def recommend_view(request):
    req = RecommendRequestSerializer(data=request.data)
    if not req.is_valid():
        return Response(req.errors, status=status.HTTP_400_BAD_REQUEST)
    data = req.validated_data

    try:
        result = recommend(
            data["likes"],
            dislikes=data.get("dislikes", []),
            exclude=data.get("exclude", []),
            filters=data.get("filters"),
            top_n=data.get("n", 10),
        )
    except TitleNotFound as e:
        return Response({"error": str(e), "unresolved": e.titles}, status=status.HTTP_404_NOT_FOUND)
    except ModelArtifactError:
        logger.exception("Model artifacts unavailable")
        return Response(
            {"error": "The recommendation model is unavailable."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    except Exception:
        logger.exception("Unexpected error in recommend_view")
        return Response(
            {"error": "Internal server error."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    # Backwards compatible: a single corrected title is also reported as `query`.
    if len(data["likes"]) == 1 and result["corrections"]:
        result["query"] = result["corrections"][0]["query"]
    return Response(result)


@extend_schema(
    parameters=[OpenApiParameter("q", str, description="What the user has typed so far")],
    responses={200: serializers.ListField(child=serializers.CharField())},
    summary="Autocomplete film titles",
)
@api_view(["GET"])
@throttle_classes([AutocompleteThrottle])
def titles_view(request):
    q = request.query_params.get("q", "")[:100]
    try:
        return Response({"titles": search_titles(q)})
    except ModelArtifactError:
        logger.exception("Model artifacts unavailable")
        return Response(
            {"error": "The recommendation model is unavailable."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


@extend_schema(responses={200: MetaSerializer}, summary="Genres and year range for filters")
@api_view(["GET"])
@throttle_classes([AutocompleteThrottle])
def meta_view(request):
    try:
        return Response(get_meta())
    except ModelArtifactError:
        logger.exception("Model artifacts unavailable")
        return Response(
            {"error": "The recommendation model is unavailable."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


@extend_schema(
    parameters=[
        OpenApiParameter("genre", str, description="Only films with this genre, e.g. sci-fi"),
        OpenApiParameter("n", int, description="How many films, 1-48 (default 18)"),
    ],
    responses={200: PopularResponseSerializer},
    summary="Well-known films for the cold-start screen",
)
@api_view(["GET"])
@throttle_classes([AutocompleteThrottle])
def popular_view(request):
    query = PopularQuerySerializer(data=request.query_params)
    if not query.is_valid():
        return Response(query.errors, status=status.HTTP_400_BAD_REQUEST)
    genre = query.validated_data.get("genre") or None
    try:
        movies = popular_movies(genre=genre, n=query.validated_data["n"])
    except ModelArtifactError:
        logger.exception("Model artifacts unavailable")
        return Response(
            {"error": "The recommendation model is unavailable."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return Response({"genre": genre, "movies": movies})
