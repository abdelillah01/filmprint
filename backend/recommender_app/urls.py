from django.urls import path

from .views import meta_view, popular_view, recommend_view, titles_view

urlpatterns = [
    path("recommend/", recommend_view, name="recommend"),
    path("titles/", titles_view, name="titles"),
    path("meta/", meta_view, name="meta"),
    path("popular/", popular_view, name="popular"),
]
