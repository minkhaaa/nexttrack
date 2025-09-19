from django.urls import path

from . import views

urlpatterns = [
    path("", views.main_page, name="main_page"),
    path("recommend/", views.recommend, name="recommend"),
    path("api/recommend/", views.RecommendationAPIView.as_view(), name="api_recommend"),
]
