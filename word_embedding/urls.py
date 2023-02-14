from django.urls import path
from .views import TwoWordEmbeddingView, ThreeWordEmbeddingView

urlpatterns = [
    path('2d/', TwoWordEmbeddingView.as_view(), name="2d Word Embedding View"),
    path('3d/', ThreeWordEmbeddingView.as_view(), name="3d Word Embedding View")
]