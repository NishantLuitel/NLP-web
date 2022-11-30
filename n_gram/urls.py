from django.urls import path
from .views import NgramView

urlpatterns = [
    path('ngram', NgramView.as_view(), name='ngram_home'),
]
