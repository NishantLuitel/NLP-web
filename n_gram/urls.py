from django.urls import path
from .views import NgramView

urlpatterns = [
    path('', NgramView.as_view(), name='ngram_home'),
]
