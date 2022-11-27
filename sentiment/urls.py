from django.urls import path
from .views import SentimentView

urlpatterns = [
    path('', SentimentView.as_view(), name='sentiment_home'),
]
