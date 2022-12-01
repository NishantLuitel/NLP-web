from django.urls import path
from .views import SentimentView, SentimentListView

urlpatterns = [
    path('', SentimentView.as_view(), name='sentiment_home'),
    path('sentiments/', SentimentListView.as_view(), name='sentiment_list'),
]
