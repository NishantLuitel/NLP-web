from django.urls import path
from .views import HomePageAPIView, AboutPageAPIView

urlpatterns = [
    path('', HomePageAPIView.as_view(), name='home'),
    path('about/', AboutPageAPIView.as_view(), name='about'),
]
