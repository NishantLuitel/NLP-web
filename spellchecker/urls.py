from django.urls import path
from .views import SpellCheckerView

urlpatterns = [
    path('', SpellCheckerView.as_view(), name='spellchecker'),
]
