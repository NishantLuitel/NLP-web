from .models import InputString
from django.forms import ModelForm


class InputStringForm(ModelForm):
    class Meta:
        model = InputString
        fields = ['body', 'n_from_ngram']
