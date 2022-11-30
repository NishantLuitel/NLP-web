from django.forms import ModelForm
from .models import InputString


class InputStringForm(ModelForm):
    class Meta:
        model = InputString
        fields = ['body']
