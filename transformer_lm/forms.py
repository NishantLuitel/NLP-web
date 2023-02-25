from django.forms import ModelForm
from .models import InputString
from django import forms


# class InputStringForm(ModelForm):
#     class Meta:
#         model = InputString
#         fields = ['body']

class InputStringForm(forms.Form):
    body = forms.CharField()
    num_words = forms.IntegerField()
