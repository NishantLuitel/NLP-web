from django.forms import ModelForm
from .models import InputString
from django import forms


# class InputStringForm(ModelForm):
#     class Meta:
#         model = InputString
#         fields = ['body']


class InputStringForm(forms.Form):
    model_choices = [('knlm', 'Kn model(faster)'), ('transformer', 'Transformer(slower)')]

    body = forms.CharField()
    models = forms.ChoiceField(choices=model_choices, widget=forms.RadioSelect())

    # num_words = forms.IntegerField()
