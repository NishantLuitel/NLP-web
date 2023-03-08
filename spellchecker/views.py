import sys
import os

from rest_framework.response import Response
from rest_framework import generics
from .serializers import InputStringSerializer
from .forms import InputStringForm
from rest_framework import renderers
from rest_framework.views import APIView
from django.http import HttpResponseRedirect
from .models import InputString
from rest_framework.permissions import AllowAny
from .load_lm import kn_lm2

absolute_path = os.path.dirname(__file__)
relative_path = "process"
full_path = os.path.join(absolute_path, relative_path)
sys.path.insert(0, os.path.join(absolute_path, relative_path))
if True:
    from corrector import extract_choices
    from transformer import model as transformer_lm


class MyBrowsableAPIRenderer(renderers.BrowsableAPIRenderer):
    # either
    def get_context(self, *args, **kwargs):
        context = super(MyBrowsableAPIRenderer, self).get_context(*args, **kwargs)
        context["post_form"] = InputStringForm()
        return context


class SpellCheckerView(APIView):
    permission_classes = [AllowAny]
    renderer_classes = [renderers.JSONRenderer, MyBrowsableAPIRenderer, ]
    # sentiments = ["Negative", "Positive", "Neutral"]

    def post(self, request):
        """
            Sentimental Classification of the input string
        """
        input = request.data.get('body')
        model_type = request.data.get('models')
        if model_type == 'knlm':
            model = kn_lm2
        elif model_type == 'transformer':
            model = transformer_lm

        form = InputStringForm(request.data)
        # check if the input string is valid
        print(form.is_valid())
        if form.is_valid():
            # First convert to indices using data_process function and unsqueeze to match dimension

            predicted_choices = extract_choices(
                input, model=model, p_lambda=1.5, trie=True, likelihood='bm', model_type=model_type)
            # generated_string = ' '.join(generated_list)

            # obj = InputString.objects.create(body=input, sentimenst=self.sentiments[sentiment_index])

            return Response({"Input String": input, "Choices list": predicted_choices[:-1]})
        return Response({"Input String": input, "Choices list": "Fill all the inputs"})
