from rest_framework.response import Response
from rest_framework import generics
from .serializers import InputStringSerializer
from .forms import InputStringForm
from rest_framework import renderers
from rest_framework.views import APIView
from .process import reconstructed_model, testOwnString
from django.http import HttpResponseRedirect
from .models import InputString
from rest_framework.permissions import AllowAny

class MyBrowsableAPIRenderer(renderers.BrowsableAPIRenderer):
    # either
    def get_context(self, *args, **kwargs):
        context = super(MyBrowsableAPIRenderer, self).get_context(*args, **kwargs)
        context["post_form"] = InputStringForm()
        return context

class SentimentView(APIView):
    permission_classes = [AllowAny]
    renderer_classes = [renderers.JSONRenderer, MyBrowsableAPIRenderer, ]
    sentiments = ["Negative", "Positive", "Neutral"]

    def post(self, request):
        """
            Sentimental Classification of the input string
        """
        input = request.data.get('body')
        form = InputStringForm(request.data)
        # check if the input string is valid
        if form.is_valid():
            # get the probabilities of the sentiments of the input string
            probabilities = testOwnString(input)
            sentiment_index = probabilities.argmax()
            obj = InputString.objects.create(body=input, sentiment=self.sentiments[sentiment_index])

            return Response({"Sentiment Type": self.sentiments[sentiment_index]})
        return Response({"Sentiment Type": "None"})


