from rest_framework.response import Response
from rest_framework import generics
from .serializers import InputStringSerializer
from .forms import InputStringForm
from rest_framework import renderers
from rest_framework.views import APIView
from .process import best_model, nonnaive_generator, data_process, device
from django.http import HttpResponseRedirect
from .models import InputString
from rest_framework.permissions import AllowAny


class MyBrowsableAPIRenderer(renderers.BrowsableAPIRenderer):
    # either
    def get_context(self, *args, **kwargs):
        context = super(MyBrowsableAPIRenderer, self).get_context(*args, **kwargs)
        context["post_form"] = InputStringForm()
        return context


class TransformerLmView(APIView):
    permission_classes = [AllowAny]
    renderer_classes = [renderers.JSONRenderer, MyBrowsableAPIRenderer, ]
    # sentiments = ["Negative", "Positive", "Neutral"]

    def post(self, request):
        """
            Sentimental Classification of the input string
        """
        input = [request.data.get('body')]
        num_words = int(request.data.get('num_words'))

        form = InputStringForm(request.data)
        # check if the input string is valid
        if form.is_valid():
            # First convert to indices using data_process function and unsqueeze to match dimension
            stoi = data_process(input)
            stoi = stoi.unsqueeze(1).to(device)

            generated_list = nonnaive_generator(best_model, stoi, no_words=num_words, k=50)
            generated_string = ' '.join(generated_list)

            # obj = InputString.objects.create(body=input, sentimenst=self.sentiments[sentiment_index])

            return Response({"Input String": input[0], "Generated String": generated_string})
        return Response({"Input String": input[0], "Generated String": "None"})
