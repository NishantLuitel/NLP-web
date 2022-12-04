from rest_framework.response import Response
from .forms import InputStringForm
from rest_framework import renderers
from rest_framework.views import APIView
from .process import suggest


class MyBrowsableAPIRenderer(renderers.BrowsableAPIRenderer):
    def get_context(self, *args, **kwargs):
        context = super(MyBrowsableAPIRenderer, self).get_context(*args, **kwargs)
        context["post_form"] = InputStringForm()
        return context

class NgramView(APIView):
    renderer_classes = [renderers.JSONRenderer, MyBrowsableAPIRenderer, ]

    def post(self, request):
        """
            Return the required no. of suggestions for the previous n-gram
        """
        input = request.data.get('body')

        if input == "":
            return Response({"Status" : "Empty String"})

        form = InputStringForm(request.data)
        if form.is_valid():
            # preprocessing of input into list of words and get the required suggestions of the word list with probabilities
            processed_input = input.split(' ')
            output = suggest(processed_input)
            return Response({"Predicted Tokens": output})
            
