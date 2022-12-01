from rest_framework.response import Response
from .forms import InputStringForm
from rest_framework import renderers
from rest_framework.views import APIView
from .process import suggest, preprocess


class MyBrowsableAPIRenderer(renderers.BrowsableAPIRenderer):
    def get_context(self, *args, **kwargs):
        context = super(MyBrowsableAPIRenderer, self).get_context(*args, **kwargs)
        context["post_form"] = InputStringForm()
        return context


class NgramView(APIView):
    renderer_classes = [renderers.JSONRenderer, MyBrowsableAPIRenderer, ]

    def get(self, request):

        return Response({"N_gram": "Probabilistic Model"})

    def post(self, request):
        input = request.data.get('body')
        if input == '':
            print("Input String", input)
        else:
            print("All Fine:", input)
        n_gram_index = request.data.get('n_from_ngram')
        print(n_gram_index)

        form = InputStringForm(request.data)
        if form.is_valid():
            print("true")
            processed_input = preprocess(input)
            print("processed_input:", input)
            output = suggest(processed_input, [int(n_gram_index)])
            print('Output:', output)
            return Response({"Predicted Tokens": output})
        return Response({"None": "None"})
