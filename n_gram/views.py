from rest_framework.response import Response
from .forms import InputStringForm
from rest_framework import renderers
from rest_framework.views import APIView
from django.shortcuts import redirect, reverse

# class SentimentViewList(generics.ListCreateAPIView):
#    serializer_class = InputStringSerializer
#    queryset = InputString.objects.all()
# class SentimentViewList(generics.CreateAPIView):
#    serializer_class = InputStringSerializer
#    queryset = InputString.objects.all()


class MyBrowsableAPIRenderer(renderers.BrowsableAPIRenderer):
    # either
    def get_context(self, *args, **kwargs):
        context = super(MyBrowsableAPIRenderer, self).get_context(*args, **kwargs)
        context["post_form"] = InputStringForm()
        return context


class NgramView(APIView):
    renderer_classes = [renderers.JSONRenderer, MyBrowsableAPIRenderer, ]
    # sentiments = ["Negative", "Positive", "Neutral"]

    def post(self, request):
        # input = request.data.get('body')
        #
        # form = InputStringForm(request.data)
        # if form.is_valid():
        #     processed_input = preprocess(input)
        #     output = reconstructed_model.predict(processed_input)
        #     sentiment_index = output.argmax()
        return Response({"Sentiment Type": "Dummy"})

        # serializer = InputStringSerializer(input, data=request.data)
        # if not serializer.is_valid():
        #     return Response({'serializer': serializer})
        # serializer.save()
        # return redirect('sentiment-home')
