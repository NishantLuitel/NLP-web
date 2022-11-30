from rest_framework.response import Response
from rest_framework import generics
from .serializers import InputStringSerializer
from .forms import InputStringForm
from rest_framework import renderers
from rest_framework.views import APIView
from .process import reconstructed_model, preprocess
from django.http import HttpResponseRedirect
from .models import InputString
from rest_framework.permissions import IsAdminUser

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


class SentimentView(APIView):
    renderer_classes = [renderers.JSONRenderer, MyBrowsableAPIRenderer, ]
    sentiments = ["Negative", "Positive", "Neutral"]

    def get(self, request):

        return Response({"Sentiment Type": "None"})

    def post(self, request):
        input = request.data.get('body')

        form = InputStringForm(request.data)
        if form.is_valid():
            processed_input = preprocess(input)
            output = reconstructed_model.predict(processed_input)
            sentiment_index = output.argmax()
            # Save for List View
            obj = InputString.objects.create(body=input, sentiment=self.sentiments[sentiment_index])

            return Response({"Sentiment Type": self.sentiments[sentiment_index]})
        return Response({"Sentiment Type": "None"})


class SentimentListView(generics.ListAPIView):
    queryset = InputString.objects.all()
    serializer_class = InputStringSerializer
    permission_classes = [IsAdminUser]

# serializer = InputStringSerializer(input, data=request.data)
# if not serializer.is_valid():
#     return Response({'serializer': serializer})
# serializer.save()
# return redirect('sentiment-home')
