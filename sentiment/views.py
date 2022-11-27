# from django.shortcuts import render
from rest_framework import generics
from rest_framework import mixins

from .models import InputString
from .serializers import InputStringSerializer
from rest_framework.response import Response
from .serializers import InputStringSerializer
from .forms import InputStringForm


from rest_framework import renderers
from rest_framework.views import APIView
# Create your views here.


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
    # renderers.TemplateHTMLRenderer,
    # renderers.MyBrowsableAPIRenderer,
    # template_name = 'rest_framework/horizontal.html'

    # template_name = 'inputform.html'

    def get(self, request):
        serializer = InputStringSerializer()
        return Response({'serializer': serializer, })

    def post(self, request):
        input = request.data.get('body')
        return Response({"your string": input})
        # serializer = InputStringSerializer(input, data=request.data)
        # if not serializer.is_valid():
        #     return Response({'serializer': serializer})
        # serializer.save()
        # return redirect('sentiment-home')
