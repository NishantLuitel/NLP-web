#from django.shortcuts import render
#from django.views.generic import TemplateView
from rest_framework.views import APIView
from rest_framework.response import Response
# Create your views here.


class HomePageAPIView(APIView):

    def get(self, request, format=None):

        BoldMessage = {'bold_message': "Let's do NLP together!"}
        return Response(BoldMessage)


class AboutPageAPIView(APIView):

    def get(self, request, format=None):

        message = {'authors': ['ab', 'ns', 'nb', 'nl']}
        return Response(message)
