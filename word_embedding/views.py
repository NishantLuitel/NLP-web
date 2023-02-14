from django.shortcuts import render

from rest_framework.views import APIView
from rest_framework.response import Response
import pandas as pd


class TwoWordEmbeddingView(APIView):
    def get(self, request):
        """
            Return the 2D and 3D coordinates
        """
        two_coordinate = pd.read_csv("NLP_Trained_models/word_embedding/2d_coordinates.csv")
        return Response({"2d" : two_coordinate.to_dict(orient="record")})


class ThreeWordEmbeddingView(APIView):
    def get(self, request):
        three_coordinate = pd.read_csv("NLP_Trained_models/word_embedding/3d_coordinates.csv")
        return Response({"3d" : three_coordinate.to_dict(orient="records")})