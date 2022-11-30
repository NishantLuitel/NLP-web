from .models import InputString
from rest_framework import serializers
from .forms import InputStringForm


class InputStringSerializer(serializers.ModelSerializer):
    class Meta:
        model = InputString
        fields = ['body', 'sentiment']
