from django.db import models

# Create your models here.


class InputString(models.Model):
    body = models.TextField()

    def __str__(self):
        return self.body[0:5]+'...'
