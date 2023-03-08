from django.db import models

# Create your models here.


class InputString(models.Model):
    body = models.TextField()

    # num_words = models.IntegerField(default=1)

    def __str__(self):
        return self.body[0:5]+'...'
