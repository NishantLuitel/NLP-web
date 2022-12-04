from django.db import models

class InputString(models.Model):
    body = models.TextField()
    n_from_ngram = models.IntegerField()

    def __str__(self):
        return self.body[0:5]+'...'
