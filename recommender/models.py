from django.db import models

# recommender/models.py


class Track(models.Model):
    track_id = models.CharField(max_length=50, unique=True)
    title = models.CharField(max_length=100)
    artist = models.CharField(max_length=100)
    genre = models.CharField(max_length=50)
    mood = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.title} by {self.artist}"
