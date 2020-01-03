from django.db import models
import uuid
from s3direct.fields import S3DirectField

# Create your models here.


class ImageUpload(models.Model):
    name = models.CharField(max_length=255, blank=False, null=False, default='pic')
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image = S3DirectField(dest='primary_destination', blank=True)
    created_at = models.DateTimeField(auto_now_add=True)



