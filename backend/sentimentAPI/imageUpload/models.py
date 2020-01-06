from django.db import models
import uuid
from s3direct.fields import S3DirectField

# Create your models here.


class ImageUpload(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    file = models.FileField(blank=False, null=False)
    created_at = models.DateTimeField(auto_now_add=True)





