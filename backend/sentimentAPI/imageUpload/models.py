import cv2
import numpy
from PIL import Image
from django.db import models
import uuid
from .imageTransform import transformColorAndDimension
from s3direct.fields import S3DirectField

# Create your models here.



class ImageUpload(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image = models.ImageField(blank=False, null=False)






