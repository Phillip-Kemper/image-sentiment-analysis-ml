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
    file = models.ImageField(blank=False, null=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        print('test')
        # there is an error here
        open_cv_image = numpy.array(self.file)
        open_cv_image = transformColorAndDimension(open_cv_image)
        print(open_cv_image)
#        img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
        self.file = Image.fromarray(open_cv_image)
        super().save(*args, **kwargs)





