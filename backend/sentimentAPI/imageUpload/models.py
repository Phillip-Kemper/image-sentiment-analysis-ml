import cv2
import numpy
from PIL import Image
from django.db import models
import uuid
import numpy as np
from keras.models import load_model

import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from s3direct.fields import S3DirectField
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True
from django.dispatch import receiver
from django.db.models.signals import post_save


# Create your models here.



class ImageUpload(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image = models.ImageField(blank=False, null=False)
    sentiment = models.IntegerField(default=0,blank=False)
    probability = models.IntegerField(default=0, blank=False)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

        sentiment, probability = self.predict(self)
        self.sentiment = sentiment
        self.probability = probability



    def predict(self,*args, **kwargs):
        tb._SYMBOLIC_SCOPE.value = True
        model = load_model('fer2013_model_2.h5')
        model.summary()
        filepath = 'media/'
        filename = self.image.name.rsplit('/', 1)[-1]


        img = Image.open(os.path.join(filepath, filename))
        img = img.resize((48, 48))
        img = np.copy(np.asarray(img)).astype(float)
        y = 0
        while (y < len(img)):
            x = 0
            while (x < len(img[y])):
                rgb = 0
                addvals = 0
                while (rgb < len(img[y][x])):
                    img[y][x][rgb] = img[y][x][rgb] / 255
                    rgb += 1
                x += 1
            y += 1
        plt.imshow(img)
        plt.show()
        print(filename)
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        print(prediction)
        sentiment = np.argmax(prediction)
        probability = prediction[0][sentiment]
        return sentiment, probability







