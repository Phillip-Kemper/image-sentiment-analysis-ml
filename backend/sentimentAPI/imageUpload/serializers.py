from rest_framework import serializers

from .models import ImageUpload

class ImageUploadSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = ImageUpload
        fields = ('name', 'image')


