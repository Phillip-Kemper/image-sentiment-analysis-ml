from rest_framework import serializers

from .models import ImageUpload


class ImageUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageUpload
        fields = '__all__'

    def create(self, validated_data):
        return ImageUpload.objects.create(**validated_data)



