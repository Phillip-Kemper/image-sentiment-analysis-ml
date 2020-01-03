from django.http import HttpResponse, Http404
from django.shortcuts import render
from django.views.generic import TemplateView
from rest_framework.views import APIView

from .serializers import ImageUploadSerializer
from rest_framework import viewsets
from .models import ImageUpload
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

# Create your views here.

#def index(self):
#    return HttpResponse("Hello, world. You're at the polls index.")
#def add(self):
#    return HttpResponse("Hello, world. You're at the polls index.")
#def upload(self):
#    return HttpResponse("Hello, world. You're at the polls index. 2")
#
class ImageUploadViewSet(viewsets.ModelViewSet):
    queryset = ImageUpload.objects.all().order_by('name')
    serializer_class = ImageUploadSerializer


class ImageUploadList(APIView):
    """
    List all snippets, or create a new snippet.
    """
    def get(self, request, format=None):
        image = ImageUpload.objects.all()
        serializer = ImageUploadSerializer(image, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ImageUploadDetail(APIView):
    """
    Retrieve, update or delete a snippet instance.
    """
    def get_object(self, pk):
        try:
            return ImageUpload.objects.get(pk=pk)
        except ImageUpload.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        image = self.get_object(pk)
        serializer = ImageUploadSerializer(image)
        return Response(serializer.data)

    def put(self, request, pk, format=None):
        snippet = self.get_object(pk)
        serializer = ImageUploadSerializer(snippet, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, format=None):
        snippet = self.get_object(pk)
        snippet.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
