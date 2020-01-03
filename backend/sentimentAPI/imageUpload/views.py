from django.http import HttpResponse
from django.shortcuts import render
from django.views.generic import TemplateView
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


@api_view(['GET', 'POST'])
def image_list(request):
    """
    List all code snippets, or create a new snippet.
    """
    if request.method == 'GET':
        image = ImageUpload.objects.all()
        serializer = ImageUploadSerializer(image, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET', 'PUT', 'DELETE'])
def image_detail(request, pk):
    """
    Retrieve, update or delete a code snippet.
    """
    try:
        image = ImageUpload.objects.get(pk=pk)
    except ImageUpload.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        serializer = ImageUploadSerializer(image)
        return Response(serializer.data)

    elif request.method == 'PUT':
        serializer = ImageUploadSerializer(image, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        image.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
