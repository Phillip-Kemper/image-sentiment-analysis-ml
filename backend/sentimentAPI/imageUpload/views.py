from django.http import HttpResponse
from django.shortcuts import render
from django.views.generic import TemplateView
from .serializers import ImageUploadSerializer
from rest_framework import viewsets
from .models import ImageUpload
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser

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

@csrf_exempt
def image_list(request):
    """
    List all code snippets, or create a new snippet.
    """
    if request.method == 'GET':
        snippets = ImageUpload.objects.all()
        serializer = ImageUploadSerializer(snippets, many=True)
        return JsonResponse(serializer.data, safe=False)

    elif request.method == 'POST':
        data = JSONParser().parse(request)
        serializer = ImageUploadSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data, status=201)
        return JsonResponse(serializer.errors, status=400)


@csrf_exempt
def image_detail(request, pk):
    """
    Retrieve, update or delete a code snippet.
    """
    try:
        snippet = ImageUpload.objects.get(pk=pk)
    except ImageUpload.DoesNotExist:
        return HttpResponse(status=404)

    if request.method == 'GET':
        serializer = ImageUploadSerializer(snippet)
        return JsonResponse(serializer.data)

    elif request.method == 'PUT':
        data = JSONParser().parse(request)
        serializer = ImageUploadSerializer(snippet, data=data)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data)
        return JsonResponse(serializer.errors, status=400)

    elif request.method == 'DELETE':
        snippet.delete()
        return HttpResponse(status=204)
