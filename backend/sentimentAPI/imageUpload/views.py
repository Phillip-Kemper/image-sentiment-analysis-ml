from django.http import HttpResponse, Http404
from django.shortcuts import render
from django.views.generic import TemplateView
from rest_framework.request import Request
from rest_framework.views import APIView

from .serializers import ImageUploadSerializer
from rest_framework import viewsets, generics
from .models import ImageUpload
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser, FileUploadParser
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
#class ImageUploadViewSet(viewsets.ModelViewSet):
#    queryset = ImageUpload.objects.all().order_by('name')
#    serializer_class = ImageUploadSerializer

class ImageUploadView(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request, *args, **kwargs):

      serializer_context = {
            'request': request,
      }
      file_serializer = ImageUploadSerializer(data=request.data,context=serializer_context)


#      print(request.data.dict()['file'])
      print(request.data)
      if file_serializer.is_valid():
          file_serializer.save()
          return Response(file_serializer.data, status=status.HTTP_201_CREATED)
      else:
          return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


#class ImageUploadList(generics.ListCreateAPIView):
#    queryset = ImageUpload.objects.all()
#    serializer_class = ImageUploadSerializer
#
#
#class ImageUploadDetail(generics.RetrieveUpdateDestroyAPIView):
#    queryset = ImageUpload.objects.all()
#    serializer_class = ImageUploadSerializer
#