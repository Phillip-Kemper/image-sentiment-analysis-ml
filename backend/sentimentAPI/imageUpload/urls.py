from django.urls import path,include

from imageUpload.views import ImageUploadView
from . import views
from rest_framework import routers

router = routers.DefaultRouter()
router.register(r'images', views.ImageUploadViewSet)

urlpatterns = [
#    path('images/', views.ImageUploadList.as_view()),
#    path('', ImageUploadView.as_view())
     path('get/<pk>/', views.ImageUploadDetail.as_view()),
     path('', include(router.urls)),
#    path('api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]


