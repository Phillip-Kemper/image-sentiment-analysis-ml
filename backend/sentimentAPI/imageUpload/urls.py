from django.urls import path,include

from . import views
from rest_framework import routers

router = routers.DefaultRouter()
router.register(r'images', views.ImageUploadViewSet)

urlpatterns = [
    path('snippets/', views.ImageUploadList.as_view()),
    path('snippets/<int:pk>/', views.ImageUploadDetail.as_view()),
    path('', include(router.urls)),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]