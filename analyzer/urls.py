from django.urls import path
from .views import index, analyze_face

urlpatterns = [
    path('', index, name='index'),
    path('analyze/', analyze_face, name='analyze_face'),
]
