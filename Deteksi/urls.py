from django.urls import path
from . import views

urlpatterns = [
    path('Deteksi/', views.predict, name='predict'),
]