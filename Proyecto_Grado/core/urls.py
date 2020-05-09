from django.urls import path
from . import views

urlpatterns = [
    path('buscar/<palabras>', views.Buscar_Twitter),
    path('Inicio', views.Inicio_Buscar),
]