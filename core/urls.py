from django.urls import path

from . import views

urlpatterns = [
    path('validate', views.ValidateView.as_view(), name='validate'),
]