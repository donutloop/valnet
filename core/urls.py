from django.conf.urls import url

from . import views

urlpatterns = [
    url('validate', views.ValidateView.as_view(), name='validate'),
]