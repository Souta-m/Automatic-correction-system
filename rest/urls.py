from django.conf.urls import url
from django.urls import path
from .import views

urlpatterns = [
    path("", views.TopView.as_view(), name="top"),
    path('forecast/', views.forecast, name='forecast'),
    path("index/", views.HomeView.as_view(), name="index"),
    path("login/", views.LoginView.as_view(), name="login"),
    path("logout/", views.LogoutView.as_view(), name="logout"),
    path("system",views.system,name="system"),
    path("problem/", views.problem, name="problem"),
    path("usage/", views.usage, name="usage"),
    path("eval_system/", views.eval_system, name="eval_syatem"),
    path("experiments/", views.experiments, name="experiments"),
    path("evaluations/", views.evaluations, name="evaluations"),
    path("final_evaluation/", views.final_evaluation, name="final_evaluation"),
    path("finish/", views.final_evaluation_write, name="finish"),
]

#    path('', views.rest, name='rest'),
