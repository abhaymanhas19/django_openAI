from apps.users import views
from django.urls import path
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('',views.Home.as_view(),name='home'),
    path('register/',views.Register.as_view(), name="register"),
    path('login/',views.LoginView.as_view(), name="login"),
    path('logout/',views.LogoutView.as_view(), name="logout"),
    path("change-password/",views.ChangePassword.as_view(), name="change-password"),
    path('user-profile/', views.UserProfile.as_view(),name="user-profile"),


    path('get-form/',views.UploadFileChoice.as_view(),name='get-form'),
    # path('show-files/<str:value>',views.ShowFiles.as_view(), name="show-files"),

    path('get-choices/',views.Getchoices.as_view(),name='get-choices'),
    path('query/',views.UserQuery.as_view(),name='query'),
    path('processing/',views.ProcessQuery.as_view(),name='processing'),
  
    
]+static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)+static(settings.STATIC_URL, document_root = settings.STATIC_ROOT)

    