import os
from django.urls import path
from . import views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),  # ✅ Home page
    path('profile-input/', views.profile_input, name='profile_input'),  # ✅ Handles username, URL, and image-based input
    path('check-profile/', views.check_profile, name='check_profile'),  # ✅ Profile check page
    path('upload-image/', views.upload_image, name='upload_image'),  # ✅ (Optional) Image upload route if you're using it separately
    
    # ✅ What's New feature
    path('whats-new/', views.whats_new, name='whats_new'),  # Renders what's new page
    path('get-cyber-news/', views.get_cyber_news, name='get_cyber_news'),  # Fetches news via API (AJAX call)
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

 


