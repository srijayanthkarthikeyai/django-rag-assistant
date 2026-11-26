from django.urls import path
from django.contrib.auth import views as auth_views
from .views import (
    chat_view,
    register_view,
    rag_stream_view,
    upload_document_view,
    document_list_view,
    document_delete_view,
    session_list_create_view,
    session_detail_view,
)

urlpatterns = [
    #Auth
    path('login/', auth_views.LoginView.as_view(template_name='documents/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('register/', register_view, name='register'),

    #Chat & RAG
    path('', chat_view, name='chat_ui'),
    path('api/rag/stream/', rag_stream_view, name='rag_stream'),
    path('api/upload/', upload_document_view, name='upload_document'),

    #Documents
    path('api/documents/', document_list_view, name='document_list'),
path('api/documents/delete/', document_delete_view, name='document_delete'),

    #Sessions
    path('api/sessions/', session_list_create_view, name='session_list_create'),
    path('api/sessions/<uuid:session_id>/', session_detail_view, name='session_detail'),
]