from django.urls import path
from .views import RAGQueryView, RAGQueryWebView

app_name = 'rag_app'

urlpatterns = [
    path('api/query/', RAGQueryView.as_view(), name='api-query'),
    path('query/', RAGQueryWebView.as_view(), name='web-query'),
]