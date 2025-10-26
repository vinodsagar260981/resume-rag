from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from django.views.generic import TemplateView
from .services import RAGService
import logging

logger = logging.getLogger(__name__)


class RAGQueryView(APIView):
    """REST API endpoint for RAG queries"""
    permission_classes = [AllowAny]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rag_service = RAGService()

    def post(self, request):
        """
        POST request handler for RAG queries
        Expected JSON: {"query": "your question", "k": 5, "score_threshold": 0.3}
        """
        try:
            query = request.data.get('query')
            k = request.data.get('k')
            score_threshold = request.data.get('score_threshold')

            if not query:
                return Response(
                    {'error': 'Query parameter is required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            result = self.rag_service.query(
                user_query=query,
                k=k,
                score_threshold=score_threshold
            )

            if result['success']:
                return Response(result, status=status.HTTP_200_OK)
            else:
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        except Exception as e:
            logger.error(f"Error in RAGQueryView: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class RAGQueryWebView(TemplateView):
    """Web interface for RAG queries"""
    template_name = 'rag_app/query.html'