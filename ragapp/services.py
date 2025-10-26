from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from django.conf import settings
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for managing embeddings"""

    def __init__(self):
        self.model_name = settings.RAG_CONFIG['embedding_model']
        self._embeddings = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            try:
                self._embeddings = OllamaEmbeddings(model=self.model_name)
                logger.info(f"Loaded embeddings model: {self.model_name}")
            except Exception as e:
                logger.error(f"Error loading embeddings: {e}")
                raise
        return self._embeddings


class VectorStoreService:
    """Service for managing the vector store"""

    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        self.persistent_directory = settings.RAG_CONFIG['persistent_directory']
        self._db = None

    @property
    def db(self):
        if self._db is None:
            try:
                self._db = Chroma(
                    persist_directory=self.persistent_directory,
                    embedding_function=self.embedding_service.embeddings,
                    collection_metadata={"hnsw:space": "cosine"}
                )
                logger.info("Vector store loaded successfully")
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
                raise
        return self._db

    def search_documents(self, query, k=None, score_threshold=None):
        """Search for relevant documents"""
        search_kwargs = settings.RAG_CONFIG['search_kwargs'].copy()

        if k is not None:
            search_kwargs['k'] = k
        if score_threshold is not None:
            search_kwargs['score_threshold'] = score_threshold

        try:
            retriever = self.db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs=search_kwargs
            )
            relevant_docs = retriever.invoke(query)
            logger.info(f"Found {len(relevant_docs)} relevant documents for query: {query}")
            return relevant_docs
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise


class LLMService:
    """Service for managing LLM interactions"""

    def __init__(self):
        self.model_name = settings.RAG_CONFIG['llm_model']
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                self._model = ChatOllama(model=self.model_name)
                logger.info(f"Loaded LLM model: {self.model_name}")
            except Exception as e:
                logger.error(f"Error loading LLM: {e}")
                raise
        return self._model

    def generate_response(self, query, documents):
        """Generate response based on query and documents"""
        doc_content = "\n".join([f"- {doc.page_content}" for doc in documents])

        combined_input = f"""Based on the following documents, please answer this question: {query}

        Documents:
        {doc_content}
        
        Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
        """

        try:
            messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=combined_input),
            ]
            result = self.model.invoke(messages)
            logger.info(f"Generated response for query: {query}")
            return result.content
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise


class RAGService:
    """Main RAG service orchestrating all components"""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store_service = VectorStoreService(self.embedding_service)
        self.llm_service = LLMService()

    def query(self, user_query, k=None, score_threshold=None):
        """Execute a complete RAG query"""
        try:
            # Search for relevant documents
            relevant_docs = self.vector_store_service.search_documents(
                user_query, k=k, score_threshold=score_threshold
            )

            if not relevant_docs:
                return {
                    'query': user_query,
                    'documents': [],
                    'response': "No relevant documents found for your query.",
                    'success': True
                }

            # Generate response
            response = self.llm_service.generate_response(user_query, relevant_docs)

            return {
                'query': user_query,
                'documents': [
                    {
                        'id': i,
                        'content': doc.page_content,
                        'metadata': doc.metadata if hasattr(doc, 'metadata') else {}
                    }
                    for i, doc in enumerate(relevant_docs, 1)
                ],
                'response': response,
                'success': True
            }
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return {
                'query': user_query,
                'documents': [],
                'response': f"Error processing query: {str(e)}",
                'success': False,
                'error': str(e)
            }