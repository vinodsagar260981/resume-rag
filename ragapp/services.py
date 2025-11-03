import logging
from django.conf import settings
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for managing embeddings"""

    def __init__(self, model_name, base_url):
        self._embeddings = OllamaEmbeddings(
            model=model_name,
            base_url=base_url
        )

    @property
    def embeddings(self):
        if self._embeddings is None:
            try:
                self._embeddings = OllamaEmbeddings(
                    model=self.model_name,
                    base_url=self.base_url,
                )
                logger.info(f"✅ Loaded embeddings model: {self.model_name}")
            except Exception as e:
                logger.exception("❌ Error initializing embeddings")
                raise
        return self._embeddings


class VectorStoreService:
    """Service for managing the vector store"""

    def __init__(self, embedding_service):
        rag_cfg = settings.RAG_CONFIG
        self.embedding_service = embedding_service
        self.persistent_directory = rag_cfg.get('persistent_directory', 'chroma_store')
        self._db = None

    @property
    def db(self):
        if self._db is None:
            try:
                self._db = Chroma(
                    persist_directory=self.persistent_directory,
                    embedding_function=self.embedding_service.embeddings,
                    collection_metadata={"hnsw:space": "cosine"},
                )
                logger.info("✅ Vector store initialized successfully")
            except Exception as e:
                logger.exception("❌ Error initializing Chroma vector store")
                raise
        return self._db

    def search_documents(self, query, k=None, score_threshold=None):
        """Search for relevant documents"""
        cfg_kwargs = settings.RAG_CONFIG.get('search_kwargs', {'k': 5, 'score_threshold': 0.5}).copy()
        if k is not None:
            cfg_kwargs['k'] = k
        if score_threshold is not None:
            cfg_kwargs['score_threshold'] = score_threshold

        try:
            retriever = self.db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs=cfg_kwargs
            )
            relevant_docs = retriever.invoke(query)
            logger.info(f"🔍 Found {len(relevant_docs)} relevant documents for query: {query}")
            return relevant_docs
        except Exception as e:
            logger.exception("❌ Error during vector search")
            raise


class LLMService:
    """Service for managing LLM interactions"""

    def __init__(self, model_name, base_url):
        self._model = ChatOllama(
            model=model_name,
            base_url=base_url
        )

    @property
    def model(self):
        if self._model is None:
            try:
                self._model = ChatOllama(
                    model=self.model_name,
                    base_url=self.base_url,
                    api_key=self.api_key,  # ✅ ChatOllama supports api_key
                )
                logger.info(f"✅ Loaded LLM model: {self.model_name}")
            except Exception as e:
                logger.exception("❌ Error initializing LLM model")
                raise
        return self._model

    def generate_response(self, query, documents):
        """Generate a response from query and retrieved documents"""
        doc_content = "\n".join([f"- {doc.page_content}" for doc in documents])

        combined_input = f"""
                        Based on the following documents, please answer this question: "{query}"
                        
                        Documents:
                        {doc_content}
                        
                        Provide a clear, helpful answer using only the given documents.
                        If not enough information is available, say: 
                        "I don’t have enough information to answer that question based on the provided documents."
                        """

        try:
            messages = [
                SystemMessage(content="You are a helpful and concise AI assistant."),
                HumanMessage(content=combined_input),
            ]
            result = self.model.invoke(messages)
            response_text = getattr(result, 'content', str(result))
            logger.info(f"💬 Generated response for query: {query}")
            return response_text.strip()
        except Exception as e:
            logger.exception("❌ Error generating LLM response")
            raise


class RAGService:
    """Main RAG service orchestrating all components"""

    def __init__(self):
        config = settings.RAG_CONFIG

        # Initialize embedding and LLM services with correct args
        self.embedding_service = EmbeddingService(
            model_name=config['embedding_model'],
            base_url=config['ollama_base_url']
        )

        self.vector_store_service = VectorStoreService(self.embedding_service)

        self.llm_service = LLMService(
            model_name=config['llm_model'],
            base_url=config['ollama_base_url']
        )

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
                        'metadata': getattr(doc, 'metadata', {})
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

