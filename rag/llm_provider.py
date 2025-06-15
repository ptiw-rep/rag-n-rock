from langchain_ollama import OllamaEmbeddings, OllamaLLM

from util import logger

class LLMProvider:
    def __init__(self, embedding_model: str = "nomic-embed-text:latest", llm_model: str = "gemma3:4b"):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        logger.info(f"Initializing LLMProvider with:")
        logger.info(f"  Embedding Model: {self.embedding_model}")
        logger.info(f"  Inference Model: {self.llm_model}")

    def get_embeddings_model(self) -> OllamaEmbeddings:
        logger.debug(f"Loading embedding model: {self.embedding_model}")
        try:
            embeddings = OllamaEmbeddings(model=self.embedding_model)
            logger.info(f"Embedding model '{self.embedding_model}' loaded successfully")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to load embedding model '{self.embedding_model}': {str(e)}", exc_info=True)
            raise

    def get_inference_model(self) -> OllamaLLM:
        logger.debug(f"Loading inference model: {self.llm_model}")
        try:
            llm = OllamaLLM(model=self.llm_model)
            logger.info(f"Inference model '{self.llm_model}' loaded successfully")
            return llm
        except Exception as e:
            logger.error(f"Failed to load inference model '{self.llm_model}': {str(e)}", exc_info=True)
            raise