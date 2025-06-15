from langchain_ollama import OllamaEmbeddings, OllamaLLM

class LLMProvider:
    def __init__(self, embedding_model: str = "nomic-embed-text", llm_model: str = "llama3"):
        self.embedding_model = embedding_model
        self.llm_model = llm_model

    def get_embeddings_model(self) -> OllamaEmbeddings:
        return OllamaEmbeddings(model=self.embedding_model)

    def get_inference_model(self) -> OllamaLLM:
        return OllamaLLM(model=self.llm_model)