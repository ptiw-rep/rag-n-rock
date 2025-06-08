import os
import logging
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader, CSVLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document

ALLOWED_FILE_EXTENSIONS = {'.pdf', '.docx', '.txt', '.csv', '.xlsx'}

class RAGPipeline:
    def __init__(self, vector_db_path: str = "./chroma_db", chunking_strategy: str = "auto"):
        """
        :param vector_db_path: Path for ChromaDB persistence
        :param chunking_strategy: 'auto', 'header', or 'character'. If 'auto', use header-based for markdown, otherwise fallback.
        """
        self.vector_db_path = vector_db_path
        # TODO: Take embedding model as function parameter
        # Soln: In main, create two objects, one for embeddings and one for LLM, and pass them here
        # Fix the types of parameters to OllamaEmbeddings and OllamaLLM type class
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = Chroma(persist_directory=self.vector_db_path, embedding_function=self.embeddings)
        # TODO: Take LLM model as function parameter
        self.llm = OllamaLLM(model="llama3")
        self.chunking_strategy = chunking_strategy
        logging.info(f"Initialized RAGPipeline.")

    def _get_text_splitter(self, docs, file_path: str):
        """
        Use MarkdownHeaderTextSplitter if markdown, else fallback to RecursiveCharacterTextSplitter.
        """
        ext = os.path.splitext(file_path)[-1].lower()
        if self.chunking_strategy == "header" or (self.chunking_strategy == "auto" and ext in ['.md', '.markdown']):
            try:
                return MarkdownHeaderTextSplitter(headers_to_split_on=["#", "##", "###"], chunk_size=1000, chunk_overlap=100)
            except Exception as e:
                logging.warning(f"Header splitter failed: {e}, falling back to character splitter.")
        # Fallback
        return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    def load_document(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.docx':
            loader = UnstructuredWordDocumentLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path)
        elif ext == '.csv':
            loader = CSVLoader(file_path)
        elif ext == '.xlsx':
            loader = UnstructuredExcelLoader(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        docs = loader.load()
        return docs

    def ingest(self, file_path: str, metadata: dict = None):
        """
        Ingests a file using adaptive chunking (header-based for markdown, otherwise character-based).
        """
        docs = self.load_document(file_path)
        splitter = self._get_text_splitter(docs, file_path)
        splits = splitter.split_documents(docs)
        # Attach file-level metadata
        for doc in splits:
            doc.metadata = doc.metadata or {}
            if metadata:
                doc.metadata.update(metadata)
            doc.metadata['source_file'] = file_path
        self.vectorstore.add_documents(splits)
        logging.info(f"Ingested {len(splits)} chunks from {file_path} using {splitter.__class__.__name__}")

    def retrieve(self, query: str, k: int = 4, keywords: Optional[list] = None, metadata_filter: Optional[dict] = None) -> List[Document]:
        """
        Hybrid retrieval: combines vector search, keyword, and metadata filtering. Always uses strict top-k retrieval (no MMR).
        :param query: user query
        :param k: number of results
        :param keywords: list of keywords to boost/filter
        :param metadata_filter: dict of metadata filters (e.g. {"source_file": ...})
        """
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        if metadata_filter:
            results = retriever.invoke(query, filter=metadata_filter)
        else:
            results = retriever.invoke(query)
        # Keyword filter/boost
        if keywords:
            keyword_results = [doc for doc in results if any(kw.lower() in doc.page_content.lower() for kw in keywords)]
            # Merge, deduplicate, and sort (keyword hits first)
            unique = {id(doc): doc for doc in keyword_results + results}
            results = list(unique.values())[:k]
        logging.info(f"Hybrid retrieval for query '{query}': {len(results)} docs (strict top-k, keywords={keywords}, metadata={metadata_filter})")
        return results
