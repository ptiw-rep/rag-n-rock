import os
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader, CSVLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

from util import logger
from rag.llm_provider import LLMProvider

ALLOWED_FILE_EXTENSIONS = {'.pdf', '.docx', '.txt', '.csv', '.xlsx'}

class RAGPipeline:
    def __init__(
            self,
            model_provider: Optional[LLMProvider] = None,
            vector_db_path: str = "./chroma_db", 
            chunking_strategy: str = "auto"
            ):
        """
        :param vector_db_path: Path for ChromaDB persistence
        :param chunking_strategy: 'auto', 'header', or 'character'. If 'auto', use header-based for markdown, otherwise fallback.
        """
        logger.info("Initializing RAGPipeline")
        self.vector_db_path = vector_db_path
        self.model_provider = model_provider or LLMProvider()
        self.embeddings = self.model_provider.get_embeddings_model()

        logger.debug(f"Loading Chroma vectorstore from: {self.vector_db_path}")
        try:
            self.vectorstore = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings
            )
            logger.info("Chroma vectorstore initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Chroma vectorstore: {str(e)}", exc_info=True)
            raise

        self.llm = self.model_provider.get_inference_model()
        self.chunking_strategy = chunking_strategy
        logger.info("RAGPipeline initialized successfully")

    def _get_text_splitter(self, docs, file_path: str):
        """
        Use MarkdownHeaderTextSplitter if markdown, else fallback to RecursiveCharacterTextSplitter.
        """
        ext = os.path.splitext(file_path)[-1].lower()
        logger.debug(f"Determining text splitter for file extension: {ext}")

        if self.chunking_strategy == "header" or (self.chunking_strategy == "auto" and ext in ['.md', '.markdown']):
            try:
                logger.info(f"Using MarkdownHeaderTextSplitter for {file_path}")
                return MarkdownHeaderTextSplitter(
                    headers_to_split_on=["#", "##", "###"],
                    chunk_size=1000,
                    chunk_overlap=100
                )
            except Exception as e:
                logger.warning(f"Header splitter failed: {e}, falling back to character splitter.")

        logger.info(f"Using RecursiveCharacterTextSplitter for {file_path}")
        return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    def load_document(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[-1].lower()
        logger.debug(f"Loading document of type '{ext}': {file_path}")
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
            logger.error(f"Unsupported file extension: {ext}")
            raise ValueError(f"Unsupported file extension: {ext}")
        
        logger.info(f"Starting document load for: {file_path}")
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} documents from: {file_path}")
        return docs

    def ingest(self, file_path: str, metadata: dict = None):
        """
        Ingests a file using adaptive chunking (header-based for markdown, otherwise character-based).
        """
        logger.info(f"Starting ingestion for file: {file_path}")
        try:
            docs = self.load_document(file_path)
            splitter = self._get_text_splitter(docs, file_path)
            splits = splitter.split_documents(docs)

            # Attach file-level metadata
            for doc in splits:
                doc.metadata = doc.metadata or {}
                if metadata:
                    doc.metadata.update(metadata)
                doc.metadata['source_file'] = file_path

            logger.debug(f"Adding {len(splits)} document chunks to vectorstore")
            self.vectorstore.add_documents(splits)
            logger.info(f"Ingested {len(splits)} chunks from {file_path} using {splitter.__class__.__name__}")
        except Exception as e:
            logger.error(f"Error during ingestion of file {file_path}: {str(e)}", exc_info=True)
            raise

    def retrieve(
            self, 
            query: str, 
            k: int = 4, 
            keywords: Optional[list] = None, 
            metadata_filter: Optional[dict] = None,
            user_id: Optional[str] = None
            ) -> List[Document]:
        """
        Hybrid retrieval: combines vector search, keyword, and metadata filtering. Always uses strict top-k retrieval (no MMR).
        :param query: user query
        :param k: number of results
        :param keywords: list of keywords to boost/filter
        :param metadata_filter: dict of metadata filters (e.g. {"source_file": ...})
        :param user_id: optional, restricts results to a specific user's files
        """
        logger.info(f"Starting hybrid retrieval for query: '{query}'")

        if user_id is not None:
            if metadata_filter is None:
                metadata_filter = {}
            metadata_filter["user_id"] = str(user_id)
            logger.debug(f"Applying user_id filter: {user_id}")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        try:
            if metadata_filter:
                logger.debug(f"Retrieving with metadata filter: {metadata_filter}")
                results = retriever.invoke(query, filter=metadata_filter)
            else:
                results = retriever.invoke(query)

            logger.debug(f"Got {len(results)} raw results from vectorstore")

            # Keyword filter/boost
            if keywords:
                logger.debug(f"Applying keyword filter: {keywords}")
                keyword_results = [
                    doc for doc in results
                    if any(kw.lower() in doc.page_content.lower() for kw in keywords)
                ]
                # Merge, deduplicate, and sort (keyword hits first)
                unique = {id(doc): doc for doc in keyword_results + results}
                results = list(unique.values())[:k]
                logger.debug(f"Filtered down to {len(results)} results after keyword matching")

            logger.info(f"Hybrid retrieval completed: {len(results)} docs returned (k={k})")
            return results
        except Exception as e:
            logger.error(f"Error during retrieval for query '{query}': {str(e)}", exc_info=True)
            raise
