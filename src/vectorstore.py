import os
import shutil
from typing import List

from sentence_transformers import SentenceTransformer

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma


CHROMA_DIRECTORY = "./data/chroma_db"
DEFAULT_COLLECTION_NAME = "documind"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class LocalSentenceTransformerEmbeddings(Embeddings):
    """
    LangChain-compatible embedding wrapper around sentence-transformers.

    The model runs locally, so embedding generation does not use
    any paid embedding API.
    """

    def __init__(self):
        self.model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(
        self,
        text: str,
    ) -> List[float]:
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()


_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = LocalSentenceTransformerEmbeddings()
    return _embedding_model


def ensure_data_directory():
    os.makedirs(CHROMA_DIRECTORY, exist_ok=True)


def create_vectorstore(
    documents: List[Document],
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """
    Create a new persistent Chroma vector database for the given
    collection_name.  Any existing data in that collection is
    removed first so a fresh index is built.
    """
    delete_collection(collection_name)
    ensure_data_directory()
    embeddings = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DIRECTORY,
        collection_metadata={"hnsw:space": "cosine"},
    )
    return vectorstore


def load_vectorstore(
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """
    Load an existing persistent Chroma collection by name.
    """
    ensure_data_directory()
    embeddings = get_embedding_model()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIRECTORY,
        collection_metadata={"hnsw:space": "cosine"},
    )
    return vectorstore


def delete_collection(collection_name: str):
    """
    Remove a single named collection from the persistent Chroma
    store without touching other collections.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIRECTORY)
        existing = [c.name for c in client.list_collections()]
        if collection_name in existing:
            client.delete_collection(collection_name)
    except Exception:
        pass


def reset_vectorstore():
    """
    Wipe the entire Chroma directory (all collections).
    Used by the "Clear everything" button.
    """
    if os.path.exists(CHROMA_DIRECTORY):
        shutil.rmtree(CHROMA_DIRECTORY, ignore_errors=True)


def build_document_filter(selected_sources):
    if not selected_sources:
        return None
    if len(selected_sources) == 1:
        return {"source": selected_sources[0]}
    return {"source": {"$in": selected_sources}}


def create_mmr_retriever(
    vectorstore,
    selected_sources,
    k=4,
    fetch_k=12,
):
    """
    MMR retriever — balances relevance with diversity.
    """
    metadata_filter = build_document_filter(selected_sources)
    search_kwargs = {
        "k": k,
        "fetch_k": fetch_k,
        "lambda_mult": 0.7,
    }
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )
    return retriever


def create_hybrid_retriever(
    vectorstore,
    all_chunks: List[Document],
    selected_sources,
    k=4,
    fetch_k=12,
):
    """
    Hybrid BM25 + MMR retriever using Reciprocal Rank Fusion.

    BM25 catches exact keyword matches (e.g. "P1", "Table 2",
    author names) that semantic embeddings miss.

    MMR catches semantically related passages even when exact
    words differ.

    Weights: 40% BM25 + 60% MMR semantic.

    Falls back to pure MMR if rank_bm25 is not installed or
    if no chunks are available.
    """
    try:
        from langchain.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever
    except ImportError:
        return create_mmr_retriever(
            vectorstore, selected_sources, k, fetch_k
        )

    # Filter in-memory chunks to match user's document selection.
    if selected_sources:
        filtered_chunks = [
            c for c in all_chunks
            if c.metadata.get("source") in selected_sources
        ]
    else:
        filtered_chunks = list(all_chunks)

    if not filtered_chunks:
        return create_mmr_retriever(
            vectorstore, selected_sources, k, fetch_k
        )

    # BM25: token-level exact matching
    bm25_retriever = BM25Retriever.from_documents(filtered_chunks)
    bm25_retriever.k = k

    # MMR: semantic similarity + diversity
    mmr_retriever = create_mmr_retriever(
        vectorstore=vectorstore,
        selected_sources=selected_sources,
        k=k,
        fetch_k=fetch_k,
    )

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, mmr_retriever],
        weights=[0.4, 0.6],
    )

    return ensemble_retriever


def get_top_relevance_score(
    vectorstore,
    query,
    selected_sources,
):
    """
    One-result similarity search for the evaluation panel.
    Does not affect retrieval.
    """
    metadata_filter = build_document_filter(selected_sources)
    try:
        if metadata_filter:
            results = vectorstore.similarity_search_with_relevance_scores(
                query, k=1, filter=metadata_filter,
            )
        else:
            results = vectorstore.similarity_search_with_relevance_scores(
                query, k=1,
            )
        if not results:
            return 0.0
        _, score = results[0]
        return max(0.0, min(1.0, float(score)))
    except Exception:
        return 0.0