import numpy as np
from src.llm_client import get_embeddings
from src.config import config


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Convert a list of text strings into embeddings.
    """
    
    # Guard: return empty array for empty input
    if not texts:
        return np.array([])

    # Use the llm client helper with model from config
    embeddings = get_embeddings(texts, model=config.embedding_model)
    return np.array(embeddings, dtype=float)


def embed_documents(documents: list[dict]) -> tuple[list[dict], np.ndarray]:
    """
    Embed all documents and return both documents and their embeddings.
    """

    if not documents:
        return documents, np.array([])

    texts: list[str] = []
    for doc in documents:
        title = doc.get("title", "") or ""
        content = doc.get("content", doc.get("text", "")) or ""
        combined = (title + "\n\n" + content).strip()
        texts.append(combined)

    embeddings = embed_texts(texts)
    return documents, embeddings
