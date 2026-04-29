import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)


def get_embedding_provider():
    configured = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    if configured:
        return configured
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "gemini"


def get_embedding_model():
    provider = get_embedding_provider()
    configured = os.getenv("EMBEDDING_MODEL", "").strip()
    if configured:
        return configured
    if provider == "openai":
        return "text-embedding-3-small"
    return "models/gemini-embedding-001"


def get_embedding_dimensions():
    configured = os.getenv("EMBEDDING_DIMENSIONS", "").strip()
    if configured:
        return int(configured)
    return 768


def create_embeddings_service():
    provider = get_embedding_provider()
    model = get_embedding_model()

    if provider == "openai":
        kwargs = {"model": model, "api_key": os.getenv("OPENAI_API_KEY")}
        if model.startswith("text-embedding-3"):
            kwargs["dimensions"] = get_embedding_dimensions()
        return OpenAIEmbeddings(**kwargs)

    if provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")


def is_embedding_quota_error(error):
    text = str(error).lower()
    return "resource_exhausted" in text or "quota" in text or "429" in text


def embed_documents_batch(embeddings_service, chunks, embeddings_enabled=True):
    if not embeddings_enabled or embeddings_service is None:
        return [None] * len(chunks), False

    try:
        return embeddings_service.embed_documents(chunks), True
    except Exception as e:
        if is_embedding_quota_error(e):
            print("  Embedding quota exhausted; continuing without embeddings.")
            return [None] * len(chunks), False
        raise


def embed_query_text(embeddings_service, text):
    return embeddings_service.embed_query(text)
