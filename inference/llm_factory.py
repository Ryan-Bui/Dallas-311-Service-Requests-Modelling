import os
from typing import Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

# Load environment variables
load_dotenv(override=True)

def get_llm(provider: Literal["openai", "groq", "vertexai"] = None, temperature: float = 0.0) -> BaseChatModel:
    """
    Factory function to initialize a LangChain ChatModel.
    
    Order of preference: Groq (highest speed), Vertex AI Gemini (stability), OpenAI.
    """
    
    # Auto-detect provider if not specified
    if not provider:
        if os.getenv("GROQ_API_KEY"):
            provider = "groq"
        elif os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT"): # Vertex AI detection
            provider = "vertexai"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            raise ValueError("No API keys found for Groq, Vertex AI, or OpenAI.")

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=temperature,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
            
    elif provider == "vertexai":
        # Using langchain-google-genai for easier key usage (AI Studio)
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
            
    elif provider == "openai":
        return ChatOpenAI(
            model="gpt-4o",
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")
