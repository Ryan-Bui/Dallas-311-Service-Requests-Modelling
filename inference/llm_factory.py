import os
from typing import Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

# Load environment variables
load_dotenv()

def get_llm(provider: Literal["openai", "groq"] = None, temperature: float = 0.0) -> BaseChatModel:
    """
    Factory function to initialize a LangChain ChatModel.
    
    Defaults to Groq if GROQ_API_KEY is found, otherwise OpenAI.
    """
    
    # Auto-detect provider if not specified
    if not provider:
        if os.getenv("GROQ_API_KEY"):
            provider = "groq"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            raise ValueError("No API keys found for Groq or OpenAI in .env file.")

    if provider == "groq":
        # Using ChatOpenAI as a bridge if langchain-groq isn't fully installed, 
        # but the prompt specifically mentioned groq library in requirements.
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model_name="llama-3.3-70b-versatile",
                temperature=temperature,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
        except ImportError:
            # Fallback for demonstration if library is missing
            return ChatOpenAI(
                base_url="https://api.groq.com/openai/v1",
                model="llama-3.3-70b-versatile",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=temperature
            )
            
    elif provider == "openai":
        return ChatOpenAI(
            model="gpt-4o",
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")
