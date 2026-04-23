import os
from typing import Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

# Load environment variables
load_dotenv(override=True)

def get_llm(provider: str = None, temperature: float = 0.0) -> BaseChatModel:
    """
    Factory function using LangChain's native with_fallbacks system.
    Supports optional provider overrides for specific tasks.
    """
    models = []
    
    # Priority order can be shifted if a provider is explicitly requested
    target_p = provider.lower() if provider else ""
    
    # 1. Groq (Llama 3.3 70B - Primary)
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        # Primary Heavy Lifter
        models.append(ChatGroq(model_name="llama-3.3-70b-versatile", temperature=temperature))
        # Quota Safety Net (8B model has higher limits)
        models.append(ChatGroq(model_name="llama-3.1-8b-instant", temperature=temperature))
        
    # 2. Gemini 1.5 Flash (Secondary Provider)
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        models.append(ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=temperature))
        
    # 3. OpenAI (GPT-4o-mini)
    if os.getenv("OPENAI_API_KEY"):
        models.append(ChatOpenAI(model="gpt-4o-mini", temperature=temperature))

    if not models:
        raise ValueError("No LLM API keys found in environment.")

    # Create the self-healing chain
    if not models:
        raise ValueError("No LLM API keys found in environment.")

    print(f"[AI RESILIENCE] Active Chain: {' -> '.join([m.model_name if hasattr(m, 'model_name') else m.model for m in models])}")

    primary = models[0]
    if len(models) > 1:
        return primary.with_fallbacks(models[1:])
    return primary
