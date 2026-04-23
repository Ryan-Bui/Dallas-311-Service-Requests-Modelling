import logging
import json
import re
import os
from .base_agent import BaseAgent
from inference.llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)

class TranscriptionAgent(BaseAgent):
    """
    Agent responsible for Stage A & B: 
    Converting raw text transcripts into structured 311 features using LLM logic.
    """

    def __init__(self, provider: str = None):
        # Delegate to self-healing factory
        self.llm = get_llm(provider=provider, temperature=0.1)
        self.output_parser = JsonOutputParser()

    def run(self, transcript: str) -> dict:
        """
        Extracts 311 features from transcript.
        
        Args:
            transcript (str): Raw text from a service call or report.
            
        Returns:
            dict: Structured features compatible with manual_infer.
        """
        logger.info("[TranscriptionAgent] Extracting features from transcript...")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the Dallas 311 Acoustic Harvester. Extract structured features from the transcript for a predictive model."),
            ("human", """
            TRANSCRIPT: "{transcript}"
            
            EXTRACT (JSON only):
            - "Service Request Type": (e.g. 'Animal Loose', 'Pothole', 'High Weeds')
            - "Department": (e.g. 'Dallas Animal Services', 'Public Works', 'Code Compliance')
            - "Priority": ('High', 'Standard', or 'Emergency')
            - "City Council District": (Integer 1-14 or null)
            - "Method Received Description": (Default to 'Phone' unless 'app' or 'web' mentioned)
            
            CRITICAL: Return only the JSON object. No intro/outro.
            """)
        ])
        
        try:
            # Step 1: Create a simple chain
            chain = prompt | self.llm
            response = chain.invoke({"transcript": transcript})
            
            # Step 2: Extract content
            content = response.content if hasattr(response, 'content') else str(response)
            print(f"[DEBUG] TranscriptionAgent Raw Response: {content}") # Direct terminal view
            
            # Step 3: Parse JSON
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                extracted = json.loads(match.group())
            else:
                extracted = self.output_parser.parse(content)
            
            # Key Repair & Mapping (Stage B: Semantic Alignment)
            schema = {
                "Service Request Type": "Unknown",
                "Department": "Unknown",
                "Priority": "Standard",
                "City Council District": None,
                "Method Received Description": "Phone"
            }
            # Merge extracted with schema defaults
            final_features = {**schema, **extracted}
                
            logger.info(f"[TranscriptionAgent] Extraction successful: {final_features}")
            return final_features
        except Exception as e:
            print(f"[ERROR] TranscriptionAgent Failed: {e}")
            logger.error(f"[TranscriptionAgent] Extraction failed: {e}")
            # Explicit fallback to ensure keys exist even in failure
            return {
                "Service Request Type": "Error/Unknown",
                "Department": "Error/Unknown",
                "Priority": "Standard",
                "City Council District": None,
                "Method Received Description": "Phone",
                "debug_error": str(e)
            }

    def validate(self) -> dict:
        # Check if LLM is reachable
        try:
            self.llm.invoke("test")
            return {"passed": True, "issues": []}
        except Exception as e:
            return {"passed": False, "issues": [str(e)]}

    def report(self) -> dict:
        """Standard reporting interface."""
        return {
            "status": "active",
            "model_type": "extraction",
            "provider": "llm"
        }
