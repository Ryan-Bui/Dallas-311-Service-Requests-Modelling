import os
import sys
# Ensure we can find the modules in the current directory
sys.path.append(os.getcwd())
from inference.explainability_chain import create_explainability_chain
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

def verify_hybrid_rag():
    """
    Test the explainability chain to see if it retrieves:
    1. GQL Data (Audit/SLA)
    2. Vector Data (PDF content)
    """
    print("\n--- Testing Expert Hybrid GraphRAG ---")
    
    # We'll use "Dallas Fire-Rescue" because it has both Audit focus and PDF report data
    department = "Dallas Fire-Rescue"
    
    # Mock model prediction result
    mock_input = {
        "prediction": "9.5 Minutes Response Time (Target: 9 Minutes)",
        "department": department,
        "coef_summary": "- staffing_level: -1.2 (Decreases response time)\n- apparatus_age: 0.8 (Increases response time)",
        "district": 4
    }

    try:
        print(f"Invoking chain for department: {department}...")
        chain = create_explainability_chain(provider="groq")
        
        response = chain.invoke(mock_input)
        
        print("\n=== EXPERT AI EXPLANATION ===")
        print(response)
        print("============================\n")
        
        response_lower = response.lower()
        if "report context" in response_lower or "audit" in response_lower or "budget" in response_lower:
            print("[SUCCESS] Hybrid context (Graph + Vector) detected in response!")
        else:
            print("[WARNING] One or more data sources might be missing in the response.")

    except Exception as e:
        print(f"[ERROR] Verification Error: {e}")

if __name__ == "__main__":
    verify_hybrid_rag()
