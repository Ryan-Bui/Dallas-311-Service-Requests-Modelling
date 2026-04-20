import os
import pytest
from google.cloud import spanner
from inference.explainability_chain import get_domain_context
from dotenv import load_dotenv

load_dotenv()

@pytest.mark.skipif(not os.getenv("GCP_PROJECT_ID"), reason="GCP credentials not configured")
def test_live_spanner_retrieval():
    """Verify that the chain can pull expert context from the live Spanner Graph."""
    
    # We'll test 'Transportation' as it was in your Audit table (Topic: 911 Service / Signal Malfunction)
    # The Ontology table has 'Transportation' for 'Signal malfunction responses'
    context = get_domain_context(department="Transportation")
    
    print(f"\nRetrieved Context:\n{context}")
    
    assert "CITY KNOWLEDGE MAP" in context
    # If the ingestion linked departments correctly, we should see an audit focus
    # (Note: In the audit table, Transportation is often grouped or listed in topics)
    assert "Transportation" in context or "General Dallas 311 guidelines" in context

if __name__ == "__main__":
    # Manual run
    print(get_domain_context(department="Transportation"))
