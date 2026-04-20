import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from inference.explainability_chain import create_explainability_chain, get_domain_context

def test_recursive_lookup_logic():
    """Verify that get_domain_context finds department, metrics, and alerts."""
    # Test Sanitation which has edges to a Metric and a Factor in data/knowledge_graph.json
    context = get_domain_context(department="Sanitation")
    
    assert "Department: Sanitation" in context
    assert "Standard: Sanitation Standard is 72 Hours" in context
    assert "Alert: Public Holiday (Adds 24-hour buffer to all routes)" in context

def test_district_lookup_logic():
    """Verify that district lookup works and includes local challenges."""
    context = get_domain_context(district=7)
    
    assert "District 7: South Dallas" in context
    assert "Note: Old truck narrow access" in context

@patch("inference.explainability_chain.get_llm")
def test_expert_chain_multi_hop(mock_get_llm):
    """Verify that the full LCEL chain receives multi-hop context."""
    mock_func = MagicMock(return_value=AIMessage(content="Expert explanation."))
    mock_llm = RunnableLambda(mock_func)
    mock_get_llm.return_value = mock_llm
    
    chain = create_explainability_chain(provider="openai")
    
    # Run for Sanitation in District 7
    response = chain.invoke({
        "prediction": "4 days",
        "coef_summary": "High volume",
        "department": "Sanitation",
        "district": 7
    })
    
    assert response == "Expert explanation."
    
    # Check if prompt received both Dept, Metric, and District info
    prompt_text = mock_func.call_args[0][0].to_string()
    assert "Sanitation" in prompt_text
    assert "72 Hours" in prompt_text
    assert "South Dallas" in prompt_text
    assert "Old truck narrow access" in prompt_text
