import pytest
import os
import json
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from inference.explainability_chain import create_explainability_chain, get_domain_context

def test_get_domain_context_sanitation():
    """Test that looking up a known department returns the correct context."""
    context = get_domain_context("Sanitation")
    assert "Department: Sanitation" in context
    assert "Target ERT: 3 Business Days" in context

def test_get_domain_context_unknown():
    """Test that unknown departments return general guidelines."""
    context = get_domain_context("Unknown Dept")
    assert "General Dallas 311 guidelines apply." in context

@patch("inference.explainability_chain.get_llm")
def test_kg_chain_integration(mock_get_llm):
    """Test that the chain correctly includes domain context in its call."""
    mock_func = MagicMock(return_value=AIMessage(content="KG-informed explanation."))
    mock_llm = RunnableLambda(mock_func)
    mock_get_llm.return_value = mock_llm
    
    chain = create_explainability_chain(provider="openai")
    
    response = chain.invoke({
        "prediction": "5 days",
        "coef_summary": "Feature Y",
        "department": "Sanitation"
    })
    
    # Verify that the input to the mock function (after prompt formatting)
    # would have included the KG context. We check if the prompt was called.
    assert response == "KG-informed explanation."
    assert mock_func.called
    
    # Check that the department was passed correctly
    input_args = mock_func.call_args[0][0]
    # The ChatPromptValue contains the messages. We check the content of the first message.
    prompt_text = input_args.to_string()
    assert "CITY KNOWLEDGE MAP:" in prompt_text
    assert "Sanitation" in prompt_text
