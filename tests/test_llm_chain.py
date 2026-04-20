import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from inference.explainability_chain import create_explainability_chain, format_coef_summary

def test_format_coef_summary_valid():
    """Test that the coefficient summary formatter correctly handles a DataFrame."""
    mock_coef = pd.DataFrame([
        {"Feature": "noise", "Coefficient": 0.5},
        {"Feature": "traffic", "Coefficient": -0.2}
    ])
    summary = format_coef_summary(mock_coef)
    assert "- noise: 0.5000 (Increases delay)" in summary
    assert "- traffic: -0.2000 (Decreases delay)" in summary

def test_format_coef_summary_empty():
    """Test empty DataFrame handling."""
    mock_coef = pd.DataFrame()
    summary = format_coef_summary(mock_coef)
    assert summary == "No coefficient data available."

@patch("inference.explainability_chain.get_llm")
def test_chain_invocation(mock_get_llm):
    """Test that the chain can be invoked and calls the LLM mock."""
    # Use a MagicMock wrapped in RunnableLambda to track calls
    mock_func = MagicMock(return_value=AIMessage(content="Mocked explanation."))
    mock_llm = RunnableLambda(mock_func)
    mock_get_llm.return_value = mock_llm
    
    chain = create_explainability_chain(provider="openai")
    
    response = chain.invoke({
        "prediction": "10 days",
        "coef_summary": "Feature X (Positive impact)"
    })
    
    assert response == "Mocked explanation."
    assert mock_func.called
