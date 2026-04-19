import pytest
import pandas as pd
import numpy as np
from agents.data_prep_agent import DataPrepAgent
from agents.transformation_agent import TransformationAgent
from agents.diagnostics_agent import DiagnosticsAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.regularization_agent import RegularizationAgent
from agents.base_agent import BaseAgent

@pytest.fixture
def sample_df():
    """Create a minimal 311-like DataFrame for testing."""
    df = pd.DataFrame({
        'Created Date': ['2025 Jan 01 12:00:00 PM', '2025 Jan 02 12:00:00 PM'],
        'Closed Date': ['2025 Jan 01 02:00:00 PM', '2025 Jan 10 12:00:00 PM'],
        'Department': ['Code Compliance', 'DPD'],
        'Service Request Type': ['Type A', 'Type B'],
        'Priority': ['High', 'Low'],
        'Method Received Description': ['Web', 'Phone'],
        'Service Request Number': ['1', '2'],
        'Address': ['A1', 'A2'],
        'Unique Key': ['UK1', 'UK2'],
        'Lat_Long Location': ['(0,0)', '(1,1)']
    })
    return df

class TestAgentInterfaces:
    """Test that all agents inherit from BaseAgent and have required methods."""
    
    @pytest.mark.parametrize("agent_class", [
        DataPrepAgent, TransformationAgent, DiagnosticsAgent, 
        ModelSelectionAgent, RegularizationAgent
    ])
    def test_inheritance(self, agent_class):
        """Verify that the agent inherits from BaseAgent."""
        assert issubclass(agent_class, BaseAgent)

    @pytest.mark.parametrize("agent_class", [
        DataPrepAgent, TransformationAgent, DiagnosticsAgent, 
        ModelSelectionAgent, RegularizationAgent
    ])
    def test_protocol_methods(self, agent_class):
        """Verify that the agent has the required interface methods."""
        agent = agent_class()
        assert hasattr(agent, "run")
        assert hasattr(agent, "validate")
        assert hasattr(agent, "report")

def test_dataprep_agent_interface(sample_df, tmp_path):
    """Smoke test for DataPrepAgent interface."""
    csv_path = tmp_path / "test.csv"
    sample_df.to_csv(csv_path, index=False)
    
    agent = DataPrepAgent(data_path=csv_path)
    # Patch sampling for test
    from src import config
    config.SAMPLE_FRAC = 1.0 
    
    df = agent.run()
    assert isinstance(df, pd.DataFrame)
    val = agent.validate()
    assert "passed" in val
    rep = agent.report()
    assert isinstance(rep, dict)

def test_transformation_agent_interface(sample_df):
    """Smoke test for TransformationAgent interface."""
    # We can just pass the fixture directly as if it was cleaned
    agent = TransformationAgent()
    
    # We need to ensure the columns expected by TransformationAgent are present
    # TransformationAgent expects Created Date and Closed Date (handled by fixture)
    df_trans = agent.run(sample_df)
    assert isinstance(df_trans, pd.DataFrame)
    assert agent.validate()["passed"]
    assert isinstance(agent.report(), dict)

def test_diagnostics_agent_interface():
    """Smoke test for DiagnosticsAgent interface."""
    df = pd.DataFrame({
        'days_to_close': np.random.normal(100, 10, 100).tolist(),
        'target': [0, 1] * 50
    })
    agent = DiagnosticsAgent()
    res = agent.run(df)
    assert isinstance(res, dict)
    assert agent.validate()["passed"]
    assert "normality_pass" in agent.report()

# We skip full model training tests in unit tests to save time, 
# focusing on interface and structural pass/fail.
