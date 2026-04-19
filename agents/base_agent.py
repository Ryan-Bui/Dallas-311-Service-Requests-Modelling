# ============================================
# Base Agent Class
# Phase 2.1 — Standardized Interface
# ============================================
from abc import ABC, abstractmethod
from typing import Any

class BaseAgent(ABC):
    """
    Abstract Base Class for all specialized agents in the pipeline.
    
    Enforces a standard interface for running tasks, validating
    outputs, and reporting results.
    """

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the core logic of the agent.
        
        This method must be implemented by all subclasses. It should
        perform its specialized task and return the appropriate result.
        """
        pass

    @abstractmethod
    def validate(self) -> dict:
        """
        Validate the output of the run() method.
        
        Returns
        -------
        dict
            Keys: 'passed' (bool), 'issues' (list[str])
        """
        pass

    @abstractmethod
    def report(self) -> dict:
        """
        Return a structured summary report of the agent's work.
        
        Returns
        -------
        dict
            Generic dictionary containing performance metrics, shapes, or statuses.
        """
        pass
