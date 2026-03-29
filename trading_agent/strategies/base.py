"""Base strategy interface."""
from abc import ABC, abstractmethod
from trading_agent.core.market_data import MarketData
from trading_agent.core.models import StrategySignal


class BaseStrategy(ABC):
    """All strategies must implement this interface."""

    def __init__(self, config: dict):
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    def analyze(self, market_data: MarketData) -> StrategySignal:
        """Analyze market data and return a signal."""
        ...

    @property
    def weight(self) -> float:
        return self.config.get("weight", 0.33)
