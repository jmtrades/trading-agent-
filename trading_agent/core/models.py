"""Core data models for the trading agent."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class Signal(Enum):
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class Candle:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class StrategySignal:
    signal: Signal
    confidence: float  # 0.0 to 1.0
    strategy_name: str
    reason: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Order:
    side: OrderSide
    order_type: OrderType
    price: float
    quantity: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class Position:
    entry_price: float
    quantity: float
    side: OrderSide
    stop_loss: float
    take_profit: float
    trailing_stop_pct: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    entry_time: float = field(default_factory=time.time)
    pnl: float = 0.0
    bars_held: int = 0
    max_bars: int = 50  # Time-based exit: close zombie trades

    def __post_init__(self):
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price
        if self.lowest_price == 0.0:
            self.lowest_price = self.entry_price

    def update(self, current_price: float) -> Optional[str]:
        """Update position with current price. Returns exit reason if stopped out."""
        if self.status == PositionStatus.CLOSED:
            return None

        self.bars_held += 1

        # Time-based exit: close zombie trades that go nowhere
        if self.bars_held >= self.max_bars:
            self.status = PositionStatus.CLOSED
            return "timeout"

        if self.side == OrderSide.BUY:
            self.highest_price = max(self.highest_price, current_price)
            self.pnl = (current_price - self.entry_price) / self.entry_price

            # Trailing stop: adjust stop loss upward
            if self.trailing_stop_pct > 0:
                trailing_stop = self.highest_price * (1 - self.trailing_stop_pct)
                self.stop_loss = max(self.stop_loss, trailing_stop)

            if current_price <= self.stop_loss:
                self.status = PositionStatus.CLOSED
                return "stop_loss"
            if current_price >= self.take_profit:
                self.status = PositionStatus.CLOSED
                return "take_profit"
        else:
            self.lowest_price = min(self.lowest_price, current_price)
            self.pnl = (self.entry_price - current_price) / self.entry_price

            if self.trailing_stop_pct > 0:
                trailing_stop = self.lowest_price * (1 + self.trailing_stop_pct)
                self.stop_loss = min(self.stop_loss, trailing_stop)

            if current_price >= self.stop_loss:
                self.status = PositionStatus.CLOSED
                return "stop_loss"
            if current_price <= self.take_profit:
                self.status = PositionStatus.CLOSED
                return "take_profit"

        return None
