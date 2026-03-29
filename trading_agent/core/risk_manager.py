"""Risk Manager - The guardian that ensures survival and consistent wins.

Key principles:
1. Never risk more than X% of capital on a single trade
2. Dynamic position sizing based on volatility (ATR)
3. Trailing stops to lock in profits
4. Maximum drawdown protection (circuit breaker)
5. Correlation-aware position limits
"""
import numpy as np
from typing import Optional
from .models import Order, OrderSide, OrderType, Position, Signal, StrategySignal
from .market_data import MarketData


class RiskManager:

    def __init__(self, config: dict, initial_capital: float = 10000.0):
        self.config = config
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.peak_capital = initial_capital
        self.positions: list[Position] = []
        self.closed_positions: list[Position] = []
        self.trade_history: list[dict] = []
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0
        self._circuit_breaker_active = False

    @property
    def open_positions(self) -> list[Position]:
        return [p for p in self.positions if p.status.value == "open"]

    @property
    def current_drawdown(self) -> float:
        if self.peak_capital == 0:
            return 0
        return (self.peak_capital - self.capital) / self.peak_capital

    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.0

    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed under current risk constraints."""
        max_dd = self.config.get("max_drawdown_pct", 0.10)
        max_positions = self.config.get("max_open_positions", 3)

        if self._circuit_breaker_active:
            return False, "Circuit breaker active - max drawdown exceeded"

        if self.current_drawdown >= max_dd:
            self._circuit_breaker_active = True
            return False, f"Max drawdown reached ({self.current_drawdown:.1%})"

        if len(self.open_positions) >= max_positions:
            return False, f"Max open positions reached ({max_positions})"

        return True, "OK"

    def calculate_position_size(self, market_data: MarketData,
                                side: OrderSide) -> float:
        """Kelly-criterion inspired position sizing with ATR adjustment.

        Sizes positions based on:
        1. Risk per trade limit (% of capital)
        2. Current ATR (volatility-adjusted)
        3. Win rate adjustment (trade smaller when losing)
        """
        risk_pct = self.config.get("risk_per_trade_pct", 0.01)
        max_position_pct = self.config.get("max_position_pct", 0.25)

        risk_amount = self.capital * risk_pct
        current_price = market_data.latest_price

        if current_price is None or current_price <= 0:
            return 0.0

        # ATR-based stop distance
        atr_vals = market_data.atr(14)
        valid_atr = atr_vals[~np.isnan(atr_vals)]
        current_atr = valid_atr[-1] if len(valid_atr) > 0 else current_price * 0.02
        stop_distance = current_atr * 2  # 2 ATR stop

        # Position size = risk amount / stop distance
        if stop_distance > 0:
            position_size = risk_amount / stop_distance
        else:
            position_size = risk_amount / (current_price * 0.02)

        # Cap at max position percentage
        max_size = (self.capital * max_position_pct) / current_price
        position_size = min(position_size, max_size)

        # Win rate adjustment: reduce size during losing streaks
        recent_trades = self.trade_history[-10:]
        if len(recent_trades) >= 5:
            recent_wins = sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
            recent_wr = recent_wins / len(recent_trades)
            if recent_wr < 0.3:
                position_size *= 0.5  # Halve size during cold streaks
            elif recent_wr > 0.7:
                position_size *= 1.2  # Slightly increase during hot streaks

        return max(position_size, 0.0)

    def create_order(self, market_data: MarketData, signal: StrategySignal,
                     side: OrderSide) -> Optional[Order]:
        """Create a risk-managed order from a signal."""
        can, reason = self.can_trade()
        if not can:
            return None

        current_price = market_data.latest_price
        if current_price is None:
            return None

        quantity = self.calculate_position_size(market_data, side)
        if quantity <= 0:
            return None

        # Dynamic stop loss and take profit based on ATR
        atr_vals = market_data.atr(14)
        valid_atr = atr_vals[~np.isnan(atr_vals)]
        current_atr = valid_atr[-1] if len(valid_atr) > 0 else current_price * 0.02

        stop_loss_pct = self.config.get("stop_loss_pct", 0.02)
        take_profit_pct = self.config.get("take_profit_pct", 0.04)
        trailing_stop = self.config.get("trailing_stop_pct", 0.015)

        # Use ATR-based stops (wider in volatile markets, tighter in calm)
        atr_stop = (current_atr * 2.0) / current_price
        stop_distance = max(stop_loss_pct, atr_stop)
        # Risk:reward 2:1 - balances win rate with payoff
        profit_distance = max(take_profit_pct, stop_distance * 2)

        if side == OrderSide.BUY:
            stop_loss = current_price * (1 - stop_distance)
            take_profit = current_price * (1 + profit_distance)
        else:
            stop_loss = current_price * (1 + stop_distance)
            take_profit = current_price * (1 - profit_distance)

        # Boost confidence: scale size with signal confidence
        quantity *= signal.confidence

        return Order(
            side=side,
            order_type=OrderType.MARKET,
            price=current_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_pct=trailing_stop,
        )

    def open_position(self, order: Order) -> Position:
        """Execute an order and open a position."""
        position = Position(
            entry_price=order.price,
            quantity=order.quantity,
            side=order.side,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            trailing_stop_pct=order.trailing_stop_pct or 0.0,
        )
        self.positions.append(position)
        cost = order.price * order.quantity
        self.capital -= cost
        return position

    def update_positions(self, current_price: float) -> list[dict]:
        """Update all open positions. Returns list of exit events."""
        exits = []
        for pos in self.open_positions:
            exit_reason = pos.update(current_price)
            if exit_reason:
                # Calculate PnL
                if pos.side == OrderSide.BUY:
                    pnl = (current_price - pos.entry_price) * pos.quantity
                else:
                    pnl = (pos.entry_price - current_price) * pos.quantity

                self.capital += pos.entry_price * pos.quantity + pnl
                self.total_pnl += pnl

                if pnl > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1

                trade_record = {
                    "entry_price": pos.entry_price,
                    "exit_price": current_price,
                    "side": pos.side.value,
                    "quantity": pos.quantity,
                    "pnl": pnl,
                    "pnl_pct": pnl / (pos.entry_price * pos.quantity),
                    "exit_reason": exit_reason,
                }
                self.trade_history.append(trade_record)
                self.closed_positions.append(pos)
                exits.append(trade_record)

        # Update peak capital
        total_value = self.capital + sum(
            p.entry_price * p.quantity * (1 + p.pnl) for p in self.open_positions
        )
        self.peak_capital = max(self.peak_capital, total_value)

        # Remove closed positions
        self.positions = [p for p in self.positions if p.status.value == "open"]

        return exits

    def get_stats(self) -> dict:
        """Return comprehensive trading statistics."""
        total_trades = self.win_count + self.loss_count
        winning_trades = [t for t in self.trade_history if t["pnl"] > 0]
        losing_trades = [t for t in self.trade_history if t["pnl"] <= 0]

        avg_win = np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t["pnl"] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win * len(winning_trades)) / abs(avg_loss * len(losing_trades)) \
            if losing_trades and avg_loss != 0 else float('inf')

        return {
            "total_trades": total_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "total_return_pct": (self.capital - self.initial_capital) / self.initial_capital,
            "max_drawdown": self.current_drawdown,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "open_positions": len(self.open_positions),
            "capital": self.capital,
        }
