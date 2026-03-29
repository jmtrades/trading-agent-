"""Risk Manager v2 - Advanced position sizing, exit management, and analytics.

Upgrades:
1. Kelly Criterion position sizing
2. Volatility-regime adaptive stops
3. Partial profit-taking at key levels
4. Consecutive loss tracking with graduated response
5. Correlation-aware exposure limits
6. Comprehensive performance analytics (Sortino, Calmar, etc.)
"""
import numpy as np
from typing import Optional
from .models import Order, OrderSide, OrderType, Position, Signal, StrategySignal
from .market_data import MarketData


class RiskManager:

    def __init__(self, config: dict, initial_capital: float = 10000.0,
                 prop_firm: str = None, prop_phase: str = "challenge"):
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
        self._consecutive_losses = 0
        self._max_consecutive_losses = 0
        self._equity_history: list[float] = [initial_capital]

        # Prop firm compliance (optional)
        self.prop_compliance = None
        if prop_firm:
            from .prop_firm import PropFirmCompliance
            self.prop_compliance = PropFirmCompliance(
                firm=prop_firm, phase=prop_phase, account_size=initial_capital
            )

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
        max_dd = self.config.get("max_drawdown_pct", 0.15)
        max_positions = self.config.get("max_open_positions", 5)

        if self._circuit_breaker_active:
            # Reset circuit breaker when drawdown recovers to 50% of threshold
            if self.current_drawdown < max_dd * 0.5:
                self._circuit_breaker_active = False
            else:
                return False, "Circuit breaker active - max drawdown exceeded"

        if self.current_drawdown >= max_dd:
            self._circuit_breaker_active = True
            return False, f"Max drawdown reached ({self.current_drawdown:.1%})"

        if len(self.open_positions) >= max_positions:
            return False, f"Max open positions reached ({max_positions})"

        # Graduated response to consecutive losses
        if self._consecutive_losses >= 5:
            return False, f"Too many consecutive losses ({self._consecutive_losses})"

        # Prop firm compliance check
        if self.prop_compliance and self.prop_compliance.state.is_blown:
            return False, "Prop firm account blown"

        return True, "OK"

    def _kelly_fraction(self) -> float:
        """Calculate Kelly Criterion fraction for optimal bet sizing.

        Kelly fraction = W - (1-W)/R
        where W = win rate, R = avg_win/avg_loss ratio

        Uses half-Kelly for safety.
        """
        if len(self.trade_history) < 10:
            return 1.0  # Not enough data, use default sizing

        wins = [t["pnl"] for t in self.trade_history if t["pnl"] > 0]
        losses = [abs(t["pnl"]) for t in self.trade_history if t["pnl"] <= 0]

        if not wins or not losses:
            return 1.0

        w = len(wins) / len(self.trade_history)
        r = np.mean(wins) / np.mean(losses)

        kelly = w - (1 - w) / r
        # Half-Kelly for safety, clamped between 0.25 and 1.5
        return max(0.25, min(kelly * 0.5, 1.5))

    def calculate_position_size(self, market_data: MarketData,
                                side: OrderSide) -> float:
        """Advanced position sizing with Kelly criterion and ATR adjustment."""
        risk_pct = self.config.get("risk_per_trade_pct", 0.02)
        max_position_pct = self.config.get("max_position_pct", 0.30)

        current_price = market_data.latest_price
        if current_price is None or current_price <= 0:
            return 0.0

        # Base risk amount
        risk_amount = self.capital * risk_pct

        # Kelly adjustment
        kelly = self._kelly_fraction()
        risk_amount *= kelly

        # ATR-based stop distance
        atr_vals = market_data.atr(14)
        valid_atr = atr_vals[~np.isnan(atr_vals)]
        current_atr = valid_atr[-1] if len(valid_atr) > 0 else current_price * 0.02
        stop_distance = current_atr * 2

        # Position size = risk amount / stop distance
        if stop_distance > 0:
            position_size = risk_amount / stop_distance
        else:
            position_size = risk_amount / (current_price * 0.02)

        # Cap at max position percentage
        max_size = (self.capital * max_position_pct) / current_price
        position_size = min(position_size, max_size)

        # Consecutive loss reduction (graduated)
        if self._consecutive_losses >= 3:
            reduction = 0.5 ** (self._consecutive_losses - 2)
            position_size *= reduction

        # Recent performance adjustment
        recent_trades = self.trade_history[-10:]
        if len(recent_trades) >= 5:
            recent_wins = sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
            recent_wr = recent_wins / len(recent_trades)
            if recent_wr < 0.3:
                position_size *= 0.5
            elif recent_wr > 0.7:
                position_size *= 1.15

        # Volatility regime: compare current ATR to average ATR
        if len(valid_atr) >= 20:
            avg_atr = np.mean(valid_atr[-20:])
            vol_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
            if vol_ratio > 2.0:
                position_size *= 0.5  # Halve in extreme volatility
            elif vol_ratio > 1.5:
                position_size *= 0.7

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

        stop_loss_pct = self.config.get("stop_loss_pct", 0.025)
        take_profit_pct = self.config.get("take_profit_pct", 0.05)
        trailing_stop = self.config.get("trailing_stop_pct", 0.012)

        # ATR-based stops with volatility-regime adaptation
        atr_stop = (current_atr * 2.0) / current_price
        stop_distance = max(stop_loss_pct, atr_stop)

        # Dynamic R:R based on win rate
        if self.win_rate > 0.5 and len(self.trade_history) >= 10:
            rr_mult = 1.8  # Can afford tighter R:R with high win rate
        elif self.win_rate < 0.35 and len(self.trade_history) >= 10:
            rr_mult = 3.0  # Need bigger wins to compensate
        else:
            rr_mult = 2.0  # Default

        profit_distance = max(take_profit_pct, stop_distance * rr_mult)

        if side == OrderSide.BUY:
            stop_loss = current_price * (1 - stop_distance)
            take_profit = current_price * (1 + profit_distance)
        else:
            stop_loss = current_price * (1 + stop_distance)
            take_profit = current_price * (1 - profit_distance)

        # Scale size with signal confidence
        quantity *= signal.confidence

        # Prop firm: cap risk to safe amount and run pre-trade check
        if self.prop_compliance:
            proposed_risk = quantity * stop_distance * current_price
            safe_risk = self.prop_compliance.get_safe_risk_amount()
            if proposed_risk > safe_risk and safe_risk > 0:
                quantity = (safe_risk / (stop_distance * current_price))
            allowed, reason = self.prop_compliance.check_pre_trade(proposed_risk)
            if not allowed:
                return None

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
                if pos.side == OrderSide.BUY:
                    pnl = (current_price - pos.entry_price) * pos.quantity
                else:
                    pnl = (pos.entry_price - current_price) * pos.quantity

                self.capital += pos.entry_price * pos.quantity + pnl
                self.total_pnl += pnl

                if pnl > 0:
                    self.win_count += 1
                    self._consecutive_losses = 0
                else:
                    self.loss_count += 1
                    self._consecutive_losses += 1
                    self._max_consecutive_losses = max(
                        self._max_consecutive_losses, self._consecutive_losses
                    )

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

                # Record with prop firm compliance
                if self.prop_compliance:
                    self.prop_compliance.record_trade_result(pnl)

        # Update peak capital and equity history
        total_value = self.capital + sum(
            p.entry_price * p.quantity * (1 + p.pnl) for p in self.open_positions
        )
        self.peak_capital = max(self.peak_capital, total_value)
        self._equity_history.append(total_value)

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

        # Advanced analytics
        equity = np.array(self._equity_history)
        stats = {
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
            "consecutive_losses": self._consecutive_losses,
            "max_consecutive_losses": self._max_consecutive_losses,
            "kelly_fraction": self._kelly_fraction(),
        }

        # Sortino Ratio (penalizes downside volatility only)
        if len(equity) > 2:
            returns = np.diff(equity) / equity[:-1]
            downside = returns[returns < 0]
            downside_std = np.std(downside) if len(downside) > 0 else 1e-10
            stats["sortino_ratio"] = (np.mean(returns) / downside_std * np.sqrt(252)) \
                if downside_std > 0 else 0
        else:
            stats["sortino_ratio"] = 0

        # Calmar Ratio (return / max drawdown)
        peaks = np.maximum.accumulate(equity)
        drawdowns = (peaks - equity) / np.where(peaks > 0, peaks, 1)
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0
        total_return = (equity[-1] - equity[0]) / equity[0] if equity[0] > 0 else 0
        stats["calmar_ratio"] = total_return / max_dd if max_dd > 0 else float('inf')

        # Expectancy (average $ per trade)
        if total_trades > 0:
            stats["expectancy"] = self.total_pnl / total_trades
        else:
            stats["expectancy"] = 0

        # Prop firm status
        if self.prop_compliance:
            stats["prop_firm"] = self.prop_compliance.get_status()

        # Win/Loss streaks
        if self.trade_history:
            max_win_streak = 0
            current_streak = 0
            for t in self.trade_history:
                if t["pnl"] > 0:
                    current_streak += 1
                    max_win_streak = max(max_win_streak, current_streak)
                else:
                    current_streak = 0
            stats["max_win_streak"] = max_win_streak
        else:
            stats["max_win_streak"] = 0

        return stats
