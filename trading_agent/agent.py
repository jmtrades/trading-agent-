"""AlphaWin Agent - The main orchestrator that combines strategies to win.

The agent:
1. Collects market data
2. Runs all strategies
3. Combines signals with weighted voting
4. Applies risk management
5. Executes winning trades
"""
import json
import logging
from pathlib import Path
from typing import Optional

from .core.models import Candle, Signal, OrderSide, StrategySignal
from .core.market_data import MarketData
from .core.risk_manager import RiskManager
from .strategies.momentum import MomentumStrategy
from .strategies.mean_reversion import MeanReversionStrategy
from .strategies.breakout import BreakoutStrategy

logger = logging.getLogger("AlphaWin")


class AlphaWinAgent:
    """The winning trading agent."""

    def __init__(self, config_path: Optional[str] = None, config: Optional[dict] = None):
        if config is not None:
            self.config = config
        elif config_path:
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            config_file = Path(__file__).parent.parent / "config" / "default.json"
            with open(config_file) as f:
                self.config = json.load(f)

        self.market_data = MarketData(max_candles=500)
        self.risk_manager = RiskManager(
            self.config.get("risk", {}),
            initial_capital=self.config.get("agent", {}).get("initial_capital", 10000.0)
        )

        # Initialize strategies
        strat_config = self.config.get("strategies", {})
        self.strategies = []

        if strat_config.get("momentum", {}).get("enabled", True):
            self.strategies.append(MomentumStrategy(strat_config["momentum"]))
        if strat_config.get("mean_reversion", {}).get("enabled", True):
            self.strategies.append(MeanReversionStrategy(strat_config["mean_reversion"]))
        if strat_config.get("breakout", {}).get("enabled", True):
            self.strategies.append(BreakoutStrategy(strat_config["breakout"]))

        self._warmup = self.config.get("agent", {}).get("warmup_periods", 50)
        logger.info(f"AlphaWin initialized with {len(self.strategies)} strategies")

    def on_candle(self, candle: Candle) -> dict:
        """Process a new candle and return action taken.

        Returns dict with: action, signal details, position info, stats
        """
        self.market_data.add_candle(candle)

        result = {
            "action": "hold",
            "price": candle.close,
            "signals": [],
            "exits": [],
        }

        # Update existing positions first
        exits = self.risk_manager.update_positions(candle.close)
        if exits:
            result["exits"] = exits
            for e in exits:
                logger.info(f"Position closed: {e['exit_reason']} | PnL: {e['pnl']:.2f}")

        # Need enough data
        if self.market_data.size < self._warmup:
            result["action"] = "warmup"
            return result

        # Run all strategies
        signals = []
        for strategy in self.strategies:
            sig = strategy.analyze(self.market_data)
            signals.append(sig)
            result["signals"].append({
                "strategy": sig.strategy_name,
                "signal": sig.signal.name,
                "confidence": sig.confidence,
                "reason": sig.reason,
            })

        # Weighted signal combination
        combined_score = self._combine_signals(signals)

        # Decision thresholds - tuned for signal range
        if combined_score >= 0.15:
            self._try_enter(OrderSide.BUY, signals, combined_score, result)
        elif combined_score <= -0.15:
            self._try_enter(OrderSide.SELL, signals, combined_score, result)
        else:
            result["action"] = "hold"
            result["combined_score"] = combined_score

        result["stats"] = self.risk_manager.get_stats()
        return result

    def _combine_signals(self, signals: list[StrategySignal]) -> float:
        """Weighted vote across all strategy signals."""
        total_score = 0.0
        total_weight = 0.0

        for sig in signals:
            strategy = next(
                (s for s in self.strategies if s.name == sig.strategy_name), None
            )
            weight = strategy.weight if strategy else 0.33
            total_score += sig.signal.value * sig.confidence * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _try_enter(self, side: OrderSide, signals: list[StrategySignal],
                   score: float, result: dict):
        """Attempt to enter a position."""
        # Use the highest confidence signal for order creation
        best_signal = max(signals, key=lambda s: s.confidence)
        best_signal.confidence = min(abs(score), 1.0)

        order = self.risk_manager.create_order(self.market_data, best_signal, side)
        if order:
            position = self.risk_manager.open_position(order)
            result["action"] = side.value
            result["order"] = {
                "side": order.side.value,
                "price": order.price,
                "quantity": order.quantity,
                "stop_loss": order.stop_loss,
                "take_profit": order.take_profit,
            }
            result["combined_score"] = score
            logger.info(
                f"Entered {side.value} @ {order.price:.2f} | "
                f"Size: {order.quantity:.4f} | Score: {score:.2f}"
            )
        else:
            result["action"] = "blocked"
            result["combined_score"] = score

    def get_stats(self) -> dict:
        return self.risk_manager.get_stats()

    def backtest(self, candles: list[Candle]) -> dict:
        """Run a backtest with the agent's current configuration."""
        from .core.backtester import Backtester
        bt = Backtester(self.config, self.strategies, self.risk_manager.initial_capital)
        return bt.run(candles, warmup=self._warmup)
