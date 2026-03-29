"""AlphaWin Agent - The main orchestrator built to win.

Trade philosophy: Be EXTREMELY selective. Only take A+ setups where
multiple independent confirmations align. The best trade is often
no trade at all.

Filters every trade through:
1. Market regime detection (is the environment favorable?)
2. Strategy consensus (do multiple strategies agree?)
3. Multi-timeframe alignment (does the higher TF confirm?)
4. Trade quality scoring (is this an A+ setup?)
5. Cooldown management (no revenge trading)
6. Risk management (can we afford this trade?)
"""
import json
import logging
from pathlib import Path
from typing import Optional

from .core.models import Candle, Signal, OrderSide, StrategySignal
from .core.market_data import MarketData
from .core.risk_manager import RiskManager
from .core.regime_detector import RegimeDetector, MarketRegime
from .core.filters import (
    MultiTimeframeFilter, ConsensusFilter,
    TradeQualityScorer, CooldownFilter, SignalPersistenceFilter,
)
from .strategies.momentum import MomentumStrategy
from .strategies.mean_reversion import MeanReversionStrategy
from .strategies.breakout import BreakoutStrategy

logger = logging.getLogger("AlphaWin")


class AlphaWinAgent:
    """The winning trading agent."""

    def __init__(self, config_path: Optional[str] = None, config: Optional[dict] = None,
                 prop_firm: Optional[str] = None, prop_phase: str = "challenge"):
        if config is not None:
            self.config = config
        elif config_path:
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            config_file = Path(__file__).parent.parent / "config" / "default.json"
            with open(config_file) as f:
                self.config = json.load(f)

        self.prop_firm = prop_firm
        self.prop_phase = prop_phase
        initial_capital = self.config.get("agent", {}).get("initial_capital", 10000.0)

        self.market_data = MarketData(max_candles=500)
        self.risk_manager = RiskManager(
            self.config.get("risk", {}),
            initial_capital=initial_capital,
            prop_firm=prop_firm,
            prop_phase=prop_phase,
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

        # Filters - every trade must pass ALL of these
        self.regime_detector = RegimeDetector(lookback=50)
        self.mtf_filter = MultiTimeframeFilter(multiplier=4)
        self.consensus_filter = ConsensusFilter(min_agreement=2)
        self.quality_scorer = TradeQualityScorer(min_quality=0.65)
        self.cooldown_filter = CooldownFilter(cooldown_bars=3, loss_cooldown_bars=8)
        self.persistence_filter = SignalPersistenceFilter(required_bars=3)

        self._warmup = self.config.get("agent", {}).get("warmup_periods", 50)
        self._bar_count = 0
        self._current_regime = MarketRegime.RANGING
        logger.info(f"AlphaWin initialized with {len(self.strategies)} strategies + 5 filters")

    def on_candle(self, candle: Candle) -> dict:
        """Process a new candle and return action taken."""
        self.market_data.add_candle(candle)
        self._bar_count += 1

        result = {
            "action": "hold",
            "price": candle.close,
            "signals": [],
            "exits": [],
            "filters": {},
        }

        # Update existing positions first
        exits = self.risk_manager.update_positions(candle.close)
        if exits:
            result["exits"] = exits
            for e in exits:
                was_loss = e.get("pnl", 0) < 0
                self.cooldown_filter.record_trade(self._bar_count, was_loss)
                logger.info(f"Position closed: {e['exit_reason']} | PnL: {e['pnl']:.2f}")

        # Need enough data
        if self.market_data.size < self._warmup:
            result["action"] = "warmup"
            return result

        # FILTER 1: Market regime
        self._current_regime, regime_conf = self.regime_detector.detect(self.market_data)
        result["filters"]["regime"] = {
            "type": self._current_regime.value,
            "confidence": regime_conf,
        }

        # Don't trade in chaos
        if self._current_regime == MarketRegime.VOLATILE_CHAOS:
            result["action"] = "hold"
            result["filters"]["blocked_by"] = "regime_chaos"
            return result

        # FILTER 2: Cooldown
        can_trade, cooldown_reason = self.cooldown_filter.can_trade(self._bar_count)
        result["filters"]["cooldown"] = cooldown_reason
        if not can_trade:
            result["action"] = "hold"
            result["filters"]["blocked_by"] = "cooldown"
            return result

        # Run all strategies with regime-adjusted weights
        regime_weights = self.regime_detector.get_strategy_weights(self._current_regime)
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

        # Weighted signal combination (regime-adaptive weights)
        combined_score = self._combine_signals(signals, regime_weights)

        # Record for persistence tracking
        self.persistence_filter.record_score(combined_score)

        # Decision
        if combined_score >= 0.25:
            self._try_enter(OrderSide.BUY, signals, combined_score, result, regime_weights)
        elif combined_score <= -0.25:
            self._try_enter(OrderSide.SELL, signals, combined_score, result, regime_weights)
        else:
            result["action"] = "hold"
            result["combined_score"] = combined_score

        result["stats"] = self.risk_manager.get_stats()
        return result

    def _combine_signals(self, signals: list[StrategySignal],
                         regime_weights: dict) -> float:
        """Regime-adaptive weighted vote across all strategy signals."""
        total_score = 0.0
        total_weight = 0.0

        for sig in signals:
            weight = regime_weights.get(sig.strategy_name, 0.33)
            # Only count signal if its strategy is favorable for current regime
            if weight > 0:
                total_score += sig.signal.value * sig.confidence * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _try_enter(self, side: OrderSide, signals: list[StrategySignal],
                   score: float, result: dict, regime_weights: dict = None):
        """Attempt to enter a position - must pass ALL filters."""
        # FILTER: Signal persistence
        persist_ok, persist_reason = self.persistence_filter.check(side)
        result["filters"]["persistence"] = persist_reason
        if not persist_ok:
            result["action"] = "filtered"
            result["filters"]["blocked_by"] = "persistence"
            result["combined_score"] = score
            return

        # Only check consensus among regime-relevant strategies
        if regime_weights:
            relevant_signals = [
                s for s in signals
                if regime_weights.get(s.strategy_name, 0) >= 0.15
            ]
        else:
            relevant_signals = signals

        # FILTER 3: Consensus - multiple strategies must agree
        consensus_ok, consensus_reason = self.consensus_filter.check(relevant_signals, side)
        result["filters"]["consensus"] = consensus_reason
        if not consensus_ok:
            result["action"] = "filtered"
            result["filters"]["blocked_by"] = "consensus"
            result["combined_score"] = score
            return

        # FILTER 4: Multi-timeframe alignment
        mtf_ok, mtf_reason = self.mtf_filter.check_alignment(self.market_data, side)
        result["filters"]["mtf"] = mtf_reason
        if not mtf_ok:
            result["action"] = "filtered"
            result["filters"]["blocked_by"] = "mtf"
            result["combined_score"] = score
            return

        # FILTER 5: Trade quality score
        quality, quality_ok, quality_reason = self.quality_scorer.score(
            signals, self.market_data, side
        )
        result["filters"]["quality"] = quality_reason
        if not quality_ok:
            result["action"] = "filtered"
            result["filters"]["blocked_by"] = "quality"
            result["combined_score"] = score
            return

        # ALL FILTERS PASSED - execute the trade
        best_signal = max(signals, key=lambda s: s.confidence)
        best_signal.confidence = min(abs(score) * quality, 1.0)

        order = self.risk_manager.create_order(self.market_data, best_signal, side)
        if order:
            position = self.risk_manager.open_position(order)
            self.cooldown_filter.record_trade(self._bar_count)
            result["action"] = side.value
            result["order"] = {
                "side": order.side.value,
                "price": order.price,
                "quantity": order.quantity,
                "stop_loss": order.stop_loss,
                "take_profit": order.take_profit,
            }
            result["combined_score"] = score
            result["trade_quality"] = quality
            logger.info(
                f"TRADE: {side.value} @ {order.price:.2f} | "
                f"Quality: {quality:.2f} | Score: {score:.2f} | "
                f"Regime: {self._current_regime.value}"
            )
        else:
            result["action"] = "blocked"
            result["filters"]["blocked_by"] = "risk_manager"
            result["combined_score"] = score

    def get_stats(self) -> dict:
        return self.risk_manager.get_stats()

    def backtest(self, candles: list[Candle]) -> dict:
        """Run a backtest with the agent's current configuration."""
        from .core.backtester import Backtester
        bt = Backtester(self.config, self.strategies, self.risk_manager.initial_capital,
                        prop_firm=self.prop_firm, prop_phase=self.prop_phase)
        return bt.run(candles, warmup=self._warmup)
