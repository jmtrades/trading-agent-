"""Tests for trade filters and regime detection."""
import numpy as np
import pytest
from trading_agent.core.models import Candle, Signal, OrderSide, StrategySignal
from trading_agent.core.market_data import MarketData
from trading_agent.core.regime_detector import RegimeDetector, MarketRegime
from trading_agent.core.filters import (
    MultiTimeframeFilter, ConsensusFilter,
    TradeQualityScorer, CooldownFilter,
)
from trading_agent.core.backtester import Backtester


def make_market_data(prices, volumes=None):
    md = MarketData()
    for i, p in enumerate(prices):
        vol = volumes[i] if volumes else 1000.0
        md.add_candle(Candle(
            timestamp=float(i * 3600),
            open=p * 0.999, high=p * 1.005,
            low=p * 0.995, close=p, volume=vol,
        ))
    return md


class TestRegimeDetector:
    def test_detects_something(self):
        detector = RegimeDetector(lookback=30)
        candles = Backtester.generate_synthetic_data(100, trend="bull")
        md = MarketData()
        md.add_candles(candles)
        regime, conf = detector.detect(md)
        assert isinstance(regime, MarketRegime)
        assert 0 <= conf <= 1

    def test_insufficient_data(self):
        detector = RegimeDetector(lookback=50)
        md = make_market_data([100] * 10)
        regime, conf = detector.detect(md)
        assert regime == MarketRegime.RANGING
        assert conf == 0.0

    def test_chaos_blocks_all_strategies(self):
        detector = RegimeDetector()
        assert not detector.is_favorable(MarketRegime.VOLATILE_CHAOS, "MomentumStrategy")
        assert not detector.is_favorable(MarketRegime.VOLATILE_CHAOS, "MeanReversionStrategy")
        assert not detector.is_favorable(MarketRegime.VOLATILE_CHAOS, "BreakoutStrategy")

    def test_regime_weights_sum(self):
        detector = RegimeDetector()
        for regime in MarketRegime:
            weights = detector.get_strategy_weights(regime)
            if regime != MarketRegime.VOLATILE_CHAOS:
                assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)


class TestConsensusFilter:
    def test_consensus_met(self):
        f = ConsensusFilter(min_agreement=2)
        signals = [
            StrategySignal(Signal.BUY, 0.8, "A", ""),
            StrategySignal(Signal.BUY, 0.6, "B", ""),
            StrategySignal(Signal.NEUTRAL, 0.0, "C", ""),
        ]
        ok, _ = f.check(signals, OrderSide.BUY)
        assert ok

    def test_consensus_not_met(self):
        f = ConsensusFilter(min_agreement=2)
        signals = [
            StrategySignal(Signal.BUY, 0.8, "A", ""),
            StrategySignal(Signal.NEUTRAL, 0.0, "B", ""),
            StrategySignal(Signal.NEUTRAL, 0.0, "C", ""),
        ]
        ok, _ = f.check(signals, OrderSide.BUY)
        assert not ok

    def test_conflicting_signals_blocked(self):
        f = ConsensusFilter(min_agreement=2)
        signals = [
            StrategySignal(Signal.BUY, 0.8, "A", ""),
            StrategySignal(Signal.BUY, 0.6, "B", ""),
            StrategySignal(Signal.SELL, 0.5, "C", ""),
        ]
        ok, _ = f.check(signals, OrderSide.BUY)
        assert not ok


class TestCooldownFilter:
    def test_allows_first_trade(self):
        f = CooldownFilter(cooldown_bars=5, loss_cooldown_bars=10)
        ok, _ = f.can_trade(0)
        assert ok

    def test_blocks_during_cooldown(self):
        f = CooldownFilter(cooldown_bars=5, loss_cooldown_bars=10)
        f.record_trade(10)
        ok, _ = f.can_trade(12)
        assert not ok

    def test_allows_after_cooldown(self):
        f = CooldownFilter(cooldown_bars=5, loss_cooldown_bars=10)
        f.record_trade(10)
        ok, _ = f.can_trade(16)
        assert ok

    def test_longer_cooldown_after_loss(self):
        f = CooldownFilter(cooldown_bars=5, loss_cooldown_bars=10)
        f.record_trade(10, was_loss=True)
        ok, _ = f.can_trade(16)
        assert not ok
        ok, _ = f.can_trade(21)
        assert ok


class TestTradeQualityScorer:
    def test_returns_score(self):
        scorer = TradeQualityScorer(min_quality=0.5)
        signals = [
            StrategySignal(Signal.BUY, 0.8, "A", ""),
            StrategySignal(Signal.BUY, 0.7, "B", ""),
        ]
        md = make_market_data(list(np.linspace(95, 105, 100)), [1000] * 100)
        quality, passes, reason = scorer.score(signals, md, OrderSide.BUY)
        assert 0 <= quality <= 1
        assert isinstance(passes, bool)
        assert "Quality=" in reason


class TestMultiTimeframeFilter:
    def test_allows_with_insufficient_data(self):
        f = MultiTimeframeFilter(multiplier=4)
        md = make_market_data([100] * 50)
        ok, _ = f.check_alignment(md, OrderSide.BUY)
        assert ok  # Should allow when not enough data

    def test_alignment_check(self):
        f = MultiTimeframeFilter(multiplier=4)
        # Uptrending market should align with buy
        prices = list(np.linspace(90, 110, 300))
        md = make_market_data(prices)
        ok, _ = f.check_alignment(md, OrderSide.BUY)
        assert ok
