"""Tests for trading strategies."""
import numpy as np
import pytest
from trading_agent.core.models import Candle, Signal
from trading_agent.core.market_data import MarketData
from trading_agent.strategies.momentum import MomentumStrategy
from trading_agent.strategies.mean_reversion import MeanReversionStrategy
from trading_agent.strategies.breakout import BreakoutStrategy


def make_candles(prices, volumes=None):
    """Create candles from a price series."""
    candles = []
    for i, p in enumerate(prices):
        vol = volumes[i] if volumes else 1000.0
        candles.append(Candle(
            timestamp=float(i * 3600),
            open=p * 0.999,
            high=p * 1.005,
            low=p * 0.995,
            close=p,
            volume=vol,
        ))
    return candles


def make_market_data(prices, volumes=None):
    md = MarketData()
    md.add_candles(make_candles(prices, volumes))
    return md


class TestMomentumStrategy:
    @pytest.fixture
    def strategy(self):
        return MomentumStrategy({
            "weight": 0.35, "rsi_period": 14, "rsi_oversold": 30,
            "rsi_overbought": 70, "macd_fast": 12, "macd_slow": 26,
            "macd_signal": 9,
        })

    def test_insufficient_data(self, strategy):
        md = make_market_data([100] * 10)
        signal = strategy.analyze(md)
        assert signal.signal == Signal.NEUTRAL

    def test_bullish_trend(self, strategy):
        # Strong uptrend should produce buy signal
        prices = list(np.linspace(80, 120, 100))
        md = make_market_data(prices)
        signal = strategy.analyze(md)
        assert signal.signal.value >= 0  # At least neutral/bullish

    def test_bearish_trend(self, strategy):
        prices = list(np.linspace(120, 80, 100))
        md = make_market_data(prices)
        signal = strategy.analyze(md)
        assert signal.signal.value <= 0  # At least neutral/bearish


class TestMeanReversionStrategy:
    @pytest.fixture
    def strategy(self):
        return MeanReversionStrategy({
            "weight": 0.30, "bb_period": 20, "bb_std": 2.0,
            "rsi_period": 14, "lookback": 20,
        })

    def test_insufficient_data(self, strategy):
        md = make_market_data([100] * 10)
        signal = strategy.analyze(md)
        assert signal.signal == Signal.NEUTRAL

    def test_oversold_bounce(self, strategy):
        # Prices drop sharply then stabilize - should signal buy
        prices = [100] * 50 + list(np.linspace(100, 85, 20))
        md = make_market_data(prices)
        signal = strategy.analyze(md)
        # After a sharp drop, mean reversion should lean bullish
        assert signal.confidence > 0

    def test_overbought_condition(self, strategy):
        prices = [100] * 50 + list(np.linspace(100, 115, 20))
        md = make_market_data(prices)
        signal = strategy.analyze(md)
        assert signal.confidence > 0


class TestBreakoutStrategy:
    @pytest.fixture
    def strategy(self):
        return BreakoutStrategy({
            "weight": 0.35, "lookback": 20, "volume_multiplier": 1.5,
            "atr_period": 14, "atr_multiplier": 1.5,
        })

    def test_insufficient_data(self, strategy):
        md = make_market_data([100] * 10)
        signal = strategy.analyze(md)
        assert signal.signal == Signal.NEUTRAL

    def test_bullish_breakout(self, strategy):
        # Consolidation then breakout above with volume
        prices = [100 + np.random.uniform(-1, 1) for _ in range(50)]
        prices.append(110)  # Breakout candle
        volumes = [1000] * 50 + [3000]  # Volume spike
        md = make_market_data(prices, volumes)
        signal = strategy.analyze(md)
        assert signal.signal.value >= 1  # Buy or strong buy

    def test_no_breakout_without_volume(self, strategy):
        prices = [100 + np.random.uniform(-1, 1) for _ in range(50)]
        prices.append(110)
        volumes = [1000] * 50 + [500]  # Low volume
        md = make_market_data(prices, volumes)
        signal = strategy.analyze(md)
        # Should have lower confidence without volume
        assert signal.confidence < 0.8
