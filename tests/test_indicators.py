"""Tests for technical indicators."""
import numpy as np
import pytest
from trading_agent.core import indicators as ind


@pytest.fixture
def price_data():
    """Generate simple price data for testing."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    return prices


@pytest.fixture
def ohlcv_data(price_data):
    """Generate OHLCV data."""
    close = price_data
    high = close + np.abs(np.random.randn(len(close))) * 0.5
    low = close - np.abs(np.random.randn(len(close))) * 0.5
    volume = np.random.uniform(500, 1500, len(close))
    return high, low, close, volume


class TestSMA:
    def test_basic(self, price_data):
        result = ind.sma(price_data, 10)
        assert len(result) == len(price_data)
        assert np.isnan(result[8])
        assert not np.isnan(result[9])

    def test_value(self, price_data):
        result = ind.sma(price_data, 5)
        expected = np.mean(price_data[:5])
        assert abs(result[4] - expected) < 1e-10


class TestEMA:
    def test_basic(self, price_data):
        result = ind.ema(price_data, 10)
        assert len(result) == len(price_data)
        assert not np.isnan(result[-1])

    def test_responds_to_price(self, price_data):
        result = ind.ema(price_data, 10)
        # EMA should exist for last element
        assert not np.isnan(result[-1])


class TestRSI:
    def test_range(self, price_data):
        result = ind.rsi(price_data, 14)
        valid = result[~np.isnan(result)]
        assert all(0 <= v <= 100 for v in valid)

    def test_rising_prices(self):
        prices = np.linspace(100, 200, 50)
        result = ind.rsi(prices, 14)
        valid = result[~np.isnan(result)]
        # Strongly rising prices should have high RSI
        assert valid[-1] > 70


class TestMACD:
    def test_shape(self, price_data):
        macd_line, signal_line, histogram = ind.macd(price_data)
        assert len(macd_line) == len(price_data)
        assert len(signal_line) == len(price_data)
        assert len(histogram) == len(price_data)


class TestBollingerBands:
    def test_band_order(self, price_data):
        upper, middle, lower = ind.bollinger_bands(price_data, 20, 2.0)
        # Where all are valid, upper > middle > lower
        for i in range(len(price_data)):
            if not np.isnan(upper[i]):
                assert upper[i] >= middle[i] >= lower[i]


class TestATR:
    def test_positive(self, ohlcv_data):
        high, low, close, _ = ohlcv_data
        result = ind.atr(high, low, close, 14)
        valid = result[~np.isnan(result)]
        assert all(v >= 0 for v in valid)


class TestVWAP:
    def test_basic(self, ohlcv_data):
        high, low, close, volume = ohlcv_data
        result = ind.vwap(high, low, close, volume)
        assert len(result) == len(close)
        assert not np.isnan(result[-1])
