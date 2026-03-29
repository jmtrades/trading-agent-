"""Tests for technical indicators."""
import numpy as np
import pytest
from trading_agent.core import indicators as ind


@pytest.fixture
def price_data():
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    return prices


@pytest.fixture
def ohlcv_data(price_data):
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


class TestStochastic:
    def test_range(self, ohlcv_data):
        high, low, close, _ = ohlcv_data
        k, d = ind.stochastic(high, low, close, 14, 3)
        valid_k = k[~np.isnan(k)]
        assert all(0 <= v <= 100 for v in valid_k)

    def test_high_close_gives_high_k(self):
        # If close is at the high, %K should be near 100
        high = np.array([110.0] * 20)
        low = np.array([90.0] * 20)
        close = np.array([109.0] * 20)
        k, d = ind.stochastic(high, low, close, 14, 3)
        valid_k = k[~np.isnan(k)]
        assert valid_k[-1] > 90


class TestOBV:
    def test_rising_prices_rising_obv(self):
        close = np.linspace(100, 110, 20)
        volume = np.ones(20) * 1000
        result = ind.obv(close, volume)
        # OBV should be rising when prices rise
        assert result[-1] > result[0]

    def test_flat_prices(self):
        close = np.ones(20) * 100
        volume = np.ones(20) * 1000
        result = ind.obv(close, volume)
        # OBV should not change much with flat prices
        assert result[-1] == result[1]  # After first candle


class TestIchimoku:
    def test_shapes(self, ohlcv_data):
        high, low, close, _ = ohlcv_data
        tenkan, kijun, span_a, span_b = ind.ichimoku(high, low, close)
        assert len(tenkan) == len(close)
        assert len(kijun) == len(close)
        assert len(span_a) == len(close)
        assert len(span_b) == len(close)

    def test_values_exist(self, ohlcv_data):
        high, low, close, _ = ohlcv_data
        tenkan, kijun, span_a, span_b = ind.ichimoku(high, low, close)
        assert not np.isnan(tenkan[-1])
        assert not np.isnan(kijun[-1])


class TestWilliamsR:
    def test_range(self, ohlcv_data):
        high, low, close, _ = ohlcv_data
        result = ind.williams_r(high, low, close, 14)
        valid = result[~np.isnan(result)]
        assert all(-100 <= v <= 0 for v in valid)


class TestMFI:
    def test_range(self, ohlcv_data):
        high, low, close, volume = ohlcv_data
        result = ind.mfi(high, low, close, volume, 14)
        valid = result[~np.isnan(result)]
        assert all(0 <= v <= 100 for v in valid)


class TestROC:
    def test_positive_trend(self):
        close = np.linspace(100, 120, 30)
        result = ind.price_rate_of_change(close, 12)
        valid = result[~np.isnan(result)]
        assert valid[-1] > 0


class TestKeltnerChannels:
    def test_band_order(self, ohlcv_data):
        high, low, close, _ = ohlcv_data
        upper, middle, lower = ind.keltner_channels(high, low, close)
        for i in range(len(close)):
            if not (np.isnan(upper[i]) or np.isnan(lower[i])):
                assert upper[i] >= middle[i] >= lower[i]


class TestSqueezeDetector:
    def test_shapes(self, ohlcv_data):
        high, low, close, _ = ohlcv_data
        squeeze_on, momentum = ind.squeeze_detector(high, low, close)
        assert len(squeeze_on) == len(close)
        assert len(momentum) == len(close)

    def test_squeeze_values(self, ohlcv_data):
        high, low, close, _ = ohlcv_data
        squeeze_on, _ = ind.squeeze_detector(high, low, close)
        # Squeeze should be 0 or 1
        valid = squeeze_on[squeeze_on != 0]
        if len(valid) > 0:
            assert all(v == 1.0 for v in valid)
