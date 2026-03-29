"""Market data manager - handles candle storage and indicator computation."""
import numpy as np
from typing import Optional
from .models import Candle
from . import indicators as ind


class MarketData:
    """Stores candles and provides computed indicators on demand."""

    def __init__(self, max_candles: int = 500):
        self.max_candles = max_candles
        self.candles: list[Candle] = []

    def add_candle(self, candle: Candle):
        self.candles.append(candle)
        if len(self.candles) > self.max_candles:
            self.candles = self.candles[-self.max_candles:]

    def add_candles(self, candles: list[Candle]):
        for c in candles:
            self.add_candle(c)

    @property
    def size(self) -> int:
        return len(self.candles)

    @property
    def close(self) -> np.ndarray:
        return np.array([c.close for c in self.candles])

    @property
    def high(self) -> np.ndarray:
        return np.array([c.high for c in self.candles])

    @property
    def low(self) -> np.ndarray:
        return np.array([c.low for c in self.candles])

    @property
    def open(self) -> np.ndarray:
        return np.array([c.open for c in self.candles])

    @property
    def volume(self) -> np.ndarray:
        return np.array([c.volume for c in self.candles])

    @property
    def latest_price(self) -> Optional[float]:
        return self.candles[-1].close if self.candles else None

    def rsi(self, period: int = 14) -> np.ndarray:
        return ind.rsi(self.close, period)

    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9):
        return ind.macd(self.close, fast, slow, signal)

    def bollinger_bands(self, period: int = 20, std_dev: float = 2.0):
        return ind.bollinger_bands(self.close, period, std_dev)

    def atr(self, period: int = 14) -> np.ndarray:
        return ind.atr(self.high, self.low, self.close, period)

    def sma(self, period: int) -> np.ndarray:
        return ind.sma(self.close, period)

    def ema(self, period: int) -> np.ndarray:
        return ind.ema(self.close, period)

    def vwap(self) -> np.ndarray:
        return ind.vwap(self.high, self.low, self.close, self.volume)

    def adx(self, period: int = 14) -> np.ndarray:
        return ind.adx(self.high, self.low, self.close, period)

    def stochastic(self, k_period: int = 14, d_period: int = 3):
        return ind.stochastic(self.high, self.low, self.close, k_period, d_period)

    def obv(self) -> np.ndarray:
        return ind.obv(self.close, self.volume)

    def ichimoku(self, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
        return ind.ichimoku(self.high, self.low, self.close, tenkan, kijun, senkou_b)

    def williams_r(self, period: int = 14) -> np.ndarray:
        return ind.williams_r(self.high, self.low, self.close, period)

    def mfi(self, period: int = 14) -> np.ndarray:
        return ind.mfi(self.high, self.low, self.close, self.volume, period)

    def roc(self, period: int = 12) -> np.ndarray:
        return ind.price_rate_of_change(self.close, period)

    def keltner_channels(self, ema_period: int = 20, atr_mult: float = 2.0):
        return ind.keltner_channels(self.high, self.low, self.close, ema_period, 14, atr_mult)

    def squeeze(self):
        return ind.squeeze_detector(self.high, self.low, self.close)
