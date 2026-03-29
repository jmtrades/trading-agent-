"""Technical indicators computed with numpy for speed."""
import numpy as np
from typing import Tuple


def sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    if len(data) < period:
        return np.full_like(data, np.nan)
    result = np.full_like(data, np.nan, dtype=float)
    cumsum = np.cumsum(data)
    result[period - 1:] = (cumsum[period - 1:] - np.concatenate(([0], cumsum[:-period]))) / period
    return result


def ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    result = np.full_like(data, np.nan, dtype=float)
    if len(data) < period:
        return result
    multiplier = 2.0 / (period + 1)
    result[period - 1] = np.mean(data[:period])
    for i in range(period, len(data)):
        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
    return result


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index."""
    result = np.full_like(close, np.nan, dtype=float)
    if len(close) < period + 1:
        return result

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return result


def macd(close: np.ndarray, fast: int = 12, slow: int = 26,
         signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD: returns (macd_line, signal_line, histogram)."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line[~np.isnan(macd_line)], signal_period)

    # Align signal line with macd_line
    full_signal = np.full_like(close, np.nan, dtype=float)
    start = np.argmax(~np.isnan(macd_line))
    valid_signal = signal_line[~np.isnan(signal_line)]
    if len(valid_signal) > 0:
        offset = start + signal_period - 1
        end = offset + len(valid_signal)
        if end <= len(full_signal):
            full_signal[offset:end] = valid_signal
        else:
            full_signal[offset:] = valid_signal[:len(full_signal) - offset]

    histogram = macd_line - full_signal
    return macd_line, full_signal, histogram


def bollinger_bands(close: np.ndarray, period: int = 20,
                    std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands: returns (upper, middle, lower)."""
    middle = sma(close, period)
    result_upper = np.full_like(close, np.nan, dtype=float)
    result_lower = np.full_like(close, np.nan, dtype=float)

    for i in range(period - 1, len(close)):
        window = close[i - period + 1:i + 1]
        std = np.std(window, ddof=0)
        result_upper[i] = middle[i] + std_dev * std
        result_lower[i] = middle[i] - std_dev * std

    return result_upper, middle, result_lower


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int = 14) -> np.ndarray:
    """Average True Range."""
    result = np.full_like(close, np.nan, dtype=float)
    if len(close) < period + 1:
        return result

    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))

    result[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, len(close)):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

    return result


def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         volume: np.ndarray) -> np.ndarray:
    """Volume Weighted Average Price."""
    typical_price = (high + low + close) / 3.0
    cumulative_tp_vol = np.cumsum(typical_price * volume)
    cumulative_vol = np.cumsum(volume)
    # Avoid division by zero
    cumulative_vol = np.where(cumulative_vol == 0, 1, cumulative_vol)
    return cumulative_tp_vol / cumulative_vol


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int = 14) -> np.ndarray:
    """Average Directional Index - measures trend strength."""
    result = np.full_like(close, np.nan, dtype=float)
    if len(close) < period * 2 + 1:
        return result

    up_move = np.diff(high)
    down_move = -np.diff(low)

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_vals = atr(high, low, close, period)

    smooth_plus_dm = ema(plus_dm, period)
    smooth_minus_dm = ema(minus_dm, period)

    # Pad to match original array length
    full_plus = np.full_like(close, np.nan, dtype=float)
    full_minus = np.full_like(close, np.nan, dtype=float)
    full_plus[1:] = smooth_plus_dm
    full_minus[1:] = smooth_minus_dm

    plus_di = np.where(atr_vals > 0, 100 * full_plus / atr_vals, 0)
    minus_di = np.where(atr_vals > 0, 100 * full_minus / atr_vals, 0)

    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = np.where(di_sum > 0, 100 * di_diff / di_sum, 0)

    # Smooth DX to get ADX
    valid_dx = dx[~np.isnan(dx)]
    if len(valid_dx) >= period:
        adx_vals = ema(valid_dx, period)
        start = len(dx) - len(valid_dx)
        valid_adx = adx_vals[~np.isnan(adx_vals)]
        adx_start = start + (len(valid_dx) - len(valid_adx))
        end = adx_start + len(valid_adx)
        if end <= len(result):
            result[adx_start:end] = valid_adx

    return result
