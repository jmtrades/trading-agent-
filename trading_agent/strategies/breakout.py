"""Breakout Strategy - Catches explosive moves from consolidation.

This strategy identifies price breaking out of ranges:
- Support/resistance from recent highs/lows
- Volume confirmation (breakouts need volume)
- ATR for volatility-adjusted thresholds
- Consolidation detection (tight ranges precede big moves)

Win edge: Filters breakouts aggressively using volume and ATR to avoid
fakeouts. Only enters when the breakout is confirmed by multiple factors.
"""
import numpy as np
from .base import BaseStrategy
from trading_agent.core.market_data import MarketData
from trading_agent.core.models import Signal, StrategySignal


class BreakoutStrategy(BaseStrategy):

    def analyze(self, market_data: MarketData) -> StrategySignal:
        cfg = self.config
        close = market_data.close
        high = market_data.high
        low = market_data.low
        volume = market_data.volume

        lookback = cfg.get("lookback", 20)
        vol_mult = cfg.get("volume_multiplier", 1.5)
        atr_mult = cfg.get("atr_multiplier", 1.5)

        if market_data.size < lookback + 10:
            return StrategySignal(Signal.NEUTRAL, 0.0, self.name, "Insufficient data")

        current_price = close[-1]
        current_volume = volume[-1]

        # Key levels from recent price action
        recent_high = np.max(high[-lookback:-1])
        recent_low = np.min(low[-lookback:-1])
        price_range = recent_high - recent_low

        # ATR for volatility context
        atr_vals = market_data.atr(cfg.get("atr_period", 14))
        current_atr = atr_vals[-1]
        if np.isnan(current_atr):
            valid_atr = atr_vals[~np.isnan(atr_vals)]
            current_atr = valid_atr[-1] if len(valid_atr) > 0 else price_range * 0.02

        # Volume analysis
        avg_volume = np.mean(volume[-lookback:-1])
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        # Consolidation detection: narrowing range over recent bars
        ranges = high[-10:] - low[-10:]
        range_trend = np.polyfit(range(len(ranges)), ranges, 1)[0]
        is_consolidating = range_trend < 0

        # Price relative to range
        range_position = (current_price - recent_low) / price_range if price_range > 0 else 0.5

        score = 0.0
        reasons = []

        # Bullish breakout
        if current_price > recent_high:
            breakout_distance = (current_price - recent_high) / current_atr
            if breakout_distance > 0:
                score += min(breakout_distance * 1.5, 3.0)
                reasons.append(f"Bullish breakout ({breakout_distance:.1f}x ATR above resistance)")

        # Bearish breakout
        elif current_price < recent_low:
            breakout_distance = (recent_low - current_price) / current_atr
            if breakout_distance > 0:
                score -= min(breakout_distance * 1.5, 3.0)
                reasons.append(f"Bearish breakout ({breakout_distance:.1f}x ATR below support)")

        # Near breakout levels (anticipation)
        elif range_position > 0.9:
            score += 0.75
            reasons.append("Testing resistance")
        elif range_position < 0.1:
            score -= 0.75
            reasons.append("Testing support")

        # Volume confirmation - critical for breakouts
        if volume_ratio >= vol_mult:
            score *= 1.5
            reasons.append(f"High volume ({volume_ratio:.1f}x average)")
        elif volume_ratio >= 1.2:
            score *= 1.1
            reasons.append(f"Above avg volume ({volume_ratio:.1f}x)")
        elif abs(score) > 1.0 and volume_ratio < 0.8:
            score *= 0.4  # Kill weak-volume breakouts
            reasons.append(f"LOW volume breakout - likely fakeout ({volume_ratio:.1f}x)")

        # Consolidation boost: breakouts from tight ranges are stronger
        if is_consolidating and abs(score) > 1.0:
            score *= 1.3
            reasons.append("Breaking from consolidation (high conviction)")

        # ATR filter: only trade when volatility supports the move
        avg_atr = np.nanmean(atr_vals[-lookback:])
        if not np.isnan(avg_atr) and current_atr > avg_atr * atr_mult:
            score *= 1.2
            reasons.append("Volatility expanding")

        # Multi-bar confirmation: price closing above/below for consecutive bars
        if len(close) >= 3:
            if all(close[-i] > recent_high for i in range(1, min(4, len(close)))):
                score += 1.0
                reasons.append("Multi-bar breakout confirmation")
            elif all(close[-i] < recent_low for i in range(1, min(4, len(close)))):
                score -= 1.0
                reasons.append("Multi-bar breakdown confirmation")

        # Convert to signal
        confidence = min(abs(score) / 5.0, 1.0)

        if score >= 3.0:
            signal = Signal.STRONG_BUY
        elif score >= 1.5:
            signal = Signal.BUY
        elif score <= -3.0:
            signal = Signal.STRONG_SELL
        elif score <= -1.5:
            signal = Signal.SELL
        else:
            signal = Signal.NEUTRAL

        return StrategySignal(
            signal=signal,
            confidence=confidence,
            strategy_name=self.name,
            reason=" | ".join(reasons),
            metadata={
                "volume_ratio": volume_ratio,
                "range_position": range_position,
                "breakout_atr": (current_price - recent_high) / current_atr if current_atr > 0 else 0
            }
        )
