"""Market Regime Detector - Only trade when conditions favor winning.

Identifies the current market regime and filters out unfavorable conditions.
This is the #1 edge: knowing WHEN not to trade is more valuable than knowing
when to trade.

Regimes:
- TRENDING_UP: Strong bullish trend, favor momentum/breakout longs
- TRENDING_DOWN: Strong bearish trend, favor momentum/breakout shorts
- RANGING: Sideways market, favor mean reversion
- VOLATILE_CHAOS: High volatility with no direction - DO NOT TRADE
- LOW_VOLATILITY: Compression, potential breakout setup
"""
import numpy as np
from enum import Enum
from .market_data import MarketData
from . import indicators as ind


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE_CHAOS = "volatile_chaos"
    LOW_VOLATILITY = "low_volatility"


class RegimeDetector:
    """Detects current market regime to filter trades."""

    def __init__(self, lookback: int = 50):
        self.lookback = lookback

    def detect(self, market_data: MarketData) -> tuple[MarketRegime, float]:
        """Returns (regime, confidence) where confidence is 0-1."""
        if market_data.size < self.lookback + 10:
            return MarketRegime.RANGING, 0.0

        close = market_data.close
        high = market_data.high
        low = market_data.low

        # 1. Trend direction via linear regression slope
        recent = close[-self.lookback:]
        x = np.arange(len(recent))
        slope, intercept = np.polyfit(x, recent, 1)
        normalized_slope = slope / np.mean(recent)  # Normalize by price level

        # 2. Trend strength via ADX
        adx_vals = market_data.adx(14)
        valid_adx = adx_vals[~np.isnan(adx_vals)]
        adx_strength = valid_adx[-1] if len(valid_adx) > 0 else 0

        # 3. Volatility regime via ATR ratio
        atr_vals = market_data.atr(14)
        valid_atr = atr_vals[~np.isnan(atr_vals)]
        if len(valid_atr) >= 20:
            current_atr = valid_atr[-1]
            avg_atr = np.mean(valid_atr[-20:])
            atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
        else:
            atr_ratio = 1.0

        # 4. Directional consistency: how often does price move in trend direction
        returns = np.diff(recent)
        if normalized_slope > 0:
            consistency = np.sum(returns > 0) / len(returns)
        else:
            consistency = np.sum(returns < 0) / len(returns)

        # 5. Range width relative to price
        price_range = (np.max(recent) - np.min(recent)) / np.mean(recent)

        # Classification logic
        if atr_ratio > 2.0 and adx_strength < 20:
            # High volatility but no trend = chaos
            return MarketRegime.VOLATILE_CHAOS, min(atr_ratio / 3.0, 1.0)

        if adx_strength > 25 and abs(normalized_slope) > 0.0001:
            confidence = min((adx_strength - 20) / 30.0, 1.0) * consistency
            if normalized_slope > 0:
                return MarketRegime.TRENDING_UP, confidence
            else:
                return MarketRegime.TRENDING_DOWN, confidence

        if atr_ratio < 0.6 and price_range < 0.03:
            return MarketRegime.LOW_VOLATILITY, min((0.6 - atr_ratio) / 0.4, 1.0)

        # Default: ranging
        confidence = 1.0 - min(adx_strength / 40.0, 1.0)
        return MarketRegime.RANGING, confidence

    def is_favorable(self, regime: MarketRegime, strategy_name: str) -> bool:
        """Check if the current regime is favorable for a given strategy."""
        # NEVER trade in chaotic conditions
        if regime == MarketRegime.VOLATILE_CHAOS:
            return False

        favorable = {
            "MomentumStrategy": [
                MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN
            ],
            "MeanReversionStrategy": [
                MarketRegime.RANGING, MarketRegime.LOW_VOLATILITY
            ],
            "BreakoutStrategy": [
                MarketRegime.LOW_VOLATILITY, MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN
            ],
        }
        return regime in favorable.get(strategy_name, [MarketRegime.RANGING])

    def get_strategy_weights(self, regime: MarketRegime) -> dict:
        """Dynamically adjust strategy weights based on regime."""
        weights = {
            MarketRegime.TRENDING_UP: {
                "MomentumStrategy": 0.50,
                "MeanReversionStrategy": 0.10,
                "BreakoutStrategy": 0.40,
            },
            MarketRegime.TRENDING_DOWN: {
                "MomentumStrategy": 0.50,
                "MeanReversionStrategy": 0.10,
                "BreakoutStrategy": 0.40,
            },
            MarketRegime.RANGING: {
                "MomentumStrategy": 0.15,
                "MeanReversionStrategy": 0.60,
                "BreakoutStrategy": 0.25,
            },
            MarketRegime.LOW_VOLATILITY: {
                "MomentumStrategy": 0.15,
                "MeanReversionStrategy": 0.25,
                "BreakoutStrategy": 0.60,
            },
            MarketRegime.VOLATILE_CHAOS: {
                "MomentumStrategy": 0.0,
                "MeanReversionStrategy": 0.0,
                "BreakoutStrategy": 0.0,
            },
        }
        return weights.get(regime, {
            "MomentumStrategy": 0.33,
            "MeanReversionStrategy": 0.34,
            "BreakoutStrategy": 0.33,
        })
