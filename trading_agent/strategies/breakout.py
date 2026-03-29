"""Breakout Strategy v2 - Enhanced with Ichimoku cloud and squeeze detection.

Uses 7+ indicators for signal generation:
- Support/resistance from recent highs/lows
- Volume confirmation
- ATR for volatility-adjusted thresholds
- Ichimoku cloud for trend context and breakout levels
- Squeeze detector (BB inside Keltner = energy building)
- OBV for volume trend confirmation
- Williams %R for momentum
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

        if market_data.size < max(lookback + 10, 60):
            return StrategySignal(Signal.NEUTRAL, 0.0, self.name, "Insufficient data")

        current_price = close[-1]
        current_volume = volume[-1]

        # Key levels
        recent_high = np.max(high[-lookback:-1])
        recent_low = np.min(low[-lookback:-1])
        price_range = recent_high - recent_low

        # ATR
        atr_vals = market_data.atr(cfg.get("atr_period", 14))
        valid_atr = atr_vals[~np.isnan(atr_vals)]
        current_atr = valid_atr[-1] if len(valid_atr) > 0 else price_range * 0.02

        # Volume
        avg_volume = np.mean(volume[-lookback:-1])
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        # Consolidation detection
        ranges = high[-10:] - low[-10:]
        range_trend = np.polyfit(range(len(ranges)), ranges, 1)[0]
        is_consolidating = range_trend < 0

        # New indicators
        tenkan, kijun, span_a, span_b = market_data.ichimoku()
        squeeze_on, squeeze_mom = market_data.squeeze()
        williams = market_data.williams_r(14)
        obv_vals = market_data.obv()

        # Ichimoku cloud bounds at current bar
        cloud_top = max(span_a[-1], span_b[-1]) if not (np.isnan(span_a[-1]) or np.isnan(span_b[-1])) else None
        cloud_bottom = min(span_a[-1], span_b[-1]) if cloud_top is not None else None

        current_williams = williams[-1] if not np.isnan(williams[-1]) else -50
        current_squeeze = squeeze_on[-1] if len(squeeze_on) > 0 else 0
        current_squeeze_mom = squeeze_mom[-1] if not np.isnan(squeeze_mom[-1]) else 0

        range_position = (current_price - recent_low) / price_range if price_range > 0 else 0.5

        score = 0.0
        reasons = []

        # --- Price breakout detection ---
        if current_price > recent_high:
            breakout_distance = (current_price - recent_high) / current_atr
            if breakout_distance > 0:
                score += min(breakout_distance * 1.5, 3.0)
                reasons.append(f"Bullish breakout ({breakout_distance:.1f}x ATR)")
        elif current_price < recent_low:
            breakout_distance = (recent_low - current_price) / current_atr
            if breakout_distance > 0:
                score -= min(breakout_distance * 1.5, 3.0)
                reasons.append(f"Bearish breakout ({breakout_distance:.1f}x ATR)")
        elif range_position > 0.9:
            score += 0.75
            reasons.append("Testing resistance")
        elif range_position < 0.1:
            score -= 0.75
            reasons.append("Testing support")

        # --- Volume confirmation ---
        if volume_ratio >= vol_mult:
            score *= 1.5
            reasons.append(f"High volume ({volume_ratio:.1f}x)")
        elif volume_ratio >= 1.2:
            score *= 1.1
        elif abs(score) > 1.0 and volume_ratio < 0.8:
            score *= 0.4
            reasons.append("Low volume - likely fakeout")

        # --- Ichimoku cloud context ---
        if cloud_top is not None:
            if current_price > cloud_top:
                if score > 0:
                    score *= 1.3
                    reasons.append("Price above Ichimoku cloud")
            elif current_price < cloud_bottom:
                if score < 0:
                    score *= 1.3
                    reasons.append("Price below Ichimoku cloud")
            else:
                # Inside the cloud - uncertain, reduce score
                score *= 0.6
                reasons.append("Price inside Ichimoku cloud (uncertain)")

        # Tenkan/Kijun cross
        if not np.isnan(tenkan[-1]) and not np.isnan(kijun[-1]):
            if tenkan[-1] > kijun[-1]:
                score += 0.5
                reasons.append("Ichimoku bullish (Tenkan > Kijun)")
            elif tenkan[-1] < kijun[-1]:
                score -= 0.5
                reasons.append("Ichimoku bearish (Tenkan < Kijun)")

        # --- Squeeze detection ---
        if current_squeeze > 0:
            # Energy building - breakout more likely
            score *= 1.4
            reasons.append("TTM Squeeze active (energy building)")
            # Squeeze momentum direction
            if current_squeeze_mom > 0 and score > 0:
                score += 0.5
                reasons.append("Squeeze momentum bullish")
            elif current_squeeze_mom < 0 and score < 0:
                score -= 0.5
                reasons.append("Squeeze momentum bearish")

        # --- Williams %R ---
        if current_williams > -20:
            score -= 0.5  # Overbought in breakout context = risky for longs
        elif current_williams < -80:
            score += 0.5  # Oversold = potential bounce breakout

        # --- OBV trend confirmation ---
        if len(obv_vals) >= 20:
            obv_recent = obv_vals[-10:]
            obv_slope = np.polyfit(range(len(obv_recent)), obv_recent, 1)[0]
            if obv_slope > 0 and score > 0:
                score *= 1.15
                reasons.append("OBV confirming bullish")
            elif obv_slope < 0 and score < 0:
                score *= 1.15
                reasons.append("OBV confirming bearish")
            elif (obv_slope > 0 and score < 0) or (obv_slope < 0 and score > 0):
                score *= 0.7
                reasons.append("OBV diverging from breakout")

        # --- Consolidation boost ---
        if is_consolidating and abs(score) > 1.0:
            score *= 1.3
            reasons.append("Breaking from consolidation")

        # --- Multi-bar confirmation ---
        if len(close) >= 3:
            if all(close[-j] > recent_high for j in range(1, min(4, len(close)))):
                score += 1.0
                reasons.append("Multi-bar breakout confirmed")
            elif all(close[-j] < recent_low for j in range(1, min(4, len(close)))):
                score -= 1.0
                reasons.append("Multi-bar breakdown confirmed")

        # Convert to signal
        confidence = min(abs(score) / 6.0, 1.0)

        if score >= 4.0:
            signal = Signal.STRONG_BUY
        elif score >= 2.0:
            signal = Signal.BUY
        elif score <= -4.0:
            signal = Signal.STRONG_SELL
        elif score <= -2.0:
            signal = Signal.SELL
        else:
            signal = Signal.NEUTRAL

        return StrategySignal(
            signal=signal,
            confidence=confidence,
            strategy_name=self.name,
            reason=" | ".join(reasons[:5]),
            metadata={
                "volume_ratio": volume_ratio,
                "range_position": range_position,
                "squeeze": current_squeeze,
                "ichimoku_cloud": "above" if cloud_top and current_price > cloud_top else
                                  "below" if cloud_bottom and current_price < cloud_bottom else "inside"
            }
        )
