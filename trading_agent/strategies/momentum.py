"""Momentum Strategy - Rides trends using RSI, MACD, and ADX confirmation.

This strategy identifies strong trends early and rides them. It combines:
- RSI for overbought/oversold detection
- MACD for trend direction and momentum shifts
- ADX for trend strength confirmation
- EMA crossovers for entry timing

Win edge: Only enters when multiple momentum indicators align, reducing
false signals. ADX filter ensures we only trade in trending markets.
"""
import numpy as np
from .base import BaseStrategy
from trading_agent.core.market_data import MarketData
from trading_agent.core.models import Signal, StrategySignal


class MomentumStrategy(BaseStrategy):

    def analyze(self, market_data: MarketData) -> StrategySignal:
        cfg = self.config
        close = market_data.close

        if market_data.size < 50:
            return StrategySignal(Signal.NEUTRAL, 0.0, self.name, "Insufficient data")

        # Core indicators
        rsi_vals = market_data.rsi(cfg.get("rsi_period", 14))
        macd_line, signal_line, histogram = market_data.macd(
            cfg.get("macd_fast", 12), cfg.get("macd_slow", 26), cfg.get("macd_signal", 9)
        )
        adx_vals = market_data.adx(14)
        ema_9 = market_data.ema(9)
        ema_21 = market_data.ema(21)

        current_rsi = rsi_vals[-1]
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        current_hist = histogram[-1]
        prev_hist = histogram[-2] if len(histogram) > 1 else 0

        # ADX trend strength (use last valid value)
        valid_adx = adx_vals[~np.isnan(adx_vals)]
        trend_strength = valid_adx[-1] if len(valid_adx) > 0 else 0

        # Scoring system: accumulate evidence
        score = 0.0
        reasons = []

        # RSI signals
        rsi_oversold = cfg.get("rsi_oversold", 30)
        rsi_overbought = cfg.get("rsi_overbought", 70)

        if current_rsi < rsi_oversold:
            score += 1.5
            reasons.append(f"RSI oversold ({current_rsi:.1f})")
        elif current_rsi < 40:
            score += 0.5
            reasons.append(f"RSI approaching oversold ({current_rsi:.1f})")
        elif current_rsi > rsi_overbought:
            score -= 1.5
            reasons.append(f"RSI overbought ({current_rsi:.1f})")
        elif current_rsi > 60:
            score -= 0.5
            reasons.append(f"RSI approaching overbought ({current_rsi:.1f})")

        # MACD signals
        if not np.isnan(current_macd) and not np.isnan(current_signal):
            if current_macd > current_signal:
                score += 1.0
                reasons.append("MACD bullish crossover")
            else:
                score -= 1.0
                reasons.append("MACD bearish crossover")

            # Histogram momentum (acceleration)
            if not np.isnan(current_hist) and not np.isnan(prev_hist):
                if current_hist > prev_hist and current_hist > 0:
                    score += 0.75
                    reasons.append("MACD histogram accelerating bullish")
                elif current_hist < prev_hist and current_hist < 0:
                    score -= 0.75
                    reasons.append("MACD histogram accelerating bearish")

        # EMA crossover
        if not np.isnan(ema_9[-1]) and not np.isnan(ema_21[-1]):
            if ema_9[-1] > ema_21[-1]:
                score += 1.0
                reasons.append("EMA 9 > EMA 21 (bullish)")
            else:
                score -= 1.0
                reasons.append("EMA 9 < EMA 21 (bearish)")

        # ADX trend filter - boost signal in strong trends
        if trend_strength > 25:
            score *= 1.3
            reasons.append(f"Strong trend (ADX={trend_strength:.1f})")
        elif trend_strength < 15:
            score *= 0.5
            reasons.append(f"Weak trend (ADX={trend_strength:.1f})")

        # Convert score to signal
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
            metadata={"score": score, "rsi": current_rsi, "adx": trend_strength}
        )
