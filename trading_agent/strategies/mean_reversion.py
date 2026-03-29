"""Mean Reversion Strategy - Profits from price returning to the mean.

This strategy identifies overextended prices and bets on reversion:
- Bollinger Bands for deviation from mean
- RSI for confirmation of extreme conditions
- Z-score for statistical significance of deviation
- Volume analysis for exhaustion signals

Win edge: Combines statistical measures with momentum exhaustion to only
enter when mean reversion is most likely. Avoids fighting strong trends
by checking trend context before entering.
"""
import numpy as np
from .base import BaseStrategy
from trading_agent.core.market_data import MarketData
from trading_agent.core.models import Signal, StrategySignal


class MeanReversionStrategy(BaseStrategy):

    def analyze(self, market_data: MarketData) -> StrategySignal:
        cfg = self.config
        close = market_data.close

        if market_data.size < 50:
            return StrategySignal(Signal.NEUTRAL, 0.0, self.name, "Insufficient data")

        # Core indicators
        bb_period = cfg.get("bb_period", 20)
        bb_std = cfg.get("bb_std", 2.0)
        upper, middle, lower = market_data.bollinger_bands(bb_period, bb_std)
        rsi_vals = market_data.rsi(cfg.get("rsi_period", 14))
        lookback = cfg.get("lookback", 20)

        current_price = close[-1]
        current_rsi = rsi_vals[-1]
        current_upper = upper[-1]
        current_lower = lower[-1]
        current_middle = middle[-1]

        if np.isnan(current_upper) or np.isnan(current_rsi):
            return StrategySignal(Signal.NEUTRAL, 0.0, self.name, "Indicators not ready")

        # Z-score: how many standard deviations from mean
        recent = close[-lookback:]
        mean = np.mean(recent)
        std = np.std(recent, ddof=1)
        z_score = (current_price - mean) / std if std > 0 else 0

        # Bandwidth: how wide are the bands (volatility context)
        bandwidth = (current_upper - current_lower) / current_middle if current_middle > 0 else 0

        # Volume analysis: look for exhaustion (declining volume at extremes)
        volume = market_data.volume
        vol_sma = np.mean(volume[-lookback:])
        recent_vol = volume[-3:]
        vol_declining = all(recent_vol[i] <= recent_vol[i - 1] for i in range(1, len(recent_vol)))

        # Scoring
        score = 0.0
        reasons = []

        # Bollinger Band position
        bb_position = (current_price - current_lower) / (current_upper - current_lower) \
            if (current_upper - current_lower) > 0 else 0.5

        if bb_position < 0.05:
            score += 2.0
            reasons.append("Price at lower BB (extreme)")
        elif bb_position < 0.2:
            score += 1.0
            reasons.append("Price near lower BB")
        elif bb_position > 0.95:
            score -= 2.0
            reasons.append("Price at upper BB (extreme)")
        elif bb_position > 0.8:
            score -= 1.0
            reasons.append("Price near upper BB")

        # Z-score confirmation
        if z_score < -2.0:
            score += 1.5
            reasons.append(f"Z-score extreme low ({z_score:.2f})")
        elif z_score < -1.0:
            score += 0.75
            reasons.append(f"Z-score low ({z_score:.2f})")
        elif z_score > 2.0:
            score -= 1.5
            reasons.append(f"Z-score extreme high ({z_score:.2f})")
        elif z_score > 1.0:
            score -= 0.75
            reasons.append(f"Z-score high ({z_score:.2f})")

        # RSI confirmation
        if current_rsi < 25:
            score += 1.5
            reasons.append(f"RSI deeply oversold ({current_rsi:.1f})")
        elif current_rsi < 35:
            score += 0.5
            reasons.append(f"RSI oversold ({current_rsi:.1f})")
        elif current_rsi > 75:
            score -= 1.5
            reasons.append(f"RSI deeply overbought ({current_rsi:.1f})")
        elif current_rsi > 65:
            score -= 0.5
            reasons.append(f"RSI overbought ({current_rsi:.1f})")

        # Volume exhaustion boost
        if vol_declining and abs(score) > 1.0:
            score *= 1.2
            reasons.append("Volume declining (exhaustion)")

        # Bandwidth filter: wide bands = high volatility = higher confidence
        if bandwidth > 0.1:
            score *= 1.1
            reasons.append("High volatility environment")
        elif bandwidth < 0.03:
            score *= 0.6
            reasons.append("Very low volatility (squeeze)")

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
            metadata={"z_score": z_score, "bb_position": bb_position, "bandwidth": bandwidth}
        )
