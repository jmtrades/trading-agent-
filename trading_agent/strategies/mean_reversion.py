"""Mean Reversion Strategy v2 - Enhanced with multi-indicator confirmation.

Uses 7 indicators for signal generation:
- Bollinger Bands for deviation from mean
- Z-score for statistical significance
- RSI for extreme conditions
- Volume analysis for exhaustion
- Stochastic for timing
- MFI for institutional money flow
- Keltner Channels for volatility context (squeeze detection)
"""
import numpy as np
from .base import BaseStrategy
from trading_agent.core.market_data import MarketData
from trading_agent.core.models import Signal, StrategySignal


class MeanReversionStrategy(BaseStrategy):

    def analyze(self, market_data: MarketData) -> StrategySignal:
        cfg = self.config
        close = market_data.close

        if market_data.size < 60:
            return StrategySignal(Signal.NEUTRAL, 0.0, self.name, "Insufficient data")

        # Core indicators
        bb_period = cfg.get("bb_period", 20)
        bb_std = cfg.get("bb_std", 2.0)
        upper, middle, lower = market_data.bollinger_bands(bb_period, bb_std)
        rsi_vals = market_data.rsi(cfg.get("rsi_period", 14))
        lookback = cfg.get("lookback", 20)

        # New indicators
        stoch_k, stoch_d = market_data.stochastic(14, 3)
        mfi_vals = market_data.mfi(14)
        squeeze_on, squeeze_mom = market_data.squeeze()

        current_price = close[-1]
        current_rsi = rsi_vals[-1]
        current_upper = upper[-1]
        current_lower = lower[-1]
        current_middle = middle[-1]

        if np.isnan(current_upper) or np.isnan(current_rsi):
            return StrategySignal(Signal.NEUTRAL, 0.0, self.name, "Indicators not ready")

        # Z-score
        recent = close[-lookback:]
        mean = np.mean(recent)
        std = np.std(recent, ddof=1)
        z_score = (current_price - mean) / std if std > 0 else 0

        # Bandwidth
        bandwidth = (current_upper - current_lower) / current_middle if current_middle > 0 else 0

        # BB position
        bb_range = current_upper - current_lower
        bb_position = (current_price - current_lower) / bb_range if bb_range > 0 else 0.5

        # Volume analysis
        volume = market_data.volume
        vol_sma = np.mean(volume[-lookback:])
        recent_vol = volume[-3:]
        vol_declining = all(recent_vol[i] <= recent_vol[i - 1] for i in range(1, len(recent_vol)))

        # Safe new indicator access
        current_stoch_k = stoch_k[-1] if not np.isnan(stoch_k[-1]) else 50
        current_mfi = mfi_vals[-1] if not np.isnan(mfi_vals[-1]) else 50
        current_squeeze = squeeze_on[-1] if len(squeeze_on) > 0 else 0

        score = 0.0
        reasons = []

        # --- Bollinger Band Position ---
        if bb_position < 0.05:
            score += 2.0
            reasons.append("Price at lower BB extreme")
        elif bb_position < 0.15:
            score += 1.0
            reasons.append("Price near lower BB")
        elif bb_position > 0.95:
            score -= 2.0
            reasons.append("Price at upper BB extreme")
        elif bb_position > 0.85:
            score -= 1.0
            reasons.append("Price near upper BB")

        # --- Z-score ---
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

        # --- RSI confirmation ---
        if current_rsi < 25:
            score += 1.5
            reasons.append(f"RSI deeply oversold ({current_rsi:.1f})")
        elif current_rsi < 35:
            score += 0.5
        elif current_rsi > 75:
            score -= 1.5
            reasons.append(f"RSI deeply overbought ({current_rsi:.1f})")
        elif current_rsi > 65:
            score -= 0.5

        # --- Stochastic confirmation ---
        if current_stoch_k < 20:
            score += 0.75
            reasons.append(f"Stoch oversold ({current_stoch_k:.0f})")
        elif current_stoch_k > 80:
            score -= 0.75
            reasons.append(f"Stoch overbought ({current_stoch_k:.0f})")

        # --- MFI (institutional money flow) ---
        if current_mfi < 20:
            score += 1.0
            reasons.append(f"MFI oversold ({current_mfi:.1f}) - institutional buying")
        elif current_mfi > 80:
            score -= 1.0
            reasons.append(f"MFI overbought ({current_mfi:.1f}) - institutional selling")

        # --- Volume exhaustion ---
        if vol_declining and abs(score) > 1.0:
            score *= 1.2
            reasons.append("Volume declining (exhaustion)")

        # --- Squeeze detection (BB inside Keltner) ---
        if current_squeeze > 0:
            # During squeeze, mean reversion is stronger
            score *= 1.3
            reasons.append("Squeeze active - mean reversion favorable")

        # --- Bandwidth filter ---
        if bandwidth > 0.1:
            score *= 1.1
        elif bandwidth < 0.03:
            score *= 0.6
            reasons.append("Very low bandwidth")

        # --- Reversal candle detection ---
        if len(close) >= 3:
            prev_body = close[-2] - market_data.open[-2]
            curr_body = close[-1] - market_data.open[-1]
            # Bullish reversal: previous bearish, current bullish with larger body
            if prev_body < 0 and curr_body > 0 and abs(curr_body) > abs(prev_body) * 0.5:
                if score > 0:
                    score += 0.75
                    reasons.append("Bullish reversal candle")
            elif prev_body > 0 and curr_body < 0 and abs(curr_body) > abs(prev_body) * 0.5:
                if score < 0:
                    score -= 0.75
                    reasons.append("Bearish reversal candle")

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
            metadata={"z_score": z_score, "bb_position": bb_position,
                       "bandwidth": bandwidth, "mfi": current_mfi}
        )
