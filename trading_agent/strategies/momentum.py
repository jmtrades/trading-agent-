"""Momentum Strategy v2 - Enhanced with multi-indicator confirmation.

Uses 7 indicators for signal generation:
- RSI for overbought/oversold
- MACD for momentum direction and acceleration
- ADX for trend strength
- EMA crossovers for timing
- Stochastic for entry precision
- OBV divergence for volume confirmation
- Rate of Change for momentum magnitude
"""
import numpy as np
from .base import BaseStrategy
from trading_agent.core.market_data import MarketData
from trading_agent.core.models import Signal, StrategySignal


class MomentumStrategy(BaseStrategy):

    def analyze(self, market_data: MarketData) -> StrategySignal:
        cfg = self.config
        close = market_data.close

        if market_data.size < 60:
            return StrategySignal(Signal.NEUTRAL, 0.0, self.name, "Insufficient data")

        # Core indicators
        rsi_vals = market_data.rsi(cfg.get("rsi_period", 14))
        macd_line, signal_line, histogram = market_data.macd(
            cfg.get("macd_fast", 12), cfg.get("macd_slow", 26), cfg.get("macd_signal", 9)
        )
        adx_vals = market_data.adx(14)
        ema_9 = market_data.ema(9)
        ema_21 = market_data.ema(21)

        # New indicators
        stoch_k, stoch_d = market_data.stochastic(14, 3)
        obv_vals = market_data.obv()
        roc_vals = market_data.roc(12)

        current_rsi = rsi_vals[-1]
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        current_hist = histogram[-1]
        prev_hist = histogram[-2] if len(histogram) > 1 else 0

        # Safe access for new indicators
        current_stoch_k = stoch_k[-1] if not np.isnan(stoch_k[-1]) else 50
        current_stoch_d = stoch_d[-1] if not np.isnan(stoch_d[-1]) else 50
        current_roc = roc_vals[-1] if not np.isnan(roc_vals[-1]) else 0

        valid_adx = adx_vals[~np.isnan(adx_vals)]
        trend_strength = valid_adx[-1] if len(valid_adx) > 0 else 0

        score = 0.0
        reasons = []

        # --- RSI Signals ---
        rsi_oversold = cfg.get("rsi_oversold", 30)
        rsi_overbought = cfg.get("rsi_overbought", 70)

        if current_rsi < rsi_oversold:
            score += 1.5
            reasons.append(f"RSI oversold ({current_rsi:.1f})")
        elif current_rsi < 40:
            score += 0.5
            reasons.append(f"RSI low ({current_rsi:.1f})")
        elif current_rsi > rsi_overbought:
            score -= 1.5
            reasons.append(f"RSI overbought ({current_rsi:.1f})")
        elif current_rsi > 60:
            score -= 0.5
            reasons.append(f"RSI high ({current_rsi:.1f})")

        # --- MACD Signals ---
        if not np.isnan(current_macd) and not np.isnan(current_signal):
            if current_macd > current_signal:
                score += 1.0
                reasons.append("MACD bullish")
            else:
                score -= 1.0
                reasons.append("MACD bearish")

            # Histogram acceleration
            if not np.isnan(current_hist) and not np.isnan(prev_hist):
                if current_hist > prev_hist and current_hist > 0:
                    score += 0.75
                    reasons.append("MACD accelerating up")
                elif current_hist < prev_hist and current_hist < 0:
                    score -= 0.75
                    reasons.append("MACD accelerating down")

        # --- EMA Crossover ---
        if not np.isnan(ema_9[-1]) and not np.isnan(ema_21[-1]):
            ema_spread = (ema_9[-1] - ema_21[-1]) / ema_21[-1]
            if ema_spread > 0.005:
                score += 1.0
                reasons.append(f"EMA bullish spread ({ema_spread:.3f})")
            elif ema_spread < -0.005:
                score -= 1.0
                reasons.append(f"EMA bearish spread ({ema_spread:.3f})")

            # Fresh crossover detection (within last 3 bars)
            if len(ema_9) >= 4 and len(ema_21) >= 4:
                for j in range(1, 4):
                    prev_9 = ema_9[-j - 1]
                    prev_21 = ema_21[-j - 1]
                    if not np.isnan(prev_9) and not np.isnan(prev_21):
                        if prev_9 <= prev_21 and ema_9[-1] > ema_21[-1]:
                            score += 0.5
                            reasons.append("Fresh EMA bullish cross")
                            break
                        elif prev_9 >= prev_21 and ema_9[-1] < ema_21[-1]:
                            score -= 0.5
                            reasons.append("Fresh EMA bearish cross")
                            break

        # --- Stochastic Confirmation ---
        if current_stoch_k < 20 and current_stoch_d < 20:
            score += 1.0
            reasons.append(f"Stoch oversold (K={current_stoch_k:.0f})")
        elif current_stoch_k > 80 and current_stoch_d > 80:
            score -= 1.0
            reasons.append(f"Stoch overbought (K={current_stoch_k:.0f})")

        # Stochastic crossover
        if current_stoch_k > current_stoch_d and current_stoch_k < 50:
            score += 0.5
            reasons.append("Stoch bullish cross from low")
        elif current_stoch_k < current_stoch_d and current_stoch_k > 50:
            score -= 0.5
            reasons.append("Stoch bearish cross from high")

        # --- OBV Divergence ---
        if len(obv_vals) >= 20 and len(close) >= 20:
            price_slope = (close[-1] - close[-20]) / close[-20]
            obv_slope = (obv_vals[-1] - obv_vals[-20])
            obv_normalized = obv_slope / abs(obv_vals[-20]) if obv_vals[-20] != 0 else 0

            # Bullish divergence: price flat/down but OBV rising
            if price_slope < 0 and obv_normalized > 0.05:
                score += 1.0
                reasons.append("OBV bullish divergence")
            # Bearish divergence: price flat/up but OBV falling
            elif price_slope > 0 and obv_normalized < -0.05:
                score -= 1.0
                reasons.append("OBV bearish divergence")

        # --- Rate of Change momentum ---
        if current_roc > 5:
            score += 0.5
            reasons.append(f"Strong upward momentum (ROC={current_roc:.1f})")
        elif current_roc < -5:
            score -= 0.5
            reasons.append(f"Strong downward momentum (ROC={current_roc:.1f})")

        # --- ADX trend filter ---
        if trend_strength > 25:
            score *= 1.3
            reasons.append(f"Strong trend (ADX={trend_strength:.1f})")
        elif trend_strength < 15:
            score *= 0.5
            reasons.append(f"Weak trend (ADX={trend_strength:.1f})")

        # Convert score to signal
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
            metadata={"score": score, "rsi": current_rsi, "adx": trend_strength,
                       "stoch_k": current_stoch_k, "roc": current_roc}
        )
