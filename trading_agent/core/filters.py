"""Trade Filters - Multiple layers of confirmation before any trade.

Every trade must pass ALL filters. This is how we maximize win rate:
reject marginal setups ruthlessly, only take A+ trades.
"""
import numpy as np
from .market_data import MarketData
from .models import OrderSide, Signal, StrategySignal


class MultiTimeframeFilter:
    """Simulates higher timeframe by aggregating candles.

    If the 1h chart says BUY but the 4h chart says SELL, don't trade.
    Higher timeframe alignment dramatically improves win rate.
    """

    def __init__(self, multiplier: int = 4):
        self.multiplier = multiplier  # e.g., 4 = use 4x candles as "higher TF"

    def check_alignment(self, market_data: MarketData, side: OrderSide) -> tuple[bool, str]:
        """Check if the higher timeframe trend agrees with the trade direction."""
        close = market_data.close
        if len(close) < self.multiplier * 50:
            return True, "Insufficient data for MTF (allowing)"

        # Build higher timeframe by sampling every Nth candle
        htf_close = close[::self.multiplier]

        if len(htf_close) < 30:
            return True, "HTF data too short (allowing)"

        # Higher TF trend: SMA 20 slope
        sma_period = 20
        if len(htf_close) < sma_period:
            return True, "HTF SMA not ready"

        sma_vals = np.convolve(htf_close, np.ones(sma_period) / sma_period, mode='valid')
        if len(sma_vals) < 5:
            return True, "HTF SMA too short"

        htf_slope = (sma_vals[-1] - sma_vals[-5]) / sma_vals[-5]

        # Higher TF momentum: price vs SMA
        htf_price_vs_sma = (htf_close[-1] - sma_vals[-1]) / sma_vals[-1]

        if side == OrderSide.BUY:
            if htf_slope > -0.005 and htf_price_vs_sma > -0.02:
                return True, f"HTF aligned bullish (slope={htf_slope:.4f})"
            return False, f"HTF bearish - blocking buy (slope={htf_slope:.4f})"
        else:
            if htf_slope < 0.005 and htf_price_vs_sma < 0.02:
                return True, f"HTF aligned bearish (slope={htf_slope:.4f})"
            return False, f"HTF bullish - blocking sell (slope={htf_slope:.4f})"


class ConsensusFilter:
    """Requires minimum number of strategies to agree.

    A single strategy screaming BUY while others are neutral = skip.
    We need consensus.
    """

    def __init__(self, min_agreement: int = 2):
        self.min_agreement = min_agreement

    def check(self, signals: list[StrategySignal], side: OrderSide) -> tuple[bool, str]:
        """Check if enough strategies agree on the direction.

        Adapts to the number of relevant strategies:
        - 3+ strategies: need min_agreement agreeing, no strong opposition
        - 2 strategies: need at least 1 agreeing, no opposition
        - 1 strategy: always passes (pre-filtered by regime)
        """
        if side == OrderSide.BUY:
            agreeing = sum(1 for s in signals if s.signal.value > 0)
            opposing = sum(1 for s in signals if s.signal.value < 0)
            strong_opposing = sum(1 for s in signals if s.signal.value <= -2)
        else:
            agreeing = sum(1 for s in signals if s.signal.value < 0)
            opposing = sum(1 for s in signals if s.signal.value > 0)
            strong_opposing = sum(1 for s in signals if s.signal.value >= 2)

        n = len(signals)

        # No strong opposition ever allowed
        if strong_opposing > 0:
            return False, f"Strong opposition: {strong_opposing} strongly oppose"

        # Adaptive agreement threshold
        required = min(self.min_agreement, max(1, n - 1))

        if agreeing >= required and opposing == 0:
            return True, f"{agreeing}/{n} strategies agree, 0 opposing"
        elif opposing > 0:
            return False, f"Conflicting: {agreeing} agree, {opposing} oppose"
        else:
            return False, f"Only {agreeing}/{required} required strategies agree"


class TradeQualityScorer:
    """Scores trade quality on multiple dimensions. Only A+ trades pass.

    Dimensions:
    1. Signal strength (combined score magnitude)
    2. Confidence alignment (are strategies confident?)
    3. Volume confirmation
    4. Volatility environment
    5. Recent price action support
    """

    def __init__(self, min_quality: float = 0.6):
        self.min_quality = min_quality

    def score(self, signals: list[StrategySignal], market_data: MarketData,
              side: OrderSide) -> tuple[float, bool, str]:
        """Returns (quality_score, passes, reason). Score is 0-1."""
        scores = {}

        # 1. Signal strength (0-1)
        avg_confidence = np.mean([s.confidence for s in signals if s.signal.value != 0])
        scores["confidence"] = avg_confidence if not np.isnan(avg_confidence) else 0

        # 2. Signal agreement strength
        if side == OrderSide.BUY:
            direction_scores = [s.signal.value * s.confidence for s in signals if s.signal.value > 0]
        else:
            direction_scores = [-s.signal.value * s.confidence for s in signals if s.signal.value < 0]
        scores["agreement"] = np.mean(direction_scores) / 2.0 if direction_scores else 0

        # 3. Volume confirmation
        volume = market_data.volume
        if len(volume) >= 20:
            vol_ratio = volume[-1] / np.mean(volume[-20:])
            scores["volume"] = min(vol_ratio / 2.0, 1.0)
        else:
            scores["volume"] = 0.5

        # 4. Price action support (candle structure)
        close = market_data.close
        if len(close) >= 3:
            recent_returns = np.diff(close[-4:]) / close[-4:-1]
            if side == OrderSide.BUY:
                # Recent candles should show buying pressure
                bullish_candles = sum(1 for r in recent_returns if r > 0)
                scores["price_action"] = bullish_candles / len(recent_returns)
            else:
                bearish_candles = sum(1 for r in recent_returns if r < 0)
                scores["price_action"] = bearish_candles / len(recent_returns)
        else:
            scores["price_action"] = 0.5

        # 5. Volatility suitability (moderate is best)
        atr_vals = market_data.atr(14)
        valid_atr = atr_vals[~np.isnan(atr_vals)]
        if len(valid_atr) >= 10:
            atr_ratio = valid_atr[-1] / np.mean(valid_atr[-10:])
            # Optimal around 1.0-1.5, penalize extremes
            if 0.8 <= atr_ratio <= 1.8:
                scores["volatility"] = 1.0
            elif atr_ratio > 2.5:
                scores["volatility"] = 0.2  # Too volatile
            else:
                scores["volatility"] = 0.6
        else:
            scores["volatility"] = 0.5

        # Weighted composite score
        weights = {
            "confidence": 0.30,
            "agreement": 0.25,
            "volume": 0.15,
            "price_action": 0.15,
            "volatility": 0.15,
        }
        quality = sum(scores[k] * weights[k] for k in weights)

        details = " | ".join(f"{k}={v:.2f}" for k, v in scores.items())
        passes = bool(quality >= self.min_quality)

        return quality, passes, f"Quality={quality:.2f} ({details})"


class SignalPersistenceFilter:
    """Only trade when the signal has been consistent for N bars.

    A real setup builds over time. If a buy signal only appears for one bar
    and flips the next, it's noise. Genuine signals persist.
    """

    def __init__(self, required_bars: int = 2):
        self.required_bars = required_bars
        self.signal_history: list[float] = []
        self.max_history = 10

    def record_score(self, score: float):
        self.signal_history.append(score)
        if len(self.signal_history) > self.max_history:
            self.signal_history = self.signal_history[-self.max_history:]

    def check(self, side: OrderSide) -> tuple[bool, str]:
        """Check if the signal direction has been consistent."""
        if len(self.signal_history) < self.required_bars:
            return True, "Not enough history (allowing)"

        recent = self.signal_history[-self.required_bars:]
        if side == OrderSide.BUY:
            consistent = all(s > 0 for s in recent)
            direction = "bullish"
        else:
            consistent = all(s < 0 for s in recent)
            direction = "bearish"

        if consistent:
            return True, f"Signal {direction} for {self.required_bars} consecutive bars"
        return False, f"Signal not consistently {direction} across last {self.required_bars} bars"


class CooldownFilter:
    """Prevents overtrading after losses.

    After a losing trade, wait N candles before trading again.
    Prevents revenge trading and emotional decisions.
    """

    def __init__(self, cooldown_bars: int = 5, loss_cooldown_bars: int = 10):
        self.cooldown_bars = cooldown_bars
        self.loss_cooldown_bars = loss_cooldown_bars
        self.last_trade_bar = -100
        self.last_loss_bar = -100

    def record_trade(self, bar: int, was_loss: bool = False):
        self.last_trade_bar = bar
        if was_loss:
            self.last_loss_bar = bar

    def can_trade(self, current_bar: int) -> tuple[bool, str]:
        """Check if cooldown period has elapsed."""
        bars_since_trade = current_bar - self.last_trade_bar
        bars_since_loss = current_bar - self.last_loss_bar

        if bars_since_loss < self.loss_cooldown_bars:
            return False, f"Loss cooldown: {self.loss_cooldown_bars - bars_since_loss} bars remaining"
        if bars_since_trade < self.cooldown_bars:
            return False, f"Trade cooldown: {self.cooldown_bars - bars_since_trade} bars remaining"
        return True, "Cooldown clear"
