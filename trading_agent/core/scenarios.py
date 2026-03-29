"""Advanced Market Scenario Generator - Train on every price pattern.

Generates realistic price data for every market condition the agent
will face in real trading. Each scenario has specific characteristics
that test different aspects of the trading system.

Scenarios:
1.  FLASH_CRASH - Sudden -5% to -15% drop, tests stop loss and panic handling
2.  GAP_UP - Overnight gap above resistance, tests gap fill vs continuation
3.  GAP_DOWN - Overnight gap below support, tests gap strategies
4.  SQUEEZE - Short squeeze / gamma squeeze, parabolic move
5.  V_RECOVERY - Sharp drop followed by equally sharp recovery
6.  DEAD_CAT_BOUNCE - Drop, fake recovery, then continued drop
7.  ACCUMULATION - Wyckoff accumulation: range with increasing volume
8.  DISTRIBUTION - Wyckoff distribution: range with decreasing volume
9.  TRENDING_PULLBACK - Strong trend with healthy pullbacks (best for momentum)
10. CHOPPY_RANGE - Whipsawing range (the account killer)
11. PARABOLIC_RUN - Exponential price acceleration
12. SLOW_BLEED - Gradual, persistent downtrend
13. BREAKOUT_FAKEOUT - False breakouts that trap traders
14. NEWS_SPIKE - Sudden volatility spike both directions
15. LOW_LIQUIDITY - Thin market with large spreads and gaps
16. CONSOLIDATION_BREAKOUT - Tight range then explosive move
17. MEAN_REVERSION_SETUP - Extreme deviation then snap back to mean
18. TREND_REVERSAL - Mature trend that reverses
"""
import numpy as np
from .models import Candle


class MarketScenario:

    @staticmethod
    def generate(scenario: str, n_candles: int = 500,
                 base_price: float = 100.0,
                 base_volume: float = 1000.0) -> list[Candle]:
        """Generate candles for a specific market scenario."""
        generators = {
            "flash_crash": MarketScenario._flash_crash,
            "gap_up": MarketScenario._gap_up,
            "gap_down": MarketScenario._gap_down,
            "squeeze": MarketScenario._squeeze,
            "v_recovery": MarketScenario._v_recovery,
            "dead_cat_bounce": MarketScenario._dead_cat_bounce,
            "accumulation": MarketScenario._accumulation,
            "distribution": MarketScenario._distribution,
            "trending_pullback": MarketScenario._trending_pullback,
            "choppy_range": MarketScenario._choppy_range,
            "parabolic_run": MarketScenario._parabolic_run,
            "slow_bleed": MarketScenario._slow_bleed,
            "breakout_fakeout": MarketScenario._breakout_fakeout,
            "news_spike": MarketScenario._news_spike,
            "low_liquidity": MarketScenario._low_liquidity,
            "consolidation_breakout": MarketScenario._consolidation_breakout,
            "mean_reversion_setup": MarketScenario._mean_reversion_setup,
            "trend_reversal": MarketScenario._trend_reversal,
        }

        if scenario not in generators:
            raise ValueError(f"Unknown scenario: {scenario}. Available: {list(generators.keys())}")

        return generators[scenario](n_candles, base_price, base_volume)

    @staticmethod
    def all_scenarios() -> list[str]:
        return [
            "flash_crash", "gap_up", "gap_down", "squeeze", "v_recovery",
            "dead_cat_bounce", "accumulation", "distribution",
            "trending_pullback", "choppy_range", "parabolic_run",
            "slow_bleed", "breakout_fakeout", "news_spike",
            "low_liquidity", "consolidation_breakout",
            "mean_reversion_setup", "trend_reversal",
        ]

    @staticmethod
    def _make_candle(i, o, c, vol, base_vol, base_price):
        """Helper to create a candle with realistic OHLC."""
        h = max(o, c) * (1 + abs(np.random.exponential(0.003)))
        l = min(o, c) * (1 - abs(np.random.exponential(0.003)))
        l = max(l, base_price * 0.01)
        return Candle(
            timestamp=float(i * 3600),
            open=round(o, 2), high=round(h, 2),
            low=round(l, 2), close=round(c, 2),
            volume=round(vol, 2),
        )

    @staticmethod
    def _flash_crash(n, bp, bv):
        """Sudden -5% to -15% crash, then stabilization."""
        np.random.seed(100)
        candles = []
        price = bp
        crash_start = int(n * 0.4)
        crash_end = crash_start + max(3, int(n * 0.01))

        for i in range(n):
            vol = bv
            if i < crash_start:
                change = np.random.normal(0.0001, 0.008)
                vol *= (1 + np.random.uniform(-0.2, 0.2))
            elif i < crash_end:
                # Crash phase: sharp selling
                change = np.random.uniform(-0.03, -0.01)
                vol *= np.random.uniform(4, 8)
            elif i < crash_end + 20:
                # Post-crash volatility
                change = np.random.normal(0.001, 0.02)
                vol *= np.random.uniform(2, 4)
            else:
                change = np.random.normal(0.0002, 0.01)
                vol *= (1 + np.random.uniform(-0.3, 0.5))

            o = price
            price *= (1 + change)
            candles.append(MarketScenario._make_candle(i, o, price, vol, bv, bp))
        return candles

    @staticmethod
    def _gap_up(n, bp, bv):
        """Price gaps up 3-5% overnight then decides direction."""
        np.random.seed(101)
        candles = []
        price = bp
        gap_bar = int(n * 0.3)

        for i in range(n):
            vol = bv
            if i == gap_bar:
                change = np.random.uniform(0.03, 0.05)
                vol *= 3
            elif i > gap_bar and i < gap_bar + 10:
                # Post-gap: slight pullback (gap fill attempt)
                change = np.random.normal(-0.002, 0.012)
                vol *= 2
            elif i > gap_bar + 10:
                change = np.random.normal(0.0003, 0.01)
            else:
                change = np.random.normal(0.0001, 0.008)

            o = price
            price *= (1 + change)
            candles.append(MarketScenario._make_candle(i, o, price, vol, bv, bp))
        return candles

    @staticmethod
    def _gap_down(n, bp, bv):
        """Price gaps down 3-5% overnight."""
        np.random.seed(102)
        candles = []
        price = bp
        gap_bar = int(n * 0.3)

        for i in range(n):
            vol = bv
            if i == gap_bar:
                change = np.random.uniform(-0.05, -0.03)
                vol *= 4
            elif i > gap_bar and i < gap_bar + 15:
                change = np.random.normal(0.001, 0.015)
                vol *= 2
            else:
                change = np.random.normal(0.0001, 0.008)

            o = price
            price *= (1 + change)
            candles.append(MarketScenario._make_candle(i, o, price, vol, bv, bp))
        return candles

    @staticmethod
    def _squeeze(n, bp, bv):
        """Short squeeze: slow build then exponential acceleration."""
        np.random.seed(103)
        candles = []
        price = bp
        squeeze_start = int(n * 0.5)

        for i in range(n):
            vol = bv
            if i < squeeze_start:
                change = np.random.normal(0.0003, 0.008)
            elif i < squeeze_start + 30:
                # Squeeze acceleration
                progress = (i - squeeze_start) / 30
                drift = 0.005 * (1 + progress * 3)
                change = np.random.normal(drift, 0.01)
                vol *= (2 + progress * 5)
            else:
                # Post-squeeze selloff
                change = np.random.normal(-0.003, 0.02)
                vol *= np.random.uniform(2, 4)

            o = price
            price *= (1 + change)
            candles.append(MarketScenario._make_candle(i, o, price, vol, bv, bp))
        return candles

    @staticmethod
    def _v_recovery(n, bp, bv):
        """Sharp drop then equally sharp recovery."""
        np.random.seed(104)
        candles = []
        price = bp
        drop_start = int(n * 0.3)
        bottom = int(n * 0.45)
        recovery_end = int(n * 0.6)

        for i in range(n):
            vol = bv
            if i < drop_start:
                change = np.random.normal(0.0001, 0.008)
            elif i < bottom:
                progress = (i - drop_start) / (bottom - drop_start)
                change = np.random.normal(-0.005 - progress * 0.005, 0.008)
                vol *= (2 + progress * 3)
            elif i < recovery_end:
                progress = (i - bottom) / (recovery_end - bottom)
                change = np.random.normal(0.005 + progress * 0.005, 0.008)
                vol *= (3 - progress * 1.5)
            else:
                change = np.random.normal(0.0002, 0.01)

            o = price
            price *= (1 + change)
            candles.append(MarketScenario._make_candle(i, o, price, vol, bv, bp))
        return candles

    @staticmethod
    def _dead_cat_bounce(n, bp, bv):
        """Drop, fake recovery, then continued drop."""
        np.random.seed(105)
        candles = []
        price = bp
        first_drop = int(n * 0.25)
        bounce_start = int(n * 0.35)
        bounce_end = int(n * 0.50)
        second_drop = int(n * 0.50)

        for i in range(n):
            vol = bv
            if i < first_drop:
                change = np.random.normal(0.0001, 0.008)
            elif i < bounce_start:
                change = np.random.normal(-0.006, 0.008)
                vol *= 3
            elif i < bounce_end:
                change = np.random.normal(0.003, 0.01)
                vol *= 1.5
            elif i < second_drop + 30:
                change = np.random.normal(-0.005, 0.01)
                vol *= 2.5
            else:
                change = np.random.normal(-0.001, 0.012)

            o = price
            price *= (1 + change)
            candles.append(MarketScenario._make_candle(i, o, price, vol, bv, bp))
        return candles

    @staticmethod
    def _accumulation(n, bp, bv):
        """Wyckoff accumulation: range with increasing volume at lows."""
        np.random.seed(106)
        candles = []
        price = bp
        range_center = bp

        for i in range(n):
            # Mean-reverting within range
            reversion = -0.3 * (price - range_center) / range_center
            change = np.random.normal(reversion * 0.01, 0.008)

            # Volume increases on dips (accumulation)
            below_center = (range_center - price) / range_center
            vol = bv * (1 + max(0, below_center) * 5 + np.random.uniform(0, 0.5))

            # Gradual shift upward in later phases
            if i > n * 0.7:
                change += 0.001

            o = price
            price *= (1 + change)
            candles.append(MarketScenario._make_candle(i, o, price, vol, bv, bp))
        return candles

    @staticmethod
    def _distribution(n, bp, bv):
        """Wyckoff distribution: range with increasing volume at highs."""
        np.random.seed(107)
        candles = []
        price = bp
        range_center = bp

        for i in range(n):
            reversion = -0.3 * (price - range_center) / range_center
            change = np.random.normal(reversion * 0.01, 0.008)

            above_center = (price - range_center) / range_center
            vol = bv * (1 + max(0, above_center) * 5 + np.random.uniform(0, 0.5))

            if i > n * 0.7:
                change -= 0.001

            o = price
            price *= (1 + change)
            candles.append(MarketScenario._make_candle(i, o, price, vol, bv, bp))
        return candles

    @staticmethod
    def _trending_pullback(n, bp, bv):
        """Strong uptrend with healthy 2-3% pullbacks every 30-50 bars."""
        np.random.seed(108)
        candles = []
        price = bp

        for i in range(n):
            cycle = i % 40
            if cycle < 30:
                change = np.random.normal(0.003, 0.008)
                vol = bv * (1 + np.random.uniform(-0.2, 0.3))
            else:
                change = np.random.normal(-0.004, 0.01)
                vol = bv * np.random.uniform(1.5, 2.5)

            o = price
            price *= (1 + change)
            candles.append(MarketScenario._make_candle(i, o, price, vol, bv, bp))
        return candles

    @staticmethod
    def _choppy_range(n, bp, bv):
        """Whipsawing range that kills trend followers."""
        np.random.seed(109)
        candles = []
        price = bp

        for i in range(n):
            # Oscillate rapidly
            wave = np.sin(2 * np.pi * i / 15) * 0.005
            change = wave + np.random.normal(0, 0.012)
            vol = bv * (1 + np.random.uniform(-0.3, 0.5))

            o = price
            price *= (1 + change)
            # Clamp to range
            price = max(bp * 0.93, min(price, bp * 1.07))
            candles.append(MarketScenario._make_candle(i, o, price, vol, bv, bp))
        return candles

    @staticmethod
    def _parabolic_run(n, bp, bv):
        """Exponential price acceleration - FOMO territory."""
        np.random.seed(110)
        candles = []
        price = bp

        for i in range(n):
            progress = i / n
            drift = 0.001 + progress * 0.008
            change = np.random.normal(drift, 0.01)
            vol = bv * (1 + progress * 3)

            o = price
            price *= (1 + change)
            candles.append(MarketScenario._make_candle(i, o, price, vol, bv, bp))
        return candles

    @staticmethod
    def _slow_bleed(n, bp, bv):
        """Gradual persistent downtrend with occasional fake bounces."""
        np.random.seed(111)
        candles = []
        price = bp

        for i in range(n):
            base_drift = -0.001
            # Occasional fake bounces
            if i % 60 >= 50:
                change = np.random.normal(0.003, 0.008)
            else:
                change = np.random.normal(base_drift, 0.007)

            vol = bv * (1 + np.random.uniform(-0.2, 0.3))

            o = price
            price *= (1 + change)
            candles.append(MarketScenario._make_candle(i, o, price, vol, bv, bp))
        return candles

    @staticmethod
    def _breakout_fakeout(n, bp, bv):
        """Repeated false breakouts that trap traders."""
        np.random.seed(112)
        candles = []
        price = bp
        range_high = bp * 1.03
        range_low = bp * 0.97

        for i in range(n):
            cycle = i % 80
            if cycle < 60:
                # Range
                reversion = -0.5 * (price - bp) / bp
                change = np.random.normal(reversion * 0.01, 0.008)
                vol = bv
            elif cycle < 65:
                # Fakeout breakout above
                change = np.random.normal(0.005, 0.005)
                vol = bv * 2.5
            elif cycle < 75:
                # Reversal back into range
                change = np.random.normal(-0.006, 0.008)
                vol = bv * 2
            else:
                change = np.random.normal(0, 0.01)
                vol = bv

            o = price
            price *= (1 + change)
            candles.append(MarketScenario._make_candle(i, o, price, vol, bv, bp))
        return candles

    @staticmethod
    def _news_spike(n, bp, bv):
        """Random high-vol spikes simulating news events."""
        np.random.seed(113)
        candles = []
        price = bp
        news_bars = sorted(np.random.choice(range(50, n - 20), size=5, replace=False))

        for i in range(n):
            is_news = any(abs(i - nb) < 3 for nb in news_bars)
            if is_news:
                change = np.random.normal(0, 0.03)
                vol = bv * np.random.uniform(5, 10)
            else:
                change = np.random.normal(0, 0.008)
                vol = bv * (1 + np.random.uniform(-0.2, 0.3))

            o = price
            price *= (1 + change)
            candles.append(MarketScenario._make_candle(i, o, price, vol, bv, bp))
        return candles

    @staticmethod
    def _low_liquidity(n, bp, bv):
        """Thin market: large random moves, wide spreads."""
        np.random.seed(114)
        candles = []
        price = bp

        for i in range(n):
            # Fat-tailed moves
            change = np.random.standard_t(3) * 0.015
            vol = bv * np.random.uniform(0.1, 0.5)

            o = price
            h = max(o, o * (1 + change)) * (1 + np.random.exponential(0.01))
            l = min(o, o * (1 + change)) * (1 - np.random.exponential(0.01))
            l = max(l, 0.01)
            price *= (1 + change)

            candles.append(Candle(
                timestamp=float(i * 3600),
                open=round(o, 2), high=round(h, 2),
                low=round(l, 2), close=round(price, 2),
                volume=round(vol, 2),
            ))
        return candles

    @staticmethod
    def _consolidation_breakout(n, bp, bv):
        """Tight range that compresses then explodes."""
        np.random.seed(115)
        candles = []
        price = bp
        breakout_bar = int(n * 0.6)

        for i in range(n):
            if i < breakout_bar:
                # Narrowing range
                compression = 1 - (i / breakout_bar) * 0.7
                change = np.random.normal(0, 0.008 * compression)
                vol = bv * (0.5 + 0.5 * (1 - compression))
            elif i < breakout_bar + 5:
                # Explosive breakout
                change = np.random.normal(0.01, 0.008)
                vol = bv * np.random.uniform(4, 8)
            else:
                change = np.random.normal(0.003, 0.012)
                vol = bv * np.random.uniform(1, 3)

            o = price
            price *= (1 + change)
            candles.append(MarketScenario._make_candle(i, o, price, vol, bv, bp))
        return candles

    @staticmethod
    def _mean_reversion_setup(n, bp, bv):
        """Extreme deviation from mean then snap back."""
        np.random.seed(116)
        candles = []
        price = bp
        deviation_start = int(n * 0.3)
        reversion_start = int(n * 0.5)

        for i in range(n):
            if i < deviation_start:
                change = np.random.normal(0, 0.008)
                vol = bv
            elif i < reversion_start:
                # Extreme move away from mean
                change = np.random.normal(0.005, 0.01)
                vol = bv * 2
            elif i < reversion_start + 20:
                # Snap back to mean
                dist_from_bp = (price - bp) / bp
                change = -dist_from_bp * 0.1 + np.random.normal(0, 0.01)
                vol = bv * 3
            else:
                change = np.random.normal(0, 0.008)
                vol = bv

            o = price
            price *= (1 + change)
            candles.append(MarketScenario._make_candle(i, o, price, vol, bv, bp))
        return candles

    @staticmethod
    def _trend_reversal(n, bp, bv):
        """Mature uptrend that tops out and reverses."""
        np.random.seed(117)
        candles = []
        price = bp
        top = int(n * 0.45)
        reversal_confirmed = int(n * 0.55)

        for i in range(n):
            if i < top:
                # Uptrend with decelerating momentum
                progress = i / top
                drift = 0.004 * (1 - progress * 0.8)
                change = np.random.normal(drift, 0.008)
                vol = bv * (1 - progress * 0.3)
            elif i < reversal_confirmed:
                # Topping: choppy, high volume
                change = np.random.normal(-0.001, 0.015)
                vol = bv * np.random.uniform(2, 3)
            else:
                # Downtrend
                change = np.random.normal(-0.003, 0.01)
                vol = bv * np.random.uniform(1, 2)

            o = price
            price *= (1 + change)
            candles.append(MarketScenario._make_candle(i, o, price, vol, bv, bp))
        return candles
