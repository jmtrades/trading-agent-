"""Tests for Market Scenario Generator and Backtester scenario training."""
import pytest
import numpy as np
from trading_agent.core.scenarios import MarketScenario
from trading_agent.core.models import Candle
from trading_agent.core.backtester import Backtester
from trading_agent.strategies.momentum import MomentumStrategy
from trading_agent.strategies.mean_reversion import MeanReversionStrategy
from trading_agent.strategies.breakout import BreakoutStrategy


class TestMarketScenario:

    def test_all_scenarios_listed(self):
        scenarios = MarketScenario.all_scenarios()
        assert len(scenarios) == 18
        assert "flash_crash" in scenarios
        assert "choppy_range" in scenarios
        assert "trend_reversal" in scenarios

    def test_invalid_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            MarketScenario.generate("fake_scenario")

    @pytest.mark.parametrize("scenario", MarketScenario.all_scenarios())
    def test_each_scenario_generates_candles(self, scenario):
        candles = MarketScenario.generate(scenario, n_candles=100)
        assert len(candles) == 100
        for c in candles:
            assert isinstance(c, Candle)
            assert c.high >= c.low
            assert c.high >= min(c.open, c.close)
            assert c.low <= max(c.open, c.close)
            assert c.volume > 0

    def test_flash_crash_has_sharp_drop(self):
        candles = MarketScenario.generate("flash_crash", n_candles=500)
        prices = [c.close for c in candles]
        # Find the worst single-bar drop
        returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
        min_return = min(returns)
        assert min_return < -0.01, "Flash crash should have at least -1% single bar drop"

    def test_parabolic_run_trends_up(self):
        candles = MarketScenario.generate("parabolic_run", n_candles=300)
        assert candles[-1].close > candles[0].close * 1.2, "Parabolic run should end much higher"

    def test_slow_bleed_trends_down(self):
        candles = MarketScenario.generate("slow_bleed", n_candles=300)
        assert candles[-1].close < candles[0].close * 0.9, "Slow bleed should end lower"

    def test_choppy_range_stays_bounded(self):
        candles = MarketScenario.generate("choppy_range", n_candles=300, base_price=100.0)
        prices = [c.close for c in candles]
        assert max(prices) < 115, "Choppy range should stay within bounds"
        assert min(prices) > 85, "Choppy range should stay within bounds"

    def test_scenario_reproducibility(self):
        """Same scenario with same seed should produce identical data."""
        c1 = MarketScenario.generate("flash_crash", n_candles=100)
        c2 = MarketScenario.generate("flash_crash", n_candles=100)
        for a, b in zip(c1, c2):
            assert a.close == b.close
            assert a.volume == b.volume


class TestScenarioBacktest:

    def _make_strategies(self):
        config = {
            "momentum": {"enabled": True},
            "mean_reversion": {"enabled": True},
            "breakout": {"enabled": True},
        }
        return [
            MomentumStrategy(config["momentum"]),
            MeanReversionStrategy(config["mean_reversion"]),
            BreakoutStrategy(config["breakout"]),
        ]

    def test_run_single_scenario(self):
        strategies = self._make_strategies()
        bt = Backtester({"risk": {}}, strategies, 10000.0)
        result = bt.run_scenario("trending_pullback", n_candles=300, warmup=50)
        assert "scenario" in result
        assert result["scenario"] == "trending_pullback"
        assert "total_trades" in result

    def test_train_all_scenarios(self):
        strategies = self._make_strategies()
        bt = Backtester({"risk": {}}, strategies, 10000.0)
        results = bt.train_all_scenarios(n_candles=200, warmup=50)
        assert "scenarios" in results
        assert "summary" in results
        assert results["summary"]["total_scenarios"] == 18
        assert len(results["scenarios"]) == 18

    def test_scenario_with_prop_firm(self):
        strategies = self._make_strategies()
        bt = Backtester({"risk": {}}, strategies, 100000.0,
                        prop_firm="ftmo", prop_phase="challenge")
        result = bt.run_scenario("trending_pullback", n_candles=300, warmup=50)
        assert "prop_firm" in result
        assert result["prop_firm"]["firm"] == "FTMO"
