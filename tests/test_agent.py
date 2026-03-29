"""Tests for the AlphaWin agent and backtester."""
import pytest
from trading_agent.agent import AlphaWinAgent
from trading_agent.core.backtester import Backtester
from trading_agent.core.models import Candle


@pytest.fixture
def default_config():
    return {
        "agent": {"warmup_periods": 50, "initial_capital": 10000.0},
        "risk": {
            "max_position_pct": 0.25,
            "max_drawdown_pct": 0.10,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "max_open_positions": 3,
            "risk_per_trade_pct": 0.01,
            "trailing_stop_pct": 0.015,
        },
        "strategies": {
            "momentum": {"enabled": True, "weight": 0.35, "rsi_period": 14,
                         "rsi_oversold": 30, "rsi_overbought": 70,
                         "macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
            "mean_reversion": {"enabled": True, "weight": 0.30, "bb_period": 20,
                               "bb_std": 2.0, "rsi_period": 14, "lookback": 20},
            "breakout": {"enabled": True, "weight": 0.35, "lookback": 20,
                         "volume_multiplier": 1.5, "atr_period": 14,
                         "atr_multiplier": 1.5},
        },
    }


class TestAgent:
    def test_initialization(self, default_config):
        agent = AlphaWinAgent(config=default_config)
        assert len(agent.strategies) == 3

    def test_warmup_period(self, default_config):
        agent = AlphaWinAgent(config=default_config)
        candle = Candle(0, 100, 101, 99, 100, 1000)
        result = agent.on_candle(candle)
        assert result["action"] == "warmup"

    def test_processes_candles(self, default_config):
        agent = AlphaWinAgent(config=default_config)
        candles = Backtester.generate_synthetic_data(100)
        results = []
        for c in candles:
            results.append(agent.on_candle(c))
        # Should have processed all candles
        assert len(results) == 100
        # After warmup, should have real actions
        non_warmup = [r for r in results if r["action"] != "warmup"]
        assert len(non_warmup) > 0


class TestBacktester:
    def test_synthetic_data_generation(self):
        candles = Backtester.generate_synthetic_data(500)
        assert len(candles) == 500
        assert all(c.high >= c.low for c in candles)
        assert all(c.volume > 0 for c in candles)

    def test_backtest_runs(self, default_config):
        agent = AlphaWinAgent(config=default_config)
        candles = Backtester.generate_synthetic_data(300, trend="bull")
        results = agent.backtest(candles)
        assert "total_trades" in results
        assert "win_rate" in results
        assert "sharpe_ratio" in results
        assert results["total_candles"] == 300

    def test_backtest_all_markets(self, default_config):
        """Validate agent performs across all market conditions."""
        agent = AlphaWinAgent(config=default_config)
        for market in ["bull", "bear", "mixed", "sideways"]:
            candles = Backtester.generate_synthetic_data(500, trend=market)
            results = agent.backtest(candles)
            assert results["total_trades"] >= 0
            # Max drawdown should be controlled
            assert results["max_drawdown_pct"] < 0.5, \
                f"Drawdown too high in {market} market: {results['max_drawdown_pct']:.1%}"

    def test_risk_management_limits_losses(self, default_config):
        """Verify risk management prevents catastrophic losses."""
        agent = AlphaWinAgent(config=default_config)
        # Bear market stress test
        candles = Backtester.generate_synthetic_data(1000, trend="bear", volatility=0.03)
        results = agent.backtest(candles)
        # Should never lose more than 50% even in harsh conditions
        assert results["capital"] > 5000, \
            f"Capital dropped below safety threshold: ${results['capital']:.2f}"
