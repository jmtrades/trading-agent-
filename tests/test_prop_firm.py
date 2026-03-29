"""Tests for PropFirmCompliance engine."""
import pytest
from trading_agent.core.prop_firm import (
    PropFirmCompliance, PropFirmPhase, ComplianceStatus,
    PROP_FIRM_PROFILES,
)


class TestPropFirmProfiles:

    def test_all_firms_have_profiles(self):
        firms = PropFirmCompliance.list_firms()
        assert len(firms) == 6
        assert "ftmo" in firms
        assert "topstep" in firms
        assert "apex" in firms

    def test_ftmo_challenge_rules(self):
        pf = PropFirmCompliance("ftmo", "challenge", 100000)
        assert pf.rules["max_daily_loss_pct"] == 0.05
        assert pf.rules["max_total_drawdown_pct"] == 0.10
        assert pf.rules["profit_target_pct"] == 0.10

    def test_invalid_firm_raises(self):
        with pytest.raises(ValueError, match="Unknown firm"):
            PropFirmCompliance("fake_firm")

    def test_invalid_phase_raises(self):
        with pytest.raises(ValueError, match="Unknown phase"):
            PropFirmCompliance("ftmo", "fake_phase")


class TestPreTradeChecks:

    def test_trade_approved_normal(self):
        pf = PropFirmCompliance("ftmo", "challenge", 100000)
        allowed, reason = pf.check_pre_trade(500)
        assert allowed
        assert reason == "Trade approved"

    def test_trade_blocked_after_blown(self):
        pf = PropFirmCompliance("ftmo", "challenge", 100000)
        pf.state.is_blown = True
        allowed, reason = pf.check_pre_trade(100)
        assert not allowed
        assert "BLOWN" in reason

    def test_daily_loss_limit_blocks(self):
        pf = PropFirmCompliance("ftmo", "challenge", 100000)
        pf.start_new_day()
        # Lose $2000 — still have $3000 of daily limit remaining
        pf.record_trade_result(-2000)
        allowed, reason = pf.check_pre_trade(100)
        assert allowed  # Still room

        # Lose more to use up the daily limit (total -5000)
        pf.record_trade_result(-3000)
        allowed, reason = pf.check_pre_trade(100)
        assert not allowed

    def test_total_drawdown_blows_account(self):
        pf = PropFirmCompliance("ftmo", "challenge", 100000)
        pf.start_new_day()
        # Lose 10% total (max dd for FTMO)
        pf.record_trade_result(-10000)
        assert pf.state.is_blown
        assert pf.state.status == ComplianceStatus.BLOWN

    def test_risk_too_large_for_dd_buffer(self):
        # Use E8 which has 8% total DD but 5% daily — lose across multiple days
        pf = PropFirmCompliance("e8_funding", "challenge", 100000)
        pf.start_new_day()
        pf.record_trade_result(-3000)  # Day 1: -3k
        pf.start_new_day()
        pf.record_trade_result(-3000)  # Day 2: -3k total = 6k, dd limit = 8k, remaining = 2k
        # Propose risk > 50% of remaining dd buffer (1000)
        allowed, reason = pf.check_pre_trade(1500)
        assert not allowed
        assert "too large" in reason


class TestTradeRecording:

    def test_record_winning_trade(self):
        pf = PropFirmCompliance("ftmo", "challenge", 100000)
        pf.start_new_day()
        pf.record_trade_result(500)
        assert pf.state.total_pnl == 500
        assert pf.state.current_equity == 100500
        assert pf.state.current_day.trades == 1

    def test_record_losing_trade(self):
        pf = PropFirmCompliance("ftmo", "challenge", 100000)
        pf.start_new_day()
        pf.record_trade_result(-300)
        assert pf.state.total_pnl == -300
        assert pf.state.current_equity == 99700

    def test_challenge_passed_on_profit_target(self):
        pf = PropFirmCompliance("ftmo", "challenge", 100000)
        # Need 4 trading days minimum
        for day in range(5):
            pf.start_new_day()
            pf.record_trade_result(2500)

        # Should have passed: 12500 profit > 10000 target, 5 trading days >= 4 min
        assert pf.state.challenge_passed


class TestConsistencyRule:

    def test_no_consistency_rule(self):
        pf = PropFirmCompliance("ftmo", "challenge", 100000)  # FTMO has no consistency
        ok, msg = pf.check_consistency()
        assert ok
        assert "No consistency rule" in msg

    def test_consistency_violation(self):
        pf = PropFirmCompliance("myfundedfx", "challenge", 100000)  # Has consistency
        # Day 1: big win
        pf.start_new_day()
        pf.record_trade_result(5000)
        # Day 2: small win
        pf.start_new_day()
        pf.record_trade_result(100)
        # Finalize last day
        pf.start_new_day()

        ok, msg = pf.check_consistency()
        assert not ok
        assert "Consistency violation" in msg

    def test_consistency_passes_even_distribution(self):
        pf = PropFirmCompliance("myfundedfx", "challenge", 100000)
        for _ in range(5):
            pf.start_new_day()
            pf.record_trade_result(1000)
        pf.start_new_day()  # finalize last day

        ok, msg = pf.check_consistency()
        assert ok


class TestSafeRiskAmount:

    def test_safe_risk_with_full_buffer(self):
        pf = PropFirmCompliance("ftmo", "challenge", 100000)
        pf.start_new_day()
        safe = pf.get_safe_risk_amount()
        assert safe > 0
        # Should be limited by the 30% of daily remaining or 15% of dd remaining
        assert safe <= 100000 * 0.05 * 0.30  # 30% of daily limit

    def test_safe_risk_decreases_with_losses(self):
        pf = PropFirmCompliance("ftmo", "challenge", 100000)
        pf.start_new_day()
        initial_safe = pf.get_safe_risk_amount()
        pf.record_trade_result(-2000)
        after_loss_safe = pf.get_safe_risk_amount()
        assert after_loss_safe < initial_safe


class TestGetStatus:

    def test_status_report(self):
        pf = PropFirmCompliance("ftmo", "challenge", 100000)
        pf.start_new_day()
        pf.record_trade_result(1000)
        status = pf.get_status()

        assert status["firm"] == "FTMO"
        assert status["phase"] == "challenge"
        assert status["account_size"] == 100000
        assert status["current_equity"] == 101000
        assert status["total_pnl"] == 1000
        assert not status["is_blown"]
        assert "daily_loss_limit" in status
        assert "total_dd_limit" in status
        assert "profit_target" in status
        assert "safe_risk_amount" in status

    def test_apex_no_daily_limit(self):
        pf = PropFirmCompliance("apex", "challenge", 100000)
        pf.start_new_day()
        status = pf.get_status()
        # Apex has no daily loss limit
        assert "daily_loss_limit" not in status
