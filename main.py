#!/usr/bin/env python3
"""AlphaWin Trading Agent - Main entry point.

Usage:
    python main.py                              # Run backtest with default config
    python main.py --market bull                 # Backtest in bull market
    python main.py --market bear                 # Backtest in bear market
    python main.py --candles 2000               # Custom candle count
    python main.py --scenarios                  # Train across ALL 18 market scenarios
    python main.py --prop-firm ftmo             # Backtest with FTMO compliance
    python main.py --prop-firm topstep --phase funded  # Topstep funded mode
    python main.py --scenarios --prop-firm ftmo  # Full scenario training with prop firm
    python main.py --list-firms                 # List available prop firms
"""
import argparse
import logging
import sys

from trading_agent.agent import AlphaWinAgent
from trading_agent.core.backtester import Backtester
from trading_agent.core.prop_firm import PropFirmCompliance
from trading_agent.core.scenarios import MarketScenario

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("main")


def print_results(results: dict, title: str = "Backtest"):
    """Print backtest results with full analytics."""
    print("\n" + "=" * 60)
    print(f"  AlphaWin {title}")
    print("=" * 60)
    print(f"  Total Candles:     {results.get('total_candles', 'N/A')}")
    print(f"  Total Trades:      {results['total_trades']}")
    print(f"  Win Rate:          {results['win_rate']:.1%}")
    print(f"  Total PnL:         ${results['total_pnl']:.2f}")
    print(f"  Total Return:      {results['total_return_pct']:.1%}")
    print(f"  Max Drawdown:      {results.get('max_drawdown_pct', results.get('max_drawdown', 0)):.1%}")
    print(f"  Sharpe Ratio:      {results.get('sharpe_ratio', 0):.2f}")
    print(f"  Sortino Ratio:     {results.get('sortino_ratio', 0):.2f}")
    print(f"  Calmar Ratio:      {results.get('calmar_ratio', 0):.2f}")
    print(f"  Profit Factor:     {results['profit_factor']:.2f}")
    print(f"  Expectancy:        ${results.get('expectancy', 0):.2f}")
    print(f"  Avg Win:           ${results['avg_win']:.2f}")
    print(f"  Avg Loss:          ${results['avg_loss']:.2f}")
    print(f"  Max Win Streak:    {results.get('max_win_streak', 0)}")
    print(f"  Max Loss Streak:   {results.get('max_consecutive_losses', 0)}")
    print(f"  Kelly Fraction:    {results.get('kelly_fraction', 1.0):.2f}")
    print(f"  Final Capital:     ${results['capital']:.2f}")

    if "filter_stats" in results:
        fs = results["filter_stats"]
        print(f"  ---")
        print(f"  Signals Generated: {fs.get('passed', 0) + fs.get('consensus_blocked', 0) + fs.get('mtf_blocked', 0) + fs.get('quality_blocked', 0)}")
        print(f"  Regime Blocked:    {fs.get('regime_blocked', 0)}")
        print(f"  Cooldown Blocked:  {fs.get('cooldown_blocked', 0)}")
        print(f"  Persistence Block: {fs.get('persistence_blocked', 0)}")
        print(f"  Consensus Blocked: {fs.get('consensus_blocked', 0)}")
        print(f"  MTF Blocked:       {fs.get('mtf_blocked', 0)}")
        print(f"  Quality Blocked:   {fs.get('quality_blocked', 0)}")
        print(f"  Trades Executed:   {fs.get('passed', 0)}")
    print("=" * 60)


def print_prop_firm_dashboard(results: dict):
    """Print prop firm compliance dashboard."""
    prop = results.get("prop_firm")
    if not prop:
        return

    status_icon = "BLOWN" if prop["is_blown"] else ("PASSED" if prop.get("challenge_passed") else "ACTIVE")

    print("\n" + "-" * 60)
    print(f"  PROP FIRM COMPLIANCE: {prop['firm']} ({prop['phase'].upper()}) [{status_icon}]")
    print("-" * 60)
    print(f"  Account Size:      ${prop['account_size']:,.0f}")
    print(f"  Current Equity:    ${prop['current_equity']:,.2f}")
    print(f"  Total PnL:         ${prop['total_pnl']:,.2f} ({prop['total_return_pct']:.2%})")
    print(f"  Trading Days:      {prop['trading_days']}")

    if "daily_loss_limit" in prop:
        used_pct = prop.get("daily_loss_pct_used", 0)
        bar = int(used_pct * 20)
        bar_str = "#" * bar + "-" * (20 - bar)
        print(f"  Daily Loss:        [{bar_str}] {used_pct:.0%} used (${prop['daily_loss_remaining']:,.0f} left)")

    if "total_dd_limit" in prop:
        dd_pct = prop.get("total_dd_pct_used", 0)
        bar = int(dd_pct * 20)
        bar_str = "#" * bar + "-" * (20 - bar)
        print(f"  Total Drawdown:    [{bar_str}] {dd_pct:.0%} used (${prop['total_dd_remaining']:,.0f} left)")

    if "profit_target" in prop:
        progress = prop.get("profit_progress", 0)
        bar = min(int(progress * 20), 20)
        bar_str = "#" * bar + "-" * (20 - bar)
        print(f"  Profit Target:     [{bar_str}] {progress:.0%} (${prop['profit_target']:,.0f} target)")
        if prop.get("days_remaining", 0) > 0:
            print(f"  Min Days Left:     {prop['days_remaining']}")

    print(f"  Consistency:       {prop.get('consistency_msg', 'N/A')}")
    print(f"  Safe Risk Amount:  ${prop.get('safe_risk_amount', 0):,.2f}")

    if prop.get("challenge_passed"):
        print(f"  >>> CHALLENGE PASSED! <<<")
    if prop.get("violations"):
        for v in prop["violations"]:
            print(f"  VIOLATION: {v}")
    print("-" * 60)


def run_backtest(market: str = "mixed", n_candles: int = 1000,
                 prop_firm: str = None, prop_phase: str = "challenge"):
    """Run backtest and print results."""
    agent = AlphaWinAgent(prop_firm=prop_firm, prop_phase=prop_phase)

    logger.info(f"Generating {n_candles} synthetic {market} market candles...")
    candles = Backtester.generate_synthetic_data(
        n_candles=n_candles,
        trend=market,
        volatility=0.02,
    )

    logger.info("Running backtest...")
    results = agent.backtest(candles)

    title = f"Results ({market.upper()} market)"
    if prop_firm:
        title += f" | {prop_firm.upper()} {prop_phase}"
    print_results(results, title)

    if "prop_firm" in results:
        print_prop_firm_dashboard(results)

    return results


def run_scenario_training(n_candles: int = 500, prop_firm: str = None,
                          prop_phase: str = "challenge"):
    """Train and validate across all 18 market scenarios."""
    agent = AlphaWinAgent(prop_firm=prop_firm, prop_phase=prop_phase)

    bt = Backtester(agent.config, agent.strategies, agent.risk_manager.initial_capital,
                    prop_firm=prop_firm, prop_phase=prop_phase)

    logger.info(f"Training across all 18 market scenarios ({n_candles} candles each)...")
    results = bt.train_all_scenarios(n_candles=n_candles, warmup=50)

    summary = results["summary"]

    print("\n" + "=" * 70)
    print("  SCENARIO TRAINING REPORT - All 18 Market Conditions")
    if prop_firm:
        print(f"  Prop Firm: {prop_firm.upper()} | Phase: {prop_phase}")
    print("=" * 70)

    # Per-scenario results table
    print(f"\n  {'Scenario':<25} {'Trades':>7} {'Win%':>7} {'PnL':>12} {'MaxDD':>8} {'Status':>10}")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*12} {'-'*8} {'-'*10}")

    for scenario_name, sr in results["scenarios"].items():
        trades = sr.get("total_trades", 0)
        wr = sr.get("win_rate", 0)
        pnl = sr.get("total_pnl", 0)
        mdd = sr.get("max_drawdown_pct", sr.get("max_drawdown", 0))

        prop = sr.get("prop_firm")
        if prop and prop.get("is_blown"):
            status = "BLOWN"
        elif prop and prop.get("challenge_passed"):
            status = "PASSED"
        elif trades == 0:
            status = "NO TRADE"
        elif pnl > 0:
            status = "PROFIT"
        elif pnl == 0:
            status = "FLAT"
        else:
            status = "LOSS"

        print(f"  {scenario_name:<25} {trades:>7} {wr:>6.1%} {pnl:>11.2f} {mdd:>7.1%} {status:>10}")

    print(f"\n  {'='*70}")
    print(f"  SUMMARY")
    print(f"  {'='*70}")
    print(f"  Total Scenarios:     {summary['total_scenarios']}")
    print(f"  Survived:            {summary['scenarios_survived']} / {summary['total_scenarios']}")
    if summary["blown_scenarios"]:
        print(f"  Blown:               {', '.join(summary['blown_scenarios'])}")
    print(f"  Total Trades:        {summary['total_trades']}")
    print(f"  Overall Win Rate:    {summary['overall_win_rate']:.1%}")
    print(f"  Total PnL:           ${summary['total_pnl']:.2f}")
    print(f"  Avg PnL/Scenario:    ${summary['avg_pnl_per_scenario']:.2f}")
    print(f"  {'='*70}")

    return results


def main():
    parser = argparse.ArgumentParser(description="AlphaWin Trading Agent")
    parser.add_argument("--market", choices=["bull", "bear", "mixed", "sideways"],
                        default="mixed", help="Market type for backtest")
    parser.add_argument("--candles", type=int, default=1000, help="Number of candles")
    parser.add_argument("--scenarios", action="store_true",
                        help="Train across all 18 market scenarios")
    parser.add_argument("--prop-firm", type=str, default=None,
                        help="Prop firm to simulate (ftmo, topstep, etc.)")
    parser.add_argument("--phase", type=str, default="challenge",
                        choices=["challenge", "verification", "funded"],
                        help="Prop firm phase")
    parser.add_argument("--list-firms", action="store_true",
                        help="List available prop firm profiles")
    args = parser.parse_args()

    if args.list_firms:
        print("\nAvailable Prop Firm Profiles:")
        print("-" * 40)
        for firm in PropFirmCompliance.list_firms():
            from trading_agent.core.prop_firm import PROP_FIRM_PROFILES
            profile = PROP_FIRM_PROFILES[firm]
            phases = [k for k in profile if k != "name"]
            print(f"  {firm:<20} ({profile['name']}) - Phases: {', '.join(phases)}")
        print()
        return

    if args.scenarios:
        run_scenario_training(args.candles, args.prop_firm, args.phase)
    else:
        results = run_backtest(args.market, args.candles, args.prop_firm, args.phase)

        # Run across all market conditions for comprehensive validation
        if args.market == "mixed" and not args.prop_firm:
            print("\n\nRunning validation across all market conditions...\n")
            for market_type in ["bull", "bear", "sideways"]:
                run_backtest(market_type, args.candles)


if __name__ == "__main__":
    main()
