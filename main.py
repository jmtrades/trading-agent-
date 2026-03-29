#!/usr/bin/env python3
"""AlphaWin Trading Agent - Main entry point.

Usage:
    python main.py                    # Run backtest with default config
    python main.py --market bull      # Backtest in bull market
    python main.py --market bear      # Backtest in bear market
    python main.py --market mixed     # Backtest in mixed market
    python main.py --candles 2000     # Custom candle count
"""
import argparse
import logging
import sys

from trading_agent.agent import AlphaWinAgent
from trading_agent.core.backtester import Backtester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("main")


def run_backtest(market: str = "mixed", n_candles: int = 1000):
    """Run backtest and print results."""
    agent = AlphaWinAgent()

    logger.info(f"Generating {n_candles} synthetic {market} market candles...")
    candles = Backtester.generate_synthetic_data(
        n_candles=n_candles,
        trend=market,
        volatility=0.02,
    )

    logger.info("Running backtest...")
    results = agent.backtest(candles)

    print("\n" + "=" * 60)
    print(f"  AlphaWin Backtest Results ({market.upper()} market)")
    print("=" * 60)
    print(f"  Total Candles:     {results['total_candles']}")
    print(f"  Total Trades:      {results['total_trades']}")
    print(f"  Win Rate:          {results['win_rate']:.1%}")
    print(f"  Total PnL:         ${results['total_pnl']:.2f}")
    print(f"  Total Return:      {results['total_return_pct']:.1%}")
    print(f"  Max Drawdown:      {results['max_drawdown_pct']:.1%}")
    print(f"  Sharpe Ratio:      {results['sharpe_ratio']:.2f}")
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
        print(f"  Consensus Blocked: {fs.get('consensus_blocked', 0)}")
        print(f"  MTF Blocked:       {fs.get('mtf_blocked', 0)}")
        print(f"  Quality Blocked:   {fs.get('quality_blocked', 0)}")
        print(f"  Trades Executed:   {fs.get('passed', 0)}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="AlphaWin Trading Agent")
    parser.add_argument("--market", choices=["bull", "bear", "mixed", "sideways"],
                        default="mixed", help="Market type for backtest")
    parser.add_argument("--candles", type=int, default=1000, help="Number of candles")
    args = parser.parse_args()

    results = run_backtest(args.market, args.candles)

    # Run across all market conditions for comprehensive validation
    if args.market == "mixed":
        print("\n\nRunning validation across all market conditions...\n")
        for market_type in ["bull", "bear", "sideways"]:
            run_backtest(market_type, args.candles)


if __name__ == "__main__":
    main()
