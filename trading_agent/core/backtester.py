"""Backtesting engine - Validates strategies against historical data.

Simulates trading with realistic conditions:
- Sequential candle processing (no lookahead bias)
- Slippage simulation
- Commission handling
- Full trade logging and statistics
"""
import json
import numpy as np
from typing import Optional
from .models import Candle, Signal, OrderSide
from .market_data import MarketData
from .risk_manager import RiskManager


class Backtester:

    def __init__(self, config: dict, strategies: list, initial_capital: float = 10000.0,
                 slippage_pct: float = 0.001, commission_pct: float = 0.001):
        self.config = config
        self.strategies = strategies
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct

    def run(self, candles: list[Candle], warmup: int = 50) -> dict:
        """Run backtest on historical candles."""
        market_data = MarketData(max_candles=500)
        risk_manager = RiskManager(self.config.get("risk", {}), self.initial_capital)

        equity_curve = []
        signals_log = []

        for i, candle in enumerate(candles):
            market_data.add_candle(candle)

            # Skip warmup period
            if i < warmup:
                equity_curve.append(self.initial_capital)
                continue

            # Update existing positions
            exits = risk_manager.update_positions(candle.close)
            for exit_info in exits:
                signals_log.append({
                    "bar": i,
                    "type": "exit",
                    "price": candle.close,
                    **exit_info
                })

            # Get signals from all strategies
            strategy_signals = []
            for strategy in self.strategies:
                sig = strategy.analyze(market_data)
                strategy_signals.append(sig)

            # Combine signals (weighted vote)
            combined_score = 0.0
            total_weight = 0.0
            for sig in strategy_signals:
                weight = next(
                    (s.weight for s in self.strategies
                     if s.name == sig.strategy_name),
                    0.33
                )
                combined_score += sig.signal.value * sig.confidence * weight
                total_weight += weight

            if total_weight > 0:
                combined_score /= total_weight

            # Execute based on combined signal
            if combined_score >= 0.15:
                side = OrderSide.BUY
                signal = strategy_signals[0]  # Use first for metadata
                signal.confidence = min(abs(combined_score), 1.0)
                order = risk_manager.create_order(market_data, signal, side)
                if order:
                    # Apply slippage
                    order.price *= (1 + self.slippage_pct)
                    pos = risk_manager.open_position(order)
                    signals_log.append({
                        "bar": i,
                        "type": "entry",
                        "side": "buy",
                        "price": order.price,
                        "quantity": order.quantity,
                        "score": combined_score,
                    })

            elif combined_score <= -0.15:
                side = OrderSide.SELL
                signal = strategy_signals[0]
                signal.confidence = min(abs(combined_score), 1.0)
                order = risk_manager.create_order(market_data, signal, side)
                if order:
                    order.price *= (1 - self.slippage_pct)
                    pos = risk_manager.open_position(order)
                    signals_log.append({
                        "bar": i,
                        "type": "entry",
                        "side": "sell",
                        "price": order.price,
                        "quantity": order.quantity,
                        "score": combined_score,
                    })

            # Track equity
            position_value = sum(
                p.entry_price * p.quantity * (1 + p.pnl)
                for p in risk_manager.open_positions
            )
            equity_curve.append(risk_manager.capital + position_value)

        # Final stats
        stats = risk_manager.get_stats()
        equity = np.array(equity_curve)

        # Sharpe ratio
        if len(equity) > 1:
            returns = np.diff(equity) / equity[:-1]
            stats["sharpe_ratio"] = (np.mean(returns) / np.std(returns) * np.sqrt(252)) \
                if np.std(returns) > 0 else 0
        else:
            stats["sharpe_ratio"] = 0

        # Max drawdown from equity curve
        peaks = np.maximum.accumulate(equity)
        drawdowns = (peaks - equity) / peaks
        stats["max_drawdown_pct"] = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0

        stats["equity_curve"] = equity_curve
        stats["signals"] = signals_log
        stats["total_candles"] = len(candles)

        return stats

    @staticmethod
    def generate_synthetic_data(n_candles: int = 1000, trend: str = "mixed",
                                volatility: float = 0.02, base_price: float = 100.0,
                                base_volume: float = 1000.0) -> list[Candle]:
        """Generate realistic synthetic candle data for testing.

        Supports trend types: 'bull', 'bear', 'mixed', 'sideways'
        """
        np.random.seed(42)
        candles = []
        price = base_price

        for i in range(n_candles):
            # Trend bias
            if trend == "bull":
                drift = 0.0003
            elif trend == "bear":
                drift = -0.0003
            elif trend == "mixed":
                cycle = np.sin(2 * np.pi * i / 200)
                drift = 0.0003 * cycle
            else:  # sideways
                drift = 0.0

            # Random walk with drift
            change = np.random.normal(drift, volatility)
            new_price = price * (1 + change)

            # Generate OHLCV
            intra_vol = abs(change) + volatility * 0.5
            o = price
            c = new_price
            h = max(o, c) * (1 + abs(np.random.normal(0, intra_vol * 0.3)))
            l = min(o, c) * (1 - abs(np.random.normal(0, intra_vol * 0.3)))
            # Volume spikes on big moves
            vol_spike = 1 + abs(change) / volatility
            v = base_volume * vol_spike * (1 + np.random.uniform(-0.3, 0.3))

            candles.append(Candle(
                timestamp=float(i * 3600),
                open=round(o, 2),
                high=round(h, 2),
                low=round(l, 2),
                close=round(c, 2),
                volume=round(v, 2),
            ))
            price = new_price

        return candles
