"""Backtesting engine - Validates strategies against historical data.

Simulates trading with realistic conditions:
- Sequential candle processing (no lookahead bias)
- Full filter pipeline (regime, consensus, MTF, quality, cooldown)
- Slippage simulation
- Commission handling
- Prop firm compliance enforcement
- Multi-scenario training across 18 market conditions
- Full trade logging and statistics
"""
import numpy as np
from .models import Candle, Signal, OrderSide
from .market_data import MarketData
from .risk_manager import RiskManager
from .regime_detector import RegimeDetector, MarketRegime
from .scenarios import MarketScenario
from .filters import (
    MultiTimeframeFilter, ConsensusFilter,
    TradeQualityScorer, CooldownFilter, SignalPersistenceFilter,
)


class Backtester:

    def __init__(self, config: dict, strategies: list, initial_capital: float = 10000.0,
                 slippage_pct: float = 0.001, commission_pct: float = 0.001,
                 prop_firm: str = None, prop_phase: str = "challenge"):
        self.config = config
        self.strategies = strategies
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct
        self.prop_firm = prop_firm
        self.prop_phase = prop_phase

    def run(self, candles: list[Candle], warmup: int = 50) -> dict:
        """Run backtest on historical candles with full filter pipeline."""
        market_data = MarketData(max_candles=500)
        risk_manager = RiskManager(self.config.get("risk", {}), self.initial_capital,
                                   prop_firm=self.prop_firm, prop_phase=self.prop_phase)

        # Initialize filters
        regime_detector = RegimeDetector(lookback=50)
        mtf_filter = MultiTimeframeFilter(multiplier=4)
        consensus_filter = ConsensusFilter(min_agreement=2)
        quality_scorer = TradeQualityScorer(min_quality=0.65)
        cooldown_filter = CooldownFilter(cooldown_bars=3, loss_cooldown_bars=8)
        persistence_filter = SignalPersistenceFilter(required_bars=3)

        equity_curve = []
        signals_log = []
        filter_stats = {"regime_blocked": 0, "cooldown_blocked": 0,
                        "consensus_blocked": 0, "mtf_blocked": 0,
                        "quality_blocked": 0, "persistence_blocked": 0,
                        "passed": 0}

        for i, candle in enumerate(candles):
            market_data.add_candle(candle)

            # Skip warmup period
            if i < warmup:
                equity_curve.append(self.initial_capital)
                continue

            # Update existing positions
            exits = risk_manager.update_positions(candle.close)
            for exit_info in exits:
                was_loss = exit_info.get("pnl", 0) < 0
                cooldown_filter.record_trade(i, was_loss)
                signals_log.append({
                    "bar": i, "type": "exit", "price": candle.close,
                    **exit_info
                })

            # FILTER 1: Market regime
            regime, regime_conf = regime_detector.detect(market_data)
            if regime == MarketRegime.VOLATILE_CHAOS:
                equity_curve.append(self._calc_equity(risk_manager))
                filter_stats["regime_blocked"] += 1
                continue

            # FILTER 2: Cooldown
            can_trade, _ = cooldown_filter.can_trade(i)
            if not can_trade:
                equity_curve.append(self._calc_equity(risk_manager))
                filter_stats["cooldown_blocked"] += 1
                continue

            # Get signals from all strategies with regime-adaptive weights
            regime_weights = regime_detector.get_strategy_weights(regime)
            strategy_signals = []
            for strategy in self.strategies:
                sig = strategy.analyze(market_data)
                strategy_signals.append(sig)

            # Combine signals (regime-adaptive weighted vote)
            combined_score = 0.0
            total_weight = 0.0
            for sig in strategy_signals:
                weight = regime_weights.get(sig.strategy_name, 0.33)
                if weight > 0:
                    combined_score += sig.signal.value * sig.confidence * weight
                    total_weight += weight

            if total_weight > 0:
                combined_score /= total_weight

            # Record score for persistence tracking
            persistence_filter.record_score(combined_score)

            # Determine side
            side = None
            if combined_score >= 0.25:
                side = OrderSide.BUY
            elif combined_score <= -0.25:
                side = OrderSide.SELL

            if side is None:
                equity_curve.append(self._calc_equity(risk_manager))
                continue

            # FILTER: Signal persistence
            persist_ok, _ = persistence_filter.check(side)
            if not persist_ok:
                equity_curve.append(self._calc_equity(risk_manager))
                filter_stats["persistence_blocked"] += 1
                continue

            # Only check consensus among regime-relevant strategies
            relevant_signals = [
                s for s in strategy_signals
                if regime_weights.get(s.strategy_name, 0) >= 0.15
            ]

            # FILTER 3: Consensus (among relevant strategies only)
            consensus_ok, _ = consensus_filter.check(relevant_signals, side)
            if not consensus_ok:
                equity_curve.append(self._calc_equity(risk_manager))
                filter_stats["consensus_blocked"] += 1
                continue

            # FILTER 4: Multi-timeframe alignment
            mtf_ok, _ = mtf_filter.check_alignment(market_data, side)
            if not mtf_ok:
                equity_curve.append(self._calc_equity(risk_manager))
                filter_stats["mtf_blocked"] += 1
                continue

            # FILTER 5: Trade quality
            quality, quality_ok, _ = quality_scorer.score(
                strategy_signals, market_data, side
            )
            if not quality_ok:
                equity_curve.append(self._calc_equity(risk_manager))
                filter_stats["quality_blocked"] += 1
                continue

            # ALL FILTERS PASSED - execute
            filter_stats["passed"] += 1
            signal = max(strategy_signals, key=lambda s: s.confidence)
            signal.confidence = min(abs(combined_score) * quality, 1.0)
            order = risk_manager.create_order(market_data, signal, side)

            if order:
                # Apply slippage
                if side == OrderSide.BUY:
                    order.price *= (1 + self.slippage_pct)
                else:
                    order.price *= (1 - self.slippage_pct)

                risk_manager.open_position(order)
                cooldown_filter.record_trade(i)
                signals_log.append({
                    "bar": i, "type": "entry", "side": side.value,
                    "price": order.price, "quantity": order.quantity,
                    "score": combined_score, "quality": quality,
                    "regime": regime.value,
                })

            equity_curve.append(self._calc_equity(risk_manager))

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
        stats["filter_stats"] = filter_stats

        return stats

    @staticmethod
    def _calc_equity(risk_manager: RiskManager) -> float:
        return risk_manager.capital + sum(
            p.entry_price * p.quantity * (1 + p.pnl)
            for p in risk_manager.open_positions
        )

    @staticmethod
    def generate_synthetic_data(n_candles: int = 1000, trend: str = "mixed",
                                volatility: float = 0.02, base_price: float = 100.0,
                                base_volume: float = 1000.0) -> list[Candle]:
        """Generate realistic synthetic candle data with market microstructure.

        Features:
        - Volatility clustering (GARCH-like): volatile periods cluster together
        - Fat tails: occasional large moves (more realistic than normal distribution)
        - Mean-reverting volatility: extreme vol reverts to normal
        - Volume-price correlation: volume spikes on big moves
        - Support/resistance levels: price tends to bounce at round numbers
        - Momentum persistence: trends last for multiple bars

        Supports trend types: 'bull', 'bear', 'mixed', 'sideways'
        """
        np.random.seed(42)
        candles = []
        price = base_price
        current_vol = volatility
        momentum = 0.0

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

            # GARCH-like volatility clustering
            vol_shock = np.random.normal(0, 0.3)
            current_vol = 0.85 * current_vol + 0.10 * volatility + 0.05 * abs(vol_shock) * volatility
            current_vol = max(volatility * 0.3, min(current_vol, volatility * 3.0))

            # Fat-tailed returns (Student's t with 4 degrees of freedom)
            t_return = np.random.standard_t(4) * current_vol / 2
            normal_return = np.random.normal(0, current_vol)
            # Mix: 80% normal, 20% fat-tailed
            raw_return = 0.8 * normal_return + 0.2 * t_return

            # Momentum persistence (autocorrelation)
            momentum = 0.3 * momentum + 0.7 * raw_return
            change = drift + momentum

            # Support/resistance at round numbers
            round_level = round(price / 10) * 10
            distance_to_round = (price - round_level) / price
            if abs(distance_to_round) < 0.005:
                # Price tends to bounce off round numbers
                change -= distance_to_round * 0.3

            new_price = price * (1 + change)
            new_price = max(new_price, base_price * 0.1)  # Floor

            # Generate OHLCV with realistic intrabar movement
            intra_vol = abs(change) + current_vol * 0.5
            o = price
            c = new_price
            h = max(o, c) * (1 + abs(np.random.exponential(intra_vol * 0.3)))
            l = min(o, c) * (1 - abs(np.random.exponential(intra_vol * 0.3)))
            l = max(l, 0.01)

            # Volume: spikes on big moves and at support/resistance
            vol_spike = 1 + abs(change) / volatility
            sr_vol_boost = 1.5 if abs(distance_to_round) < 0.01 else 1.0
            v = base_volume * vol_spike * sr_vol_boost * (1 + np.random.exponential(0.3))

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

    def run_scenario(self, scenario: str, n_candles: int = 500,
                     warmup: int = 50) -> dict:
        """Run backtest on a specific market scenario."""
        candles = MarketScenario.generate(scenario, n_candles)
        result = self.run(candles, warmup=warmup)
        result["scenario"] = scenario
        return result

    def train_all_scenarios(self, n_candles: int = 500,
                            warmup: int = 50) -> dict:
        """Train and validate across ALL 18 market scenarios.

        Returns aggregate results plus per-scenario breakdown.
        """
        scenarios = MarketScenario.all_scenarios()
        results = {}
        total_trades = 0
        total_wins = 0
        total_losses = 0
        total_pnl = 0.0
        scenarios_survived = 0
        blown_scenarios = []

        for scenario in scenarios:
            # Fresh backtester state for each scenario
            bt = Backtester(self.config, self.strategies, self.initial_capital,
                            self.slippage_pct, self.commission_pct,
                            self.prop_firm, self.prop_phase)
            result = bt.run_scenario(scenario, n_candles, warmup)
            results[scenario] = result

            trades = result.get("total_trades", 0)
            total_trades += trades
            wins = int(result.get("win_rate", 0) * trades)
            total_wins += wins
            total_losses += trades - wins
            total_pnl += result.get("total_pnl", 0)

            # Check if prop firm account survived
            prop = result.get("prop_firm")
            if prop and prop.get("is_blown"):
                blown_scenarios.append(scenario)
            else:
                scenarios_survived += 1

        overall_wr = total_wins / total_trades if total_trades > 0 else 0

        return {
            "scenarios": results,
            "summary": {
                "total_scenarios": len(scenarios),
                "scenarios_survived": scenarios_survived,
                "blown_scenarios": blown_scenarios,
                "total_trades": total_trades,
                "total_wins": total_wins,
                "total_losses": total_losses,
                "overall_win_rate": overall_wr,
                "total_pnl": total_pnl,
                "avg_pnl_per_scenario": total_pnl / len(scenarios),
            },
        }
