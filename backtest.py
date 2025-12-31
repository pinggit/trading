#!/usr/bin/env python3
"""
Backtesting framework for trading strategy experiments.
Tests strategies against historical data before using them live.

Setup:
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate

    # Install dependencies
    pip install yfinance pandas numpy

Usage:
    # Run with default settings (tests all strategies on SPY 2022-2024)
    ./backtest.py

    # Or import and use programmatically:
    from backtest import Backtester, fetch_data, sma_crossover

    data = fetch_data("AAPL", "2023-01-01", "2024-01-01")
    bt = Backtester(data, initial_capital=10000)
    result = bt.run(lambda d: sma_crossover(d, fast=10, slow=30))
    print(f"Return: {result.total_return:.2%}")

Available strategies:
    sma_crossover(data, fast=10, slow=30)
        Buy when fast SMA crosses above slow SMA, sell on reverse cross.

    rsi_strategy(data, period=14, oversold=30, overbought=70)
        Buy when RSI < oversold, sell when RSI > overbought.

    bollinger_bands(data, period=20, std_dev=2.0)
        Buy at lower band, sell at upper band (mean reversion).

Output metrics:
    Total Return      Strategy's overall return
    Buy & Hold        Return if you just held the stock
    Outperformance    Strategy return minus buy-and-hold
    Number of Trades  How many round-trip trades executed
    Win Rate          Percentage of profitable trades
    Max Drawdown      Largest peak-to-trough decline
    Sharpe Ratio      Risk-adjusted return (higher = better)

Educational purposes only - not financial advice.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Callable
from datetime import datetime


@dataclass
class TradeResult:
    """Results from a backtest run."""
    total_return: float
    buy_hold_return: float
    num_trades: int
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    trades: pd.DataFrame


class Backtester:
    """Simple backtesting engine."""

    def __init__(self, data: pd.DataFrame, initial_capital: float = 10000):
        """
        Args:
            data: DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume'
            initial_capital: Starting capital in dollars
        """
        self.data = data.copy()
        self.initial_capital = initial_capital

    def run(self, strategy: Callable[[pd.DataFrame], pd.Series]) -> TradeResult:
        """
        Run backtest with given strategy.

        Args:
            strategy: Function that takes price data, returns Series of signals
                      1 = buy, -1 = sell, 0 = hold
        """
        # Generate signals
        signals = strategy(self.data)
        self.data['signal'] = signals

        # Calculate positions (1 = long, 0 = out)
        self.data['position'] = 0
        position = 0
        for i in range(len(self.data)):
            if self.data['signal'].iloc[i] == 1:
                position = 1
            elif self.data['signal'].iloc[i] == -1:
                position = 0
            self.data.iloc[i, self.data.columns.get_loc('position')] = position

        # Calculate returns
        self.data['daily_return'] = self.data['Close'].pct_change()
        self.data['strategy_return'] = self.data['daily_return'] * \
            self.data['position'].shift(1)

        # Calculate cumulative returns
        self.data['cumulative_return'] = (1 + self.data['strategy_return']).cumprod()
        self.data['buy_hold_return'] = (1 + self.data['daily_return']).cumprod()

        # Extract trades
        trades = self._extract_trades()

        # Calculate metrics
        total_return = self.data['cumulative_return'].iloc[-1] - 1
        buy_hold = self.data['buy_hold_return'].iloc[-1] - 1
        max_dd = self._max_drawdown()
        sharpe = self._sharpe_ratio()
        win_rate = (trades['pnl'] > 0).mean() if len(trades) > 0 else 0

        return TradeResult(
            total_return=total_return,
            buy_hold_return=buy_hold,
            num_trades=len(trades),
            win_rate=win_rate,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            trades=trades
        )

    def _extract_trades(self) -> pd.DataFrame:
        """Extract individual trades from signals."""
        trades = []
        entry_price = None
        entry_date = None

        for i in range(1, len(self.data)):
            prev_pos = self.data['position'].iloc[i-1]
            curr_pos = self.data['position'].iloc[i]

            if prev_pos == 0 and curr_pos == 1:  # Entry
                entry_price = self.data['Close'].iloc[i]
                entry_date = self.data.index[i]
            elif prev_pos == 1 and curr_pos == 0:  # Exit
                if entry_price is not None:
                    exit_price = self.data['Close'].iloc[i]
                    pnl = (exit_price - entry_price) / entry_price
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': self.data.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl
                    })
                    entry_price = None

        return pd.DataFrame(trades)

    def _max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        cumulative = self.data['cumulative_return']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio."""
        returns = self.data['strategy_return'].dropna()
        if returns.std() == 0:
            return 0
        excess_returns = returns.mean() - risk_free_rate / 252
        return excess_returns / returns.std() * np.sqrt(252)


# ============================================================================
# Example Strategies
# ============================================================================

def sma_crossover(data: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.Series:
    """
    Simple Moving Average crossover strategy.
    Buy when fast SMA crosses above slow SMA.
    Sell when fast SMA crosses below slow SMA.
    """
    fast_sma = data['Close'].rolling(window=fast).mean()
    slow_sma = data['Close'].rolling(window=slow).mean()

    signals = pd.Series(0, index=data.index)
    signals[fast_sma > slow_sma] = 1   # Buy signal when fast > slow
    signals[fast_sma <= slow_sma] = -1  # Sell signal when fast <= slow

    # Only signal on crossovers
    prev_fast = fast_sma.shift(1)
    prev_slow = slow_sma.shift(1)

    buy_cross = (prev_fast <= prev_slow) & (fast_sma > slow_sma)
    sell_cross = (prev_fast > prev_slow) & (fast_sma <= slow_sma)

    signals = pd.Series(0, index=data.index)
    signals[buy_cross] = 1
    signals[sell_cross] = -1

    return signals


def rsi_strategy(data: pd.DataFrame, period: int = 14,
                 oversold: int = 30, overbought: int = 70) -> pd.Series:
    """
    RSI mean reversion strategy.
    Buy when RSI < oversold (default 30).
    Sell when RSI > overbought (default 70).
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    signals = pd.Series(0, index=data.index)
    signals[rsi < oversold] = 1   # Buy when oversold
    signals[rsi > overbought] = -1  # Sell when overbought

    return signals


def bollinger_bands(data: pd.DataFrame, period: int = 20,
                    std_dev: float = 2.0) -> pd.Series:
    """
    Bollinger Bands mean reversion.
    Buy when price touches lower band.
    Sell when price touches upper band.
    """
    sma = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()

    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)

    signals = pd.Series(0, index=data.index)
    signals[data['Close'] <= lower_band] = 1   # Buy at lower band
    signals[data['Close'] >= upper_band] = -1  # Sell at upper band

    return signals


# ============================================================================
# Data Fetching
# ============================================================================

def fetch_data(symbol: str, start: str, end: str = None) -> pd.DataFrame:
    """
    Fetch historical data using yfinance.

    Args:
        symbol: Ticker symbol (e.g., 'AAPL', 'SPY')
        start: Start date 'YYYY-MM-DD'
        end: End date 'YYYY-MM-DD' (default: today)
    """
    try:
        import yfinance as yf
    except ImportError:
        print("Install yfinance: pip install yfinance")
        raise

    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start, end=end)
    return data


def print_results(result: TradeResult, strategy_name: str = "Strategy"):
    """Print backtest results in readable format."""
    print(f"\n{'='*50}")
    print(f" {strategy_name} Backtest Results")
    print(f"{'='*50}")
    print(f" Total Return:     {result.total_return:>10.2%}")
    print(f" Buy & Hold:       {result.buy_hold_return:>10.2%}")
    print(f" Outperformance:   {result.total_return - result.buy_hold_return:>10.2%}")
    print(f" Number of Trades: {result.num_trades:>10}")
    print(f" Win Rate:         {result.win_rate:>10.2%}")
    print(f" Max Drawdown:     {result.max_drawdown:>10.2%}")
    print(f" Sharpe Ratio:     {result.sharpe_ratio:>10.2f}")
    print(f"{'='*50}\n")

    if len(result.trades) > 0:
        print("Last 5 trades:")
        print(result.trades.tail().to_string())


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Fetch 2 years of SPY data
    print("Fetching data...")
    data = fetch_data("SPY", "2022-01-01", "2024-01-01")
    print(f"Loaded {len(data)} days of data")

    # Test SMA crossover
    bt = Backtester(data)
    result = bt.run(lambda d: sma_crossover(d, fast=10, slow=30))
    print_results(result, "SMA Crossover (10/30)")

    # Test RSI strategy
    bt = Backtester(data)
    result = bt.run(rsi_strategy)
    print_results(result, "RSI (14, 30/70)")

    # Test Bollinger Bands
    bt = Backtester(data)
    result = bt.run(bollinger_bands)
    print_results(result, "Bollinger Bands")
