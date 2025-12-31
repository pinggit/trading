#!/usr/bin/env python3
"""
Live trading bot using Alpaca API.
Supports paper trading (simulation) and live trading.

Setup:
    1. Create account at https://alpaca.markets
    2. Get API keys from dashboard
    3. Copy config.json.example to config.json and add your keys

    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate

    # Install dependencies
    pip install alpaca-trade-api pandas numpy

    # Copy config and add your API keys
    cp config.json.example config.json

Usage:
    # Test connection and show account status
    ./trader.py test

    # Show account balance and positions
    ./trader.py status

    # Run bot continuously (executes trades based on strategy signals)
    ./trader.py run

    # Manual trading
    ./trader.py buy -s AAPL -a 1000      # Buy $1000 of AAPL
    ./trader.py buy -s AAPL -q 10        # Buy 10 shares of AAPL
    ./trader.py sell -s AAPL             # Sell all AAPL shares
    ./trader.py sell -s AAPL -q 5        # Sell 5 shares of AAPL
    ./trader.py close -s AAPL            # Close AAPL position
    ./trader.py close-all                # Close all positions

Options:
    --config, -c     Config file path (default: config.json)
    --symbol, -s     Stock symbol for buy/sell/close
    --qty, -q        Number of shares
    --amount, -a     Dollar amount to invest
    --strategy       Override strategy: sma_crossover, rsi, macd

Config file options (config.json):
    api_key          Alpaca API key
    api_secret       Alpaca API secret
    base_url         API endpoint (paper-api or api.alpaca.markets)
    symbols          List of symbols to trade ["SPY", "AAPL", ...]
    strategy         Trading strategy: sma_crossover, rsi, macd
    check_interval   Seconds between strategy checks (default: 60)
    position_size    Fraction of capital per trade (default: 0.1 = 10%)
    max_positions    Maximum concurrent positions (default: 5)
    timeframe        Bar timeframe: 1Min, 5Min, 15Min, 1Hour, 1Day
    signal_mode      "crossover" (signal on cross) or "position" (fast>slow)
    use_margin       Use margin buying power (default: false, safer)

IMPORTANT: Start with paper trading. Never risk money you can't afford to lose.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np

# Alpaca API
from alpaca_trade_api import REST as AlpacaREST
from alpaca_trade_api.stream import Stream

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)


@dataclass
class Config:
    """Trading bot configuration."""
    api_key: str
    api_secret: str
    base_url: str = "https://paper-api.alpaca.markets"  # Paper trading default
    symbols: list = None
    strategy: str = "sma_crossover"
    check_interval: int = 60  # seconds
    position_size: float = 0.1  # 10% of portfolio per trade
    max_positions: int = 5
    timeframe: str = "1Day"  # "1Min", "5Min", "15Min", "1Hour", "1Day"
    signal_mode: str = "crossover"  # "crossover" (on cross) or "position" (fast > slow)
    use_margin: bool = False  # If False, only use available cash (safer)

    @classmethod
    def from_file(cls, path: str = "config.json") -> "Config":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_env(cls) -> "Config":
        """Load config from environment variables."""
        return cls(
            api_key=os.environ.get("ALPACA_API_KEY", ""),
            api_secret=os.environ.get("ALPACA_API_SECRET", ""),
            base_url=os.environ.get(
                "ALPACA_BASE_URL",
                "https://paper-api.alpaca.markets"
            ),
            symbols=os.environ.get("TRADING_SYMBOLS", "SPY,AAPL,MSFT").split(",")
        )


class TradingBot:
    """Main trading bot class."""

    def __init__(self, config: Config):
        self.config = config
        self.api = AlpacaREST(
            key_id=config.api_key,
            secret_key=config.api_secret,
            base_url=config.base_url
        )
        self.running = False
        self.strategies = {
            "sma_crossover": self.strategy_sma_crossover,
            "rsi": self.strategy_rsi,
            "macd": self.strategy_macd,
        }

    # =========================================================================
    # Account & Market Info
    # =========================================================================

    def get_account(self) -> dict:
        """Get account information."""
        account = self.api.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "last_equity": float(account.last_equity),
            "pattern_day_trader": account.pattern_day_trader,
            "trading_blocked": account.trading_blocked,
        }

    def get_positions(self) -> list:
        """Get current positions."""
        positions = self.api.list_positions()
        return [{
            "symbol": p.symbol,
            "qty": float(p.qty),
            "avg_entry": float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "unrealized_pl": float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc),
        } for p in positions]

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        clock = self.api.get_clock()
        return clock.is_open

    def get_bars(self, symbol: str, timeframe: str = "1Day",
                 limit: int = 100) -> pd.DataFrame:
        """
        Get historical bars for a symbol.

        Args:
            symbol: Stock ticker
            timeframe: '1Min', '5Min', '15Min', '1Hour', '1Day'
            limit: Number of bars to fetch
        """
        bars = self.api.get_bars(
            symbol,
            timeframe,
            limit=limit
        ).df

        if len(bars) == 0:
            return pd.DataFrame()

        # Reset index to get timestamp as column
        bars = bars.reset_index()
        bars.columns = ['timestamp', 'open', 'high', 'low', 'close',
                        'volume', 'trade_count', 'vwap']
        return bars

    def get_latest_price(self, symbol: str) -> float:
        """Get latest trade price for a symbol."""
        trade = self.api.get_latest_trade(symbol)
        return float(trade.price)

    # =========================================================================
    # Order Execution
    # =========================================================================

    def buy(self, symbol: str, qty: int = None, notional: float = None,
            limit_price: float = None) -> dict:
        """
        Place a buy order.

        Args:
            symbol: Stock ticker
            qty: Number of shares (mutually exclusive with notional)
            notional: Dollar amount to buy (mutually exclusive with qty)
            limit_price: Limit price (market order if None)
        """
        order_type = "limit" if limit_price else "market"

        order_params = {
            "symbol": symbol,
            "side": "buy",
            "type": order_type,
            "time_in_force": "day",
        }

        if qty:
            order_params["qty"] = qty
        elif notional:
            order_params["notional"] = notional
        else:
            raise ValueError("Must specify qty or notional")

        if limit_price:
            order_params["limit_price"] = limit_price

        order = self.api.submit_order(**order_params)
        if notional:
            log.info(f"BUY order placed: {symbol} ${notional}")
        else:
            log.info(f"BUY order placed: {symbol} qty={qty}")
        return self._order_to_dict(order)

    def sell(self, symbol: str, qty: int = None,
             limit_price: float = None) -> dict:
        """
        Place a sell order.

        Args:
            symbol: Stock ticker
            qty: Number of shares (None = sell all)
            limit_price: Limit price (market order if None)
        """
        if qty is None:
            # Sell entire position
            positions = {p.symbol: p for p in self.api.list_positions()}
            if symbol not in positions:
                log.warning(f"No position in {symbol} to sell")
                return None
            qty = float(positions[symbol].qty)

        order_type = "limit" if limit_price else "market"

        order = self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell",
            type=order_type,
            time_in_force="day",
            limit_price=limit_price
        )
        log.info(f"SELL order placed: {symbol} qty={qty}")
        return self._order_to_dict(order)

    def close_position(self, symbol: str) -> dict:
        """Close entire position for a symbol."""
        try:
            order = self.api.close_position(symbol)
            log.info(f"Closed position: {symbol}")
            return self._order_to_dict(order)
        except Exception as e:
            log.error(f"Failed to close {symbol}: {e}")
            return None

    def close_all_positions(self) -> list:
        """Close all open positions."""
        results = []
        positions = self.api.list_positions()
        for p in positions:
            result = self.close_position(p.symbol)
            if result:
                results.append(result)
        return results

    def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        self.api.cancel_all_orders()
        log.info("Cancelled all open orders")

    def _order_to_dict(self, order) -> dict:
        """Convert order object to dict."""
        return {
            "id": order.id,
            "symbol": order.symbol,
            "side": order.side,
            "qty": order.qty,
            "type": order.type,
            "status": order.status,
            "filled_qty": order.filled_qty,
            "filled_avg_price": order.filled_avg_price,
        }

    # =========================================================================
    # Strategies
    # =========================================================================

    def strategy_sma_crossover(self, symbol: str,
                               fast: int = 10, slow: int = 30) -> int:
        """
        SMA crossover strategy.
        Returns: 1 (buy), -1 (sell), 0 (hold)

        Modes (via config.signal_mode):
        - "crossover": signal only when lines cross (rare but precise)
        - "position": signal when fast > slow (more frequent)
        """
        bars = self.get_bars(symbol, self.config.timeframe, limit=slow + 5)
        if len(bars) < slow:
            return 0

        close = bars['close']
        fast_sma = close.rolling(window=fast).mean()
        slow_sma = close.rolling(window=slow).mean()

        if self.config.signal_mode == "position":
            # Position mode: buy when fast > slow, sell when fast < slow
            if fast_sma.iloc[-1] > slow_sma.iloc[-1]:
                return 1  # Buy signal
            elif fast_sma.iloc[-1] < slow_sma.iloc[-1]:
                return -1  # Sell signal
        else:
            # Crossover mode: signal only on actual crossover
            if fast_sma.iloc[-1] > slow_sma.iloc[-1] and \
               fast_sma.iloc[-2] <= slow_sma.iloc[-2]:
                return 1  # Buy signal
            elif fast_sma.iloc[-1] < slow_sma.iloc[-1] and \
                 fast_sma.iloc[-2] >= slow_sma.iloc[-2]:
                return -1  # Sell signal

        return 0

    def strategy_rsi(self, symbol: str, period: int = 14,
                     oversold: int = 30, overbought: int = 70) -> int:
        """
        RSI mean reversion strategy.
        Returns: 1 (buy), -1 (sell), 0 (hold)
        """
        bars = self.get_bars(symbol, self.config.timeframe, limit=period + 5)
        if len(bars) < period:
            return 0

        close = bars['close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        if current_rsi < oversold:
            return 1
        elif current_rsi > overbought:
            return -1
        return 0

    def strategy_macd(self, symbol: str,
                      fast: int = 12, slow: int = 26, signal: int = 9) -> int:
        """
        MACD crossover strategy.
        Returns: 1 (buy), -1 (sell), 0 (hold)
        """
        bars = self.get_bars(symbol, self.config.timeframe, limit=slow + signal + 5)
        if len(bars) < slow + signal:
            return 0

        close = bars['close']
        exp_fast = close.ewm(span=fast, adjust=False).mean()
        exp_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = exp_fast - exp_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        # Check for crossover
        if macd_line.iloc[-1] > signal_line.iloc[-1] and \
           macd_line.iloc[-2] <= signal_line.iloc[-2]:
            return 1
        elif macd_line.iloc[-1] < signal_line.iloc[-1] and \
             macd_line.iloc[-2] >= signal_line.iloc[-2]:
            return -1

        return 0

    # =========================================================================
    # Main Trading Loop
    # =========================================================================

    def calculate_position_size(self) -> float:
        """Calculate dollar amount to invest per trade."""
        account = self.get_account()
        # Use buying_power (includes margin) or cash only based on config
        available = account['buying_power'] if self.config.use_margin else account['cash']
        if available <= 0:
            return 0
        return round(available * self.config.position_size, 2)

    def run_once(self) -> dict:
        """Run one iteration of the trading loop."""
        if not self.is_market_open():
            log.info("Market is closed")
            return {"status": "market_closed"}

        results = {"trades": [], "signals": {}}
        strategy_fn = self.strategies.get(self.config.strategy)

        if not strategy_fn:
            log.error(f"Unknown strategy: {self.config.strategy}")
            return {"error": "unknown_strategy"}

        positions = {p.symbol: p for p in self.api.list_positions()}
        account = self.get_account()

        for symbol in self.config.symbols:
            try:
                signal = strategy_fn(symbol)
                results["signals"][symbol] = signal

                if signal == 1 and symbol not in positions:
                    # Buy signal and no position
                    if len(positions) < self.config.max_positions:
                        amount = self.calculate_position_size()
                        order = self.buy(symbol, notional=amount)
                        results["trades"].append(order)
                    else:
                        log.info(f"Max positions reached, skipping {symbol}")

                elif signal == -1 and symbol in positions:
                    # Sell signal and have position
                    order = self.close_position(symbol)
                    if order:
                        results["trades"].append(order)

            except Exception as e:
                log.error(f"Error processing {symbol}: {e}")

        return results

    def run(self) -> None:
        """Run the trading bot continuously."""
        log.info("Starting trading bot...")
        log.info(f"Strategy: {self.config.strategy}")
        log.info(f"Timeframe: {self.config.timeframe}")
        log.info(f"Signal mode: {self.config.signal_mode}")
        log.info(f"Symbols: {self.config.symbols}")
        log.info(f"Check interval: {self.config.check_interval}s")

        account = self.get_account()
        log.info(f"Account equity: ${account['equity']:,.2f}")
        log.info(f"Buying power: ${account['buying_power']:,.2f}")

        self.running = True

        while self.running:
            try:
                results = self.run_once()
                log.info(f"Signals: {results.get('signals', {})}")

                if results.get("trades"):
                    for trade in results["trades"]:
                        log.info(f"Trade executed: {trade}")

                time.sleep(self.config.check_interval)

            except KeyboardInterrupt:
                log.info("Shutting down...")
                self.running = False
            except Exception as e:
                log.error(f"Error in main loop: {e}")
                time.sleep(10)

    def stop(self) -> None:
        """Stop the trading bot."""
        self.running = False


# =============================================================================
# CLI
# =============================================================================

def print_account(bot: TradingBot):
    """Print account summary."""
    account = bot.get_account()
    positions = bot.get_positions()

    # Calculate P/L
    today_pl = account['equity'] - account['last_equity']
    today_pl_pct = (today_pl / account['last_equity']) * 100 if account['last_equity'] else 0
    total_unrealized = sum(p['unrealized_pl'] for p in positions)

    print("\n" + "="*50)
    print(" ACCOUNT SUMMARY")
    print("="*50)
    print(f" Equity:        ${account['equity']:>12,.2f}")
    print(f" Cash:          ${account['cash']:>12,.2f}")
    print(f" Buying Power:  ${account['buying_power']:>12,.2f}")
    print("-"*50)
    pl_sign = "+" if today_pl >= 0 else ""
    print(f" Today's P/L:   {pl_sign}${today_pl:>11,.2f} ({pl_sign}{today_pl_pct:.2f}%)")
    ur_sign = "+" if total_unrealized >= 0 else ""
    print(f" Unrealized:    {ur_sign}${total_unrealized:>11,.2f}")
    print("="*50)


def print_positions(bot: TradingBot):
    """Print current positions."""
    positions = bot.get_positions()
    print("\n" + "="*60)
    print(" POSITIONS")
    print("="*60)
    if not positions:
        print(" No open positions")
    else:
        print(f" {'Symbol':<8} {'Qty':>10} {'Entry':>10} {'Current':>10} {'P/L':>12}")
        print("-"*60)
        for p in positions:
            print(f" {p['symbol']:<8} {p['qty']:>10.4f} "
                  f"${p['avg_entry']:>9.2f} ${p['current_price']:>9.2f} "
                  f"${p['unrealized_pl']:>10.2f}")
    print("="*60)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Trading Bot")
    parser.add_argument("command", choices=["run", "status", "buy", "sell",
                                            "close", "close-all", "test"],
                        help="Command to execute")
    parser.add_argument("--symbol", "-s", help="Stock symbol")
    parser.add_argument("--qty", "-q", type=int, help="Quantity (shares)")
    parser.add_argument("--amount", "-a", type=float, help="Dollar amount")
    parser.add_argument("--config", "-c", default="config.json",
                        help="Config file path")
    parser.add_argument("--strategy", choices=["sma_crossover", "rsi", "macd"],
                        default="sma_crossover", help="Trading strategy")

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = Config.from_file(str(config_path))
    else:
        config = Config.from_env()

    if args.strategy:
        config.strategy = args.strategy

    if not config.api_key or not config.api_secret:
        print("ERROR: API keys not configured")
        print("\nSet environment variables:")
        print("  export ALPACA_API_KEY='your_key'")
        print("  export ALPACA_API_SECRET='your_secret'")
        print("\nOr create config.json with api_key and api_secret")
        return 1

    bot = TradingBot(config)

    if args.command == "status":
        print_account(bot)
        print_positions(bot)

    elif args.command == "run":
        bot.run()

    elif args.command == "buy":
        if not args.symbol:
            print("ERROR: --symbol required")
            return 1
        if args.qty:
            order = bot.buy(args.symbol, qty=args.qty)
        else:
            amount = args.amount or bot.calculate_position_size()
            order = bot.buy(args.symbol, notional=amount)
        print(f"Order placed: {order}")

    elif args.command == "sell":
        if not args.symbol:
            print("ERROR: --symbol required")
            return 1
        order = bot.sell(args.symbol, qty=args.qty)
        print(f"Order placed: {order}")

    elif args.command == "close":
        if not args.symbol:
            print("ERROR: --symbol required")
            return 1
        order = bot.close_position(args.symbol)
        print(f"Position closed: {order}")

    elif args.command == "close-all":
        orders = bot.close_all_positions()
        print(f"Closed {len(orders)} positions")

    elif args.command == "test":
        print("Testing connection...")
        print_account(bot)
        print(f"\nMarket open: {bot.is_market_open()}")
        if config.symbols:
            symbol = config.symbols[0]
            print(f"\nLatest price for {symbol}: ${bot.get_latest_price(symbol):.2f}")
            signal = bot.strategies[config.strategy](symbol)
            print(f"Strategy signal: {signal} (1=buy, -1=sell, 0=hold)")

    return 0


if __name__ == "__main__":
    exit(main())
