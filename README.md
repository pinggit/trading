# Trading Bot

Automated trading bot using Alpaca API with backtesting support.

## Setup

1. Create free account at https://alpaca.markets
2. Go to dashboard → Paper Trading → API Keys → Generate
3. Clone and configure:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install alpaca-trade-api yfinance pandas numpy

# Configure API keys
cp config.json.example config.json
# Edit config.json with your API key and secret from Alpaca dashboard
```

## Usage

### Backtest strategies on historical data
```bash
./backtest.py
```

### Live/Paper trading
```bash
./trader.py test      # Test connection
./trader.py status    # Show account & positions
./trader.py run       # Run bot continuously
```

### Manual trades
```bash
./trader.py buy -s AAPL -a 1000   # Buy $1000 of AAPL
./trader.py sell -s AAPL          # Sell all AAPL
./trader.py close-all             # Close all positions
```

## Strategies

- **sma_crossover**: Buy when fast SMA > slow SMA
- **rsi**: Buy when RSI < 30, sell when > 70
- **macd**: Buy/sell on MACD crossovers

## Config Options

| Option | Description | Default |
|--------|-------------|---------|
| symbols | Stocks to trade | ["SPY", "AAPL", ...] |
| strategy | Trading strategy | sma_crossover |
| position_size | Fraction per trade | 0.1 (10%) |
| max_positions | Max concurrent | 5 |
| timeframe | Bar interval | 5Min |
| signal_mode | crossover or position | position |

Start with paper trading (`paper-api.alpaca.markets`). Not financial advice.
