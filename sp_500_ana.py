"""
S&P 500 Stock Analyzer using Massive API
Technical analysis and buy/sell recommendations using real market data.

Requirements:
    pip install massive pandas tabulate colorama

Usage:
    1. Get your free API key from https://massive.com
    2. Set environment variable: export MASSIVE_API_KEY='your_key'
    3. Run: python stock_analyzer.py
"""

import os
import sys
import requests
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
from tabulate import tabulate
from colorama import init, Fore, Style
import warnings

warnings.filterwarnings('ignore')

# Import Massive client
try:
    from massive import RESTClient
except ImportError:
    print("Error: massive library not installed.")
    print("Run: pip install massive")
    sys.exit(1)

# Initialize colorama
init()

# ============================================
# CONFIGURATION
# ============================================
MASSIVE_API_KEY = os.environ.get('MASSIVE_API_KEY', 'YOUR_API_KEY_HERE')

FALLBACK_SP500_STOCKS = [
    # Top 50 S&P 500 stocks by market cap (as of 2024)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH', 'JNJ',
    'JPM', 'V', 'XOM', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'LLY',
    'PEP', 'KO', 'COST', 'AVGO', 'WMT', 'MCD', 'CSCO', 'ACN', 'ABT', 'TMO',
    'CRM', 'DHR', 'NKE', 'NFLX', 'DIS', 'ADBE', 'AMD', 'INTC', 'VZ', 'CMCSA',
    'PM', 'TXN', 'NEE', 'WFC', 'RTX', 'BMY', 'QCOM', 'UPS', 'ORCL', 'IBM',
    # Additional major stocks
    'AMGN', 'CAT', 'GS', 'BA', 'HON', 'SBUX', 'GE', 'LMT', 'DE', 'MMM',
    'AXP', 'ISRG', 'MDLZ', 'GILD', 'ADI', 'SYK', 'BKNG', 'PLD', 'VRTX', 'REGN',
    'C', 'BLK', 'SCHW', 'CB', 'ZTS', 'PANW', 'SO', 'DUK', 'MO', 'CI'
]


# ============================================
# S&P 500 LIST FETCHER
# ============================================
class SP500ListFetcher:
    """Fetch S&P 500 constituents dynamically from multiple sources."""
    
    # GitHub raw CSV URLs (maintained datasets)
    GITHUB_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    GITHUB_CSV_URL_ALT = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500.csv"
    
    _cache = None
    _cache_time = None
    _cache_duration = timedelta(hours=24)
    _sectors_cache = None
    
    @classmethod
    def get_sp500_list(cls, use_cache: bool = True, client: RESTClient = None) -> list[str]:
        if use_cache and cls._is_cache_valid():
            print(f"{Fore.CYAN}Using cached S&P 500 list ({len(cls._cache)} stocks){Style.RESET_ALL}")
            return cls._cache
        
        # Method 1: Try GitHub CSV (most reliable)
        stocks, sectors = cls._fetch_from_github()
        if stocks:
            cls._update_cache(stocks, sectors)
            return stocks
        
        # Method 2: Try alternative GitHub CSV
        stocks, sectors = cls._fetch_from_github_alt()
        if stocks:
            cls._update_cache(stocks, sectors)
            return stocks
        
        # Method 3: Try Massive API to get top stocks by market cap
        if client:
            stocks = cls._fetch_from_massive_api(client)
            if stocks:
                cls._update_cache(stocks, {})
                return stocks
        
        # Method 4: Use fallback hardcoded list
        print(f"{Fore.YELLOW}Using fallback stock list{Style.RESET_ALL}")
        return FALLBACK_SP500_STOCKS
    
    @classmethod
    def _is_cache_valid(cls) -> bool:
        return cls._cache is not None and cls._cache_time is not None and \
               datetime.now() - cls._cache_time < cls._cache_duration
    
    @classmethod
    def _update_cache(cls, stocks: list[str], sectors: dict = None):
        cls._cache = stocks
        cls._cache_time = datetime.now()
        cls._sectors_cache = sectors
    
    @classmethod
    def _fetch_from_github(cls) -> tuple:
        """Fetch S&P 500 list from GitHub datasets repo."""
        try:
            print(f"{Fore.CYAN}Fetching S&P 500 list from GitHub...{Style.RESET_ALL}")
            import requests
            response = requests.get(cls.GITHUB_CSV_URL, timeout=15)
            response.raise_for_status()
            
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            # Find symbol and sector columns
            symbol_col = next((c for c in df.columns if c.lower() in ['symbol', 'ticker']), df.columns[0])
            sector_col = next((c for c in df.columns if 'sector' in c.lower()), None)
            
            symbols, sectors = [], {}
            for _, row in df.iterrows():
                s = row[symbol_col]
                if pd.isna(s):
                    continue
                s = str(s).strip()
                if s and len(s) <= 6:
                    symbols.append(s)
                    if sector_col and not pd.isna(row[sector_col]):
                        sector = str(row[sector_col]).strip()
                        sectors.setdefault(sector, []).append(s)
            
            if len(symbols) > 400:
                print(f"{Fore.GREEN}✓ Fetched {len(symbols)} stocks from GitHub{Style.RESET_ALL}")
                return symbols, sectors
        except Exception as e:
            print(f"{Fore.YELLOW}GitHub fetch failed: {e}{Style.RESET_ALL}")
        return None, None
    
    @classmethod
    def _fetch_from_github_alt(cls) -> tuple:
        """Fetch S&P 500 list from alternative GitHub repo."""
        try:
            print(f"{Fore.CYAN}Trying alternative GitHub source...{Style.RESET_ALL}")
            import requests
            response = requests.get(cls.GITHUB_CSV_URL_ALT, timeout=15)
            response.raise_for_status()
            
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            symbol_col = next((c for c in df.columns if c.lower() in ['symbol', 'ticker']), df.columns[0])
            sector_col = next((c for c in df.columns if 'sector' in c.lower()), None)
            
            symbols, sectors = [], {}
            for _, row in df.iterrows():
                s = row[symbol_col]
                if pd.isna(s):
                    continue
                s = str(s).strip()
                if s and len(s) <= 6:
                    symbols.append(s)
                    if sector_col and not pd.isna(row[sector_col]):
                        sector = str(row[sector_col]).strip()
                        sectors.setdefault(sector, []).append(s)
            
            if len(symbols) > 400:
                print(f"{Fore.GREEN}✓ Fetched {len(symbols)} stocks from GitHub (alt){Style.RESET_ALL}")
                return symbols, sectors
        except Exception as e:
            print(f"{Fore.YELLOW}Alternative GitHub fetch failed: {e}{Style.RESET_ALL}")
        return None, None
    
    @classmethod
    def _fetch_from_massive_api(cls, client: RESTClient, limit: int = 500) -> list[str]:
        """Fetch top stocks by market cap from Massive API."""
        try:
            print(f"{Fore.CYAN}Fetching top stocks from Massive API...{Style.RESET_ALL}")
            tickers = []
            for t in client.list_tickers(
                market="stocks",
                type="CS",  # Common Stock
                active=True,
                limit=limit,
                sort="market_cap",
                order="desc"
            ):
                if hasattr(t, 'ticker'):
                    tickers.append(t.ticker)
                if len(tickers) >= limit:
                    break
            
            if len(tickers) > 100:
                print(f"{Fore.GREEN}✓ Fetched {len(tickers)} stocks from Massive API{Style.RESET_ALL}")
                return tickers
        except Exception as e:
            print(f"{Fore.YELLOW}Massive API fetch failed: {e}{Style.RESET_ALL}")
        return None
    
    @classmethod
    def get_sp500_by_sector(cls) -> dict:
        if cls._sectors_cache:
            return cls._sectors_cache
        # Try to fetch if not cached
        stocks, sectors = cls._fetch_from_github()
        if stocks:
            cls._update_cache(stocks, sectors)
            return sectors or {}
        return {}
    
    @classmethod
    def refresh_list(cls, client: RESTClient = None) -> list[str]:
        return cls.get_sp500_list(use_cache=False, client=client)


# ============================================
# MASSIVE API DATA FETCHER
# ============================================
def fetch_stock_data(client: RESTClient, symbol: str) -> dict | None:
    """Fetch stock data using Massive API."""
    try:
        print(f"  Fetching {symbol}...")
        
        # Initialize defaults
        name, market_cap, sector, exchange = symbol, 0, 'N/A', 'N/A'
        current_price = previous_close = volume = high = low = 0
        
        # Get ticker details
        try:
            details = client.get_ticker_details(symbol)
            if details:
                name = getattr(details, 'name', symbol) or symbol
                market_cap = getattr(details, 'market_cap', 0) or 0
                sector = getattr(details, 'sic_description', 'N/A') or 'N/A'
                exchange = getattr(details, 'primary_exchange', 'N/A') or 'N/A'
        except Exception as e:
            print(f"{Fore.YELLOW}    Details error: {e}{Style.RESET_ALL}")
        
        # Get snapshot for current price
        try:
            snapshot = client.get_snapshot_ticker("stocks", symbol)
            if snapshot:
                if hasattr(snapshot, 'day') and snapshot.day:
                    current_price = getattr(snapshot.day, 'close', 0) or 0
                    volume = getattr(snapshot.day, 'volume', 0) or 0
                    high = getattr(snapshot.day, 'high', 0) or 0
                    low = getattr(snapshot.day, 'low', 0) or 0
                if hasattr(snapshot, 'prev_day') and snapshot.prev_day:
                    previous_close = getattr(snapshot.prev_day, 'close', 0) or current_price
        except Exception as e:
            print(f"{Fore.YELLOW}    Snapshot error: {e}{Style.RESET_ALL}")
        
        # Fallback to previous close
        if current_price == 0:
            try:
                prev_list = client.get_previous_close_agg(symbol)
                if prev_list and len(prev_list) > 0:
                    p = prev_list[0]
                    current_price = getattr(p, 'close', 0) or 0
                    previous_close = getattr(p, 'open', 0) or current_price
                    volume = getattr(p, 'volume', 0) or 0
                    high = getattr(p, 'high', 0) or current_price
                    low = getattr(p, 'low', 0) or current_price
            except Exception as e:
                print(f"{Fore.RED}    Previous close error: {e}{Style.RESET_ALL}")
                return None
        
        if current_price == 0:
            print(f"{Fore.RED}    No price data for {symbol}{Style.RESET_ALL}")
            return None
        
        if previous_close == 0:
            previous_close = current_price
        
        # Get historical data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        high_52, low_52, avg_volume = high, low, volume
        ma50 = ma200 = ema20 = current_price
        rsi = 50
        macd = macd_signal = macd_histogram = 0
        
        try:
            aggs = list(client.list_aggs(
                ticker=symbol, multiplier=1, timespan="day",
                from_=start_date, to=end_date, limit=5000
            ))
            
            if aggs:
                closes = [getattr(a, 'close', 0) or 0 for a in aggs]
                highs = [getattr(a, 'high', 0) or 0 for a in aggs]
                lows = [l for l in [getattr(a, 'low', 0) or 0 for a in aggs] if l > 0]
                volumes = [getattr(a, 'volume', 0) or 0 for a in aggs]
                
                high_52 = max(highs) if highs else high
                low_52 = min(lows) if lows else low
                avg_volume = sum(volumes) / len(volumes) if volumes else volume
                
                n = len(closes)
                ma50 = sum(closes[-50:]) / min(50, n) if n > 0 else current_price
                ma200 = sum(closes[-200:]) / min(200, n) if n > 0 else current_price
                
                # Calculate EMA 20
                ema20 = calculate_ema(closes, 20)
                
                # Calculate RSI
                rsi = calculate_rsi(closes)
                
                # Calculate MACD
                macd, macd_signal, macd_histogram = calculate_macd(closes)
        except Exception as e:
            print(f"{Fore.YELLOW}    Historical data error: {e}{Style.RESET_ALL}")
        
        # Try to get technical indicators from API
        try:
            sma = client.get_sma(symbol, timespan="day", window=50, series_type="close")
            if sma and hasattr(sma, 'values') and sma.values:
                ma50 = sma.values[0].value or ma50
        except:
            pass
        
        try:
            ema = client.get_ema(symbol, timespan="day", window=20, series_type="close")
            if ema and hasattr(ema, 'values') and ema.values:
                ema20 = ema.values[0].value or ema20
        except:
            pass
        
        try:
            rsi_data = client.get_rsi(symbol, timespan="day", window=14, series_type="close")
            if rsi_data and hasattr(rsi_data, 'values') and rsi_data.values:
                rsi = rsi_data.values[0].value or rsi
        except:
            pass
        
        try:
            macd_data = client.get_macd(symbol, timespan="day", short_window=12, long_window=26, signal_window=9)
            if macd_data and hasattr(macd_data, 'values') and macd_data.values:
                v = macd_data.values[0]
                macd = getattr(v, 'value', 0) or macd
                macd_signal = getattr(v, 'signal', 0) or macd_signal
                macd_histogram = getattr(v, 'histogram', 0) or macd_histogram
        except:
            pass
        
        change = current_price - previous_close
        change_pct = (change / previous_close * 100) if previous_close > 0 else 0
        
        return {
            'symbol': symbol, 'name': name,
            'current_price': current_price, 'previous_close': previous_close,
            'change': change, 'change_percent': change_pct,
            'high_52': high_52, 'low_52': low_52,
            'ma50': ma50, 'ma200': ma200, 'ema20': ema20,
            'rsi': rsi, 'macd': macd, 'macd_signal': macd_signal, 'macd_histogram': macd_histogram,
            'market_cap': market_cap, 'volume': volume, 'avg_volume': avg_volume,
            'sector': sector, 'exchange': exchange
        }
    except Exception as e:
        print(f"{Fore.RED}Error fetching {symbol}: {e}{Style.RESET_ALL}")
        return None


def calculate_ema(prices: list, period: int) -> float:
    """Calculate Exponential Moving Average."""
    if len(prices) < period:
        return prices[-1] if prices else 0
    
    multiplier = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    
    return ema


def calculate_rsi(prices: list, period: int = 14) -> float:
    """Calculate Relative Strength Index."""
    if len(prices) < period + 1:
        return 50
    
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    
    if avg_loss == 0:
        return 100 if avg_gain > 0 else 50
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices: list) -> tuple:
    """Calculate MACD, Signal, and Histogram."""
    if len(prices) < 35:
        return 0, 0, 0
    
    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
    macd_line = ema12 - ema26
    
    # Approximate signal line
    macd_values = []
    for i in range(26, len(prices)):
        e12 = calculate_ema(prices[:i+1], 12)
        e26 = calculate_ema(prices[:i+1], 26)
        macd_values.append(e12 - e26)
    
    signal = calculate_ema(macd_values, 9) if len(macd_values) >= 9 else macd_line
    histogram = macd_line - signal
    
    return macd_line, signal, histogram


# ============================================
# ANALYSIS ENGINE
# ============================================
def analyze_stock(data: dict) -> dict:
    """Generate buy/sell recommendation based on technical analysis."""
    score = 50
    signals = []
    
    # Moving averages
    if data['current_price'] > data['ma50']:
        score += 8
        signals.append(('bullish', 'Price above 50-day SMA'))
    else:
        score -= 8
        signals.append(('bearish', 'Price below 50-day SMA'))
    
    if data['current_price'] > data['ma200']:
        score += 10
        signals.append(('bullish', 'Price above 200-day SMA'))
    else:
        score -= 10
        signals.append(('bearish', 'Price below 200-day SMA'))
    
    if data['current_price'] > data['ema20']:
        score += 4
        signals.append(('bullish', 'Price above 20-day EMA'))
    else:
        score -= 4
        signals.append(('bearish', 'Price below 20-day EMA'))
    
    # Golden/Death Cross
    if data['ma50'] > data['ma200']:
        score += 5
        signals.append(('bullish', 'Golden cross (50SMA > 200SMA)'))
    else:
        score -= 5
        signals.append(('bearish', 'Death cross (50SMA < 200SMA)'))
    
    # RSI
    rsi = data['rsi']
    if rsi < 30:
        score += 12
        signals.append(('bullish', f'RSI oversold ({rsi:.1f})'))
    elif rsi > 70:
        score -= 12
        signals.append(('bearish', f'RSI overbought ({rsi:.1f})'))
    else:
        signals.append(('neutral', f'RSI neutral ({rsi:.1f})'))
    
    # MACD
    if data['macd'] != 0:
        if data['macd'] > data['macd_signal']:
            score += 6
            signals.append(('bullish', 'MACD above signal'))
        else:
            score -= 6
            signals.append(('bearish', 'MACD below signal'))
    
    # 52-week position
    if data['high_52'] > data['low_52']:
        range_pct = (data['current_price'] - data['low_52']) / (data['high_52'] - data['low_52'])
        if range_pct < 0.3:
            score += 7
            signals.append(('bullish', f'Near 52-week low ({range_pct*100:.1f}%)'))
        elif range_pct > 0.85:
            score -= 5
            signals.append(('bearish', f'Near 52-week high ({range_pct*100:.1f}%)'))
    
    score = max(0, min(100, score))
    
    if score >= 70:
        rec, color = 'STRONG BUY', Fore.GREEN + Style.BRIGHT
    elif score >= 58:
        rec, color = 'BUY', Fore.GREEN
    elif score >= 42:
        rec, color = 'HOLD', Fore.YELLOW
    elif score >= 30:
        rec, color = 'SELL', Fore.RED
    else:
        rec, color = 'STRONG SELL', Fore.RED + Style.BRIGHT
    
    return {'score': score, 'recommendation': rec, 'color': color, 'signals': signals}


# ============================================
# DISPLAY FUNCTIONS
# ============================================
def format_number(num: float) -> str:
    if num >= 1e12: return f"${num/1e12:.2f}T"
    if num >= 1e9: return f"${num/1e9:.2f}B"
    if num >= 1e6: return f"${num/1e6:.2f}M"
    return f"${num:.2f}"


def display_analysis(data: dict, analysis: dict):
    print("\n" + "=" * 60)
    print(f"{Style.BRIGHT}{data['symbol']} - {data['name']}{Style.RESET_ALL}")
    print(f"Sector: {data['sector']} | Exchange: {data['exchange']}")
    print("=" * 60)
    
    color = Fore.GREEN if data['change'] >= 0 else Fore.RED
    sign = '+' if data['change'] >= 0 else ''
    
    print(f"\n{Style.BRIGHT}Price:{Style.RESET_ALL} ${data['current_price']:.2f}")
    print(f"Change: {color}{sign}{data['change']:.2f} ({sign}{data['change_percent']:.2f}%){Style.RESET_ALL}")
    
    print(f"\n{Style.BRIGHT}Recommendation:{Style.RESET_ALL} {analysis['color']}{analysis['recommendation']}{Style.RESET_ALL} (Score: {analysis['score']}/100)")
    
    filled = int(30 * analysis['score'] / 100)
    print(f"[{'█' * filled}{'░' * (30-filled)}]")
    
    print(f"\n{Style.BRIGHT}Key Metrics:{Style.RESET_ALL}")
    print(tabulate([
        ['Market Cap', format_number(data['market_cap']) if data['market_cap'] > 0 else 'N/A'],
        ['52-Week High', f"${data['high_52']:.2f}"],
        ['52-Week Low', f"${data['low_52']:.2f}"],
        ['Volume', f"{data['volume']:,.0f}"],
        ['Avg Volume', f"{data['avg_volume']:,.0f}"]
    ], tablefmt='simple'))
    
    print(f"\n{Style.BRIGHT}Technical Indicators:{Style.RESET_ALL}")
    print(tabulate([
        ['20-Day EMA', f"${data['ema20']:.2f}"],
        ['50-Day SMA', f"${data['ma50']:.2f}"],
        ['200-Day SMA', f"${data['ma200']:.2f}"],
        ['RSI (14)', f"{data['rsi']:.1f}"],
        ['MACD', f"{data['macd']:.4f}"]
    ], tablefmt='simple'))
    
    print(f"\n{Style.BRIGHT}Signals:{Style.RESET_ALL}")
    for t, txt in analysis['signals']:
        icon = f"{Fore.GREEN}▲" if t == 'bullish' else f"{Fore.RED}▼" if t == 'bearish' else f"{Fore.YELLOW}●"
        print(f"  {icon}{Style.RESET_ALL} {txt}")
    
    print(f"\n{Fore.YELLOW}⚠ Educational purposes only. Consult a financial advisor.{Style.RESET_ALL}")


def display_menu():
    print(f"\n{Style.BRIGHT}{'=' * 50}")
    print("       S&P 500 STOCK ANALYZER v5.0")
    print("       Powered by Massive API")
    print(f"{'=' * 50}{Style.RESET_ALL}")
    print("\n1. Analyze a single stock")
    print("2. Scan top stocks")
    print("3. List S&P 500 stocks")
    print("4. List by sector")
    print("5. Refresh stock list")
    print("6. Compare stocks")
    print("7. Exit")


def scan_stocks(client: RESTClient, stocks: list[str], n: int = 10):
    print(f"\n{Style.BRIGHT}Scanning {n} stocks...{Style.RESET_ALL}")
    results = []
    
    for i, symbol in enumerate(stocks[:n]):
        data = fetch_stock_data(client, symbol)
        if data:
            analysis = analyze_stock(data)
            results.append({
                'symbol': symbol,
                'price': data['current_price'],
                'change': data['change_percent'],
                'rsi': data['rsi'],
                'score': analysis['score'],
                'rec': analysis['recommendation']
            })
        print(f"  Progress: {i+1}/{n}")
    
    if not results:
        print(f"{Fore.RED}No results.{Style.RESET_ALL}")
        return
    
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n{Style.BRIGHT}Results (Ranked by Score):{Style.RESET_ALL}")
    table = [[r['symbol'], f"${r['price']:.2f}", f"{'+' if r['change'] >= 0 else ''}{r['change']:.2f}%",
              f"{r['rsi']:.1f}", r['score'], r['rec']] for r in results]
    print(tabulate(table, headers=['Symbol', 'Price', 'Change', 'RSI', 'Score', 'Rec'], tablefmt='grid'))


def compare_stocks(client: RESTClient, symbols: list[str]):
    print(f"\n{Style.BRIGHT}Comparing: {', '.join(symbols)}{Style.RESET_ALL}")
    results = []
    
    for symbol in symbols:
        data = fetch_stock_data(client, symbol)
        if data:
            results.append({'data': data, 'analysis': analyze_stock(data)})
    
    if not results:
        print(f"{Fore.RED}No data to compare.{Style.RESET_ALL}")
        return
    
    headers = ['Metric'] + [r['data']['symbol'] for r in results]
    rows = [
        ['Price'] + [f"${r['data']['current_price']:.2f}" for r in results],
        ['Change %'] + [f"{r['data']['change_percent']:+.2f}%" for r in results],
        ['RSI'] + [f"{r['data']['rsi']:.1f}" for r in results],
        ['50-Day SMA'] + [f"${r['data']['ma50']:.2f}" for r in results],
        ['Score'] + [str(r['analysis']['score']) for r in results],
        ['Recommendation'] + [r['analysis']['recommendation'] for r in results],
    ]
    print(tabulate(rows, headers=headers, tablefmt='grid'))


# ============================================
# MAIN PROGRAM
# ============================================
def main():
    print(f"\n{Fore.CYAN}{Style.BRIGHT}")
    print("  ╔═══════════════════════════════════════════╗")
    print("  ║     S&P 500 STOCK ANALYZER v5.0           ║")
    print("  ║     Powered by Massive API                ║")
    print("  ╚═══════════════════════════════════════════╝")
    print(f"{Style.RESET_ALL}")
    
    if MASSIVE_API_KEY == 'YOUR_API_KEY_HERE':
        print(f"{Fore.RED}{'=' * 50}")
        print("  ERROR: API KEY NOT SET")
        print(f"{'=' * 50}{Style.RESET_ALL}")
        print("\nGet your free API key from https://massive.com")
        print("\nThen set it:")
        print("  export MASSIVE_API_KEY='your_key_here'")
        return
    
    # Initialize client
    try:
        client = RESTClient(api_key=MASSIVE_API_KEY)
        print(f"{Fore.GREEN}✓ Massive API client initialized{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Failed to initialize client: {e}{Style.RESET_ALL}")
        return
    
    # Get S&P 500 list
    sp500 = SP500ListFetcher.get_sp500_list(client=client)
    
    while True:
        display_menu()
        choice = input("\nChoice (1-7): ").strip()
        
        if choice == '1':
            symbol = input("Stock symbol (e.g., AAPL): ").strip().upper()
            if symbol:
                data = fetch_stock_data(client, symbol)
                if data:
                    display_analysis(data, analyze_stock(data))
        
        elif choice == '2':
            n = int(input("How many stocks? (default 10): ").strip() or "10")
            scan_stocks(client, sp500, min(max(1, n), len(sp500)))
        
        elif choice == '3':
            print(f"\n{Style.BRIGHT}S&P 500 Stocks ({len(sp500)}):{Style.RESET_ALL}")
            for i in range(0, min(50, len(sp500)), 10):
                print("  " + ", ".join(sp500[i:i+10]))
            if len(sp500) > 50:
                print(f"  ... and {len(sp500) - 50} more")
        
        elif choice == '4':
            sectors = SP500ListFetcher.get_sp500_by_sector()
            if sectors:
                for sector, tickers in sorted(sectors.items()):
                    print(f"\n{Fore.CYAN}{sector} ({len(tickers)}){Style.RESET_ALL}")
                    print("  " + ", ".join(tickers[:10]) + ("..." if len(tickers) > 10 else ""))
        
        elif choice == '5':
            sp500 = SP500ListFetcher.refresh_list(client=client)
            print(f"{Fore.GREEN}✓ Refreshed: {len(sp500)} stocks{Style.RESET_ALL}")
        
        elif choice == '6':
            symbols = input("Symbols (comma-separated, e.g., AAPL,MSFT,GOOGL): ").strip().upper()
            symbols = [s.strip() for s in symbols.split(',') if s.strip()]
            if symbols:
                compare_stocks(client, symbols)
        
        elif choice == '7':
            print(f"\n{Fore.CYAN}Goodbye!{Style.RESET_ALL}")
            break


if __name__ == "__main__":
    main()