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
from typing import Dict, List, Any
import json
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
# ETF LIST MANAGER
# ============================================
class ETFListManager:
    """Manage popular ETF lists by category."""

    # Comprehensive ETF list organized by category
    ETF_CATEGORIES = {
        'Broad Market': {
            'SPY': 'S&P 500 ETF Trust',
            'VOO': 'Vanguard S&P 500 ETF',
            'IVV': 'iShares Core S&P 500 ETF',
            'QQQ': 'Invesco QQQ (Nasdaq-100)',
            'VTI': 'Vanguard Total Stock Market ETF',
            'DIA': 'SPDR Dow Jones Industrial Average ETF',
            'IWM': 'iShares Russell 2000 ETF',
            'VEA': 'Vanguard FTSE Developed Markets ETF',
            'VWO': 'Vanguard FTSE Emerging Markets ETF',
        },
        'Technology': {
            'XLK': 'Technology Select Sector SPDR Fund',
            'VGT': 'Vanguard Information Technology ETF',
            'SOXX': 'iShares Semiconductor ETF',
            'ARKK': 'ARK Innovation ETF',
            'IGV': 'iShares Expanded Tech-Software Sector ETF',
            'WCLD': 'WisdomTree Cloud Computing Fund',
        },
        'Growth & Value': {
            'VUG': 'Vanguard Growth ETF',
            'IWF': 'iShares Russell 1000 Growth ETF',
            'VTV': 'Vanguard Value ETF',
            'IWD': 'iShares Russell 1000 Value ETF',
            'SCHG': 'Schwab U.S. Large-Cap Growth ETF',
        },
        'Sector Specific': {
            'XLF': 'Financial Select Sector SPDR Fund',
            'XLE': 'Energy Select Sector SPDR Fund',
            'XLV': 'Health Care Select Sector SPDR Fund',
            'XLI': 'Industrial Select Sector SPDR Fund',
            'XLP': 'Consumer Staples Select Sector SPDR Fund',
            'XLY': 'Consumer Discretionary Select Sector SPDR',
            'XLU': 'Utilities Select Sector SPDR Fund',
            'XLRE': 'Real Estate Select Sector SPDR Fund',
            'XLB': 'Materials Select Sector SPDR Fund',
            'XLC': 'Communication Services Select Sector SPDR',
        },
        'Bond ETFs': {
            'AGG': 'iShares Core U.S. Aggregate Bond ETF',
            'BND': 'Vanguard Total Bond Market ETF',
            'TLT': 'iShares 20+ Year Treasury Bond ETF',
            'IEF': 'iShares 7-10 Year Treasury Bond ETF',
            'LQD': 'iShares iBoxx Investment Grade Corporate',
            'HYG': 'iShares iBoxx High Yield Corporate Bond',
            'MUB': 'iShares National Muni Bond ETF',
            'TIP': 'iShares TIPS Bond ETF',
        },
        'Dividend': {
            'VYM': 'Vanguard High Dividend Yield ETF',
            'SCHD': 'Schwab U.S. Dividend Equity ETF',
            'DVY': 'iShares Select Dividend ETF',
            'NOBL': 'ProShares S&P 500 Dividend Aristocrats',
            'SDY': 'SPDR S&P Dividend ETF',
            'VIG': 'Vanguard Dividend Appreciation ETF',
        },
        'International': {
            'EFA': 'iShares MSCI EAFE ETF',
            'IEFA': 'iShares Core MSCI EAFE ETF',
            'EEM': 'iShares MSCI Emerging Markets ETF',
            'IEMG': 'iShares Core MSCI Emerging Markets ETF',
            'FXI': 'iShares China Large-Cap ETF',
            'EWJ': 'iShares MSCI Japan ETF',
        },
        'Commodities & Real Assets': {
            'GLD': 'SPDR Gold Shares',
            'SLV': 'iShares Silver Trust',
            'USO': 'United States Oil Fund',
            'DBC': 'Invesco DB Commodity Index Tracking Fund',
            'VNQ': 'Vanguard Real Estate ETF',
            'PDBC': 'Invesco Optimum Yield Diversified Commodity',
        },
        'Thematic & Innovation': {
            'ARKW': 'ARK Next Generation Internet ETF',
            'ARKF': 'ARK Fintech Innovation ETF',
            'ARKG': 'ARK Genomic Revolution ETF',
            'BOTZ': 'Global X Robotics & AI ETF',
            'ICLN': 'iShares Global Clean Energy ETF',
            'TAN': 'Invesco Solar ETF',
            'LIT': 'Global X Lithium & Battery Tech ETF',
        },
        'Low Volatility': {
            'SPLV': 'Invesco S&P 500 Low Volatility ETF',
            'USMV': 'iShares MSCI USA Min Vol Factor ETF',
            'EEMV': 'iShares MSCI Emerging Markets Min Vol',
        },
    }

    @classmethod
    def get_all_etfs(cls) -> list[str]:
        """Get list of all ETF symbols."""
        etfs = []
        for category in cls.ETF_CATEGORIES.values():
            etfs.extend(category.keys())
        return sorted(etfs)

    @classmethod
    def get_etfs_by_category(cls, category: str) -> dict:
        """Get ETFs in a specific category."""
        return cls.ETF_CATEGORIES.get(category, {})

    @classmethod
    def get_all_categories(cls) -> list[str]:
        """Get list of all ETF categories."""
        return list(cls.ETF_CATEGORIES.keys())

    @classmethod
    def find_etf_category(cls, symbol: str) -> str:
        """Find which category an ETF belongs to."""
        for category, etfs in cls.ETF_CATEGORIES.items():
            if symbol.upper() in etfs:
                return category
        return 'Unknown'

    @classmethod
    def get_etf_name(cls, symbol: str) -> str:
        """Get the full name of an ETF."""
        for category in cls.ETF_CATEGORIES.values():
            if symbol.upper() in category:
                return category[symbol.upper()]
        return symbol

    @classmethod
    def display_all_etfs(cls):
        """Display all ETFs organized by category."""
        print(f"\n{Style.BRIGHT}{'=' * 60}")
        print(f"POPULAR ETF LIST - {len(cls.get_all_etfs())} ETFs")
        print(f"{'=' * 60}{Style.RESET_ALL}\n")

        for category, etfs in cls.ETF_CATEGORIES.items():
            print(f"{Fore.CYAN}{Style.BRIGHT}{category} ({len(etfs)} ETFs):{Style.RESET_ALL}")
            for symbol, name in etfs.items():
                print(f"  {symbol:6} - {name}")
            print()


# ============================================
# Config data from file
# ============================================

class ConfigManager:
    """Manager for reading and accessing configuration from config.json"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the config manager with the path to config.json
        
        Args:
            config_path: Path to the config.json file (default: "config.json")
        """
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file
        
        Returns:
            Dictionary containing all configuration values
        """
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            print(f"✓ Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            print(f"Error: {self.config_path} not found")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {self.config_path}: {e}")
            return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key
        
        Args:
            key: Configuration key to retrieve
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    # Alpaca API credentials
    @property
    def api_key(self) -> str:
        """Get Alpaca API key"""
        return self.config.get('api_key', '')
    
    @property
    def api_secret(self) -> str:
        """Get Alpaca API secret"""
        return self.config.get('api_secret', '')
    
    @property
    def base_url(self) -> str:
        """Get Alpaca base URL"""
        return self.config.get('base_url', 'https://paper-api.alpaca.markets')
    
    # Trading symbols
    @property
    def symbols(self) -> List[str]:
        """Get list of trading symbols"""
        return self.config.get('symbols', [])
    
    # Strategy settings
    @property
    def strategy(self) -> str:
        """Get trading strategy name"""
        return self.config.get('strategy', 'sma_crossover')
    
    @property
    def check_interval(self) -> int:
        """Get check interval in seconds"""
        return self.config.get('check_interval', 60)
    
    @property
    def position_size(self) -> float:
        """Get position size as fraction of portfolio"""
        return self.config.get('position_size', 0.1)
    
    @property
    def max_positions(self) -> int:
        """Get maximum number of positions"""
        return self.config.get('max_positions', 9)
    
    @property
    def timeframe(self) -> str:
        """Get timeframe for data (e.g., '5Min', '1Hour', '1Day')"""
        return self.config.get('timeframe', '5Min')
    
    @property
    def signal_mode(self) -> str:
        """Get signal mode ('position' or other)"""
        return self.config.get('signal_mode', 'position')
    
    @property
    def use_margin(self) -> bool:
        """Get whether to use margin trading"""
        return self.config.get('use_margin', False)
    
    # Massive API
    @property
    def massive_api(self) -> str:
        """Get Massive API key"""
        return self.config.get('massive_api', '')
    
    @property
    def long_term_stocks(self) -> List[str]:
        """Get list of long-term stocks"""
        return self.config.get('long_term_stocks', [])
    
    def save_config(self) -> bool:
        """Save current configuration back to JSON file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"✓ Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def update(self, key: str, value: Any) -> None:
        """Update a configuration value
        
        Args:
            key: Configuration key to update
            value: New value
        """
        self.config[key] = value
    
    def display_config(self) -> None:
        """Display all configuration values in a readable format"""
        print("\n=== Configuration ===")
        print(f"Alpaca API Key: {self.api_key[:10]}..." if self.api_key else "Not set")
        print(f"Alpaca API Secret: {self.api_secret[:10]}..." if self.api_secret else "Not set")
        print(f"Base URL: {self.base_url}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Strategy: {self.strategy}")
        print(f"Check Interval: {self.check_interval}s")
        print(f"Position Size: {self.position_size * 100}%")
        print(f"Max Positions: {self.max_positions}")
        print(f"Timeframe: {self.timeframe}")
        print(f"Signal Mode: {self.signal_mode}")
        print(f"Use Margin: {self.use_margin}")
        print(f"Massive API: {self.massive_api[:10]}..." if self.massive_api else "Not set")
        print(f"Long-term Stocks: {', '.join(self.long_term_stocks) if self.long_term_stocks else 'None'}")
        print("=" * 40)

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
        bb_upper = bb_lower = current_price
        stoch_k = stoch_d = 50
        adx = 20
        momentum = 0
        volatility = 0

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

                # Calculate new indicators
                bb_upper, _, bb_lower = calculate_bollinger_bands(closes, 20)
                stoch_k, stoch_d = calculate_stochastic(highs, lows, closes, 14)
                adx = calculate_adx(highs, lows, closes, 14)
                momentum = calculate_momentum(closes, 10)
                volatility = calculate_volatility(closes, 20)
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
            'bb_upper': bb_upper, 'bb_lower': bb_lower,
            'stoch_k': stoch_k, 'stoch_d': stoch_d,
            'adx': adx, 'momentum': momentum, 'volatility': volatility,
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


def calculate_bollinger_bands(prices: list, period: int = 20, num_std: float = 2.0) -> tuple:
    """Calculate Bollinger Bands (upper, middle, lower).

    Reference: John Bollinger's "Bollinger on Bollinger Bands" (2001)
    """
    if len(prices) < period:
        return 0, 0, 0

    # Calculate SMA (middle band)
    sma = sum(prices[-period:]) / period

    # Calculate standard deviation
    variance = sum((p - sma) ** 2 for p in prices[-period:]) / period
    std_dev = variance ** 0.5

    upper_band = sma + (num_std * std_dev)
    lower_band = sma - (num_std * std_dev)

    return upper_band, sma, lower_band


def calculate_stochastic(highs: list, lows: list, closes: list, period: int = 14) -> tuple:
    """Calculate Stochastic Oscillator (%K and %D).

    Reference: George Lane, 1950s
    """
    if len(closes) < period or len(highs) < period or len(lows) < period:
        return 50, 50

    # Get last 'period' values
    recent_highs = highs[-period:]
    recent_lows = lows[-period:]
    current_close = closes[-1]

    highest_high = max(recent_highs)
    lowest_low = min(recent_lows)

    if highest_high == lowest_low:
        return 50, 50

    # %K calculation
    k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100

    # %D is 3-period SMA of %K (simplified here as we'd need to calculate multiple %K values)
    # For simplicity, we'll calculate %K for last 3 periods and average
    k_values = []
    for i in range(max(0, len(closes) - 3), len(closes)):
        if i >= period - 1:
            h = max(highs[i-period+1:i+1])
            l = min(lows[i-period+1:i+1])
            if h != l:
                k_values.append(((closes[i] - l) / (h - l)) * 100)

    d_percent = sum(k_values) / len(k_values) if k_values else k_percent

    return k_percent, d_percent


def calculate_adx(highs: list, lows: list, closes: list, period: int = 14) -> float:
    """Calculate Average Directional Index (ADX) for trend strength.

    Reference: J. Welles Wilder Jr. - "New Concepts in Technical Trading Systems"
    Simplified version - returns value 0-100 where >25 indicates strong trend
    """
    if len(closes) < period + 1:
        return 20  # Neutral value

    # Calculate True Range and Directional Movement (simplified)
    tr_values = []
    plus_dm = []
    minus_dm = []

    for i in range(1, min(len(closes), period + 1)):
        high_diff = highs[-i] - highs[-i-1] if i < len(highs) else 0
        low_diff = lows[-i-1] - lows[-i] if i < len(lows) else 0

        plus_dm.append(high_diff if high_diff > low_diff and high_diff > 0 else 0)
        minus_dm.append(low_diff if low_diff > high_diff and low_diff > 0 else 0)

        high_low = highs[-i] - lows[-i]
        high_close = abs(highs[-i] - closes[-i-1]) if i < len(closes) else 0
        low_close = abs(lows[-i] - closes[-i-1]) if i < len(closes) else 0
        tr_values.append(max(high_low, high_close, low_close))

    # Calculate smoothed averages
    avg_tr = sum(tr_values) / len(tr_values) if tr_values else 1
    avg_plus_dm = sum(plus_dm) / len(plus_dm) if plus_dm else 0
    avg_minus_dm = sum(minus_dm) / len(minus_dm) if minus_dm else 0

    # Calculate DI+ and DI-
    di_plus = (avg_plus_dm / avg_tr * 100) if avg_tr > 0 else 0
    di_minus = (avg_minus_dm / avg_tr * 100) if avg_tr > 0 else 0

    # Calculate DX and ADX (simplified)
    di_sum = di_plus + di_minus
    if di_sum == 0:
        return 20

    dx = abs(di_plus - di_minus) / di_sum * 100

    return dx  # Simplified ADX approximation


def calculate_momentum(prices: list, period: int = 10) -> float:
    """Calculate price momentum (Rate of Change).

    Returns percentage change over period
    """
    if len(prices) < period + 1:
        return 0

    current_price = prices[-1]
    past_price = prices[-period-1]

    if past_price == 0:
        return 0

    return ((current_price - past_price) / past_price) * 100


def calculate_volatility(prices: list, period: int = 20) -> float:
    """Calculate volatility as standard deviation of returns."""
    if len(prices) < period + 1:
        return 0

    recent_prices = prices[-period-1:]
    returns = [(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
               for i in range(1, len(recent_prices))]

    if not returns:
        return 0

    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)

    return (variance ** 0.5) * 100  # Return as percentage


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

    # NEW INDICATORS:

    # 1. Volume Analysis (±8 points)
    if data['volume'] > data['avg_volume'] * 1.5:
        score += 5
        signals.append(('bullish', f'High volume ({data["volume"]/data["avg_volume"]:.1f}x avg)'))
    elif data['volume'] > data['avg_volume'] * 2:
        score += 3  # Extra bonus for volume spike
        signals.append(('bullish', 'Volume spike detected'))

    # 2. Bollinger Bands (±6 points)
    if data['bb_upper'] > data['bb_lower']:
        bb_position = (data['current_price'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        if bb_position < 0.2:
            score += 6
            signals.append(('bullish', f'Near lower Bollinger Band ({bb_position*100:.1f}%)'))
        elif bb_position > 0.8:
            score -= 6
            signals.append(('bearish', f'Near upper Bollinger Band ({bb_position*100:.1f}%)'))

    # 3. Stochastic Oscillator (±5 points)
    stoch_k = data.get('stoch_k', 50)
    stoch_d = data.get('stoch_d', 50)
    if stoch_k < 20:
        if stoch_k > stoch_d:
            score += 5
            signals.append(('bullish', f'Stochastic oversold & turning up ({stoch_k:.1f})'))
        else:
            score += 3
            signals.append(('bullish', f'Stochastic oversold ({stoch_k:.1f})'))
    elif stoch_k > 80:
        if stoch_k < stoch_d:
            score -= 5
            signals.append(('bearish', f'Stochastic overbought & turning down ({stoch_k:.1f})'))
        else:
            score -= 3
            signals.append(('bearish', f'Stochastic overbought ({stoch_k:.1f})'))

    # 4. ADX Trend Strength (±5 points)
    adx = data.get('adx', 20)
    if adx > 25:
        # Strong trend - check if it's bullish or bearish
        if data['current_price'] > data['ma50']:
            score += 5
            signals.append(('bullish', f'Strong uptrend (ADX: {adx:.1f})'))
        else:
            score -= 5
            signals.append(('bearish', f'Strong downtrend (ADX: {adx:.1f})'))
    elif adx < 20:
        signals.append(('neutral', f'Weak/no trend (ADX: {adx:.1f})'))

    # 5. Momentum (±7 points)
    momentum = data.get('momentum', 0)
    if momentum > 5:
        score += 7
        signals.append(('bullish', f'Strong positive momentum (+{momentum:.1f}%)'))
    elif momentum > 2:
        score += 4
        signals.append(('bullish', f'Positive momentum (+{momentum:.1f}%)'))
    elif momentum < -5:
        score -= 7
        signals.append(('bearish', f'Strong negative momentum ({momentum:.1f}%)'))
    elif momentum < -2:
        score -= 4
        signals.append(('bearish', f'Negative momentum ({momentum:.1f}%)'))

    # 6. Volatility Score (±5 points)
    volatility = data.get('volatility', 0)
    if volatility < 2 and data['current_price'] > data['ma50']:
        score += 5
        signals.append(('bullish', f'Low volatility uptrend ({volatility:.1f}%)'))
    elif volatility > 5:
        score -= 3
        signals.append(('bearish', f'High volatility/risk ({volatility:.1f}%)'))

    # 7. MACD Histogram Trend (±4 points)
    macd_hist = data.get('macd_histogram', 0)
    if macd_hist > 0 and data['macd'] > data['macd_signal']:
        score += 4
        signals.append(('bullish', 'MACD momentum strengthening'))
    elif macd_hist < 0 and data['macd'] < data['macd_signal']:
        score -= 4
        signals.append(('bearish', 'MACD momentum weakening'))

    # 8. Multiple MA Alignment Bonus (±3 points)
    if (data['current_price'] > data['ema20'] > data['ma50'] > data['ma200']):
        score += 3
        signals.append(('bullish', 'Perfect MA alignment (bullish stack)'))
    elif (data['current_price'] < data['ema20'] < data['ma50'] < data['ma200']):
        score -= 3
        signals.append(('bearish', 'Perfect MA alignment (bearish stack)'))

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

    print(f"\n{Style.BRIGHT}Advanced Indicators:{Style.RESET_ALL}")
    print(tabulate([
        ['Bollinger Upper', f"${data.get('bb_upper', 0):.2f}"],
        ['Bollinger Lower', f"${data.get('bb_lower', 0):.2f}"],
        ['Stochastic %K', f"{data.get('stoch_k', 50):.1f}"],
        ['Stochastic %D', f"{data.get('stoch_d', 50):.1f}"],
        ['ADX (Trend)', f"{data.get('adx', 20):.1f}"],
        ['Momentum (10d)', f"{data.get('momentum', 0):+.2f}%"],
        ['Volatility', f"{data.get('volatility', 0):.2f}%"]
    ], tablefmt='simple'))

    print(f"\n{Style.BRIGHT}Signals:{Style.RESET_ALL}")
    for t, txt in analysis['signals']:
        icon = f"{Fore.GREEN}▲" if t == 'bullish' else f"{Fore.RED}▼" if t == 'bearish' else f"{Fore.YELLOW}●"
        print(f"  {icon}{Style.RESET_ALL} {txt}")
    
    print(f"\n{Fore.YELLOW}⚠ Educational purposes only. Consult a financial advisor.{Style.RESET_ALL}")


def display_menu():
    print(f"\n{Style.BRIGHT}{'=' * 50}")
    print("       S&P 500 STOCK ANALYZER v6.0")
    print("       Powered by Massive API - Enhanced Scoring")
    print(f"{'=' * 50}{Style.RESET_ALL}")
    print("\n{Fore.YELLOW}Stocks:{Style.RESET_ALL}")
    print("  1. Analyze a single stock/ETF")
    print("  2. Scan top stocks")
    print("  3. List S&P 500 stocks")
    print("  4. List by sector")
    print("  5. My personal list")
    print("\n{Fore.CYAN}ETFs:{Style.RESET_ALL}")
    print("  6. Browse all ETFs by category")
    print("  7. Scan ETFs by category")
    print("  8. Compare multiple ETFs")
    print("\n{Fore.GREEN}Other:{Style.RESET_ALL}")
    print("  9. Compare stocks/ETFs")
    print(" 10. Refresh stock list")
    print("  0. Exit")


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


def scan_etfs_by_category(client: RESTClient, category: str):
    """Scan all ETFs in a specific category."""
    etfs = ETFListManager.get_etfs_by_category(category)

    if not etfs:
        print(f"{Fore.RED}Category not found.{Style.RESET_ALL}")
        return

    print(f"\n{Style.BRIGHT}Scanning {category} ETFs ({len(etfs)} total)...{Style.RESET_ALL}")
    results = []

    for i, (symbol, name) in enumerate(etfs.items()):
        data = fetch_stock_data(client, symbol)
        if data:
            analysis = analyze_stock(data)
            results.append({
                'symbol': symbol,
                'name': name[:30],  # Truncate long names
                'price': data['current_price'],
                'change': data['change_percent'],
                'rsi': data['rsi'],
                'score': analysis['score'],
                'rec': analysis['recommendation']
            })
        print(f"  Progress: {i+1}/{len(etfs)}")

    if not results:
        print(f"{Fore.RED}No results.{Style.RESET_ALL}")
        return

    results.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n{Style.BRIGHT}{category} ETFs - Results (Ranked by Score):{Style.RESET_ALL}")
    table = [[r['symbol'], r['name'], f"${r['price']:.2f}",
              f"{'+' if r['change'] >= 0 else ''}{r['change']:.2f}%",
              f"{r['rsi']:.1f}", r['score'], r['rec']] for r in results]
    print(tabulate(table, headers=['Symbol', 'Name', 'Price', 'Change', 'RSI', 'Score', 'Rec'], tablefmt='grid'))


def browse_etf_categories():
    """Display ETF categories and let user choose one to explore."""
    categories = ETFListManager.get_all_categories()

    print(f"\n{Style.BRIGHT}ETF Categories:{Style.RESET_ALL}")
    for i, category in enumerate(categories, 1):
        etf_count = len(ETFListManager.get_etfs_by_category(category))
        print(f"  {i}. {category} ({etf_count} ETFs)")

    choice = input(f"\nSelect category (1-{len(categories)}) or 'all' to see full list: ").strip().lower()

    if choice == 'all':
        ETFListManager.display_all_etfs()
    elif choice.isdigit() and 1 <= int(choice) <= len(categories):
        category = categories[int(choice) - 1]
        etfs = ETFListManager.get_etfs_by_category(category)
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{category} ETFs:{Style.RESET_ALL}")
        for symbol, name in etfs.items():
            print(f"  {symbol:6} - {name}")
    else:
        print(f"{Fore.RED}Invalid choice.{Style.RESET_ALL}")


# ============================================
# MAIN PROGRAM
# ============================================
def main():
    print(f"\n{Fore.CYAN}{Style.BRIGHT}")
    print("  ╔═══════════════════════════════════════════╗")
    print("  ║  STOCK & ETF ANALYZER v6.0                ║")
    print("  ║  Powered by Massive API                   ║")
    print("  ║  Enhanced Multi-Factor Scoring            ║")
    print("  ║  80+ ETFs across 9 Categories             ║")
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
    
    #cfg data
    obj=ConfigManager()

    # Initialize client
    try:
        client = RESTClient(api_key=MASSIVE_API_KEY)
        print(f"{Fore.GREEN}✓ Massive API client initialized{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Failed to initialize client: {e}{Style.RESET_ALL}")
        return
    
    # Get S&P 500 list
    sp500 = SP500ListFetcher.get_sp500_list(client=client)
    
    #Get personal list
    personal_stock=obj.long_term_stocks

    while True:
        display_menu()
        choice = input("\nChoice: ").strip()

        if choice == '1':
            symbol = input("Stock/ETF symbol (e.g., AAPL, SPY): ").strip().upper()
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
            if personal_stock:
                scan_stocks(client, personal_stock, len(personal_stock))
            else:
                print(f"{Fore.YELLOW}No stocks in your personal list. Add them to config.json{Style.RESET_ALL}")

        elif choice == '6':
            browse_etf_categories()

        elif choice == '7':
            categories = ETFListManager.get_all_categories()
            print(f"\n{Style.BRIGHT}Select ETF Category to Scan:{Style.RESET_ALL}")
            for i, category in enumerate(categories, 1):
                etf_count = len(ETFListManager.get_etfs_by_category(category))
                print(f"  {i}. {category} ({etf_count} ETFs)")

            cat_choice = input(f"\nCategory number (1-{len(categories)}): ").strip()
            if cat_choice.isdigit() and 1 <= int(cat_choice) <= len(categories):
                category = categories[int(cat_choice) - 1]
                scan_etfs_by_category(client, category)
            else:
                print(f"{Fore.RED}Invalid choice.{Style.RESET_ALL}")

        elif choice == '8':
            symbols = input("ETF symbols (comma-separated, e.g., SPY,QQQ,VTI): ").strip().upper()
            symbols = [s.strip() for s in symbols.split(',') if s.strip()]
            if symbols:
                compare_stocks(client, symbols)

        elif choice == '9':
            symbols = input("Symbols (comma-separated, e.g., AAPL,SPY,MSFT): ").strip().upper()
            symbols = [s.strip() for s in symbols.split(',') if s.strip()]
            if symbols:
                compare_stocks(client, symbols)

        elif choice == '10':
            sp500 = SP500ListFetcher.refresh_list(client=client)
            print(f"{Fore.GREEN}✓ Refreshed: {len(sp500)} stocks{Style.RESET_ALL}")

        elif choice == '0':
            print(f"\n{Fore.CYAN}Goodbye!{Style.RESET_ALL}")
            break

        else:
            print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()