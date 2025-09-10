"""
Data Fetcher Module for SMC Trading Signals Generator
Handles data fetching, validation, and watchlist management
"""

import yfinance as yf
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple, Optional
import re


# Comprehensive watchlist covering major liquid instruments
WATCHLIST = {
    'forex': [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'AUDUSD=X', 'USDCAD=X',
        'NZDUSD=X', 'EURJPY=X', 'GBPJPY=X', 'EURGBP=X', 'AUDJPY=X', 'CHFJPY=X',
        'EURAUD=X', 'EURCHF=X', 'GBPCHF=X', 'AUDCAD=X', 'AUDCHF=X', 'CADCHF=X',
        'NZDJPY=X', 'EURNZD=X', 'GBPNZD=X', 'AUDNZD=X', 'CADJPY=X', 'NZDCHF=X'
    ],
    'stocks': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
        'UNH', 'JNJ', 'V', 'PG', 'JPM', 'XOM', 'HD', 'CVX', 'MA', 'PFE',
        'ABBV', 'BAC', 'KO', 'AVGO', 'PEP', 'TMO', 'COST', 'WMT', 'DHR',
        'VZ', 'ADBE', 'NFLX', 'CRM', 'ACN', 'TXN', 'QCOM', 'NKE', 'MRK',
        'ABT', 'T', 'HON', 'LIN', 'PM', 'UNP', 'RTX', 'LOW', 'SPGI'
    ],
    'crypto': [
        'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD',
        'DOGE-USD', 'DOT-USD', 'AVAX-USD', 'SHIB-USD', 'MATIC-USD', 'LTC-USD',
        'UNI-USD', 'LINK-USD', 'ATOM-USD', 'XLM-USD', 'BCH-USD', 'FIL-USD',
        'TRX-USD', 'ETC-USD', 'ALGO-USD', 'VET-USD', 'ICP-USD', 'THETA-USD',
        'FTM-USD', 'HBAR-USD', 'NEAR-USD', 'QNT-USD', 'FLOW-USD', 'MANA-USD'
    ],
    'indices': [
        '^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX', '^TNX', '^FVX', '^TYX',
        '^N225', '^HSI', '^FTSE', '^GDAXI', '^FCHI', '^STOXX50E', '^AEX',
        '^IBEX', '^MXX', '^BVSP', '^MERV', '^AXJO', '^NZ50', '^KS11',
        '^TWII', '^JKSE', '^SET.BK', '^NSEI', '^BSESN', '^KLSE'
    ],
    'commodities': [
        'GC=F', 'SI=F', 'CL=F', 'NG=F', 'PL=F', 'PA=F', 'HG=F', 'ZC=F',
        'ZS=F', 'ZW=F', 'KC=F', 'SB=F', 'CC=F', 'CT=F', 'LB=F', 'OJ=F',
        'HE=F', 'LE=F', 'GF=F', 'KE=F', 'RR=F', 'BO=F', 'SM=F', 'RS=F'
    ]
}


def detect_market_type(symbol: str) -> str:
    """
    Auto-detect market type from symbol format
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Market type string
    """
    symbol_upper = symbol.upper()
    
    if symbol_upper.endswith('=X'):
        return 'forex'
    elif symbol_upper.endswith('-USD'):
        return 'crypto'
    elif symbol_upper.startswith('^'):
        return 'indices'
    elif symbol_upper.endswith('=F'):
        return 'commodities'
    else:
        return 'stocks'


def validate_symbol(symbol: str) -> Tuple[bool, str]:
    """
    Validate if symbol exists and is tradeable
    
    Args:
        symbol: Trading symbol to validate
        
    Returns:
        Tuple of (is_valid, market_type)
    """
    try:
        # First check if symbol is in our known watchlist
        market_type = detect_market_type(symbol)
        if market_type != 'unknown':
            return True, market_type
        
        # If not in watchlist, try to fetch data with yfinance
        ticker = yf.Ticker(symbol)
        
        # Try to get basic info first
        info = ticker.info
        if info and 'symbol' in info:
            return True, market_type
        
        # If info fails, try to fetch recent data
        hist = ticker.history(period='5d', interval='1d')
        if not hist.empty:
            return True, market_type
            
        return False, 'unknown'
        
    except Exception:
        return False, 'unknown'


def get_symbol_examples() -> Dict[str, List[str]]:
    """
    Get example symbols for each market type
    
    Returns:
        Dictionary with market types as keys and example symbols as values
    """
    return {
        'Forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X'],
        'Stocks': ['AAPL', 'MSFT', 'GOOGL'],
        'Crypto': ['BTC-USD', 'ETH-USD', 'BNB-USD'],
        'Indices': ['^GSPC', '^DJI', '^IXIC'],
        'Commodities': ['GC=F', 'SI=F', 'CL=F']
    }


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_ohlcv(symbol: str, period: str = '1mo', interval: str = '1h') -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data using yfinance with caching
    
    Args:
        symbol: Trading symbol
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            return None
            
        # Ensure proper column names
        df.columns = [col.lower() for col in df.columns]
        
        # Add volume if missing (some instruments don't have volume)
        if 'volume' not in df.columns:
            df['volume'] = 0
            
        # Remove timezone info for consistency
        df.index = df.index.tz_localize(None)
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None


def get_multi_timeframe_data(symbol: str, htf_interval: str = '1d', ltf_interval: str = '1h') -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Fetch data for both higher timeframe (bias) and lower timeframe (signals)
    
    Args:
        symbol: Trading symbol
        htf_interval: Higher timeframe interval for bias determination
        ltf_interval: Lower timeframe interval for signal generation
        
    Returns:
        Tuple of (htf_data, ltf_data)
    """
    # Determine period based on intervals
    if htf_interval in ['1d', '1wk']:
        period = '2y'
    elif htf_interval in ['4h', '1h']:
        period = '1y'
    else:
        period = '6mo'
    
    htf_data = fetch_ohlcv(symbol, period=period, interval=htf_interval)
    ltf_data = fetch_ohlcv(symbol, period=period, interval=ltf_interval)
    
    return htf_data, ltf_data


def get_watchlist_symbols(market_type: str = 'all') -> List[str]:
    """
    Get symbols from watchlist by market type
    
    Args:
        market_type: Market type filter ('all', 'forex', 'stocks', etc.)
        
    Returns:
        List of symbols
    """
    if market_type == 'all':
        all_symbols = []
        for symbols in WATCHLIST.values():
            all_symbols.extend(symbols)
        return all_symbols
    else:
        return WATCHLIST.get(market_type.lower(), [])


def update_watchlist(new_watchlist: Dict[str, List[str]]) -> bool:
    """
    Update the watchlist (for future enhancement)
    
    Args:
        new_watchlist: New watchlist dictionary
        
    Returns:
        Success status
    """
    try:
        # Validate the new watchlist structure
        required_keys = ['forex', 'stocks', 'crypto', 'indices', 'commodities']
        if not all(key in new_watchlist for key in required_keys):
            return False
            
        # Validate symbols
        for market_type, symbols in new_watchlist.items():
            if not isinstance(symbols, list):
                return False
            for symbol in symbols:
                if not isinstance(symbol, str) or not symbol.strip():
                    return False
                    
        # Update global watchlist
        global WATCHLIST
        WATCHLIST = new_watchlist
        return True
        
    except Exception:
        return False


def get_market_hours_info(symbol: str) -> Dict[str, str]:
    """
    Get market hours information for a symbol
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Dictionary with market hours info
    """
    market_type = detect_market_type(symbol)
    
    market_hours = {
        'forex': '24/5 (Sunday 5 PM - Friday 5 PM EST)',
        'crypto': '24/7',
        'stocks': '9:30 AM - 4:00 PM EST (Mon-Fri)',
        'indices': '9:30 AM - 4:00 PM EST (Mon-Fri)',
        'commodities': 'Varies by commodity (typically 6 PM - 5 PM EST)'
    }
    
    return {
        'market_type': market_type.title(),
        'trading_hours': market_hours.get(market_type, 'Unknown'),
        'timezone': 'EST/EDT'
    }
