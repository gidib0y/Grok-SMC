"""
Utilities and Risk Management Module for SMC Trading Signals Generator
Handles position sizing, risk calculations, and utility functions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import streamlit as st


def calc_position_size(account_balance: float, risk_percentage: float, entry_price: float, stop_loss: float, market_type: str = 'forex') -> Dict[str, float]:
    """
    Calculate position size based on risk management rules
    
    Args:
        account_balance: Total account balance
        risk_percentage: Risk percentage (e.g., 0.01 for 1%)
        entry_price: Entry price
        stop_loss: Stop loss price
        market_type: Market type for appropriate scaling
        
    Returns:
        Dictionary with position size calculations
    """
    risk_amount = account_balance * risk_percentage
    price_risk = abs(entry_price - stop_loss)
    
    if price_risk == 0:
        return {'size': 0, 'risk_amount': 0, 'risk_per_share': 0}
    
    # For crypto, scale position size appropriately
    if market_type.lower() == 'crypto':
        # Crypto prices are much higher, so we need to scale down position size
        position_size = (risk_amount / price_risk) * 0.01  # Scale down by 100x
    else:
        position_size = risk_amount / price_risk
    
    risk_per_share = price_risk
    
    return {
        'size': position_size,
        'risk_amount': risk_amount,
        'risk_per_share': risk_per_share,
        'position_value': position_size * entry_price
    }


def calculate_risk_reward_ratio(entry: float, stop_loss: float, take_profit: float, direction: str) -> float:
    """
    Calculate risk-reward ratio
    
    Args:
        entry: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        direction: 'Buy' or 'Sell'
        
    Returns:
        Risk-reward ratio
    """
    if direction == 'Buy':
        risk = entry - stop_loss
        reward = take_profit - entry
    else:  # Sell
        risk = stop_loss - entry
        reward = entry - take_profit
    
    if risk <= 0:
        return 0
    
    return reward / risk


def validate_signal_quality(signal: Dict) -> Dict[str, bool]:
    """
    Validate signal quality and confluences
    
    Args:
        signal: Trading signal dictionary
        
    Returns:
        Dictionary with validation results
    """
    validations = {
        'has_entry': signal.get('entry', 0) > 0,
        'has_stop_loss': signal.get('sl', 0) > 0,
        'has_take_profit': signal.get('tp', 0) > 0,
        'min_confluences': signal.get('confluences', 0) >= 3,
        'valid_rr': signal.get('rr', 0) >= 1.0,
        'has_rationale': bool(signal.get('rationale', '').strip())
    }
    
    validations['overall_valid'] = all(validations.values())
    
    return validations


def format_price(price: float, market_type: str = 'forex') -> str:
    """
    Format price based on market type
    
    Args:
        price: Price to format
        market_type: Market type for appropriate decimal places
        
    Returns:
        Formatted price string
    """
    if market_type == 'forex':
        return f"{price:.5f}"
    elif market_type == 'crypto':
        return f"{price:.2f}"
    elif market_type == 'stocks':
        return f"${price:.2f}"
    elif market_type == 'commodities':
        return f"${price:.2f}"
    else:
        return f"{price:.4f}"


def calculate_slippage_and_commission(entry_price: float, market_type: str, position_size: float) -> Dict[str, float]:
    """
    Calculate estimated slippage and commission costs
    
    Args:
        entry_price: Entry price
        market_type: Market type
        position_size: Position size
        
    Returns:
        Dictionary with cost estimates
    """
    # Base slippage and commission rates
    rates = {
        'forex': {'slippage': 0.0001, 'commission': 0.0001},
        'crypto': {'slippage': 0.0005, 'commission': 0.001},
        'stocks': {'slippage': 0.0002, 'commission': 0.005},
        'commodities': {'slippage': 0.0003, 'commission': 0.0002},
        'indices': {'slippage': 0.0002, 'commission': 0.0001}
    }
    
    rate = rates.get(market_type, rates['forex'])
    
    slippage_cost = entry_price * rate['slippage'] * position_size
    commission_cost = entry_price * rate['commission'] * position_size
    
    return {
        'slippage': slippage_cost,
        'commission': commission_cost,
        'total_cost': slippage_cost + commission_cost
    }


def generate_alert_message(signal: Dict, symbol: str) -> str:
    """
    Generate alert message for signal
    
    Args:
        signal: Trading signal
        symbol: Trading symbol
        
    Returns:
        Formatted alert message
    """
    signal_type = signal.get('type', 'Unknown')
    direction = signal.get('direction', 'Unknown')
    # Clean and convert numeric values, removing currency symbols
    entry = float(str(signal.get('entry', 0)).replace('$', '').replace(',', ''))
    sl = float(str(signal.get('sl', 0)).replace('$', '').replace(',', ''))
    tp = float(str(signal.get('tp', 0)).replace('$', '').replace(',', ''))
    rr = float(str(signal.get('rr', 0)).replace('$', '').replace(',', ''))
    confluences = int(signal.get('confluences', 0))
    
    if signal_type == 'Market':
        return f"🚨 EXECUTE NOW: {direction} {symbol} at {entry:.5f} | SL: {sl:.5f} | TP: {tp:.5f} | RR: {rr:.2f} | Confluences: {confluences}"
    else:
        return f"⏳ PENDING: Set {direction} Limit at {entry:.5f} for {symbol} | SL: {sl:.5f} | TP: {tp:.5f} | RR: {rr:.2f} | Confluences: {confluences}"


def simulate_alert(signal: Dict, symbol: str):
    """
    Simulate alert (print for now, can be extended to email/SMS)
    
    Args:
        signal: Trading signal
        symbol: Trading symbol
    """
    alert_message = generate_alert_message(signal, symbol)
    print(f"ALERT: {alert_message}")
    
    # Future enhancement: Send email via smtplib
    # Future enhancement: Send SMS via Twilio
    # Future enhancement: Send to Discord/Slack webhook


def get_market_session_info() -> Dict[str, str]:
    """
    Get current market session information
    
    Returns:
        Dictionary with session info
    """
    import datetime
    
    now = datetime.datetime.now()
    hour = now.hour
    
    # Market sessions (EST/EDT)
    if 0 <= hour < 8:
        session = "Asian"
        status = "Open" if 0 <= hour < 6 else "Closing"
    elif 8 <= hour < 16:
        session = "London"
        status = "Open" if 8 <= hour < 12 else "Closing"
    elif 16 <= hour < 22:
        session = "New York"
        status = "Open" if 16 <= hour < 20 else "Closing"
    else:
        session = "Asian"
        status = "Opening"
    
    return {
        'session': session,
        'status': status,
        'time': now.strftime("%H:%M:%S"),
        'date': now.strftime("%Y-%m-%d")
    }


def calculate_portfolio_metrics(trades: List[Dict]) -> Dict[str, float]:
    """
    Calculate portfolio performance metrics
    
    Args:
        trades: List of trade dictionaries with P&L
        
    Returns:
        Dictionary with performance metrics
    """
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_rr': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
    
    pnls = [trade.get('pnl', 0) for trade in trades]
    winning_trades = [pnl for pnl in pnls if pnl > 0]
    losing_trades = [pnl for pnl in pnls if pnl < 0]
    
    total_trades = len(trades)
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    total_pnl = sum(pnls)
    
    # Calculate average R:R
    rrs = [trade.get('rr', 0) for trade in trades if trade.get('rr', 0) > 0]
    avg_rr = np.mean(rrs) if rrs else 0
    
    # Calculate max drawdown
    cumulative_pnl = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = running_max - cumulative_pnl
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # Calculate Sharpe ratio (simplified)
    if len(pnls) > 1:
        sharpe_ratio = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0
    else:
        sharpe_ratio = 0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_rr': avg_rr,
        'total_pnl': total_pnl,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades)
    }


def export_signals_to_csv(signals: List[Dict], filename: str = 'smc_signals.csv'):
    """
    Export signals to CSV file
    
    Args:
        signals: List of signal dictionaries
        filename: Output filename
        
    Returns:
        Success status
    """
    try:
        if not signals:
            return False
        
        df = pd.DataFrame(signals)
        df.to_csv(filename, index=False)
        return True
        
    except Exception as e:
        st.error(f"Error exporting signals: {str(e)}")
        return False


def get_timeframe_multiplier(timeframe: str) -> int:
    """
    Get multiplier for timeframe conversion
    
    Args:
        timeframe: Timeframe string (1h, 4h, 1d, etc.)
        
    Returns:
        Multiplier for calculations
    """
    multipliers = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '4h': 240,
        '1d': 1440,
        '1wk': 10080
    }
    
    return multipliers.get(timeframe, 60)


def validate_timeframe(timeframe: str) -> bool:
    """
    Validate if timeframe is supported
    
    Args:
        timeframe: Timeframe string
        
    Returns:
        True if valid, False otherwise
    """
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1wk']
    return timeframe in valid_timeframes


def get_optimal_timeframe(market_type: str) -> str:
    """
    Get optimal timeframe for market type
    
    Args:
        market_type: Market type
        
    Returns:
        Recommended timeframe
    """
    recommendations = {
        'forex': '1h',
        'crypto': '4h',
        'stocks': '1d',
        'indices': '1d',
        'commodities': '4h'
    }
    
    return recommendations.get(market_type, '1h')


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage
    
    Args:
        value: Value to format
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def get_risk_color(risk_level: float) -> str:
    """
    Get color for risk level visualization
    
    Args:
        risk_level: Risk percentage (0-1)
        
    Returns:
        Color string
    """
    if risk_level <= 0.01:
        return "green"
    elif risk_level <= 0.02:
        return "orange"
    else:
        return "red"


def create_signal_summary(signals: List[Dict]) -> Dict[str, int]:
    """
    Create summary of signals by type and direction
    
    Args:
        signals: List of signals
        
    Returns:
        Summary dictionary
    """
    summary = {
        'total': len(signals),
        'market_buy': 0,
        'market_sell': 0,
        'limit_buy': 0,
        'limit_sell': 0,
        'high_confluence': 0  # >= 4 confluences
    }
    
    for signal in signals:
        signal_type = signal.get('type', '')
        direction = signal.get('direction', '')
        confluences = signal.get('confluences', 0)
        
        if signal_type == 'Market' and direction == 'Buy':
            summary['market_buy'] += 1
        elif signal_type == 'Market' and direction == 'Sell':
            summary['market_sell'] += 1
        elif signal_type == 'Limit' and direction == 'Buy':
            summary['limit_buy'] += 1
        elif signal_type == 'Limit' and direction == 'Sell':
            summary['limit_sell'] += 1
        
        if confluences >= 4:
            summary['high_confluence'] += 1
    
    return summary
