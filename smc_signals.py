"""
SMC (Smart Money Concepts) Signal Generation Module
Implements rule-based trading signals using market structure, FVG, order blocks, and liquidity concepts
Fixed indentation errors permanently - All syntax errors resolved - Confidence scoring added
"""

import pandas as pd
import numpy as np
import talib
from scipy.signal import argrelextrema
from typing import Dict, List, Tuple, Optional
import streamlit as st
from order_flow_analyzer import OrderFlowAnalyzer


class SMCSignalGenerator:
    """
    Smart Money Concepts signal generator with rule-based logic
    """
    
    def __init__(self, market_type: str = 'forex'):
        self.market_type = market_type.lower()
        self.fvg_threshold = 0.001  # 0.1% default
        if self.market_type == 'crypto':
            self.fvg_threshold = 0.005  # 0.5% for crypto
        
        # Initialize order flow analyzer
        self.order_flow_analyzer = OrderFlowAnalyzer(market_type)
        
    def _get_optimal_entry_timeframe(self, analysis_timeframe: str) -> str:
        """
        Automatically determine optimal entry timeframe based on analysis timeframe
        
        Args:
            analysis_timeframe: The timeframe used for analysis ('1h', '4h', '1d')
            
        Returns:
            Optimal entry timeframe for precise execution
        """
        # Professional multi-timeframe mapping
        timeframe_mapping = {
            '1d': '15m',    # Daily analysis → 15min entry (institutional standard)
            '4h': '5m',     # 4-hour analysis → 5min entry (professional day trading)
            '1h': '1m'      # 1-hour analysis → 1min entry (scalping precision)
        }
        
        return timeframe_mapping.get(analysis_timeframe, '15m')
        
    def identify_swing_points(self, df: pd.DataFrame, order: int = 5) -> Tuple[List[int], List[int]]:
        """
        Identify swing highs and lows using scipy argrelextrema
        
        Args:
            df: OHLCV DataFrame
            order: Minimum distance between swing points
            
        Returns:
            Tuple of (swing_highs, swing_lows) indices
        """
        highs = df['high'].values
        lows = df['low'].values
        
        # Find swing highs and lows
        swing_highs = argrelextrema(highs, np.greater, order=order)[0]
        swing_lows = argrelextrema(lows, np.less, order=order)[0]
        
        return swing_highs.tolist(), swing_lows.tolist()
    
    def determine_bias(self, df: pd.DataFrame, swing_highs: List[int], swing_lows: List[int]) -> str:
        """
        Determine market bias based on market structure
        
        Args:
            df: OHLCV DataFrame
            swing_highs: List of swing high indices
            swing_lows: List of swing low indices
            
        Returns:
            Market bias: 'Bullish', 'Bearish', or 'Neutral'
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'Neutral'
        
        # Get recent swing points (last 4)
        recent_highs = swing_highs[-4:] if len(swing_highs) >= 4 else swing_highs
        recent_lows = swing_lows[-4:] if len(swing_lows) >= 4 else swing_lows
        
        # Check for Higher Highs and Higher Lows (Bullish)
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            latest_high = df['high'].iloc[recent_highs[-1]]
            prev_high = df['high'].iloc[recent_highs[-2]]
            latest_low = df['low'].iloc[recent_lows[-1]]
            prev_low = df['low'].iloc[recent_lows[-2]]
            
            if latest_high > prev_high and latest_low > prev_low:
                return 'Bullish'
            elif latest_high < prev_high and latest_low < prev_low:
                return 'Bearish'
        
        return 'Neutral'
    
    def identify_fvg(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Identify Fair Value Gaps (FVG)
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Dictionary with bullish and bearish FVGs
        """
        bullish_fvgs = []
        bearish_fvgs = []
        
        for i in range(2, len(df)):
            # Bullish FVG: low[0] > high[2]
            if df['low'].iloc[i] > df['high'].iloc[i-2]:
                gap_size = (df['low'].iloc[i] - df['high'].iloc[i-2]) / df['close'].iloc[i-2]
                if gap_size > self.fvg_threshold:
                    bullish_fvgs.append({
                        'index': i,
                        'top': df['low'].iloc[i],
                        'bottom': df['high'].iloc[i-2],
                        'size': gap_size,
                        'mitigated': False
                    })
            
            # Bearish FVG: high[0] < low[2]
            if df['high'].iloc[i] < df['low'].iloc[i-2]:
                gap_size = (df['low'].iloc[i-2] - df['high'].iloc[i]) / df['close'].iloc[i-2]
                if gap_size > self.fvg_threshold:
                    bearish_fvgs.append({
                        'index': i,
                        'top': df['low'].iloc[i-2],
                        'bottom': df['high'].iloc[i],
                        'size': gap_size,
                        'mitigated': False
                    })
        
        # Check for mitigation
        self._check_fvg_mitigation(df, bullish_fvgs, bearish_fvgs)
        
        return {'bullish': bullish_fvgs, 'bearish': bearish_fvgs}
    
    def _check_fvg_mitigation(self, df: pd.DataFrame, bullish_fvgs: List[Dict], bearish_fvgs: List[Dict]):
        """Check if FVGs have been mitigated"""
        for fvg in bullish_fvgs:
            for i in range(fvg['index'] + 1, len(df)):
                if df['low'].iloc[i] <= fvg['bottom']:
                    fvg['mitigated'] = True
                    break
        
        for fvg in bearish_fvgs:
            for i in range(fvg['index'] + 1, len(df)):
                if df['high'].iloc[i] >= fvg['top']:
                    fvg['mitigated'] = True
                    break
    
    def identify_order_blocks(self, df: pd.DataFrame, swing_highs: List[int], swing_lows: List[int]) -> Dict[str, List[Dict]]:
        """
        Identify Order Blocks (OB)
        
        Args:
            df: OHLCV DataFrame
            swing_highs: List of swing high indices
            swing_lows: List of swing low indices
            
        Returns:
            Dictionary with bullish and bearish order blocks
        """
        bullish_obs = []
        bearish_obs = []
        
        # Calculate average volume for volume filter
        avg_volume = df['volume'].rolling(20).mean() if self.market_type == 'stocks' else df['volume'].rolling(10).mean()
        volume_threshold = avg_volume * 1.5
        
        for i in range(10, len(df) - 5):
            # Bullish OB: Last bearish candle before bullish impulse
            if (df['close'].iloc[i] < df['open'].iloc[i] and  # Bearish candle
                df['close'].iloc[i+1] > df['open'].iloc[i+1] and  # Next candle bullish
                df['close'].iloc[i+1] > df['high'].iloc[i] and  # Break above bearish candle
                df['volume'].iloc[i] > volume_threshold.iloc[i]):  # Volume filter
                
                bullish_obs.append({
                    'index': i,
                    'high': df['high'].iloc[i],
                    'low': df['low'].iloc[i],
                    'mitigated': False
                })
            
            # Bearish OB: Last bullish candle before bearish impulse
            if (df['close'].iloc[i] > df['open'].iloc[i] and  # Bullish candle
                df['close'].iloc[i+1] < df['open'].iloc[i+1] and  # Next candle bearish
                df['close'].iloc[i+1] < df['low'].iloc[i] and  # Break below bullish candle
                df['volume'].iloc[i] > volume_threshold.iloc[i]):  # Volume filter
                
                bearish_obs.append({
                    'index': i,
                    'high': df['high'].iloc[i],
                    'low': df['low'].iloc[i],
                    'mitigated': False
                })
        
        # Check for mitigation
        self._check_ob_mitigation(df, bullish_obs, bearish_obs)
        
        return {'bullish': bullish_obs, 'bearish': bearish_obs}
    
    def _check_ob_mitigation(self, df: pd.DataFrame, bullish_obs: List[Dict], bearish_obs: List[Dict]):
        """Check if Order Blocks have been mitigated"""
        for ob in bullish_obs:
            for i in range(ob['index'] + 1, len(df)):
                if df['close'].iloc[i] <= ob['low']:
                    ob['mitigated'] = True
                    break
        
        for ob in bearish_obs:
            for i in range(ob['index'] + 1, len(df)):
                if df['close'].iloc[i] >= ob['high']:
                    ob['mitigated'] = True
                    break
    
    def identify_liquidity_grabs(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Identify Liquidity Grabs
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Dictionary with bullish and bearish liquidity grabs
        """
        bullish_grabs = []
        bearish_grabs = []
        
        for i in range(5, len(df)):
            body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
            lower_wick = df['open'].iloc[i] - df['low'].iloc[i] if df['close'].iloc[i] > df['open'].iloc[i] else df['close'].iloc[i] - df['low'].iloc[i]
            upper_wick = df['high'].iloc[i] - df['open'].iloc[i] if df['close'].iloc[i] < df['open'].iloc[i] else df['high'].iloc[i] - df['close'].iloc[i]
            
            # Bullish liquidity grab: Lower wick > 2x body and close > open
            if (lower_wick > 2 * body_size and 
                df['close'].iloc[i] > df['open'].iloc[i] and
                df['close'].iloc[i] > df['close'].iloc[i-1]):  # Sweep and recovery
                
                bullish_grabs.append({
                    'index': i,
                    'price': df['low'].iloc[i],
                    'recovery_price': df['close'].iloc[i]
                })
            
            # Bearish liquidity grab: Upper wick > 2x body and close < open
            if (upper_wick > 2 * body_size and 
                df['close'].iloc[i] < df['open'].iloc[i] and
                df['close'].iloc[i] < df['close'].iloc[i-1]):  # Sweep and rejection
                
                bearish_grabs.append({
                    'index': i,
                    'price': df['high'].iloc[i],
                    'rejection_price': df['close'].iloc[i]
                })
        
        return {'bullish': bullish_grabs, 'bearish': bearish_grabs}
    
    def identify_poi(self, df: pd.DataFrame, swing_highs: List[int], swing_lows: List[int]) -> Dict[str, List[float]]:
        """
        Identify Points of Interest (POI) - Support/Resistance levels
        
        Args:
            df: OHLCV DataFrame
            swing_highs: List of swing high indices
            swing_lows: List of swing low indices
            
        Returns:
            Dictionary with support and resistance levels
        """
        resistance_levels = []
        support_levels = []
        
        # Add swing highs as resistance
        for idx in swing_highs[-10:]:  # Last 10 swing highs
            resistance_levels.append(df['high'].iloc[idx])
        
        # Add swing lows as support
        for idx in swing_lows[-10:]:  # Last 10 swing lows
            support_levels.append(df['low'].iloc[idx])
        
        # Add Fibonacci retracement levels
        if len(swing_highs) >= 1 and len(swing_lows) >= 1:
            recent_high = df['high'].iloc[swing_highs[-1]]
            recent_low = df['low'].iloc[swing_lows[-1]]
            fib_range = recent_high - recent_low
            
            # 50% retracement
            fib_50 = recent_low + (fib_range * 0.5)
            resistance_levels.append(fib_50)
            support_levels.append(fib_50)
        
        return {'support': support_levels, 'resistance': resistance_levels}
    
    def get_price_action_signals(self, df: pd.DataFrame) -> List[Dict]:
        """
        Get price action confirmation signals using TA-Lib
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            List of price action signals
        """
        signals = []
        
        # Convert to numpy arrays for TA-Lib
        open_prices = df['open'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        
        # Engulfing patterns
        bullish_engulfing = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
        bearish_engulfing = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
        
        # Hammer patterns
        hammer = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
        hanging_man = talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices)
        
        # Doji patterns
        doji = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
        
        for i in range(len(df)):
            current_signals = []
            
            if bullish_engulfing[i] > 0:
                current_signals.append('Bullish Engulfing')
            if bearish_engulfing[i] < 0:
                current_signals.append('Bearish Engulfing')
            if hammer[i] > 0:
                current_signals.append('Hammer')
            if hanging_man[i] < 0:
                current_signals.append('Hanging Man')
            if doji[i] != 0:
                current_signals.append('Doji')
            
            if current_signals:
                signals.append({
                    'index': i,
                    'signals': current_signals,
                    'price': df['close'].iloc[i]
                })
        
        return signals
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        atr_values = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
        return pd.Series(atr_values, index=df.index)
    
    def generate_signals(self, df: pd.DataFrame, bias_filter: str = 'All', quality_threshold: int = 3, 
                        analysis_timeframe: str = '1h') -> List[Dict]:
        """
        Generate comprehensive SMC trading signals with multi-timeframe entry analysis
        
        Args:
            df: OHLCV DataFrame (analysis timeframe)
            bias_filter: Filter signals by bias ('All', 'Bullish', 'Bearish')
            quality_threshold: Minimum confluences required
            entry_timeframe: Lower timeframe for precise entry (e.g., '15m', '5m')
            
        Returns:
            List of trading signals with precise entry timing
        """
        if len(df) < 50:
            return []
        
        signals = []
        
        # Identify market structure
        swing_highs, swing_lows = self.identify_swing_points(df)
        bias = self.determine_bias(df, swing_highs, swing_lows)
        
        # Apply bias filter
        if bias_filter != 'All' and bias != bias_filter:
            return []
        
        # Get SMC components
        fvgs = self.identify_fvg(df)
        order_blocks = self.identify_order_blocks(df, swing_highs, swing_lows)
        liquidity_grabs = self.identify_liquidity_grabs(df)
        poi = self.identify_poi(df, swing_highs, swing_lows)
        price_action = self.get_price_action_signals(df)
        atr = self.calculate_atr(df)
        
        current_price = df['close'].iloc[-1]
        current_atr = atr.iloc[-1]
        
        # Assess market conditions for professional filtering
        market_conditions = self._assess_market_conditions(df)
        
        # Professional market condition filtering
        if not market_conditions['is_trending'] and market_conditions['market_quality'] == 'Medium':
            # Skip signal generation in poor market conditions
            return []
        
        # Generate Market Buy signals
        if bias == 'Bullish':
            market_buy_signals = self._generate_market_buy_signals(
                df, fvgs, order_blocks, liquidity_grabs, price_action, 
                poi, current_price, current_atr, swing_lows, quality_threshold, order_flow_confluence
            )
            signals.extend(market_buy_signals)
        
        # Generate Market Sell signals
        if bias == 'Bearish':
            market_sell_signals = self._generate_market_sell_signals(
                df, fvgs, order_blocks, liquidity_grabs, price_action,
                poi, current_price, current_atr, swing_highs, quality_threshold, order_flow_confluence
            )
            signals.extend(market_sell_signals)
        
        # Generate Limit Buy signals
        limit_buy_signals = self._generate_limit_buy_signals(
            df, fvgs, order_blocks, poi, current_price, current_atr, quality_threshold
        )
        signals.extend(limit_buy_signals)
        
        # Generate Limit Sell signals
        limit_sell_signals = self._generate_limit_sell_signals(
            df, fvgs, order_blocks, poi, current_price, current_atr, quality_threshold
        )
        signals.extend(limit_sell_signals)
        
        # Get order flow confluence analysis for signal enhancement
        order_flow_confluence = self.order_flow_analyzer.analyze_order_flow_confluence(df)
        
        # Filter signals by minimum confluences (higher quality)
        signals = [s for s in signals if s['confluences'] >= quality_threshold]
        
        # Additional quality filters
        signals = self._apply_quality_filters(signals, df, current_price, current_atr)
        
        # Rank signals by quality
        signals = self._rank_signals_by_quality(signals)
        
        # Diversify targets when multiple signals hit same level
        signals = self._diversify_signal_targets(signals)
        
        # Apply automated multi-timeframe entry analysis
        if signals:
            # Automatically determine optimal entry timeframe
            entry_timeframe = self._get_optimal_entry_timeframe(analysis_timeframe)
            signals = self._apply_multi_timeframe_entry_analysis(signals, entry_timeframe)
        
        # If no signals found, try with lower confluence threshold
        if not signals and quality_threshold > 2:
            # Temporarily lower the threshold for individual signal generation
            temp_threshold = max(2, quality_threshold - 1)
            st.warning(f"⚠️ No signals found with {quality_threshold} confluences. Trying with {temp_threshold} confluences...")
            
            # Regenerate signals with lower threshold
            if bias == 'Bullish':
                market_buy_signals = self._generate_market_buy_signals(
                    df, fvgs, order_blocks, liquidity_grabs, price_action, 
                    poi, current_price, current_atr, swing_lows, temp_threshold
                )
                signals.extend(market_buy_signals)
            
            if bias == 'Bearish':
                market_sell_signals = self._generate_market_sell_signals(
                    df, fvgs, order_blocks, liquidity_grabs, price_action,
                    poi, current_price, current_atr, swing_highs, temp_threshold
                )
                signals.extend(market_sell_signals)
            
            limit_buy_signals = self._generate_limit_buy_signals(
                df, fvgs, order_blocks, poi, current_price, current_atr, temp_threshold
            )
            signals.extend(limit_buy_signals)
            
            limit_sell_signals = self._generate_limit_sell_signals(
                df, fvgs, order_blocks, poi, current_price, current_atr, temp_threshold
            )
            signals.extend(limit_sell_signals)
            
            # Filter by the lower threshold
            signals = [s for s in signals if s['confluences'] >= temp_threshold]
        
        # If still no signals found, return empty list
        # This ensures only real, high-quality signals are shown
        
        # Calculate confidence scores for all signals
        for signal in signals:
            signal['confidence'] = self._calculate_signal_confidence(signal, df, current_atr)
        
        # Sort signals by confidence score (highest first)
        signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return signals
    
    def _generate_market_buy_signals(self, df, fvgs, order_blocks, liquidity_grabs, price_action, poi, current_price, atr, swing_lows, quality_threshold=3, order_flow_confluence=None):
        """Generate Market Buy signals with order flow enhancement"""
        signals = []
        
        # Check for bullish confluences
        confluences = 0
        rationale_parts = []
        
        # FVG confluence (extend lookback period)
        recent_bullish_fvgs = [fvg for fvg in fvgs['bullish'] if not fvg['mitigated'] and fvg['index'] >= len(df) - 50]
        if recent_bullish_fvgs:
            confluences += 1
            rationale_parts.append("Bullish FVG")
        
        # Liquidity grab confluence (extend lookback period)
        recent_bullish_grabs = [grab for grab in liquidity_grabs['bullish'] if grab['index'] >= len(df) - 20]
        if recent_bullish_grabs:
            confluences += 1
            rationale_parts.append("Bullish Liquidity Grab")
        
        # Price action confluence (extend lookback period)
        recent_pa = [pa for pa in price_action if pa['index'] >= len(df) - 10 and 'Bullish' in str(pa['signals'])]
        if recent_pa:
            confluences += 1
            rationale_parts.append("Bullish Price Action")
        
        # Order block confluence (extend lookback period)
        recent_bullish_obs = [ob for ob in order_blocks['bullish'] if not ob['mitigated'] and ob['index'] >= len(df) - 100]
        if recent_bullish_obs:
            confluences += 1
            rationale_parts.append("Bullish Order Block")
        
        # Additional confluences for higher quality
        # Support level confluence
        if poi['support'] and current_price > min(poi['support']):
            confluences += 1
            rationale_parts.append("Support Level")
        
        # Enhanced trend confluence with multiple timeframes
        if len(df) >= 50:
            # Multiple moving averages for better trend confirmation
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            ema_12 = df['close'].ewm(span=12).mean().iloc[-1]
            ema_26 = df['close'].ewm(span=26).mean().iloc[-1]
            
            trend_score = 0
            trend_parts = []
            
            # Price above/below moving averages
            if current_price > sma_20:
                trend_score += 1
                trend_parts.append("Above SMA 20")
            if current_price > sma_50:
                trend_score += 1
                trend_parts.append("Above SMA 50")
            if current_price > ema_12:
                trend_score += 1
                trend_parts.append("Above EMA 12")
            if current_price > ema_26:
                trend_score += 1
                trend_parts.append("Above EMA 26")
            
            # Moving average alignment (bullish when shorter > longer)
            if sma_20 > sma_50:
                trend_score += 1
                trend_parts.append("SMA 20 > SMA 50")
            if ema_12 > ema_26:
                trend_score += 1
                trend_parts.append("EMA 12 > EMA 26")
            
            # Add trend confluence if we have at least 3 trend confirmations
            if trend_score >= 3:
                confluences += 1
                rationale_parts.append("Strong Bullish Trend")
            elif trend_score >= 2:
                confluences += 1
                rationale_parts.append("Moderate Bullish Trend")
        
        # Enhanced volume confluence with multiple confirmations
        if 'volume' in df.columns and len(df) >= 20:
            recent_volume = df['volume'].tail(5).mean()
            avg_volume_20 = df['volume'].tail(20).mean()
            avg_volume_50 = df['volume'].mean() if len(df) >= 50 else avg_volume_20
            
            volume_score = 0
            volume_parts = []
            
            # Recent volume vs 20-period average
            if recent_volume > avg_volume_20 * 1.2:
                volume_score += 1
                volume_parts.append("High Recent Volume")
            elif recent_volume > avg_volume_20 * 1.1:
                volume_score += 0.5
                volume_parts.append("Above Average Volume")
            
            # Recent volume vs 50-period average
            if recent_volume > avg_volume_50 * 1.3:
                volume_score += 1
                volume_parts.append("Strong Volume Surge")
            elif recent_volume > avg_volume_50 * 1.1:
                volume_score += 0.5
                volume_parts.append("Volume Above Average")
            
            # Volume trend (increasing volume)
            volume_trend = df['volume'].tail(5).mean() / df['volume'].tail(10).mean()
            if volume_trend > 1.1:
                volume_score += 1
                volume_parts.append("Increasing Volume Trend")
            
            # Add volume confluence based on score
            if volume_score >= 2.5:
                confluences += 1
                rationale_parts.append("Strong Volume Confirmation")
            elif volume_score >= 1.5:
                confluences += 1
                rationale_parts.append("Moderate Volume Confirmation")
        
        # Order Flow Confluence Enhancement
        if order_flow_confluence and order_flow_confluence['confluence_score'] > 0:
            # Add order flow confluences based on strength
            of_score = order_flow_confluence['confluence_score']
            
            if of_score >= 2.0:
                confluences += 2
                rationale_parts.append("Strong Order Flow Confirmation")
            elif of_score >= 1.0:
                confluences += 1
                rationale_parts.append("Order Flow Confirmation")
            
            # Add specific order flow factors
            if order_flow_confluence['imbalances']:
                bullish_imbalances = [im for im in order_flow_confluence['imbalances'] if im['type'] == 'buying_imbalance']
                if bullish_imbalances:
                    confluences += 1
                    rationale_parts.append("Buying Imbalance")
            
            if order_flow_confluence['absorptions']:
                bullish_absorptions = [abs for abs in order_flow_confluence['absorptions'] if abs['type'] == 'bullish_absorption']
                if bullish_absorptions:
                    confluences += 1
                    rationale_parts.append("Bullish Absorption")
            
            if order_flow_confluence['aggressive_orders']:
                bullish_aggressive = [agg for agg in order_flow_confluence['aggressive_orders'] if agg['type'] == 'aggressive_buy']
                if bullish_aggressive:
                    confluences += 1
                    rationale_parts.append("Aggressive Buying")
        
        if confluences >= quality_threshold:
            # Calculate stop loss and take profit with proper validation
            # For buy signals, stop loss should be below entry price
            sl = current_price * 0.98  # 2% below current price (conservative)
            
            # Try to use swing low if available and reasonable
            if len(swing_lows) >= 3:
                recent_swing_low = min(swing_lows[-3:])
                # Ensure swing low is reasonable (between 50% and 100% of current price)
                if current_price * 0.5 < recent_swing_low < current_price:
                    sl = recent_swing_low - (atr * 0.5)
                    # Ensure stop loss is still below entry
                    if sl >= current_price:
                        sl = current_price * 0.98
            
            # Calculate take profit
            tp = current_price * 1.04  # 4% above current price (1:2 RR)
            
            # Try to use resistance level if available and reasonable
            if poi['resistance'] and max(poi['resistance']) > current_price:
                resistance_level = max(poi['resistance'])
                # Ensure resistance is reasonable (not more than 10% above current price)
                if resistance_level <= current_price * 1.10:
                    tp = resistance_level
            
            # Calculate risk-reward ratio
            price_risk = current_price - sl
            price_reward = tp - current_price
            rr = price_reward / price_risk if price_risk > 0 else 0
            
            # Ensure reasonable RR (between 0.5 and 5.0)
            if rr > 5.0:
                tp = current_price + (price_risk * 3.0)  # Cap at 1:3 RR
                rr = 3.0
            elif rr < 0.5:
                tp = current_price + (price_risk * 1.5)  # Minimum 1:1.5 RR
                rr = 1.5
            
            signals.append({
                'type': 'Market',
                'direction': 'Buy',
                'entry': current_price,
                'sl': sl,
                'tp': tp,
                'rr': rr,
                'confluences': confluences,
                'rationale': ' + '.join(rationale_parts)
            })
        
        return signals
    
    def _generate_market_sell_signals(self, df, fvgs, order_blocks, liquidity_grabs, price_action, poi, current_price, atr, swing_highs, quality_threshold=3, order_flow_confluence=None):
        """Generate Market Sell signals with order flow enhancement"""
        signals = []
        
        # Check for bearish confluences
        confluences = 0
        rationale_parts = []
        
        # FVG confluence (extend lookback period)
        recent_bearish_fvgs = [fvg for fvg in fvgs['bearish'] if not fvg['mitigated'] and fvg['index'] >= len(df) - 50]
        if recent_bearish_fvgs:
            confluences += 1
            rationale_parts.append("Bearish FVG")
        
        # Liquidity grab confluence (extend lookback period)
        recent_bearish_grabs = [grab for grab in liquidity_grabs['bearish'] if grab['index'] >= len(df) - 20]
        if recent_bearish_grabs:
            confluences += 1
            rationale_parts.append("Bearish Liquidity Grab")
        
        # Price action confluence (extend lookback period)
        recent_pa = [pa for pa in price_action if pa['index'] >= len(df) - 10 and 'Bearish' in str(pa['signals'])]
        if recent_pa:
            confluences += 1
            rationale_parts.append("Bearish Price Action")
        
        # Order block confluence (extend lookback period)
        recent_bearish_obs = [ob for ob in order_blocks['bearish'] if not ob['mitigated'] and ob['index'] >= len(df) - 100]
        if recent_bearish_obs:
            confluences += 1
            rationale_parts.append("Bearish Order Block")
        
        # Additional confluences for higher quality
        # Resistance level confluence
        if poi['resistance'] and current_price < max(poi['resistance']):
            confluences += 1
            rationale_parts.append("Resistance Level")
        
        # Enhanced trend confluence with multiple timeframes
        if len(df) >= 50:
            # Multiple moving averages for better trend confirmation
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            ema_12 = df['close'].ewm(span=12).mean().iloc[-1]
            ema_26 = df['close'].ewm(span=26).mean().iloc[-1]
            
            trend_score = 0
            trend_parts = []
            
            # Price below moving averages (bearish)
            if current_price < sma_20:
                trend_score += 1
                trend_parts.append("Below SMA 20")
            if current_price < sma_50:
                trend_score += 1
                trend_parts.append("Below SMA 50")
            if current_price < ema_12:
                trend_score += 1
                trend_parts.append("Below EMA 12")
            if current_price < ema_26:
                trend_score += 1
                trend_parts.append("Below EMA 26")
            
            # Moving average alignment (bearish when shorter < longer)
            if sma_20 < sma_50:
                trend_score += 1
                trend_parts.append("SMA 20 < SMA 50")
            if ema_12 < ema_26:
                trend_score += 1
                trend_parts.append("EMA 12 < EMA 26")
            
            # Add trend confluence if we have at least 3 trend confirmations
            if trend_score >= 3:
                confluences += 1
                rationale_parts.append("Strong Bearish Trend")
            elif trend_score >= 2:
                confluences += 1
                rationale_parts.append("Moderate Bearish Trend")
        
        # Enhanced volume confluence with multiple confirmations
        if 'volume' in df.columns and len(df) >= 20:
            recent_volume = df['volume'].tail(5).mean()
            avg_volume_20 = df['volume'].tail(20).mean()
            avg_volume_50 = df['volume'].mean() if len(df) >= 50 else avg_volume_20
            
            volume_score = 0
            volume_parts = []
            
            # Recent volume vs 20-period average
            if recent_volume > avg_volume_20 * 1.2:
                volume_score += 1
                volume_parts.append("High Recent Volume")
            elif recent_volume > avg_volume_20 * 1.1:
                volume_score += 0.5
                volume_parts.append("Above Average Volume")
            
            # Recent volume vs 50-period average
            if recent_volume > avg_volume_50 * 1.3:
                volume_score += 1
                volume_parts.append("Strong Volume Surge")
            elif recent_volume > avg_volume_50 * 1.1:
                volume_score += 0.5
                volume_parts.append("Volume Above Average")
            
            # Volume trend (increasing volume)
            volume_trend = df['volume'].tail(5).mean() / df['volume'].tail(10).mean()
            if volume_trend > 1.1:
                volume_score += 1
                volume_parts.append("Increasing Volume Trend")
            
            # Add volume confluence based on score
            if volume_score >= 2.5:
                confluences += 1
                rationale_parts.append("Strong Volume Confirmation")
            elif volume_score >= 1.5:
                confluences += 1
                rationale_parts.append("Moderate Volume Confirmation")
        
        # Order Flow Confluence Enhancement
        if order_flow_confluence and order_flow_confluence['confluence_score'] > 0:
            # Add order flow confluences based on strength
            of_score = order_flow_confluence['confluence_score']
            
            if of_score >= 2.0:
                confluences += 2
                rationale_parts.append("Strong Order Flow Confirmation")
            elif of_score >= 1.0:
                confluences += 1
                rationale_parts.append("Order Flow Confirmation")
            
            # Add specific order flow factors
            if order_flow_confluence['imbalances']:
                bearish_imbalances = [im for im in order_flow_confluence['imbalances'] if im['type'] == 'selling_imbalance']
                if bearish_imbalances:
                    confluences += 1
                    rationale_parts.append("Selling Imbalance")
            
            if order_flow_confluence['absorptions']:
                bearish_absorptions = [abs for abs in order_flow_confluence['absorptions'] if abs['type'] == 'bearish_absorption']
                if bearish_absorptions:
                    confluences += 1
                    rationale_parts.append("Bearish Absorption")
            
            if order_flow_confluence['aggressive_orders']:
                bearish_aggressive = [agg for agg in order_flow_confluence['aggressive_orders'] if agg['type'] == 'aggressive_sell']
                if bearish_aggressive:
                    confluences += 1
                    rationale_parts.append("Aggressive Selling")
        
        if confluences >= quality_threshold:
            # Calculate stop loss and take profit with proper validation
            # For sell signals, stop loss should be above entry price
            sl = current_price * 1.02  # 2% above current price (conservative)
            
            # Try to use swing high if available and reasonable
            if len(swing_highs) >= 3:
                recent_swing_high = max(swing_highs[-3:])
                # Ensure swing high is reasonable (between 100% and 150% of current price)
                if current_price < recent_swing_high < current_price * 1.5:
                    sl = recent_swing_high + (atr * 0.5)
                    # Ensure stop loss is still above entry
                    if sl <= current_price:
                        sl = current_price * 1.02
            
            # Calculate take profit
            tp = current_price * 0.96  # 4% below current price (1:2 RR)
            
            # Try to use support level if available and reasonable
            if poi['support'] and min(poi['support']) < current_price:
                support_level = min(poi['support'])
                # Ensure support is reasonable (not more than 10% below current price)
                if support_level >= current_price * 0.90:
                    tp = support_level
            
            # Calculate risk-reward ratio
            price_risk = sl - current_price
            price_reward = current_price - tp
            rr = price_reward / price_risk if price_risk > 0 else 0
            
            # Ensure reasonable RR (between 0.5 and 5.0)
            if rr > 5.0:
                tp = current_price - (price_risk * 3.0)  # Cap at 1:3 RR
                rr = 3.0
            elif rr < 0.5:
                tp = current_price - (price_risk * 1.5)  # Minimum 1:1.5 RR
                rr = 1.5
            
            signals.append({
                'type': 'Market',
                'direction': 'Sell',
                'entry': current_price,
                'sl': sl,
                'tp': tp,
                'rr': rr,
                'confluences': confluences,
                'rationale': ' + '.join(rationale_parts)
            })
        
        return signals
    
    def _generate_limit_buy_signals(self, df, fvgs, order_blocks, poi, current_price, atr, quality_threshold=3):
        """Generate Limit Buy signals with institutional-grade enhancements"""
        signals = []
        
        # FVG Limit Buy signals
        for fvg in fvgs['bullish']:
            if not fvg['mitigated'] and fvg['index'] >= len(df) - 20:
                confluences = 1
                rationale_parts = ["Unmitigated Bullish FVG"]
                
                # Check for support confluence
                if poi['support']:
                    support_levels = [s for s in poi['support'] if abs(s - fvg['bottom']) / fvg['bottom'] < 0.01]
                    if support_levels:
                        confluences += 1
                        rationale_parts.append("Support Level")
                
                # Enhanced entry price logic with professional buffer zones
                if confluences >= quality_threshold:
                    # Professional entry price calculation
                    base_entry = fvg['bottom']
                    buffer_zone = atr * 0.1  # 10% of ATR buffer
                    entry = base_entry + buffer_zone  # Small buffer above FVG bottom
                    
                    # Debug: Log FVG details
                    print(f"DEBUG FVG: bottom={fvg['bottom']:.2f}, top={fvg['top']:.2f}, atr={atr:.2f}, entry={entry:.2f}")
                    
                    # Market structure confirmation
                    market_structure_score = self._assess_market_structure(df, fvg['index'], 'bullish')
                    if market_structure_score > 0.6:  # Strong market structure
                        confluences += 1
                        rationale_parts.append("Strong Market Structure")
                    
                    # Enhanced stop loss with market structure consideration
                    sl = entry - (atr * 2.5)  # Slightly wider stop for limit orders
                    tp = entry + (atr * 4)  # 4x ATR take profit (1:2 RR)
                    
                    # Try to use resistance level if available and reasonable
                    if poi['resistance'] and max(poi['resistance']) > entry:
                        resistance_level = max(poi['resistance'])
                        if resistance_level <= entry * 1.10:  # Not more than 10% above entry
                            tp = resistance_level
                    
                    # Calculate risk-reward ratio
                    price_risk = entry - sl
                    price_reward = tp - entry
                    rr = price_reward / price_risk if price_risk > 0 else 0
                    
                    # Ensure reasonable RR (between 0.5 and 5.0)
                    if rr > 5.0:
                        tp = entry + (price_risk * 3.0)  # Cap at 1:3 RR
                        rr = 3.0
                    elif rr < 0.5:
                        tp = entry + (price_risk * 1.5)  # Minimum 1:1.5 RR
                        rr = 1.5
                    
                    # Professional order management parameters
                    order_expiry_hours = 24  # 24-hour order expiry
                    max_slippage = 0.001  # 0.1% max slippage
                    partial_fill_allowed = True
                    
                    # Volume confirmation
                    volume_score = self._assess_volume_confirmation(df, fvg['index'])
                    if volume_score > 0.7:
                        confluences += 1
                        rationale_parts.append("Volume Confirmation")
                    
                    signals.append({
                        'type': 'Limit',
                        'direction': 'Buy',
                        'entry': entry,  # Enhanced entry with buffer
                        'sl': sl,
                        'tp': tp,
                        'rr': rr,
                        'confluences': confluences,
                        'rationale': ' + '.join(rationale_parts),
                        # Professional order management
                        'order_expiry_hours': order_expiry_hours,
                        'max_slippage': max_slippage,
                        'partial_fill_allowed': partial_fill_allowed,
                        'market_structure_score': market_structure_score,
                        'volume_score': volume_score,
                        'entry_quality': 'Professional' if confluences >= 4 else 'Standard'
                    })
        
        # Order Block Limit Buy signals
        for ob in order_blocks['bullish']:
            if not ob['mitigated'] and ob['index'] >= len(df) - 20:
                confluences = 1
                rationale_parts = ["Unmitigated Bullish Order Block"]
                
                # Check for support confluence
                if poi['support']:
                    support_levels = [s for s in poi['support'] if abs(s - ob['low']) / ob['low'] < 0.01]
                    if support_levels:
                        confluences += 1
                        rationale_parts.append("Support Level")
                
                if confluences >= quality_threshold:
                    # Professional entry price calculation for Order Blocks
                    base_entry = ob['low']
                    buffer_zone = atr * 0.05  # 5% of ATR buffer for Order Blocks
                    entry = base_entry + buffer_zone  # Small buffer above OB low
                    
                    # Debug: Log Order Block details
                    print(f"DEBUG OB: low={ob['low']:.2f}, high={ob['high']:.2f}, atr={atr:.2f}, entry={entry:.2f}")
                    
                    # Market structure confirmation
                    market_structure_score = self._assess_market_structure(df, ob['index'], 'bullish')
                    if market_structure_score > 0.6:
                        confluences += 1
                        rationale_parts.append("Strong Market Structure")
                    
                    # Enhanced stop loss and take profit
                    sl = entry - (atr * 2.5)  # Professional stop loss
                    tp = max(poi['resistance']) if poi['resistance'] else entry + (atr * 4)
                    
                    # Calculate risk-reward ratio
                    price_risk = entry - sl
                    price_reward = tp - entry
                    rr = price_reward / price_risk if price_risk > 0 else 0
                    
                    # Ensure reasonable RR
                    if rr > 5.0:
                        tp = entry + (price_risk * 3.0)
                        rr = 3.0
                    elif rr < 0.5:
                        tp = entry + (price_risk * 1.5)
                        rr = 1.5
                    
                    # Volume confirmation
                    volume_score = self._assess_volume_confirmation(df, ob['index'])
                    if volume_score > 0.7:
                        confluences += 1
                        rationale_parts.append("Volume Confirmation")
                    
                    # Professional order management parameters
                    order_expiry_hours = 24
                    max_slippage = 0.001
                    partial_fill_allowed = True
                    
                    signals.append({
                        'type': 'Limit',
                        'direction': 'Buy',
                        'entry': entry,  # Enhanced entry with buffer
                        'sl': sl,
                        'tp': tp,
                        'rr': rr,
                        'confluences': confluences,
                        'rationale': ' + '.join(rationale_parts),
                        # Professional order management
                        'order_expiry_hours': order_expiry_hours,
                        'max_slippage': max_slippage,
                        'partial_fill_allowed': partial_fill_allowed,
                        'market_structure_score': market_structure_score,
                        'volume_score': volume_score,
                        'entry_quality': 'Professional' if confluences >= 4 else 'Standard'
                    })
        
        return signals
    
    def _generate_limit_sell_signals(self, df, fvgs, order_blocks, poi, current_price, atr, quality_threshold=3):
        """Generate Limit Sell signals"""
        signals = []
        
        # FVG Limit Sell signals
        for fvg in fvgs['bearish']:
            if not fvg['mitigated'] and fvg['index'] >= len(df) - 20:
                confluences = 1
                rationale_parts = ["Unmitigated Bearish FVG"]
                
                # Check for resistance confluence
                if poi['resistance']:
                    resistance_levels = [r for r in poi['resistance'] if abs(r - fvg['top']) / fvg['top'] < 0.01]
                    if resistance_levels:
                        confluences += 1
                        rationale_parts.append("Resistance Level")
                
                if confluences >= quality_threshold:
                    entry = fvg['top']
                    sl = entry + (atr * 2)  # 2x ATR stop loss
                    tp = entry - (atr * 4)  # 4x ATR take profit (1:2 RR)
                    
                    # Try to use support level if available and reasonable
                    if poi['support'] and min(poi['support']) < entry:
                        support_level = min(poi['support'])
                        if support_level >= entry * 0.90:  # Not more than 10% below entry
                            tp = support_level
                    
                    # Calculate risk-reward ratio
                    price_risk = sl - entry
                    price_reward = entry - tp
                    rr = price_reward / price_risk if price_risk > 0 else 0
                    
                    # Ensure reasonable RR (between 0.5 and 5.0)
                    if rr > 5.0:
                        tp = entry - (price_risk * 3.0)  # Cap at 1:3 RR
                        rr = 3.0
                    elif rr < 0.5:
                        tp = entry - (price_risk * 1.5)  # Minimum 1:1.5 RR
                        rr = 1.5
                    
                    signals.append({
                        'type': 'Limit',
                        'direction': 'Sell',
                        'entry': fvg['top'],
                        'sl': sl,
                        'tp': tp,
                        'rr': rr,
                        'confluences': confluences,
                        'rationale': ' + '.join(rationale_parts)
                    })
        
        # Order Block Limit Sell signals
        for ob in order_blocks['bearish']:
            if not ob['mitigated'] and ob['index'] >= len(df) - 20:
                confluences = 1
                rationale_parts = ["Unmitigated Bearish Order Block"]
                
                # Check for resistance confluence
                if poi['resistance']:
                    resistance_levels = [r for r in poi['resistance'] if abs(r - ob['high']) / ob['high'] < 0.01]
                    if resistance_levels:
                        confluences += 1
                        rationale_parts.append("Resistance Level")
                
                if confluences >= quality_threshold:
                    sl = ob['high'] + atr
                    tp = min(poi['support']) if poi['support'] else ob['low'] * 0.97
                    rr = (ob['high'] - tp) / (sl - ob['high']) if sl > ob['high'] else 0
                    
                    signals.append({
                        'type': 'Limit',
                        'direction': 'Sell',
                        'entry': ob['high'],
                        'sl': sl,
                        'tp': tp,
                        'rr': rr,
                        'confluences': confluences,
                        'rationale': ' + '.join(rationale_parts)
                    })
        
        return signals
    
    def _diversify_signal_targets(self, signals: List[Dict]) -> List[Dict]:
        """Diversify targets when multiple signals hit same level"""
        if len(signals) <= 1:
            return signals
        
        # Group signals by similar target levels
        target_groups = {}
        tolerance = 0.01  # 1% tolerance for similar targets
        
        for i, signal in enumerate(signals):
            target = signal['tp']
            grouped = False
            
            for group_target, group_signals in target_groups.items():
                if abs(target - group_target) / group_target <= tolerance:
                    target_groups[group_target].append((i, signal))
                    grouped = True
                    break
            
            if not grouped:
                target_groups[target] = [(i, signal)]
        
        # Diversify targets within each group
        diversified_signals = []
        for group_target, group_signals in target_groups.items():
            if len(group_signals) > 1:
                # Sort by quality score if available, otherwise by confluences
                group_signals.sort(key=lambda x: x[1].get('quality_score', x[1]['confluences']), reverse=True)
                
                # Keep the best signal with original target
                best_signal = group_signals[0][1]
                diversified_signals.append(best_signal)
                
                # Diversify remaining signals
                for j, (_, signal) in enumerate(group_signals[1:], 1):
                    # Create diversified target
                    entry = signal['entry']
                    sl = signal['sl']
                    price_risk = abs(entry - sl)
                    
                    # Create different target levels
                    if signal['direction'] == 'Buy':
                        # Diversify upward targets
                        diversified_target = entry + (price_risk * (2.0 + j * 0.2))  # 2.0, 2.2, 2.4, etc.
                    else:
                        # Diversify downward targets
                        diversified_target = entry - (price_risk * (2.0 + j * 0.2))  # 2.0, 2.2, 2.4, etc.
                    
                    # Update signal with diversified target
                    signal['tp'] = diversified_target
                    price_reward = abs(diversified_target - entry)
                    signal['rr'] = price_reward / price_risk if price_risk > 0 else 0
                    
                    # Add diversification note to rationale
                    signal['rationale'] += f" + Diversified Target {j}"
                    
                    diversified_signals.append(signal)
            else:
                # Single signal in group, keep as is
                diversified_signals.append(group_signals[0][1])
        
        # Sort by quality score again
        diversified_signals.sort(key=lambda x: x.get('quality_score', x['confluences']), reverse=True)
        
        return diversified_signals
    
    def _apply_quality_filters(self, signals: List[Dict], df: pd.DataFrame, current_price: float, atr: float) -> List[Dict]:
        """Apply additional quality filters to improve signal quality"""
        if not signals:
            return signals
        
        filtered_signals = []
        
        for signal in signals:
            # Filter 1: Minimum risk-reward ratio
            if signal['rr'] < 1.5:  # Require at least 1:1.5 R:R
                continue
            
            # Filter 2: Maximum risk-reward ratio (avoid unrealistic ratios)
            if signal['rr'] > 10:  # Cap at 1:10 R:R
                continue
            
            # Filter 3: Stop loss not too close to current price
            price_risk = abs(signal['entry'] - signal['sl'])
            if price_risk < (atr * 0.5):  # Stop loss too close
                continue
            
            # Filter 4: Take profit not too far from current price
            profit_distance = abs(signal['tp'] - signal['entry'])
            if profit_distance > (atr * 5):  # Take profit too far
                continue
            
            # Filter 5: Entry price should be reasonable relative to current price
            entry_distance = abs(signal['entry'] - current_price) / current_price
            if entry_distance > 0.05:  # Entry more than 5% away from current price
                continue
            
            # Filter 6: Check for recent price action confirmation
            recent_volume = df['volume'].tail(5).mean()
            if recent_volume > 0 and df['volume'].iloc[-1] < (recent_volume * 0.5):  # Low volume
                continue
            
            filtered_signals.append(signal)
        
        return filtered_signals
    
    def _rank_signals_by_quality(self, signals: List[Dict]) -> List[Dict]:
        """Rank signals by quality score"""
        if not signals:
            return signals
        
        for signal in signals:
            # Calculate quality score
            score = 0
            
            # Base score from confluences
            score += signal['confluences'] * 10
            
            # Risk-reward bonus
            if signal['rr'] >= 2.0:
                score += 20
            elif signal['rr'] >= 1.5:
                score += 10
            
            # Market type bonus (forex gets slight preference)
            if signal.get('market_type') == 'forex':
                score += 5
            
            # Add quality score to signal
            signal['quality_score'] = score
        
        # Sort by quality score (highest first)
        signals.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return signals
    
    def _assess_market_structure(self, df: pd.DataFrame, signal_index: int, direction: str) -> float:
        """
        Assess market structure strength for professional signal validation
        
        Args:
            df: OHLCV DataFrame
            signal_index: Index of the signal
            direction: 'bullish' or 'bearish'
            
        Returns:
            Market structure score (0-1)
        """
        try:
            # Look at recent price action around signal
            lookback = min(20, signal_index)
            recent_data = df.iloc[signal_index-lookback:signal_index+1]
            
            if len(recent_data) < 10:
                return 0.5  # Neutral if insufficient data
            
            # Calculate higher highs and higher lows for bullish
            if direction == 'bullish':
                highs = recent_data['high'].values
                lows = recent_data['low'].values
                
                # Count higher highs
                higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
                higher_lows = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
                
                # Calculate structure score
                structure_score = (higher_highs + higher_lows) / (len(highs) - 1) / 2
                
            else:  # bearish
                highs = recent_data['high'].values
                lows = recent_data['low'].values
                
                # Count lower highs and lower lows
                lower_highs = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
                lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
                
                # Calculate structure score
                structure_score = (lower_highs + lower_lows) / (len(highs) - 1) / 2
            
            return min(max(structure_score, 0), 1)  # Clamp between 0 and 1
            
        except Exception:
            return 0.5  # Neutral on error
    
    def _assess_volume_confirmation(self, df: pd.DataFrame, signal_index: int) -> float:
        """
        Assess volume confirmation for professional signal validation
        
        Args:
            df: OHLCV DataFrame
            signal_index: Index of the signal
            
        Returns:
            Volume confirmation score (0-1)
        """
        try:
            if 'volume' not in df.columns:
                return 0.5  # Neutral if no volume data
            
            # Look at recent volume around signal
            lookback = min(10, signal_index)
            recent_volume = df['volume'].iloc[signal_index-lookback:signal_index+1]
            
            if len(recent_volume) < 5:
                return 0.5  # Neutral if insufficient data
            
            # Calculate volume metrics
            current_volume = recent_volume.iloc[-1]
            avg_volume = recent_volume.mean()
            volume_trend = recent_volume.pct_change().mean()
            
            # Volume confirmation score
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            volume_score = min(volume_ratio / 2, 1)  # Cap at 1.0
            
            # Bonus for increasing volume trend
            if volume_trend > 0:
                volume_score += 0.2
            
            return min(max(volume_score, 0), 1)  # Clamp between 0 and 1
            
        except Exception:
            return 0.5  # Neutral on error
    
    def _assess_market_conditions(self, df: pd.DataFrame) -> Dict:
        """
        Assess current market conditions for professional filtering
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Dictionary with market condition scores
        """
        try:
            # Calculate volatility
            if len(df) >= 20:
                returns = df['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            else:
                volatility = 0.02  # Default 2%
            
            # Calculate trend strength
            if len(df) >= 50:
                sma_20 = df['close'].rolling(20).mean().iloc[-1]
                sma_50 = df['close'].rolling(50).mean().iloc[-1]
                trend_strength = abs(sma_20 - sma_50) / sma_50 if sma_50 > 0 else 0
            else:
                trend_strength = 0.01  # Default 1%
            
            # Market condition assessment
            conditions = {
                'volatility': volatility,
                'trend_strength': trend_strength,
                'is_trending': trend_strength > 0.02,  # 2% trend threshold
                'is_volatile': volatility > 0.15,  # 15% volatility threshold
                'market_quality': 'High' if 0.05 < volatility < 0.25 and trend_strength > 0.01 else 'Medium'
            }
            
            return conditions
            
        except Exception:
            return {
                'volatility': 0.02,
                'trend_strength': 0.01,
                'is_trending': True,
                'is_volatile': False,
                'market_quality': 'Medium'
            }
    
    def _calculate_signal_confidence(self, signal: Dict, df: pd.DataFrame, atr: float) -> float:
        """
        Calculate comprehensive confidence score for a signal
        
        Args:
            signal: Signal dictionary
            df: OHLCV DataFrame
            atr: Current ATR value
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            confidence_factors = []
            
            # 1. Confluence Score (0-0.3)
            confluence_score = min(signal.get('confluences', 0) / 8.0, 0.3)
            confidence_factors.append(confluence_score)
        
            # 2. Risk-Reward Score (0-0.2)
            rr = signal.get('rr', 0)
            if rr >= 2.0:
                rr_score = 0.2
            elif rr >= 1.5:
                rr_score = 0.15
            elif rr >= 1.0:
                rr_score = 0.1
            else:
                rr_score = 0.05
            confidence_factors.append(rr_score)
        
            # 3. Market Structure Score (0-0.2)
            market_structure_score = signal.get('market_structure_score', 0)
            # Ensure market_structure_score is a scalar value
            if isinstance(market_structure_score, str):
                market_structure_score = 0.0  # Default for string values
            elif hasattr(market_structure_score, '__len__') and not isinstance(market_structure_score, str):
                market_structure_score = float(market_structure_score) if len(market_structure_score) > 0 else 0
            else:
                try:
                    market_structure_score = float(market_structure_score) if market_structure_score is not None else 0
                except (ValueError, TypeError):
                    market_structure_score = 0.0
            confidence_factors.append(market_structure_score * 0.2)
        
            # 4. Volume Confirmation Score (0-0.15)
            volume_score = signal.get('volume_score', 0)
            # Ensure volume_score is a scalar value
            if isinstance(volume_score, str):
                volume_score = 0.0  # Default for string values
            elif hasattr(volume_score, '__len__') and not isinstance(volume_score, str):
                volume_score = float(volume_score) if len(volume_score) > 0 else 0
            else:
                try:
                    volume_score = float(volume_score) if volume_score is not None else 0
                except (ValueError, TypeError):
                    volume_score = 0.0
            confidence_factors.append(volume_score * 0.15)
            
            # 5. Entry Quality Score (0-0.1)
            entry_quality = signal.get('entry_quality', 0)
            # Ensure entry_quality is a scalar value
            if isinstance(entry_quality, str):
                # Handle string values - convert to numeric based on quality level
                if entry_quality.lower() in ['high', 'excellent', 'premium']:
                    entry_quality = 1.0
                elif entry_quality.lower() in ['medium', 'good', 'standard']:
                    entry_quality = 0.5
                elif entry_quality.lower() in ['low', 'poor', 'basic']:
                    entry_quality = 0.2
                else:
                    entry_quality = 0.0
            elif hasattr(entry_quality, '__len__') and not isinstance(entry_quality, str):
                entry_quality = float(entry_quality) if len(entry_quality) > 0 else 0
            else:
                try:
                    entry_quality = float(entry_quality) if entry_quality is not None else 0
                except (ValueError, TypeError):
                    entry_quality = 0.0
            confidence_factors.append(entry_quality * 0.1)
            
            # 6. Market Conditions Score (0-0.05)
            market_conditions = self._assess_market_conditions(df)
            if market_conditions['market_quality'] == 'High':
                conditions_score = 0.05
            elif market_conditions['market_quality'] == 'Medium':
                conditions_score = 0.03
            else:
                conditions_score = 0.01
            confidence_factors.append(conditions_score)
        
            # Calculate total confidence
            total_confidence = sum(confidence_factors)
            
            # Ensure confidence is between 0.0 and 1.0
            return min(max(total_confidence, 0.0), 1.0)
            
        except Exception as e:
            # Fallback to basic confluence-based confidence
            confluences = signal.get('confluences', 0)
            return min(confluences / 8.0, 0.8)  # Max 80% confidence on fallback
    
    def _apply_multi_timeframe_entry_analysis(self, signals: List[Dict], entry_timeframe: str) -> List[Dict]:
        """
        Apply multi-timeframe analysis for precise entry timing
        
        Args:
            signals: List of signals from higher timeframe
            entry_timeframe: Lower timeframe for entry analysis (e.g., '15m', '5m')
            
        Returns:
            Enhanced signals with precise entry timing
        """
        enhanced_signals = []
        
        for signal in signals:
            try:
                # Get symbol from signal or use default
                symbol = signal.get('symbol', 'EURUSD=X')
                
                # Fetch lower timeframe data for entry analysis
                entry_df = self._fetch_entry_timeframe_data(symbol, entry_timeframe)
                
                if entry_df is None or len(entry_df) < 20:
                    # Fallback to original signal if lower timeframe data unavailable
                    enhanced_signals.append(signal)
                    continue
                
                # Find precise entry point on lower timeframe
                precise_entry = self._find_precise_entry_point(
                    signal, entry_df, entry_timeframe
                )
                
                if precise_entry:
                    # Update signal with precise entry information
                    enhanced_signal = signal.copy()
                    enhanced_signal.update({
                        'precise_entry': precise_entry['price'],
                        'entry_timeframe': entry_timeframe,
                        'entry_confidence': precise_entry['confidence'],
                        'entry_trigger': precise_entry['trigger'],
                        'entry_rationale': precise_entry['rationale'],
                        'original_entry': signal['entry'],  # Keep original for reference
                        'entry_quality': 'Multi-Timeframe' if precise_entry['confidence'] > 0.7 else 'Standard'
                    })
                    enhanced_signals.append(enhanced_signal)
                else:
                    # Keep original signal if no precise entry found
                    enhanced_signals.append(signal)
                    
            except Exception as e:
                # Fallback to original signal on error
                enhanced_signals.append(signal)
        
        return enhanced_signals
    
    def _fetch_entry_timeframe_data(self, symbol: str, entry_timeframe: str) -> pd.DataFrame:
        """
        Fetch data for entry timeframe analysis
        
        Args:
            symbol: Trading symbol
            entry_timeframe: Timeframe for entry analysis
            
        Returns:
            DataFrame with entry timeframe data
        """
        try:
            # Map timeframe to yfinance interval
            timeframe_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d'
            }
            
            interval = timeframe_map.get(entry_timeframe, '15m')
            
            # Fetch data (this would need to be implemented with actual data fetching)
            # For now, return None to indicate data not available
            return None
            
        except Exception:
            return None
    
    def _find_precise_entry_point(self, signal: Dict, entry_df: pd.DataFrame, 
                                 entry_timeframe: str) -> Optional[Dict]:
        """
        Find precise entry point on lower timeframe
        
        Args:
            signal: Original signal from higher timeframe
            entry_df: Lower timeframe data
            entry_timeframe: Entry timeframe
            
        Returns:
            Dictionary with precise entry information
        """
        try:
            if entry_df is None or len(entry_df) < 20:
                return None
            
            # Get signal parameters
            direction = signal.get('direction', 'Buy')
            target_entry = signal.get('entry', 0)
            sl = signal.get('sl', 0)
            
            # Find entry zone on lower timeframe
            entry_zone = self._identify_entry_zone(entry_df, target_entry, direction)
            
            if not entry_zone:
                return None
            
            # Calculate entry confidence based on lower timeframe confluence
            confidence = self._calculate_entry_confidence(entry_df, entry_zone, direction)
            
            # Determine entry trigger
            trigger = self._determine_entry_trigger(entry_df, entry_zone, direction)
            
            return {
                'price': entry_zone['price'],
                'confidence': confidence,
                'trigger': trigger,
                'rationale': f"Precise entry on {entry_timeframe} timeframe"
            }
            
        except Exception:
            return None
    
    def _identify_entry_zone(self, df: pd.DataFrame, target_price: float, 
                           direction: str) -> Optional[Dict]:
        """
        Identify entry zone on lower timeframe
        
        Args:
            df: Lower timeframe data
            target_price: Target entry price from higher timeframe
            direction: Trade direction
            
        Returns:
            Entry zone information
        """
        try:
            # Look for price action around target price
            tolerance = target_price * 0.002  # 0.2% tolerance
            
            if direction == 'Buy':
                # Look for support levels near target price
                recent_lows = df['low'].tail(20)
                entry_candidates = recent_lows[
                    (recent_lows >= target_price - tolerance) & 
                    (recent_lows <= target_price + tolerance)
                ]
                
                if not entry_candidates.empty:
                    return {
                        'price': entry_candidates.iloc[-1],
                        'type': 'support',
                        'strength': 'medium'
                    }
            else:  # Sell
                # Look for resistance levels near target price
                recent_highs = df['high'].tail(20)
                entry_candidates = recent_highs[
                    (recent_highs >= target_price - tolerance) & 
                    (recent_highs <= target_price + tolerance)
                ]
                
                if not entry_candidates.empty:
                    return {
                        'price': entry_candidates.iloc[-1],
                        'type': 'resistance',
                        'strength': 'medium'
                    }
            
            return None
            
        except Exception:
            return None
    
    def _calculate_entry_confidence(self, df: pd.DataFrame, entry_zone: Dict, 
                                  direction: str) -> float:
        """
        Calculate confidence for entry point
        
        Args:
            df: Lower timeframe data
            entry_zone: Entry zone information
            direction: Trade direction
            
        Returns:
            Confidence score (0-1)
        """
        try:
            confidence = 0.5  # Base confidence
            
            # Add confidence based on volume
            if 'volume' in df.columns:
                recent_volume = df['volume'].tail(10).mean()
                avg_volume = df['volume'].mean()
                if recent_volume > avg_volume * 1.2:
                    confidence += 0.2
            
            # Add confidence based on price action
            if direction == 'Buy':
                # Check for bullish patterns
                recent_closes = df['close'].tail(5)
                if recent_closes.iloc[-1] > recent_closes.iloc[-2]:
                    confidence += 0.1
            else:
                # Check for bearish patterns
                recent_closes = df['close'].tail(5)
                if recent_closes.iloc[-1] < recent_closes.iloc[-2]:
                    confidence += 0.1
            
            return min(max(confidence, 0), 1)
            
        except Exception:
            return 0.5
    
    def _determine_entry_trigger(self, df: pd.DataFrame, entry_zone: Dict, 
                               direction: str) -> str:
        """
        Determine entry trigger condition
        
        Args:
            df: Lower timeframe data
            entry_zone: Entry zone information
            direction: Trade direction
            
        Returns:
            Entry trigger description
        """
        try:
            if direction == 'Buy':
                return "Price breaks above entry zone with volume"
            else:
                return "Price breaks below entry zone with volume"
        except Exception:
            return "Price reaches entry zone"
