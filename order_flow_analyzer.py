"""
Order Flow Analysis Module for SMC Trading Signals
Implements advanced order flow analysis including delta, absorption, imbalance detection, and volume profile
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import streamlit as st
from scipy import stats
from collections import defaultdict


class OrderFlowAnalyzer:
    """
    Advanced order flow analysis for institutional trading patterns
    """
    
    def __init__(self, market_type: str = 'forex'):
        self.market_type = market_type.lower()
        self.volume_profile_levels = {}
        self.delta_threshold = 0.6  # 60% buying/selling pressure threshold
        self.absorption_threshold = 2.0  # 2x average volume for absorption
        
    def calculate_volume_delta(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate volume delta (buying vs selling pressure)
        
        Args:
            df: OHLCV DataFrame
            period: Lookback period for delta calculation
            
        Returns:
            Volume delta series (positive = buying pressure, negative = selling pressure)
        """
        if 'volume' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # Calculate price movement direction
        price_change = df['close'].diff()
        
        # Estimate buying vs selling volume based on price movement
        # When price goes up, assume more buying volume
        # When price goes down, assume more selling volume
        buying_volume = np.where(price_change > 0, df['volume'], 0)
        selling_volume = np.where(price_change < 0, df['volume'], 0)
        
        # Calculate rolling delta
        buying_volume_ma = pd.Series(buying_volume, index=df.index).rolling(period).sum()
        selling_volume_ma = pd.Series(selling_volume, index=df.index).rolling(period).sum()
        
        # Volume delta as percentage
        total_volume = buying_volume_ma + selling_volume_ma
        delta = (buying_volume_ma - selling_volume_ma) / total_volume.replace(0, 1)
        
        return delta.fillna(0)
    
    def detect_volume_imbalances(self, df: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """
        Detect volume imbalances (significant buying or selling pressure)
        
        Args:
            df: OHLCV DataFrame
            threshold: Delta threshold for imbalance detection
            
        Returns:
            List of imbalance events
        """
        if 'volume' not in df.columns:
            return []
        
        delta = self.calculate_volume_delta(df)
        imbalances = []
        
        for i in range(1, len(df)):
            current_delta = delta.iloc[i]
            
            # Strong buying imbalance
            if current_delta > threshold:
                imbalances.append({
                    'index': i,
                    'timestamp': df.index[i],
                    'type': 'buying_imbalance',
                    'delta': current_delta,
                    'price': df['close'].iloc[i],
                    'volume': df['volume'].iloc[i],
                    'strength': min(current_delta / threshold, 2.0)  # Cap at 2.0
                })
            
            # Strong selling imbalance
            elif current_delta < -threshold:
                imbalances.append({
                    'index': i,
                    'timestamp': df.index[i],
                    'type': 'selling_imbalance',
                    'delta': current_delta,
                    'price': df['close'].iloc[i],
                    'volume': df['volume'].iloc[i],
                    'strength': min(abs(current_delta) / threshold, 2.0)
                })
        
        return imbalances
    
    def detect_volume_absorption(self, df: pd.DataFrame, lookback: int = 20) -> List[Dict]:
        """
        Detect volume absorption (large orders being absorbed without price movement)
        
        Args:
            df: OHLCV DataFrame
            lookback: Period for average volume calculation
            
        Returns:
            List of absorption events
        """
        if 'volume' not in df.columns:
            return []
        
        avg_volume = df['volume'].rolling(lookback).mean()
        absorptions = []
        
        for i in range(lookback, len(df)):
            current_volume = df['volume'].iloc[i]
            avg_vol = avg_volume.iloc[i]
            price_change = abs(df['close'].iloc[i] - df['close'].iloc[i-1])
            
            # High volume with low price movement = absorption
            if (current_volume > avg_vol * self.absorption_threshold and 
                price_change < df['close'].iloc[i] * 0.001):  # Less than 0.1% price change
                
                # Determine absorption type based on price direction
                price_direction = df['close'].iloc[i] - df['close'].iloc[i-1]
                absorption_type = 'bullish_absorption' if price_direction > 0 else 'bearish_absorption'
                
                absorptions.append({
                    'index': i,
                    'timestamp': df.index[i],
                    'type': absorption_type,
                    'volume': current_volume,
                    'avg_volume': avg_vol,
                    'volume_ratio': current_volume / avg_vol,
                    'price': df['close'].iloc[i],
                    'price_change': price_change
                })
        
        return absorptions
    
    def calculate_volume_profile(self, df: pd.DataFrame, bins: int = 50) -> Dict:
        """
        Calculate volume profile (volume at each price level)
        
        Args:
            df: OHLCV DataFrame
            bins: Number of price bins for profile
            
        Returns:
            Volume profile data
        """
        if 'volume' not in df.columns:
            return {'profile': {}, 'poc': None, 'value_area': None}
        
        # Create price bins
        min_price = df['low'].min()
        max_price = df['high'].max()
        price_bins = np.linspace(min_price, max_price, bins + 1)
        
        # Calculate volume at each price level
        volume_profile = defaultdict(float)
        
        for i in range(len(df)):
            low = df['low'].iloc[i]
            high = df['high'].iloc[i]
            volume = df['volume'].iloc[i]
            
            # Distribute volume across price range
            if high != low:
                price_range = high - low
                volume_per_price = volume / price_range
                
                for j in range(len(price_bins) - 1):
                    bin_low = price_bins[j]
                    bin_high = price_bins[j + 1]
                    
                    # Check if price range overlaps with bin
                    if not (high < bin_low or low > bin_high):
                        overlap_low = max(low, bin_low)
                        overlap_high = min(high, bin_high)
                        overlap_ratio = (overlap_high - overlap_low) / price_range
                        
                        bin_center = (bin_low + bin_high) / 2
                        volume_profile[bin_center] += volume_per_price * overlap_ratio
        
        # Find Point of Control (POC) - price with highest volume
        if volume_profile:
            poc_price = max(volume_profile.keys(), key=lambda k: volume_profile[k])
            poc_volume = volume_profile[poc_price]
            
            # Calculate Value Area (70% of volume)
            total_volume = sum(volume_profile.values())
            target_volume = total_volume * 0.7
            
            # Sort prices by volume (descending)
            sorted_prices = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
            
            value_area_volume = 0
            value_area_prices = []
            
            for price, vol in sorted_prices:
                value_area_volume += vol
                value_area_prices.append(price)
                if value_area_volume >= target_volume:
                    break
            
            value_area_high = max(value_area_prices) if value_area_prices else poc_price
            value_area_low = min(value_area_prices) if value_area_prices else poc_price
            
            return {
                'profile': dict(volume_profile),
                'poc': poc_price,
                'poc_volume': poc_volume,
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'total_volume': total_volume
            }
        
        return {'profile': {}, 'poc': None, 'value_area': None}
    
    def detect_aggressive_orders(self, df: pd.DataFrame, threshold: float = 1.5) -> List[Dict]:
        """
        Detect aggressive orders (market orders that move price significantly)
        
        Args:
            df: OHLCV DataFrame
            threshold: Volume threshold multiplier for aggressive orders
            
        Returns:
            List of aggressive order events
        """
        if 'volume' not in df.columns:
            return []
        
        avg_volume = df['volume'].rolling(20).mean()
        aggressive_orders = []
        
        for i in range(1, len(df)):
            current_volume = df['volume'].iloc[i]
            avg_vol = avg_volume.iloc[i]
            price_change = abs(df['close'].iloc[i] - df['close'].iloc[i-1])
            price_change_pct = price_change / df['close'].iloc[i-1]
            
            # High volume with significant price movement = aggressive order
            if (current_volume > avg_vol * threshold and 
                price_change_pct > 0.002):  # More than 0.2% price change
                
                order_type = 'aggressive_buy' if df['close'].iloc[i] > df['close'].iloc[i-1] else 'aggressive_sell'
                
                aggressive_orders.append({
                    'index': i,
                    'timestamp': df.index[i],
                    'type': order_type,
                    'volume': current_volume,
                    'avg_volume': avg_vol,
                    'volume_ratio': current_volume / avg_vol,
                    'price': df['close'].iloc[i],
                    'price_change_pct': price_change_pct,
                    'strength': min(current_volume / avg_vol, 5.0)  # Cap at 5x
                })
        
        return aggressive_orders
    
    def analyze_order_flow_confluence(self, df: pd.DataFrame) -> Dict:
        """
        Analyze order flow confluence for signal strength
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Order flow confluence analysis
        """
        # Get all order flow signals
        imbalances = self.detect_volume_imbalances(df)
        absorptions = self.detect_volume_absorption(df)
        aggressive_orders = self.detect_aggressive_orders(df)
        volume_profile = self.calculate_volume_profile(df)
        
        # Calculate confluence score
        confluence_score = 0
        confluence_factors = []
        
        # Recent imbalances (last 10 bars)
        recent_imbalances = [im for im in imbalances if im['index'] >= len(df) - 10]
        if recent_imbalances:
            confluence_score += len(recent_imbalances) * 0.3
            confluence_factors.append(f"{len(recent_imbalances)} Recent Imbalances")
        
        # Recent absorptions
        recent_absorptions = [abs for abs in absorptions if abs['index'] >= len(df) - 10]
        if recent_absorptions:
            confluence_score += len(recent_absorptions) * 0.4
            confluence_factors.append(f"{len(recent_absorptions)} Volume Absorptions")
        
        # Recent aggressive orders
        recent_aggressive = [agg for agg in aggressive_orders if agg['index'] >= len(df) - 10]
        if recent_aggressive:
            confluence_score += len(recent_aggressive) * 0.3
            confluence_factors.append(f"{len(recent_aggressive)} Aggressive Orders")
        
        # Volume profile confluence
        current_price = df['close'].iloc[-1]
        if volume_profile['poc']:
            poc_distance = abs(current_price - volume_profile['poc']) / current_price
            if poc_distance < 0.02:  # Within 2% of POC
                confluence_score += 0.5
                confluence_factors.append("Near Point of Control")
        
        # Value area confluence
        if volume_profile['value_area_high'] and volume_profile['value_area_low']:
            if volume_profile['value_area_low'] <= current_price <= volume_profile['value_area_high']:
                confluence_score += 0.3
                confluence_factors.append("Within Value Area")
        
        return {
            'confluence_score': min(confluence_score, 5.0),  # Cap at 5.0
            'confluence_factors': confluence_factors,
            'imbalances': recent_imbalances,
            'absorptions': recent_absorptions,
            'aggressive_orders': recent_aggressive,
            'volume_profile': volume_profile,
            'total_signals': len(recent_imbalances) + len(recent_absorptions) + len(recent_aggressive)
        }
    
    def get_order_flow_signals(self, df: pd.DataFrame) -> List[Dict]:
        """
        Generate order flow trading signals
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            List of order flow signals
        """
        signals = []
        confluence_analysis = self.analyze_order_flow_confluence(df)
        
        # Only generate signals if there's sufficient confluence
        if confluence_analysis['confluence_score'] < 1.0:
            return signals
        
        current_price = df['close'].iloc[-1]
        atr = self._calculate_atr(df)
        current_atr = atr.iloc[-1]
        
        # Analyze recent order flow patterns
        recent_imbalances = confluence_analysis['imbalances']
        recent_absorptions = confluence_analysis['absorptions']
        recent_aggressive = confluence_analysis['aggressive_orders']
        
        # Generate signals based on order flow patterns
        for imbalance in recent_imbalances:
            if imbalance['type'] == 'buying_imbalance' and imbalance['strength'] > 1.2:
                signals.append({
                    'type': 'Order Flow Buy',
                    'direction': 'Buy',
                    'entry': current_price,
                    'sl': current_price - (current_atr * 1.5),
                    'tp': current_price + (current_atr * 2.0),
                    'rr': 2.0 / 1.5,
                    'confluences': int(confluence_analysis['confluence_score']),
                    'rationale': f"Strong buying imbalance (Delta: {imbalance['delta']:.2f})",
                    'order_flow_type': 'imbalance',
                    'strength': imbalance['strength'],
                    'volume': imbalance['volume']
                })
            
            elif imbalance['type'] == 'selling_imbalance' and imbalance['strength'] > 1.2:
                signals.append({
                    'type': 'Order Flow Sell',
                    'direction': 'Sell',
                    'entry': current_price,
                    'sl': current_price + (current_atr * 1.5),
                    'tp': current_price - (current_atr * 2.0),
                    'rr': 2.0 / 1.5,
                    'confluences': int(confluence_analysis['confluence_score']),
                    'rationale': f"Strong selling imbalance (Delta: {imbalance['delta']:.2f})",
                    'order_flow_type': 'imbalance',
                    'strength': imbalance['strength'],
                    'volume': imbalance['volume']
                })
        
        # Absorption signals
        for absorption in recent_absorptions:
            if absorption['type'] == 'bullish_absorption' and absorption['volume_ratio'] > 2.5:
                signals.append({
                    'type': 'Order Flow Buy',
                    'direction': 'Buy',
                    'entry': current_price,
                    'sl': current_price - (current_atr * 1.0),
                    'tp': current_price + (current_atr * 1.5),
                    'rr': 1.5,
                    'confluences': int(confluence_analysis['confluence_score']),
                    'rationale': f"Bullish absorption (Volume: {absorption['volume_ratio']:.1f}x avg)",
                    'order_flow_type': 'absorption',
                    'strength': absorption['volume_ratio'],
                    'volume': absorption['volume']
                })
            
            elif absorption['type'] == 'bearish_absorption' and absorption['volume_ratio'] > 2.5:
                signals.append({
                    'type': 'Order Flow Sell',
                    'direction': 'Sell',
                    'entry': current_price,
                    'sl': current_price + (current_atr * 1.0),
                    'tp': current_price - (current_atr * 1.5),
                    'rr': 1.5,
                    'confluences': int(confluence_analysis['confluence_score']),
                    'rationale': f"Bearish absorption (Volume: {absorption['volume_ratio']:.1f}x avg)",
                    'order_flow_type': 'absorption',
                    'strength': absorption['volume_ratio'],
                    'volume': absorption['volume']
                })
        
        return signals
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr, index=df.index).rolling(period).mean()
        
        return atr.fillna(method='bfill')
    
    def get_volume_profile_levels(self, df: pd.DataFrame) -> Dict:
        """
        Get key volume profile levels for support/resistance
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Volume profile levels
        """
        volume_profile = self.calculate_volume_profile(df)
        
        if not volume_profile['profile']:
            return {}
        
        # Sort by volume to find key levels
        sorted_levels = sorted(volume_profile['profile'].items(), key=lambda x: x[1], reverse=True)
        
        # Get top 5 volume levels
        key_levels = sorted_levels[:5]
        
        return {
            'poc': volume_profile['poc'],
            'value_area_high': volume_profile['value_area_high'],
            'value_area_low': volume_profile['value_area_low'],
            'key_levels': key_levels,
            'total_volume': volume_profile['total_volume']
        }
