"""
ML Data Collector for Signal Performance Tracking
Collects real-time signal performance data for ML training
"""

import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List
import json
import os


class MLDataCollector:
    """
    Collects and manages signal performance data for ML training
    """
    
    def __init__(self, data_file: str = "ml_training_data.json"):
        self.data_file = data_file
        self.signals_data = self.load_data()
    
    def load_data(self) -> List[Dict]:
        """Load existing training data"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            st.warning(f"Could not load training data: {e}")
        return []
    
    def save_data(self):
        """Save training data to file"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.signals_data, f, indent=2, default=str)
        except Exception as e:
            st.error(f"Could not save training data: {e}")
    
    def add_signal_tracking(self, signal: Dict, market_data: pd.DataFrame) -> str:
        """
        Add a signal for tracking and return tracking ID
        
        Args:
            signal: Trading signal to track
            market_data: Market data at signal time
            
        Returns:
            Tracking ID for the signal
        """
        tracking_id = f"{signal.get('symbol', '')}_{datetime.now().timestamp()}"
        
        tracking_entry = {
            'tracking_id': tracking_id,
            'signal': signal,
            'market_data_snapshot': {
                'timestamp': datetime.now().isoformat(),
                'price': market_data['close'].iloc[-1],
                'volume': market_data['volume'].iloc[-1] if 'volume' in market_data.columns else 0,
                'high': market_data['high'].iloc[-1],
                'low': market_data['low'].iloc[-1]
            },
            'status': 'pending',
            'created_at': datetime.now().isoformat(),
            'outcome': None,
            'actual_pnl': None,
            'closed_at': None
        }
        
        self.signals_data.append(tracking_entry)
        self.save_data()
        
        return tracking_id
    
    def update_signal_outcome(self, tracking_id: str, outcome: str, actual_pnl: float = None):
        """
        Update signal outcome when trade is closed
        
        Args:
            tracking_id: Signal tracking ID
            outcome: 'win', 'loss', or 'breakeven'
            actual_pnl: Actual profit/loss
        """
        for entry in self.signals_data:
            if entry['tracking_id'] == tracking_id:
                entry['status'] = 'closed'
                entry['outcome'] = outcome
                entry['actual_pnl'] = actual_pnl
                entry['closed_at'] = datetime.now().isoformat()
                break
        
        self.save_data()
    
    def get_pending_signals(self) -> List[Dict]:
        """Get all pending signals"""
        return [entry for entry in self.signals_data if entry['status'] == 'pending']
    
    def get_training_data(self) -> List[Dict]:
        """Get all closed signals for training"""
        return [entry for entry in self.signals_data if entry['status'] == 'closed']
    
    def display_tracking_interface(self):
        """Display the signal tracking interface"""
        st.subheader("📊 Signal Performance Tracking")
        
        # Show pending signals
        pending_signals = self.get_pending_signals()
        if pending_signals:
            st.write(f"**Pending Signals:** {len(pending_signals)}")
            
            for signal in pending_signals[-5:]:  # Show last 5
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.write(f"**{signal['signal'].get('symbol', 'Unknown')}**")
                    st.write(f"{signal['signal'].get('direction', 'N/A')} {signal['signal'].get('type', 'N/A')}")
                
                with col2:
                    st.write(f"Entry: {signal['signal'].get('entry', 0):.5f}")
                    st.write(f"SL: {signal['signal'].get('sl', 0):.5f}")
                
                with col3:
                    st.write(f"TP: {signal['signal'].get('tp', 0):.5f}")
                    st.write(f"RR: {signal['signal'].get('rr', 0):.2f}")
                
                with col4:
                    # Outcome buttons
                    if st.button("✅ Win", key=f"win_{signal['tracking_id']}"):
                        self.update_signal_outcome(signal['tracking_id'], 'win', 100)
                        st.rerun()
                    
                    if st.button("❌ Loss", key=f"loss_{signal['tracking_id']}"):
                        self.update_signal_outcome(signal['tracking_id'], 'loss', -50)
                        st.rerun()
                    
                    if st.button("⚖️ Breakeven", key=f"breakeven_{signal['tracking_id']}"):
                        self.update_signal_outcome(signal['tracking_id'], 'breakeven', 0)
                        st.rerun()
        else:
            st.info("No pending signals to track")
        
        # Show training data stats
        training_data = self.get_training_data()
        if training_data:
            st.write(f"**Training Data:** {len(training_data)} signals")
            
            # Calculate performance stats
            wins = len([s for s in training_data if s['outcome'] == 'win'])
            losses = len([s for s in training_data if s['outcome'] == 'loss'])
            breakevens = len([s for s in training_data if s['outcome'] == 'breakeven'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Wins", wins)
            with col2:
                st.metric("Losses", losses)
            with col3:
                st.metric("Breakevens", breakevens)
            
            if wins + losses > 0:
                win_rate = wins / (wins + losses) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Export training data
        if st.button("📤 Export Training Data"):
            if training_data:
                # Convert to format suitable for ML training
                ml_data = []
                for entry in training_data:
                    ml_entry = {
                        'signal': entry['signal'],
                        'outcome': entry['outcome'],
                        'actual_pnl': entry['actual_pnl'],
                        'market_data': entry['market_data_snapshot']
                    }
                    ml_data.append(ml_entry)
                
                # Save as JSON
                with open('ml_export.json', 'w') as f:
                    json.dump(ml_data, f, indent=2, default=str)
                
                st.success("Training data exported to ml_export.json")
            else:
                st.warning("No training data to export")
