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
import yfinance as yf
import threading
import time


class MLDataCollector:
    """
    Collects and manages signal performance data for ML training
    """
    
    def __init__(self, data_file: str = "ml_training_data.json"):
        self.data_file = data_file
        self.signals_data = self.load_data()
        # Start a single background tracker thread per user session
        if not st.session_state.get('ml_tracker_thread_started'):
            try:
                thread = threading.Thread(target=self._background_worker, daemon=True)
                thread.start()
                st.session_state['ml_tracker_thread_started'] = True
            except Exception:
                pass
    
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
    
    def add_signal_tracking(self, signal: Dict, market_data: pd.DataFrame = None) -> str:
        """
        Add a signal for tracking and return tracking ID
        
        Args:
            signal: Trading signal to track
            market_data: Market data at signal time (optional)
            
        Returns:
            Tracking ID for the signal
        """
        tracking_id = f"{signal.get('symbol', '')}_{datetime.now().timestamp()}"
        
        # Handle case when market_data is None (e.g., from scan market)
        if market_data is not None:
            market_snapshot = {
                'timestamp': datetime.now().isoformat(),
                'price': market_data['close'].iloc[-1],
                'volume': market_data['volume'].iloc[-1] if 'volume' in market_data.columns else 0,
                'high': market_data['high'].iloc[-1],
                'low': market_data['low'].iloc[-1]
            }
        else:
            # Use signal data when market_data is not available
            market_snapshot = {
                'timestamp': datetime.now().isoformat(),
                'price': signal.get('entry', 0),
                'volume': 0,
                'high': signal.get('entry', 0),
                'low': signal.get('entry', 0)
            }
        
        tracking_entry = {
            'tracking_id': tracking_id,
            'signal': signal,
            'market_data_snapshot': market_snapshot,
            'status': 'pending',
            'created_at': datetime.now().isoformat(),
            'outcome': None,
            'actual_pnl': None,
            'closed_at': None
        }
        
        self.signals_data.append(tracking_entry)
        self.save_data()
        
        return tracking_id
    
    def _compute_actual_pnl(self, signal: Dict, outcome: str) -> float:
        """Compute a simple $ P&L approximation based on entry/SL/TP."""
        entry = float(signal.get('entry', 0) or 0)
        sl = float(signal.get('sl', 0) or 0)
        tp = float(signal.get('tp', 0) or 0)
        direction = (signal.get('direction') or '').lower()
        if entry == 0:
            return 0.0
        if outcome == 'breakeven':
            return 0.0
        if direction == 'buy':
            return (tp - entry) if outcome == 'win' else (sl - entry)
        if direction == 'sell':
            return (entry - tp) if outcome == 'win' else (entry - sl)
        return 0.0

    def check_and_update_pending(self):
        """Auto-check pending signals and close them if TP/SL has been hit since creation.

        Uses yfinance intraday/hourly data where possible. Falls back to daily.
        """
        updated = 0
        now = datetime.utcnow()
        for entry in list(self.signals_data):
            try:
                if entry.get('status') != 'pending':
                    continue
                signal = entry.get('signal', {})
                symbol = signal.get('symbol') or ''
                if not symbol or symbol == 'Unknown':
                    continue
                created_iso = entry.get('created_at')
                created_dt = None
                if created_iso:
                    try:
                        created_dt = datetime.fromisoformat(created_iso.replace('Z', '+00:00'))
                    except Exception:
                        created_dt = now - timedelta(days=14)
                else:
                    created_dt = now - timedelta(days=14)

                # Fetch recent data since creation (cap max 30 days)
                start_dt = max(created_dt, now - timedelta(days=30))
                # Try hourly, if empty fallback to daily
                df = yf.download(symbol, start=start_dt, end=now + timedelta(days=1), interval='60m', progress=False)
                if df is None or df.empty:
                    df = yf.download(symbol, start=start_dt, end=now + timedelta(days=1), interval='1d', progress=False)
                if df is None or df.empty:
                    continue

                direction = (signal.get('direction') or '').lower()
                entry_price = float(signal.get('entry', 0) or 0)
                sl = float(signal.get('sl', 0) or 0)
                tp = float(signal.get('tp', 0) or 0)
                if entry_price == 0 or sl == 0 or tp == 0 or direction not in ('buy', 'sell'):
                    continue

                hit_tp = False
                hit_sl = False
                highs = df['High'] if 'High' in df.columns else df['high']
                lows = df['Low'] if 'Low' in df.columns else df['low']

                if direction == 'buy':
                    hit_tp = (highs >= tp).any()
                    hit_sl = (lows <= sl).any()
                else:  # sell
                    hit_tp = (lows <= tp).any()
                    hit_sl = (highs >= sl).any()

                # If both hit, decide by which occurred first
                if hit_tp and hit_sl:
                    # Find first time each was hit
                    tp_idx = None
                    sl_idx = None
                    for ts, row in df.iterrows():
                        hi = row['High'] if 'High' in df.columns else row['high']
                        lo = row['Low'] if 'Low' in df.columns else row['low']
                        if tp_idx is None:
                            if (direction == 'buy' and hi >= tp) or (direction == 'sell' and lo <= tp):
                                tp_idx = ts
                        if sl_idx is None:
                            if (direction == 'buy' and lo <= sl) or (direction == 'sell' and hi >= sl):
                                sl_idx = ts
                        if tp_idx is not None and sl_idx is not None:
                            break
                    outcome = 'win' if (sl_idx is None or (tp_idx is not None and tp_idx <= sl_idx)) else 'loss'
                elif hit_tp:
                    outcome = 'win'
                elif hit_sl:
                    outcome = 'loss'
                else:
                    continue

                entry['status'] = 'closed'
                entry['outcome'] = outcome
                entry['actual_pnl'] = self._compute_actual_pnl(signal, outcome)
                entry['closed_at'] = now.isoformat()
                updated += 1
            except Exception:
                continue

        if updated:
            self.save_data()
        return updated

    def _background_worker(self):
        """Background loop that periodically checks pending signals.

        Runs every 10 minutes to keep the training dataset updated automatically.
        """
        while True:
            try:
                self.check_and_update_pending()
            except Exception:
                pass
            # Sleep 10 minutes between checks
            time.sleep(600)
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
        # Show pending signals
        pending_signals = self.get_pending_signals()
        if pending_signals:
            st.write(f"**Pending Signals:** {len(pending_signals)}")
            if st.button("🔁 Auto-check pending signals now"):
                updated = self.check_and_update_pending()
                if updated:
                    st.success(f"✅ Updated {updated} signals to completed")
                else:
                    st.info("No TP/SL hits detected for pending signals")
                st.rerun()
            
            for signal in pending_signals[-5:]:  # Show last 5
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # Get symbol from tracking_id or signal data
                    symbol = signal['signal'].get('symbol', 'Unknown')
                    if symbol == 'Unknown' and '_' in signal['tracking_id']:
                        # Extract symbol from tracking_id if available
                        symbol = signal['tracking_id'].split('_')[0] or 'Unknown'
                    
                    st.write(f"**{symbol}**")
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
            
            # Manual trade completion section
            st.markdown("---")
            st.subheader("📝 Record Completed Trades")
            st.write("**Did you complete any trades that hit TP? Record them here for ML training:**")
            
            with st.expander("Add Completed Trade", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    symbol = st.text_input("Symbol (e.g., XRP, SOL)", value="XRP")
                    direction = st.selectbox("Direction", ["Buy", "Sell"])
                    signal_type = st.selectbox("Signal Type", ["Market", "Limit"])
                    entry_price = st.number_input("Entry Price", value=0.0, format="%.5f")
                    stop_loss = st.number_input("Stop Loss", value=0.0, format="%.5f")
                
                with col2:
                    take_profit = st.number_input("Take Profit", value=0.0, format="%.5f")
                    actual_pnl = st.number_input("Actual P&L", value=0.0, format="%.2f")
                    outcome = st.selectbox("Outcome", ["win", "loss", "breakeven"])
                    confluences = st.number_input("Confluences", value=3, min_value=1, max_value=10)
                
                if st.button("Record Completed Trade"):
                    if symbol and entry_price > 0 and take_profit > 0:
                        # Create a completed signal entry
                        completed_signal = {
                            'symbol': symbol,
                            'type': signal_type,
                            'direction': direction,
                            'entry': entry_price,
                            'sl': stop_loss,
                            'tp': take_profit,
                            'rr': abs(take_profit - entry_price) / abs(entry_price - stop_loss) if stop_loss != entry_price else 0,
                            'confluences': confluences,
                            'rationale': f"Manually recorded completed {direction} {signal_type} trade",
                            'confidence': 1.0  # High confidence for completed trades
                        }
                        
                        # Add to tracking system as completed
                        tracking_id = f"{symbol}_{datetime.now().timestamp()}"
                        completed_entry = {
                            'tracking_id': tracking_id,
                            'signal': completed_signal,
                            'market_data_snapshot': {
                                'timestamp': datetime.now().isoformat(),
                                'price': entry_price,
                                'volume': 0,
                                'high': entry_price,
                                'low': entry_price
                            },
                            'status': 'closed',
                            'created_at': datetime.now().isoformat(),
                            'outcome': outcome,
                            'actual_pnl': actual_pnl,
                            'closed_at': datetime.now().isoformat()
                        }
                        
                        self.signals_data.append(completed_entry)
                        self.save_data()
                        st.success(f"✅ Recorded completed {symbol} trade: {outcome.upper()} with P&L: ${actual_pnl:.2f}")
                        st.rerun()
                    else:
                        st.error("Please fill in all required fields (Symbol, Entry Price, Take Profit)")
        
        else:
            st.info("No pending signals to track")
        
        
        # Show completed trades
        completed_trades = self.get_training_data()
        if completed_trades:
            st.markdown("---")
            st.subheader("📊 Completed Trades")
            st.write(f"**Total Completed:** {len(completed_trades)} trades")
            
            # Show recent completed trades
            recent_trades = completed_trades[-10:]  # Show last 10
            for trade in reversed(recent_trades):  # Show newest first
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    symbol = trade['signal'].get('symbol', 'Unknown')
                    if symbol == 'Unknown' and '_' in trade['tracking_id']:
                        symbol = trade['tracking_id'].split('_')[0] or 'Unknown'
                    st.write(f"**{symbol}**")
                    st.write(f"{trade['signal'].get('direction', 'N/A')} {trade['signal'].get('type', 'N/A')}")
                
                with col2:
                    st.write(f"Entry: {trade['signal'].get('entry', 0):.5f}")
                    st.write(f"SL: {trade['signal'].get('sl', 0):.5f}")
                
                with col3:
                    st.write(f"TP: {trade['signal'].get('tp', 0):.5f}")
                    st.write(f"RR: {trade['signal'].get('rr', 0):.2f}")
                
                with col4:
                    # Color code the outcome
                    outcome = trade.get('outcome', 'unknown')
                    if outcome == 'win':
                        st.success(f"✅ {outcome.upper()}")
                    elif outcome == 'loss':
                        st.error(f"❌ {outcome.upper()}")
                    else:
                        st.warning(f"⚖️ {outcome.upper()}")
                    
                    pnl = trade.get('actual_pnl', 0)
                    if pnl > 0:
                        st.write(f"P&L: +${pnl:.2f}")
                    else:
                        st.write(f"P&L: ${pnl:.2f}")
                
                with col5:
                    closed_at = trade.get('closed_at', 'Unknown')
                    if closed_at != 'Unknown':
                        try:
                            from datetime import datetime
                            dt = datetime.fromisoformat(closed_at.replace('Z', '+00:00'))
                            st.write(f"Closed: {dt.strftime('%m/%d %H:%M')}")
                        except:
                            st.write(f"Closed: {closed_at}")
            
            # Calculate performance stats
            wins = len([s for s in completed_trades if s['outcome'] == 'win'])
            losses = len([s for s in completed_trades if s['outcome'] == 'loss'])
            breakevens = len([s for s in completed_trades if s['outcome'] == 'breakeven'])
            
            st.markdown("---")
            st.subheader("📈 Performance Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Wins", wins)
            with col2:
                st.metric("Losses", losses)
            with col3:
                st.metric("Breakevens", breakevens)
            with col4:
                if wins + losses > 0:
                    win_rate = wins / (wins + losses) * 100
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                else:
                    st.metric("Win Rate", "N/A")
            
            # Calculate total P&L
            total_pnl = sum([trade.get('actual_pnl', 0) for trade in completed_trades])
            avg_pnl = total_pnl / len(completed_trades) if completed_trades else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total P&L", f"${total_pnl:.2f}")
            with col2:
                st.metric("Avg P&L per Trade", f"${avg_pnl:.2f}")
        else:
            st.markdown("---")
            st.subheader("📊 Completed Trades")
            st.info("No completed trades yet. Complete some pending signals to see them here.")
        
        # Data management section
        st.markdown("---")
        st.subheader("🔧 Data Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📤 Export Training Data"):
                if completed_trades:
                    # Convert to format suitable for ML training
                    ml_data = []
                    for entry in completed_trades:
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
        
        with col2:
            if st.button("🗑️ Clear All Data"):
                if st.session_state.get('confirm_clear', False):
                    self.signals_data = []
                    self.save_data()
                    st.success("✅ All data cleared!")
                    st.session_state.confirm_clear = False
                    st.rerun()
                else:
                    st.session_state.confirm_clear = True
                    st.warning("⚠️ Click again to confirm clearing all data")
        
        with col3:
            if st.button("🔄 Refresh Data"):
                self.load_data()
                st.success("✅ Data refreshed!")
                st.rerun()
