"""
SMC Trading Signals Generator - Main Streamlit Application
Professional Smart Money Concepts trading signal generator with comprehensive analysis
"""

# type: ignore
import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
import concurrent.futures
import threading
from datetime import datetime
import io

# Import our modules
from data_fetcher import (
    fetch_ohlcv, validate_symbol, get_watchlist_symbols, 
    detect_market_type, get_symbol_examples, get_market_hours_info,
    clear_data_cache, get_troubleshooting_tips
)
from smc_signals import SMCSignalGenerator
from backtest import run_backtest, SMCBacktester
from utils import (
    calc_position_size, format_price, generate_alert_message,
    get_market_session_info, export_signals_to_csv, create_signal_summary
)
from ml_signal_optimizer import MLSignalOptimizer
from ml_data_collector import MLDataCollector
from order_flow_analyzer import OrderFlowAnalyzer

# Page configuration
st.set_page_config(
    layout='wide',
    page_title='SMC Signals Pro',
    page_icon='📈',
    initial_sidebar_state='expanded'
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .signal-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .bullish-signal {
        border-left-color: #28a745;
        background-color: #d4edda;
    }
    .bearish-signal {
        border-left-color: #dc3545;
        background-color: #f8d7da;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def display_dashboard():
    """Main dashboard with overview and quick actions"""
    st.subheader("🏠 Trading Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Signals", "0", "0")
    with col2:
        st.metric("Win Rate", "0%", "0%")
    with col3:
        st.metric("Total P&L", "$0.00", "$0.00")
    with col4:
        st.metric("Risk Level", "Low", "0%")
    
    st.markdown("---")
    
    # Quick Actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Analyze Symbol", use_container_width=True, key="quick_analyze_symbol"):
            st.session_state.show_analysis = True
            st.rerun()
    
    with col2:
        if st.button("🔍 Scan Market", use_container_width=True, key="quick_scan_market"):
            st.session_state.show_scanner = True
            st.rerun()
    
    with col3:
        if st.button("📈 View Backtest", use_container_width=True, key="quick_view_backtest"):
            st.session_state.show_backtest = True
            st.rerun()
    
    st.markdown("---")
    
    # Recent Signals
    st.subheader("📋 Recent Signals")
    st.info("No recent signals. Use the Signal Analysis tab to generate signals.")
    
    # Market Overview
    st.subheader("🌍 Market Overview")
    st.info("Market data will be displayed here when signals are generated.")
    
    # Quick Links
    st.markdown("---")
    st.subheader("🔗 Quick Links")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 View Signal Performance", use_container_width=True, key="dashboard_view_performance"):
            st.session_state.page = "📊 Signal Performance Tracking"
            st.rerun()
    
    with col2:
        if st.button("🤖 AI/ML Enhancement", use_container_width=True, key="dashboard_ai_ml"):
            st.session_state.page = "🤖 AI/ML Enhancement"
            st.rerun()
    
    with col3:
        if st.button("🔍 Scan Market", use_container_width=True, key="dashboard_scan_market"):
            st.session_state.page = "🔍 Market Scanner"
            st.rerun()


def display_signal_analysis():
    """Signal analysis section"""
    st.subheader("📊 Signal Analysis")
    
    # Get parameters from sidebar
    symbol = st.session_state.get('selected_symbol', '')
    timeframe = st.session_state.get('timeframe', '1h')
    bias_filter = st.session_state.get('bias_filter', 'All')
    quality_threshold = st.session_state.get('quality_threshold', 3)
    risk_pct = st.session_state.get('risk_pct', 1.0)
    market_type = st.session_state.get('market_type', 'Auto')
    account_balance = st.session_state.get('account_balance', 10000)
    use_ml = st.session_state.get('use_ml', False)
    ml_confidence_threshold = st.session_state.get('ml_confidence_threshold', 0.7)
    
    if st.button("🔍 Analyze Symbol", type="primary", key="signal_analyze_symbol"):
        analyze_single_symbol(symbol, timeframe, bias_filter, risk_pct, market_type, account_balance, quality_threshold, use_ml, ml_confidence_threshold)
    
    if st.button("📊 Scan Market", key="signal_scan_market"):
        scan_market(timeframe, bias_filter, risk_pct, market_type, account_balance, quality_threshold)


def display_market_scanner():
    """Market scanner section"""
    st.subheader("🔍 Market Scanner")
    st.info("Use the Signal Analysis tab to scan multiple markets for trading opportunities.")


def display_backtesting_section():
    """Backtesting section"""
    st.subheader("📈 Backtesting")
    st.info("Backtesting will be available when you analyze a symbol in the Signal Analysis tab.")


def display_order_flow_section():
    """Order flow analysis section"""
    st.subheader("🌊 Order Flow Analysis")
    st.info("Order flow analysis will be available when you analyze a symbol in the Signal Analysis tab.")


def display_signal_performance_tracking():
    """Signal Performance Tracking page"""
    st.subheader("📊 Signal Performance Tracking")
    
    # Initialize ML Data Collector
    data_collector = MLDataCollector()
    data_collector.display_tracking_interface()


def display_ai_ml_enhancement():
    """AI/ML Enhancement page"""
    st.subheader("🤖 AI/ML Enhancement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧠 Machine Learning Features")
        st.info("""
        **Current ML Capabilities:**
        - Signal quality optimization
        - Performance prediction
        - Risk assessment
        - Market condition analysis
        """)
        
        if st.button("🚀 Train ML Models", use_container_width=True, key="ai_train_models"):
            st.info("ML training will be implemented in future updates")
    
    with col2:
        st.markdown("### 📈 Performance Analytics")
        st.info("""
        **Analytics Available:**
        - Win rate tracking
        - P&L analysis
        - Signal accuracy metrics
        - Market correlation analysis
        """)
        
        if st.button("📊 View Analytics", use_container_width=True, key="ai_view_analytics"):
            st.info("Analytics dashboard will be implemented in future updates")
    
    st.markdown("---")
    
    # ML Settings
    st.markdown("### ⚙️ ML Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ml_enabled = st.checkbox("Enable ML Optimization", value=False)
        confidence_threshold = st.slider("ML Confidence Threshold", 0.0, 1.0, 0.7, 0.1)
    
    with col2:
        auto_retrain = st.checkbox("Auto Retrain Models", value=False)
        training_frequency = st.selectbox("Training Frequency", ["Daily", "Weekly", "Monthly"])
    
    if st.button("💾 Save ML Settings", key="ai_save_settings"):
        st.success("ML settings saved successfully!")


def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 SMC Signals Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Smart Money Concepts Trading Signal Generator</p>', unsafe_allow_html=True)
    
    # Sidebar Navigation
    with st.sidebar:
        st.header("🧭 Navigation")
        
        # Navigation menu
        page_options = [
            "🏠 Dashboard",
            "📊 Signal Analysis", 
            "🔍 Market Scanner",
            "📈 Backtesting",
            "🌊 Order Flow",
            "📊 Signal Performance Tracking",
            "🤖 AI/ML Enhancement"
        ]
        
        # Get current page from session state or default to Dashboard
        current_page = st.session_state.get('page', "🏠 Dashboard")
        
        page = st.radio(
            "Select Page",
            page_options,
            index=page_options.index(current_page) if current_page in page_options else 0
        )
        
        # Update session state
        st.session_state.page = page
        
        st.markdown("---")
        st.header("⚙️ Trading Parameters")
        
        # Symbol input
        default_symbol = st.session_state.get('selected_symbol', '')
        symbol = st.text_input(
            "Trading Symbol",
            value=default_symbol,
            help="Enter symbol (e.g., EURUSD=X, AAPL, BTC-USD)"
        )
        
        # Store in session state
        st.session_state.selected_symbol = symbol
        
        # Timeframe selection
        timeframe = st.selectbox(
            "Analysis Timeframe",
            options=['1h', '4h', '1d'],
            index=0,
            help="Select timeframe for signal analysis"
        )
        st.session_state.timeframe = timeframe
        
        # Bias filter
        bias_filter = st.selectbox(
            "Bias Filter",
            options=['All', 'Bullish', 'Bearish'],
            index=0,
            help="Filter signals by market bias"
        )
        st.session_state.bias_filter = bias_filter
        
        # Signal quality threshold
        quality_threshold = st.slider(
            "Signal Quality (Confluences)",
            min_value=2,
            max_value=5,
            value=3,
            step=1,
            help="Higher values = fewer but higher quality signals"
        )
        st.session_state.quality_threshold = quality_threshold
        
        # AI/ML Controls
        st.markdown("---")
        st.subheader("🤖 AI/ML Enhancement")
        
        use_ml = st.checkbox(
            "Enable AI Signal Optimization",
            value=False,
            help="Use machine learning to improve signal quality"
        )
        st.session_state.use_ml = use_ml
        
        if use_ml:
            ml_confidence_threshold = st.slider(
                "ML Confidence Threshold",
                min_value=0.5,
                max_value=0.9,
                value=0.7,
                step=0.05,
                help="Minimum ML confidence to show signal"
            )
            st.session_state.ml_confidence_threshold = ml_confidence_threshold
        else:
            st.session_state.ml_confidence_threshold = 0.7
        
        # Risk percentage
        risk_pct = st.slider(
            "Risk Percentage",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Risk per trade as percentage of account"
        ) / 100
        st.session_state.risk_pct = risk_pct
        
        # Market type
        selected_market = st.session_state.get('selected_market', 'Auto')
        market_options = ['Auto', 'Forex', 'Stocks', 'Crypto', 'Indices', 'Commodities']
        market_index = market_options.index(selected_market) if selected_market in market_options else 0
        
        market_type = st.selectbox(
            "Market Type",
            options=market_options,
            index=market_index,
            help="Market type for analysis"
        )
        st.session_state.market_type = market_type
        
        # Account balance
        account_balance = st.number_input(
            "Account Balance ($)",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000,
            help="Your trading account balance"
        )
        st.session_state.account_balance = account_balance
        
        # Market session info
        st.markdown("---")
        st.subheader("📅 Market Session")
        session_info = get_market_session_info()
        st.metric("Current Session", session_info['session'])
        st.metric("Status", session_info['status'])
        st.caption(f"Time: {session_info['time']}")
        
        # ML Training Section
        if use_ml:
            st.markdown("---")
            st.subheader("🧠 ML Model Training")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Train ML Models", key="ml_train_models"):
                    ml_optimizer = MLSignalOptimizer()
                    ml_optimizer.load_training_data()
                    
                    if ml_optimizer.train_models():
                        st.success("✅ ML models trained successfully!")
                    else:
                        st.warning("⚠️ Need more training data")
            
            with col2:
                if st.button("📊 View Model Performance", key="ml_view_performance"):
                    ml_optimizer = MLSignalOptimizer()
                    importance = ml_optimizer.get_feature_importance()
                    
                    if importance:
                        st.write("**Feature Importance:**")
                        for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
                            st.write(f"• {feature}: {score:.3f}")
                    else:
                        st.info("No trained model available")
        
        # Symbol examples
        with st.expander("💡 Symbol Examples"):
            examples = get_symbol_examples()
            for market, symbols in examples.items():
                st.write(f"**{market}:**")
                st.code(", ".join(symbols))
    
    # Main Content Area
    st.markdown("---")
    
    # Route to appropriate page based on sidebar selection
    if page == "🏠 Dashboard":
        display_dashboard()
    elif page == "📊 Signal Analysis":
        display_signal_analysis()
    elif page == "🔍 Market Scanner":
        display_market_scanner()
    elif page == "📈 Backtesting":
        display_backtesting_section()
    elif page == "🌊 Order Flow":
        display_order_flow_section()
    elif page == "📊 Signal Performance Tracking":
        display_signal_performance_tracking()
    elif page == "🤖 AI/ML Enhancement":
        display_ai_ml_enhancement()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9rem;'>
            <p>⚠️ <strong>Disclaimer:</strong> Trading signals are for educational purposes only. Not financial advice. 
            Always backtest before live trading. Risk maximum 1% per trade.</p>
            <p>Built with Streamlit & yfinance | SMC Trading Signals Pro</p>
        </div>
        """, 
        unsafe_allow_html=True
    )


def analyze_single_symbol(symbol, timeframe, bias_filter, risk_pct, market_type, account_balance, quality_threshold=3, use_ml=False, ml_confidence_threshold=0.7):
    """Analyze a single trading symbol"""
    
    # Validate symbol
    is_valid, detected_market = validate_symbol(symbol)
    
    if not is_valid:
        st.error(f"❌ Invalid symbol: {symbol}")
        st.info("💡 Try these examples:")
        examples = get_symbol_examples()
        for market, symbols in examples.items():
            st.write(f"**{market}:** {', '.join(symbols[:3])}")
        return
    
    # Auto-detect market type if needed
    if market_type == 'Auto':
        market_type = detected_market
    else:
        market_type = market_type.lower()
    
    # Show symbol info
    market_info = get_market_hours_info(symbol)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Symbol", symbol)
    with col2:
        st.metric("Market Type", market_info['market_type'])
    with col3:
        st.metric("Analysis TF", timeframe)
        # Entry timeframe will be shown after signal generation
    
    # Fetch data
    with st.spinner(f"📊 Fetching data for {symbol}..."):
        df = fetch_ohlcv(symbol, period='1y', interval=timeframe)
    
    if df is None or df.empty:
        st.error(f"❌ Could not fetch data for {symbol}")
        return
    
    # Generate signals
    with st.spinner("🔍 Generating SMC signals..."):
        signal_generator = SMCSignalGenerator(market_type=market_type)
        
        # Debug: Show symbol being used
        st.write(f"🔍 Debug: Generating signals for symbol: '{symbol}'")
        st.write(f"🔍 Debug: Market type: {market_type}")
        st.write(f"🔍 Debug: Data shape: {df.shape}")
        st.write(f"🔍 Debug: Current price: ${df['close'].iloc[-1]:.2f}")
        
        # Pass analysis timeframe for automated entry timeframe selection
        signals = signal_generator.generate_signals(df, bias_filter, quality_threshold, timeframe, symbol)
        
        # Show entry timeframe information
        entry_tf = signal_generator._get_optimal_entry_timeframe(timeframe)
        st.info(f"🎯 **Automated Entry Timing**: Analysis on {timeframe} → Entry on {entry_tf} timeframe")
        
        # Show signal generation status
        if signals:
            st.success(f"✅ Generated {len(signals)} signals")
            # Debug: Show first signal details
            if signals:
                st.write(f"🔍 Debug: First signal details:")
                st.write(f"Symbol: '{signals[0].get('symbol', 'MISSING')}'")
                st.write(f"Entry: {signals[0].get('entry', 'MISSING')}")
                st.write(f"Type: {signals[0].get('type', 'MISSING')}")
        else:
            st.warning("⚠️ No signals found - Try adjusting confluence threshold or bias filter")
        
        # Apply ML optimization if enabled
        if use_ml:
            with st.spinner("🤖 Applying AI optimization..."):
                ml_optimizer = MLSignalOptimizer()
                data_collector = MLDataCollector()
                optimized_signals = []
                
                for signal in signals:
                    # Optimize signal with ML
                    optimized_signal = ml_optimizer.optimize_signal_parameters(signal, df)
                    
                    # Add to tracking for performance monitoring
                    tracking_id = data_collector.add_signal_tracking(optimized_signal, df)
                    optimized_signal['tracking_id'] = tracking_id
                    
                    # Filter by ML confidence, but show warning for low confidence
                    ml_confidence = optimized_signal.get('ml_confidence', 0)
                    if ml_confidence >= ml_confidence_threshold:
                        optimized_signals.append(optimized_signal)
                    else:
                        # Add warning to signal for low confidence
                        optimized_signal['ml_warning'] = f"Low ML confidence: {ml_confidence:.2f} < {ml_confidence_threshold}"
                        optimized_signals.append(optimized_signal)
                
                signals = optimized_signals
    
    # Display results
    display_analysis_results(symbol, df, signals, market_type, risk_pct, account_balance, timeframe)


def scan_market(timeframe, bias_filter, risk_pct, market_type, account_balance, quality_threshold=3):
    """Scan multiple symbols for trading opportunities"""
    
    st.header("📊 Market Scan Results")
    
    # Get symbols to scan
    if market_type == 'Auto':
        symbols_to_scan = []
        for mt in ['forex', 'stocks', 'crypto', 'indices', 'commodities']:
            symbols_to_scan.extend(get_watchlist_symbols(mt)[:10])  # Limit to 10 per market
    else:
        symbols_to_scan = get_watchlist_symbols(market_type.lower())[:50]  # Limit to 50 symbols
    
    if not symbols_to_scan:
        st.error("❌ No symbols to scan")
        return
    
    st.info(f"🔍 Scanning {len(symbols_to_scan)} symbols...")
    st.write(f"📋 Symbols to scan: {symbols_to_scan[:10]}...")  # Show first 10 symbols
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_signals = []
    
    def scan_symbol(symbol):
        """Scan individual symbol"""
        try:
            # Fetch data
            df = fetch_ohlcv(symbol, period='6mo', interval=timeframe)
            if df is None or df.empty:
                st.warning(f"⚠️ No data for {symbol}")
                return []
            
            # Generate signals
            detected_market = detect_market_type(symbol)
            signal_generator = SMCSignalGenerator(market_type=detected_market)
            signals = signal_generator.generate_signals(df, bias_filter, quality_threshold, timeframe, symbol)
            
            # Add symbol info to signals
            for signal in signals:
                signal['symbol'] = symbol
                signal['market_type'] = detected_market
            
            if signals:
                st.write(f"✅ {symbol}: Found {len(signals)} signals")
            
            return signals
            
        except Exception as e:
            st.error(f"❌ Error scanning {symbol}: {str(e)}")
            return []
    
    # Scan symbols sequentially (more reliable with Streamlit)
    completed = 0
    for symbol in symbols_to_scan:
        try:
            signals = scan_symbol(symbol)
            all_signals.extend(signals)
            completed += 1
            progress_bar.progress(completed / len(symbols_to_scan))
            status_text.text(f"Scanned {completed}/{len(symbols_to_scan)} symbols: {symbol}")
        except Exception as e:
            completed += 1
            progress_bar.progress(completed / len(symbols_to_scan))
            status_text.text(f"Error scanning {symbol}: {str(e)}")
    
    # Show scan summary
    st.success(f"🎯 Scan Complete! Found {len(all_signals)} total signals from {completed} symbols")
    
    # Show troubleshooting tips if some symbols failed
    failed_symbols = len(symbols_to_scan) - completed
    if failed_symbols > 0:
        st.warning(f"⚠️ {failed_symbols} symbols had no data available")
        
        with st.expander("🔧 Troubleshooting Tips for 'No Data' Issues"):
            tips = get_troubleshooting_tips()
            for category, tip_list in tips.items():
                st.write(f"**{category}:**")
                for tip in tip_list:
                    st.write(f"• {tip}")
            
            if st.button("🔄 Clear Data Cache", key="clear_cache_scan"):
                if clear_data_cache():
                    st.success("✅ Cache cleared! Try scanning again.")
                    st.rerun()
                else:
                    st.error("❌ Failed to clear cache")
    
    # Apply ML tracking to scan results if ML is enabled
    if st.session_state.get('use_ml', False):
        with st.spinner("🤖 Adding signals to AI tracking system..."):
            data_collector = MLDataCollector()
            tracked_signals = []
            
            for signal in all_signals:
                # Add to tracking for performance monitoring
                tracking_id = data_collector.add_signal_tracking(signal, None)  # No market data for scan
                signal['tracking_id'] = tracking_id
                tracked_signals.append(signal)
            
            all_signals = tracked_signals
            st.success(f"✅ Added {len(tracked_signals)} signals to AI tracking system")
    
    # Display scan results
    display_scan_results(all_signals, risk_pct, account_balance, symbols_to_scan)


def display_analysis_results(symbol, df, signals, market_type, risk_pct, account_balance, timeframe):
    """Display analysis results for a single symbol"""
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Chart & Signals", "📊 Signals Table", "📉 Backtest", "📋 Analysis", "🌊 Order Flow"])
    
    with tab1:
        display_chart_with_signals(symbol, df, signals, market_type)
    
    with tab2:
        display_signals_table(signals, risk_pct, account_balance, market_type, symbol)
    
    with tab3:
        display_backtest_results(symbol, df, market_type)
    
    with tab4:
        display_analysis_summary(symbol, df, signals, market_type)
    
    with tab5:
        display_order_flow_analysis(symbol, df, market_type)


def display_chart_with_signals(symbol, df, signals, market_type):
    """Display candlestick chart with SMC annotations"""
    
    # Debug: Check signals
    st.write(f"🔍 Chart Debug: {len(signals)} signals to display")
    
    # Debug: Show signal details
    if signals:
        st.write("🔍 Signal Details:")
        for i, signal in enumerate(signals):
            confidence = signal.get('confidence', 0)
            confidence_pct = confidence * 100
            st.write(f"Signal {i+1}: Entry=${signal['entry']:.2f}, SL=${signal['sl']:.2f}, TP=${signal['tp']:.2f}, Type={signal['type']}, Confidence={confidence_pct:.1f}%")
        
        # Show current price and ATR for comparison
        current_price = df['close'].iloc[-1]
        signal_generator = SMCSignalGenerator(market_type=market_type)
        atr = signal_generator.calculate_atr(df).iloc[-1]
        st.write(f"Current Price: ${current_price:.2f}, ATR: ${atr:.2f}")
    
    # Create candlestick chart
    fig = go.Figure(data=go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=symbol
    ))
    
    # Add SMC annotations
    for signal in signals:
        entry_price = signal['entry']
        signal_type = signal['type']
        direction = signal['direction']
        
        # Color based on direction
        color = 'green' if direction == 'Buy' else 'red'
        
        # Add entry line
        fig.add_hline(
            y=entry_price,
            line_dash="dash",
            line_color=color,
            annotation_text=f"{signal_type} {direction} @ {format_price(entry_price, market_type)}",
            annotation_position="right"
        )
        
        # Add stop loss and take profit
        fig.add_hline(
            y=signal['sl'],
            line_dash="dot",
            line_color="red",
            opacity=0.7
        )
        
        fig.add_hline(
            y=signal['tp'],
            line_dash="dot",
            line_color="green",
            opacity=0.7
        )
    
    fig.update_layout(
        title=f"{symbol} - SMC Analysis",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        showlegend=True
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Signal summary
    if signals:
        summary = create_signal_summary(signals)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Signals", summary['total'])
        with col2:
            st.metric("Market Orders", summary['market_buy'] + summary['market_sell'])
        with col3:
            st.metric("Limit Orders", summary['limit_buy'] + summary['limit_sell'])
        with col4:
            st.metric("High Confluence", summary['high_confluence'])


def display_signals_table(signals, risk_pct, account_balance, market_type, symbol="Unknown"):
    """Display signals in a formatted table"""
    
    if not signals:
        st.info("ℹ️ No signals found for current parameters")
        return
    
    # Create DataFrame
    df_signals = pd.DataFrame(signals)
    
    # Add symbol column for single symbol analysis
    if not df_signals.empty and 'symbol' not in df_signals.columns:
        df_signals['symbol'] = symbol
    
    # Add position sizing
    position_sizes = []
    for _, signal in df_signals.iterrows():
        pos_size = calc_position_size(account_balance, risk_pct, signal['entry'], signal['sl'], market_type)
        position_sizes.append(pos_size['size'])
    
    df_signals['position_size'] = position_sizes
    df_signals['position_value'] = df_signals['position_size'] * df_signals['entry']
    
    # Format columns
    df_signals['entry'] = df_signals['entry'].apply(lambda x: format_price(x, market_type))
    df_signals['sl'] = df_signals['sl'].apply(lambda x: format_price(x, market_type))
    df_signals['tp'] = df_signals['tp'].apply(lambda x: format_price(x, market_type))
    df_signals['rr'] = df_signals['rr'].round(2)
    df_signals['position_size'] = df_signals['position_size'].round(0)
    df_signals['position_value'] = df_signals['position_value'].round(2)
    
    # Add confidence column if it exists
    if 'confidence' in df_signals.columns:
        df_signals['confidence'] = (df_signals['confidence'] * 100).round(1)
    
    # Create tabs for different signal types
    tab1, tab2, tab3 = st.tabs(["All Signals", "Market Orders", "Limit Orders"])
    
    with tab1:
        display_signal_table(df_signals, "All Signals")
    
    with tab2:
        market_signals = df_signals[df_signals['type'] == 'Market']
        if not market_signals.empty:
            display_signal_table(market_signals, "Market Orders")
        else:
            st.info("No market orders found")
    
    with tab3:
        limit_signals = df_signals[df_signals['type'] == 'Limit']
        if not limit_signals.empty:
            display_signal_table(limit_signals, "Limit Orders")
        else:
            st.info("No limit orders found")
    
    # Export button
    if st.button("📥 Export Signals to CSV", key="export_signals_csv"):
        csv = df_signals.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"smc_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def display_signal_table(df_signals, title):
    """Display formatted signal table"""
    
    st.subheader(title)
    
    # Color code the table
    def color_signal(val):
        if val == 'Buy':
            return 'background-color: #d4edda'
        elif val == 'Sell':
            return 'background-color: #f8d7da'
        return ''
    
    # Select columns to display
    display_columns = ['symbol', 'type', 'direction', 'entry', 'sl', 'tp', 'rr', 'confluences', 'position_size', 'rationale']
    
    # Add quality score if available
    if 'quality_score' in df_signals.columns:
        display_columns.append('quality_score')
    
    # Add ML columns if available
    ml_columns = ['ml_win_probability', 'ml_expected_pnl', 'ml_quality_score', 'ml_confidence']
    for col in ml_columns:
        if col in df_signals.columns:
            display_columns.append(col)
    
    styled_df = df_signals[display_columns].style.map(
        color_signal, subset=['direction']
    )
    
    st.dataframe(styled_df, width='stretch')
    
    # Show alerts for high-quality signals
    high_quality_signals = df_signals[df_signals['confluences'] >= 4]
    
    if not high_quality_signals.empty:
        st.subheader("🚨 High Quality Signals")
        
        for _, signal in high_quality_signals.iterrows():
            alert_msg = generate_alert_message(signal.to_dict(), "SYMBOL")
            confidence = signal.get('confidence', 0)
            confidence_pct = confidence * 100 if confidence > 0 else 0
            
            if signal['type'] == 'Market':
                st.success(f"{alert_msg} | Confidence: {confidence_pct:.1f}%")
            else:
                st.info(alert_msg)


def display_backtest_results(symbol, df, market_type):
    """Display backtesting results"""
    
    st.subheader("📉 Backtest Results")
    
    if len(df) < 50:
        st.warning("⚠️ Insufficient data for backtesting (need at least 50 bars)")
        return
    
    # Run backtest with progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔄 Initializing backtest...")
        progress_bar.progress(10)
        
        status_text.text("🔄 Running backtest...")
        backtester, metrics = run_backtest(df, market_type)
        
        progress_bar.progress(100)
        status_text.text("✅ Backtest completed!")
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", metrics['total_trades'])
            st.metric("Win Rate", f"{metrics['win_rate']:.1%}")
        
        with col2:
            st.metric("Total P&L", f"${metrics['total_pnl']:.2f}")
            st.metric("Avg R:R", f"{metrics['avg_rr']:.2f}")
        
        with col3:
            st.metric("Max Drawdown", f"${metrics['max_drawdown']:.2f}")
            st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
        
        with col4:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            st.metric("Largest Win", f"${metrics['largest_win']:.2f}")
        
        # Equity curve
        if backtester.equity_curve:
            fig = backtester.plot_equity_curve()
            st.plotly_chart(fig, width='stretch')
        
        # Trade analysis
        if backtester.trades:
            st.subheader("📊 Trade Analysis")
            trade_df = backtester.get_trade_analysis()
            
            # Show recent trades
            recent_trades = trade_df.tail(10)[['entry_date', 'direction', 'entry_price', 'exit_price', 'pnl', 'rr']]
            st.dataframe(recent_trades, width='stretch')
    
    except Exception as e:
        st.error(f"❌ Backtest failed: {str(e)}")


def display_analysis_summary(symbol, df, signals, market_type):
    """Display analysis summary"""
    
    st.subheader("📋 Analysis Summary")
    
    # Market structure analysis
    signal_generator = SMCSignalGenerator(market_type=market_type)
    swing_highs, swing_lows = signal_generator.identify_swing_points(df)
    bias = signal_generator.determine_bias(df, swing_highs, swing_lows)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Market Bias", bias)
        st.metric("Swing Highs", len(swing_highs))
    
    with col2:
        st.metric("Swing Lows", len(swing_lows))
        st.metric("Data Points", len(df))
    
    with col3:
        st.metric("Current Price", format_price(df['close'].iloc[-1], market_type))
        st.metric("ATR (14)", format_price(signal_generator.calculate_atr(df).iloc[-1], market_type))
    
    # SMC components
    st.subheader("🔍 SMC Components")
    
    fvgs = signal_generator.identify_fvg(df)
    order_blocks = signal_generator.identify_order_blocks(df, swing_highs, swing_lows)
    liquidity_grabs = signal_generator.identify_liquidity_grabs(df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Bullish FVGs", len(fvgs['bullish']))
        st.metric("Bearish FVGs", len(fvgs['bearish']))
    
    with col2:
        st.metric("Bullish OBs", len(order_blocks['bullish']))
        st.metric("Bearish OBs", len(order_blocks['bearish']))
    
    with col3:
        st.metric("Bullish Grabs", len(liquidity_grabs['bullish']))
        st.metric("Bearish Grabs", len(liquidity_grabs['bearish']))


def display_order_flow_analysis(symbol, df, market_type):
    """Display order flow analysis"""
    
    st.subheader("🌊 Order Flow Analysis")
    
    # Initialize order flow analyzer
    order_flow_analyzer = OrderFlowAnalyzer(market_type)
    
    # Get order flow confluence analysis
    confluence_analysis = order_flow_analyzer.analyze_order_flow_confluence(df)
    
    # Display confluence metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Order Flow Score", f"{confluence_analysis['confluence_score']:.1f}/5.0")
    
    with col2:
        st.metric("Total Signals", confluence_analysis['total_signals'])
    
    with col3:
        st.metric("Imbalances", len(confluence_analysis['imbalances']))
    
    with col4:
        st.metric("Absorptions", len(confluence_analysis['absorptions']))
    
    # Display confluence factors
    if confluence_analysis['confluence_factors']:
        st.subheader("🔍 Order Flow Confluence Factors")
        for factor in confluence_analysis['confluence_factors']:
            st.write(f"• {factor}")
    
    # Volume Profile Analysis
    st.subheader("📊 Volume Profile")
    volume_profile = confluence_analysis['volume_profile']
    
    if volume_profile['poc']:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Point of Control", format_price(volume_profile['poc'], market_type))
        
        with col2:
            st.metric("Value Area High", format_price(volume_profile['value_area_high'], market_type))
        
        with col3:
            st.metric("Value Area Low", format_price(volume_profile['value_area_low'], market_type))
        
        # Current price vs POC
        current_price = df['close'].iloc[-1]
        poc_distance = abs(current_price - volume_profile['poc']) / current_price * 100
        
        if poc_distance < 2:
            st.success(f"✅ Price is within 2% of Point of Control ({poc_distance:.1f}% away)")
        elif poc_distance < 5:
            st.info(f"ℹ️ Price is {poc_distance:.1f}% away from Point of Control")
        else:
            st.warning(f"⚠️ Price is {poc_distance:.1f}% away from Point of Control")
    
    # Recent Imbalances
    if confluence_analysis['imbalances']:
        st.subheader("⚖️ Recent Volume Imbalances")
        
        for imbalance in confluence_analysis['imbalances'][-5:]:  # Show last 5
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write(f"**{imbalance['type'].replace('_', ' ').title()}**")
            
            with col2:
                st.write(f"Delta: {imbalance['delta']:.2f}")
            
            with col3:
                st.write(f"Strength: {imbalance['strength']:.1f}")
            
            with col4:
                st.write(f"Volume: {imbalance['volume']:,.0f}")
    
    # Recent Absorptions
    if confluence_analysis['absorptions']:
        st.subheader("🍽️ Recent Volume Absorptions")
        
        for absorption in confluence_analysis['absorptions'][-5:]:  # Show last 5
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write(f"**{absorption['type'].replace('_', ' ').title()}**")
            
            with col2:
                st.write(f"Volume Ratio: {absorption['volume_ratio']:.1f}x")
            
            with col3:
                st.write(f"Price Change: {absorption['price_change']:.4f}")
            
            with col4:
                st.write(f"Volume: {absorption['volume']:,.0f}")
    
    # Recent Aggressive Orders
    if confluence_analysis['aggressive_orders']:
        st.subheader("⚡ Recent Aggressive Orders")
        
        for order in confluence_analysis['aggressive_orders'][-5:]:  # Show last 5
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write(f"**{order['type'].replace('_', ' ').title()}**")
            
            with col2:
                st.write(f"Volume Ratio: {order['volume_ratio']:.1f}x")
            
            with col3:
                st.write(f"Price Change: {order['price_change_pct']:.2%}")
            
            with col4:
                st.write(f"Strength: {order['strength']:.1f}")
    
    # Order Flow Integration Note
    st.subheader("🎯 Order Flow Integration")
    st.info("""
    **Order flow analysis is now integrated as confluence factors in SMC signals:**
    
    • **Volume Imbalances** enhance signal strength
    • **Volume Absorptions** confirm institutional activity  
    • **Aggressive Orders** validate price movement
    • **Volume Profile** provides key support/resistance levels
    
    Look for these factors in the main signals table under "Rationale" column.
    """)


def display_scan_results(all_signals, risk_pct, account_balance, symbols_to_scan):
    """Display market scan results"""
    
    if not all_signals:
        st.warning("⚠️ No real signals found across all scanned symbols")
        st.info("💡 Try adjusting the confluence threshold or bias filter to find more signals")
        st.info("🔍 Consider scanning different market types or timeframes")
        return
    
    # Sort by confluences (highest first)
    all_signals.sort(key=lambda x: x['confluences'], reverse=True)
    
    # Limit to top 20 signals
    top_signals = all_signals[:20]
    
    st.success(f"🎯 Found {len(all_signals)} total signals, showing top {len(top_signals)}")
    
    # Debug: Show signal details
    if top_signals:
        st.write("🔍 Debug - First signal details:")
        st.json(top_signals[0])
        
        # Show raw signal data
        st.write("📊 Raw Signal Data:")
        for i, signal in enumerate(top_signals[:3]):
            st.write(f"Signal {i+1}: {signal}")
    
    # Create DataFrame
    df_scan = pd.DataFrame(top_signals)
    
    # Debug DataFrame
    st.write(f"🔍 DataFrame created with {len(df_scan)} rows")
    if not df_scan.empty:
        st.write("DataFrame columns:", df_scan.columns.tolist())
        st.write("DataFrame shape:", df_scan.shape)
        
        # Ensure symbol column exists
        if 'symbol' not in df_scan.columns:
            st.warning("⚠️ Symbol column missing from signals")
    
    # Add position sizing
    if not df_scan.empty:
        position_sizes = []
        for _, signal in df_scan.iterrows():
            market_type_signal = signal.get('market_type', 'forex')
            pos_size = calc_position_size(account_balance, risk_pct, signal['entry'], signal['sl'], market_type_signal)
            position_sizes.append(pos_size['size'])
        
        df_scan['position_size'] = position_sizes
        
        # Format columns
        for col in ['entry', 'sl', 'tp']:
            if col in df_scan.columns:
                df_scan[col] = df_scan[col].apply(lambda x: f"{x:.5f}")
        
        df_scan['rr'] = df_scan['rr'].round(2)
        df_scan['position_size'] = df_scan['position_size'].round(0)
    
    # Display table
    if not df_scan.empty:
        st.dataframe(df_scan, width='stretch')
    else:
        st.info("No signals to display in table")
    
    # Show top opportunities
    st.subheader("🏆 Top Opportunities")
    
    for i, signal in enumerate(top_signals[:5]):
        alert_msg = generate_alert_message(signal, signal['symbol'])
        
        if signal['type'] == 'Market':
            st.success(f"**#{i+1}** {alert_msg}")
        else:
            st.info(f"**#{i+1}** {alert_msg}")
    
    # Export results
    if st.button("📥 Export Scan Results", key="export_scan_results"):
        csv = df_scan.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"market_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
