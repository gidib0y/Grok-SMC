"""
SMC Trading Signals Generator - Main Streamlit Application
Professional Smart Money Concepts trading signal generator with comprehensive analysis
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
import threading
from datetime import datetime
import io

# Import our modules
from data_fetcher import (
    fetch_ohlcv, validate_symbol, get_watchlist_symbols, 
    detect_market_type, get_symbol_examples, get_market_hours_info
)
from smc_signals import SMCSignalGenerator
from backtest import run_backtest, SMCBacktester
from utils import (
    calc_position_size, format_price, generate_alert_message,
    get_market_session_info, export_signals_to_csv, create_signal_summary
)
from ml_signal_optimizer import MLSignalOptimizer
from ml_data_collector import MLDataCollector

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

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 SMC Signals Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Smart Money Concepts Trading Signal Generator</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Trading Parameters")
        
        # Symbol input
        default_symbol = st.session_state.get('selected_symbol', '')
        symbol = st.text_input(
            "Trading Symbol",
            value=default_symbol,
            help="Enter symbol (e.g., EURUSD=X, AAPL, BTC-USD) or use quick selection below"
        )
        
        # Timeframe selection
        timeframe = st.selectbox(
            "Analysis Timeframe",
            options=['1h', '4h', '1d'],
            index=0,
            help="Select timeframe for signal analysis. Entry timeframe will be automatically optimized."
        )
        
        # Bias filter
        bias_filter = st.selectbox(
            "Bias Filter",
            options=['All', 'Bullish', 'Bearish'],
            index=0,
            help="Filter signals by market bias"
        )
        
        # Signal quality threshold
        quality_threshold = st.slider(
            "Signal Quality (Confluences)",
            min_value=2,
            max_value=5,
            value=3,
            step=1,
            help="Higher values = fewer but higher quality signals"
        )
        
        # AI/ML Controls
        st.markdown("---")
        st.subheader("🤖 AI/ML Enhancement")
        
        use_ml = st.checkbox(
            "Enable AI Signal Optimization",
            value=False,
            help="Use machine learning to improve signal quality"
        )
        
        if use_ml:
            ml_confidence_threshold = st.slider(
                "ML Confidence Threshold",
                min_value=0.5,
                max_value=0.9,
                value=0.7,
                step=0.05,
                help="Minimum ML confidence to show signal"
            )
        else:
            ml_confidence_threshold = 0.7  # Default value when ML is disabled
        
        # Risk percentage
        risk_pct = st.slider(
            "Risk Percentage",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Risk per trade as percentage of account"
        ) / 100
        
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
        
        # Account balance
        account_balance = st.number_input(
            "Account Balance ($)",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000,
            help="Your trading account balance"
        )
        
        # Buttons
        col1, col2 = st.columns(2)
        
        with col1:
            analyze_btn = st.button("🔍 Analyze Symbol", type="primary")
        
        with col2:
            scan_btn = st.button("📊 Scan Market")
        
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
                if st.button("🔄 Train ML Models"):
                    ml_optimizer = MLSignalOptimizer()
                    ml_optimizer.load_training_data()
                    
                    if ml_optimizer.train_models():
                        st.success("✅ ML models trained successfully!")
                    else:
                        st.warning("⚠️ Need more training data")
            
            with col2:
                if st.button("📊 View Model Performance"):
                    ml_optimizer = MLSignalOptimizer()
                    importance = ml_optimizer.get_feature_importance()
                    
                    if importance:
                        st.write("**Feature Importance:**")
                        for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
                            st.write(f"• {feature}: {score:.3f}")
                    else:
                        st.info("No trained model available")
            
            # Data Collection Interface
            st.markdown("---")
            data_collector = MLDataCollector()
            data_collector.display_tracking_interface()
        
        # Symbol examples
        with st.expander("💡 Symbol Examples"):
            examples = get_symbol_examples()
            for market, symbols in examples.items():
                st.write(f"**{market}:**")
                st.code(", ".join(symbols))
    
    # Main content area
    if analyze_btn:
            analyze_single_symbol(symbol, timeframe, bias_filter, risk_pct, market_type, account_balance, quality_threshold, use_ml, ml_confidence_threshold)
    
    if scan_btn:
        scan_market(timeframe, bias_filter, risk_pct, market_type, account_balance, quality_threshold)
    
    # Default view - show instructions instead of auto-generating signals
    if not analyze_btn and not scan_btn:
        st.info("👆 Please select a symbol above and click 'Analyze Symbol' to generate trading signals")
        
        # Show market selection interface
        st.markdown("---")
        st.subheader("📊 Quick Market Selection")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Forex Pairs**")
            forex_symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X"]
            for symbol in forex_symbols:
                if st.button(f"📈 {symbol}", key=f"forex_{symbol}"):
                    st.session_state.selected_symbol = symbol
                    st.session_state.selected_market = "forex"
                    st.rerun()
        
        with col2:
            st.markdown("**Crypto**")
            crypto_symbols = ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD", "DOT-USD", "MATIC-USD"]
            for symbol in crypto_symbols:
                if st.button(f"₿ {symbol}", key=f"crypto_{symbol}"):
                    st.session_state.selected_symbol = symbol
                    st.session_state.selected_market = "crypto"
                    st.rerun()
        
        with col3:
            st.markdown("**Stocks**")
            stock_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA"]
            for symbol in stock_symbols:
                if st.button(f"📊 {symbol}", key=f"stock_{symbol}"):
                    st.session_state.selected_symbol = symbol
                    st.session_state.selected_market = "stocks"
                    st.rerun()
        
        # Show selected symbol if any
        if hasattr(st.session_state, 'selected_symbol'):
            st.success(f"Selected: {st.session_state.selected_symbol} ({st.session_state.get('selected_market', 'unknown')})")
            st.info("Click 'Analyze Symbol' to generate signals for the selected symbol")
    
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
        
        # Pass analysis timeframe for automated entry timeframe selection
        signals = signal_generator.generate_signals(df, bias_filter, quality_threshold, timeframe)
        
        # Show entry timeframe information
        entry_tf = signal_generator._get_optimal_entry_timeframe(timeframe)
        st.info(f"🎯 **Automated Entry Timing**: Analysis on {timeframe} → Entry on {entry_tf} timeframe")
        
        # Show signal generation status
        if signals:
            st.success(f"✅ Generated {len(signals)} signals")
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
                return []
            
            # Generate signals
            detected_market = detect_market_type(symbol)
            signal_generator = SMCSignalGenerator(market_type=detected_market)
            signals = signal_generator.generate_signals(df, bias_filter, quality_threshold, timeframe)
            
            # Add symbol info to signals
            for signal in signals:
                signal['symbol'] = symbol
                signal['market_type'] = detected_market
            
            return signals
            
        except Exception as e:
            return []
    
    # Use threading for concurrent scanning
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_symbol = {executor.submit(scan_symbol, symbol): symbol for symbol in symbols_to_scan}
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                signals = future.result()
                all_signals.extend(signals)
                completed += 1
                progress_bar.progress(completed / len(symbols_to_scan))
                status_text.text(f"Scanned {completed}/{len(symbols_to_scan)} symbols")
            except Exception as e:
                completed += 1
                progress_bar.progress(completed / len(symbols_to_scan))
    
    # Display scan results
    display_scan_results(all_signals, risk_pct, account_balance, symbols_to_scan)


def display_analysis_results(symbol, df, signals, market_type, risk_pct, account_balance, timeframe):
    """Display analysis results for a single symbol"""
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Chart & Signals", "📊 Signals Table", "📉 Backtest", "📋 Analysis"])
    
    with tab1:
        display_chart_with_signals(symbol, df, signals, market_type)
    
    with tab2:
        display_signals_table(signals, risk_pct, account_balance, market_type, symbol)
    
    with tab3:
        display_backtest_results(symbol, df, market_type)
    
    with tab4:
        display_analysis_summary(symbol, df, signals, market_type)


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
    if st.button("📥 Export Signals to CSV"):
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
    if st.button("📥 Export Scan Results"):
        csv = df_scan.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"market_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
