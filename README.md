# SMC Trading Signals Generator

A professional Streamlit application for generating Smart Money Concepts (SMC) trading signals across multiple asset classes including Forex, Stocks, Crypto, Indices, and Commodities.

## Features

### 🎯 Core Functionality
- **Multi-Asset Support**: Forex, Stocks, Crypto, Indices, Commodities
- **Smart Money Concepts**: Market structure, FVG, Order Blocks, Liquidity Grabs
- **Signal Types**: Market orders and Limit orders
- **Risk Management**: Position sizing, stop loss, take profit calculations
- **Real-time Analysis**: Live market data via yfinance

### 📊 Analysis Tools
- **Market Structure Analysis**: Swing highs/lows, BOS, ChoCH identification
- **Fair Value Gaps (FVG)**: Bullish and bearish gap detection
- **Order Blocks**: Institutional order block identification
- **Liquidity Analysis**: Liquidity grab detection
- **Price Action**: TA-Lib pattern recognition

### 🔍 Signal Generation
- **Rule-based Logic**: Emotion-free, systematic approach
- **Confluence System**: Minimum 3 confluences required
- **Risk-Reward Ratios**: Automatic R:R calculation
- **Market Bias Filter**: Bullish, Bearish, or All signals

### 📈 Backtesting
- **Historical Performance**: 2-year backtesting capability
- **Performance Metrics**: Win rate, Sharpe ratio, max drawdown
- **Equity Curve**: Visual performance tracking
- **Trade Analysis**: Detailed trade breakdown

### 🚀 Market Scanning
- **Multi-Symbol Scan**: Scan 50+ liquid instruments
- **Concurrent Processing**: Threaded scanning for speed
- **Top Opportunities**: Ranked by confluence strength
- **Export Functionality**: CSV export for further analysis

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd smc-trading-signals
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
streamlit run app.py
```

## Usage

### Single Symbol Analysis
1. Enter a trading symbol (e.g., `EURUSD=X`, `AAPL`, `BTC-USD`)
2. Select timeframe (1h, 4h, 1d)
3. Choose bias filter (All, Bullish, Bearish)
4. Set risk percentage (0.5% - 2.0%)
5. Click "Analyze Symbol"

### Market Scanning
1. Configure parameters in sidebar
2. Click "Scan Market"
3. Review top opportunities
4. Export results for further analysis

### Backtesting
1. Navigate to "Backtest" tab
2. Review performance metrics
3. Analyze equity curve
4. Study trade history

## Supported Symbols

### Forex
- Major pairs: `EURUSD=X`, `GBPUSD=X`, `USDJPY=X`
- Cross pairs: `EURJPY=X`, `GBPJPY=X`, `EURGBP=X`
- Exotic pairs: `USDCHF=X`, `AUDUSD=X`, `USDCAD=X`

### Stocks
- US stocks: `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `TSLA`
- Tech stocks: `META`, `NVDA`, `NFLX`, `CRM`
- Blue chips: `JNJ`, `V`, `PG`, `JPM`

### Crypto
- Major coins: `BTC-USD`, `ETH-USD`, `BNB-USD`
- Altcoins: `ADA-USD`, `SOL-USD`, `DOT-USD`
- Meme coins: `DOGE-USD`, `SHIB-USD`

### Indices
- US indices: `^GSPC`, `^DJI`, `^IXIC`
- International: `^N225`, `^HSI`, `^FTSE`

### Commodities
- Precious metals: `GC=F`, `SI=F`
- Energy: `CL=F`, `NG=F`
- Agriculture: `ZC=F`, `ZS=F`

## Signal Types

### Market Orders
- **Market Buy**: Execute immediately at current price
- **Market Sell**: Execute immediately at current price
- **Requirements**: Bullish/Bearish bias + 3+ confluences

### Limit Orders
- **Limit Buy**: Set buy order at FVG/OB level
- **Limit Sell**: Set sell order at FVG/OB level
- **Requirements**: Unmitigated zones + 2+ confluences

## Risk Management

### Position Sizing
- Risk-based position sizing
- Maximum 1-2% risk per trade
- Automatic stop loss calculation
- Risk-reward ratio optimization

### Confluence Requirements
- **Market Orders**: Minimum 3 confluences
- **Limit Orders**: Minimum 2 confluences
- **High Quality**: 4+ confluences highlighted

## Performance Metrics

### Backtesting Metrics
- **Win Rate**: Percentage of profitable trades
- **Average R:R**: Average risk-reward ratio
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Maximum peak-to-trough decline
- **Profit Factor**: Gross profit / Gross loss

## Technical Implementation

### Architecture
- **Modular Design**: Separate modules for data, signals, backtesting
- **Caching**: 5-minute data caching for performance
- **Threading**: Concurrent market scanning
- **Error Handling**: Robust error handling and validation

### Dependencies
- **Streamlit**: Web application framework
- **yfinance**: Market data provider
- **pandas/numpy**: Data manipulation
- **TA-Lib**: Technical analysis library
- **plotly**: Interactive charts
- **scipy**: Scientific computing

## Disclaimer

⚠️ **Important**: This application is for educational purposes only. Trading signals are not financial advice. Always:

- Backtest strategies before live trading
- Risk maximum 1% per trade
- Understand market risks
- Consult financial professionals
- Never trade with money you can't afford to lose

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ using Streamlit and Smart Money Concepts**
