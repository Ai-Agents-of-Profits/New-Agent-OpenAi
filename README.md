# Crypto Trading Assistant

A multi-agent AI system that provides comprehensive cryptocurrency market analysis and trading insights through specialized agent coordination. This system features a React frontend for intuitive interaction and a Flask backend powering multiple specialized agents.

## System Architecture

The Trading Assistant operates on a multi-agent architecture:

- **Orchestration Agent**: Coordinates all specialized agents, processes user requests, and manages responses
- **Specialized Agents**:
  - **Market Data Agent**: Retrieves real-time price and market information
  - **Technical Analysis Agent**: Analyzes indicators and chart patterns
  - **Orderbook Analysis Agent**: Examines market depth and liquidity
  - **Token Dashboard Agent**: Provides comprehensive token analysis
  - **Execution Agent**: Handles trading execution and position management

## Features

- **Comprehensive Market Analysis**:
  - Real-time data from multiple exchanges
  - Technical indicator calculations and interpretations
  - Orderbook depth and liquidity analysis
  - Support & resistance level identification
  - Multi-timeframe trend analysis

- **Advanced Trading Features**:
  - Strategy-based trade setup generation
  - Clear entry, stop-loss, and take-profit recommendations
  - Time-bound signal validity specifications
  - Risk management suggestions
  - Position monitoring

- **Interactive User Interface**:
  - Modern React frontend with Tailwind CSS
  - Mobile-responsive design
  - Intuitive conversation interface
  - Markdown-formatted agent responses
  - Syntax-highlighted code snippets

## Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn
- API keys for supported exchanges (Binance USDM, Binance, etc.)

### Backend Setup

1. **Clone the repository**
   ```
   git clone [repository-url]
   cd openai-agent-demo
   ```

2. **Create a virtual environment**
   ```
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Unix/MacOS
   source venv/bin/activate
   ```

3. **Install Python dependencies**
   ```
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   - Create a `.env` file based on `.env.example`
   - Add your OpenAI API key
   - Add your exchange API credentials:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     BINANCE_API_KEY=your_binance_api_key_here
     BINANCE_API_SECRET=your_binance_secret_here
     ```

### Frontend Setup

1. **Navigate to the frontend directory**
   ```
   cd frontend
   ```

2. **Install Node.js dependencies**
   ```
   npm install
   # or
   yarn install
   ```

3. **Configure proxy settings (if needed)**
   - The frontend proxy is configured to connect to the backend at `127.0.0.1:8000`
   - Edit `frontend/package.json` if you need to change the proxy address

## Running the Application

### Start the Backend Server

```
python web_app.py
```
This will start the Flask backend on `http://127.0.0.1:8000`.

### Start the Frontend Development Server

In a separate terminal:
```
cd frontend
npm start
# or
yarn start
```
This will launch the React development server on `http://localhost:3000`.

## Using the Trading Assistant

### Frontend Interface

1. **Accessing the application**
   - Open your browser and navigate to `http://localhost:3000`
   - You will see the Trading Assistant interface with a conversation panel

2. **Starting a conversation**
   - Type your question or request in the input field at the bottom
   - Press Enter or click the send button to submit

3. **Agent Interactions**
   - The Orchestration Agent will process your request
   - Responses will be formatted with markdown, including:
     - Syntax-highlighted code blocks
     - Snippet identifiers
     - Formatted headings and text

4. **Trading Strategy Analysis**
   - Request analysis using strategies like:
     - Breakout
     - Trend Following
     - Mean Reversion
     - Comprehensive
     - Swing Trade
     - Scalping

5. **Clearing Conversations**
   - Use the "Clear Conversation" button to start a new session

### Example Prompts

**General Market Information**:
```
What's the current price and market summary for BTC/USDT?
```

**Technical Analysis**:
```
Perform technical analysis on ETH/USDT on the 4h timeframe.
```

**Trading Strategy Setups**:
```
Use the comprehensive strategy to analyze SOL/USDT on the 1h timeframe.
```

**Token Dashboard**:
```
Generate a complete token dashboard for DOGE/USDT with high detail level.
```

**Combined Analysis**:
```
Analyze BNB/USDT for breakout opportunities on the 1h timeframe, and include orderbook analysis to confirm liquidity at key levels.
```

## Troubleshooting

### Common Issues

1. **API Connection Issues**
   - Ensure your API keys are correct in the `.env` file
   - Check that your account has API access enabled on the exchange
   - For Binance timestamp errors, the system uses an increased `recvWindow` parameter (60 seconds)

2. **Frontend Proxy Connection**
   - If experiencing connection issues, ensure the backend is running
   - Check that the proxy in `package.json` points to the correct backend address (use `127.0.0.1` instead of `localhost`)

3. **Tailwind CSS Styling Issues**
   - Run `npm run build:css` or `yarn build:css` to rebuild the CSS
   - Check that the Tailwind configuration is correctly set up

## Extending the System

### Adding New Trading Strategies

Edit `frontend/src/data/tradingStrategies.ts` to add new strategy prompts and descriptions.

### Modifying Agent Behavior

1. **Agent Logic**: Customize agent behavior in `src/agents/` directory
2. **Prompts**: Edit system prompts in `src/trading/prompts.py`
3. **Exchange Integration**: Modify exchange parameters in `src/exchange/connector.py`

### Adding Support for New Exchanges

1. Extend the `ExchangeConnector` class in `src/exchange/connector.py`
2. Add new exchange configuration in `src/exchange/config.py`
3. Update the frontend trading pair selector to include the new exchange

## Security Considerations

- API keys are stored in the `.env` file and should never be committed to version control
- Use read-only API keys when possible for analysis-only operations
- Enable IP restrictions on your exchange API keys
- Review code for security issues before deploying in a production environment

## License

[Include appropriate license information]
