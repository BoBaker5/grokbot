import asyncio
import datetime
import json
import logging
import os
import pickle
import sqlite3
import time
import traceback
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Union
import random
import numpy as np
import pandas as pd
import krakenex
import requests
from pykrakenapi import KrakenAPI
from dotenv import load_dotenv
import sys 

# ML/AI imports
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Configure pandas settings
pd.set_option("future.no_silent_downcasting", True)

# Load environment variables
load_dotenv()

class KryptosTradingBot:
    """
    Streamlined and enhanced crypto trading bot with sentiment analysis, ML/AI models,
    and comprehensive backtesting capabilities.
    """
    def __init__(self, config_path=None):
        """Initialize the trading bot with configuration and systems."""
        # Setup logging first
        self.logger = self._setup_logging()
        self.logger.info("Initializing KryptosTradingBot...")

        # Load configuration
        self.config = self._load_config(config_path)
        
        # Create necessary directories
        self._create_directories()
        
        # Initialize database
        self.db_name = self.config.get('database', {}).get('name', 'kryptos_trading.db')
        self.init_database()
        
        # Set initial state variables
        self.initial_capital = float(self.config.get('initial_capital', 1000000.0))
        self.balance = {'ZUSD': self.initial_capital}
        self.positions = {}
        self.trade_history = []
        self.portfolio_history = [{
            'timestamp': datetime.now(),
            'balance': self.initial_capital,
            'equity': self.initial_capital,
            'drawdown': 0.0
        }]
        
        # Load saved trading state BEFORE initializing other components
        self._load_saved_state()
        
        # Set trading parameters
        self._configure_trading_parameters()
        
        # Connect to API
        self._initialize_api()
        
        # Initialize components
        self._initialize_components()
        
        # State variables
        self.is_running = True
        self.is_backtesting = False
        self.is_models_loaded = False  # Force model training
        
        # Load models
        self._load_models()

        # Initialize risk management module
        self._initialize_risk_management()
        
        # NOW update risk management with current drawdown
        if hasattr(self, 'portfolio_history') and self.portfolio_history:
            latest_entry = self.portfolio_history[-1]
            if 'drawdown' in latest_entry and hasattr(self, 'risk_management'):
                self.risk_management['current_drawdown'] = latest_entry['drawdown']
                
                # Activate risk protection if needed
                if 'drawdown_protection' in self.risk_management:
                    if latest_entry['drawdown'] >= self.risk_management['drawdown_protection'].get('tier1_drawdown', 0.1):
                        self.risk_management['active_risk_protection'] = True
                        
                        if latest_entry['drawdown'] >= self.risk_management['drawdown_protection'].get('tier2_drawdown', 0.15):
                            self.risk_management['risk_reduction_factor'] = 1.0 - self.risk_management['drawdown_protection'].get('tier2_risk_reduction', 0.5)
                        else:
                            self.risk_management['risk_reduction_factor'] = 1.0 - self.risk_management['drawdown_protection'].get('tier1_risk_reduction', 0.25)
        
        # Initialize cache for historical price patterns
        self._initialize_pattern_recognition_cache()
        
        # Load disk cache
        self._load_disk_cache()
        
        # NOW load market regimes
        if hasattr(self, 'db_name'):
            try:
                conn = sqlite3.connect(self.db_name)
                c = conn.cursor()
                c.execute('SELECT * FROM market_regimes ORDER BY timestamp DESC LIMIT 50')
                regime_data = c.fetchall()
                conn.close()
                
                if regime_data and hasattr(self, 'market_regimes'):
                    for row in regime_data:
                        try:
                            timestamp = datetime.fromisoformat(row[0]) if isinstance(row[0], str) else row[0]
                            symbol = row[1]
                            regime = row[2]
                            
                            # Store the most recent regime for each symbol
                            if 'symbols' not in self.market_regimes:
                                self.market_regimes['symbols'] = {}
                                
                            if symbol not in self.market_regimes['symbols'] or timestamp > datetime.fromisoformat(self.market_regimes['symbols'][symbol]['timestamp']) if isinstance(self.market_regimes['symbols'][symbol].get('timestamp'), str) else timestamp:
                                self.market_regimes['symbols'][symbol] = {
                                    'regime': regime,
                                    'timestamp': row[0]
                                }
                        except Exception as regime_err:
                            self.logger.error(f"Error loading market regime: {str(regime_err)}")
            except Exception as e:
                self.logger.error(f"Error loading market regimes: {str(e)}")
        
        # Initialize API keys
        self.newsapi_key = os.environ.get('NEWS_API_KEY')
        self.newsapi_available = bool(self.newsapi_key)
        self.cryptocompare_api_key = os.environ.get('CRYPTOCOMPARE_API_KEY')
        self.cryptocompare_available = bool(self.cryptocompare_api_key)
        self.coinmarketcap_key = os.environ.get('COINMARKETCAP_API_KEY')
        self.coinmarketcap_available = bool(self.coinmarketcap_key)

        # Check API keys
        self._check_api_keys()
        
        # Performance analytics tracker
        self.performance_analytics = {
            'strategy_performance': {},
            'signal_accuracy': {},
            'win_rate_by_cycle': {}
        }
        
        # Initialize market regime detector
        self._initialize_market_regime_detector()
        self._load_market_regimes_from_db()
        
        # Register signal handlers for clean shutdown
        self._register_signal_handlers()
        
        # Start background tasks
        self._start_background_tasks()
        
        # NOW calculate total equity
        total_equity = self.calculate_total_equity()
        self.logger.info(f"KryptosTradingBot initialized successfully with balance: ${self.balance.get('ZUSD', 0):.2f}, Total Equity: ${total_equity:.2f}")

    def _initialize_market_regime_detector(self):
        """Initialize the market regime detector for adaptive strategies."""
        self.market_regimes = {
            'global': 'neutral',  # Current global market regime
            'symbols': {},        # Symbol-specific regimes
            'regime_history': []  # Track regime changes for analysis
        }

        for symbol in self.symbols:
            self.market_regimes['symbols'][symbol] = {
                'regime': 'neutral',
                'volatility': 0.0,
                'trend_strength': 0.0,
                'timestamp': datetime.now().isoformat()
            }
        
        # Regime-specific parameter sets
        self.regime_parameters = {
            'bull_trend': {
                'BUY_THRESHOLD_COEF': 0.95,  # More aggressive buying
                'STOP_LOSS_PCT': 0.015,      # Wider stops
                'TAKE_PROFIT_PCT': 0.045,    # Higher targets
                'TRAILING_STOP_PCT': 0.020,  # Wider trailing stops
                'MAX_POSITION_SIZE': 0.20,   # Larger positions
                'DCA_PARTS': 2              # Fewer DCA parts
            },
            'bear_trend': {
                'BUY_THRESHOLD_COEF': 1.05,  # More conservative buying
                'STOP_LOSS_PCT': 0.010,      # Tighter stops
                'TAKE_PROFIT_PCT': 0.025,    # Lower targets
                'TRAILING_STOP_PCT': 0.012,  # Tighter trailing stops
                'MAX_POSITION_SIZE': 0.10,   # Smaller positions
                'DCA_PARTS': 4              # More DCA parts
            },
            'ranging': {
                'BUY_THRESHOLD_COEF': 1.00,  # Standard buying
                'STOP_LOSS_PCT': 0.010,      # Standard stops
                'TAKE_PROFIT_PCT': 0.025,    # Standard targets
                'TRAILING_STOP_PCT': 0.010,  # Standard trailing stops
                'MAX_POSITION_SIZE': 0.08,   # Standard positions
                'DCA_PARTS': 3              # Standard DCA parts
            },
            'high_volatility': {
                'BUY_THRESHOLD_COEF': 1.10,  # Very conservative buying
                'STOP_LOSS_PCT': 0.015,      # Wider stops for volatility
                'TAKE_PROFIT_PCT': 0.030,    # Higher targets
                'TRAILING_STOP_PCT': 0.018,  # Wider trailing stops
                'MAX_POSITION_SIZE': 0.05,   # Small positions
                'DCA_PARTS': 5              # Many DCA parts
            },
            'low_volatility': {
                'BUY_THRESHOLD_COEF': 0.97,  # Slightly aggressive buying
                'STOP_LOSS_PCT': 0.008,      # Standard stops
                'TAKE_PROFIT_PCT': 0.020,    # Lower targets
                'TRAILING_STOP_PCT': 0.008,  # Tighter trailing stops
                'MAX_POSITION_SIZE': 0.10,   # Larger positions
                'DCA_PARTS': 2              # Fewer DCA parts
            }
        }
        
        # Last regime check time
        self.last_regime_check = datetime.now() - timedelta(hours=1)  # Force initial check
    
    def calculate_kelly_position_size(self, symbol, signal):
        """Calculate optimal position size using the Kelly Criterion."""
        # Get symbol-specific performance if available
        win_rate = 0.5  # Default 50% win rate
        avg_win = 0.02  # Default 2% gain
        avg_loss = 0.01  # Default 1% loss
        
        # Get actual metrics if available
        if hasattr(self, 'performance_analytics') and 'symbol_performance' in self.performance_analytics:
            if symbol in self.performance_analytics['symbol_performance']:
                perf = self.performance_analytics['symbol_performance'][symbol]
                trades = perf.get('trades', 0)
                if trades >= 10:  # Only use if we have enough data
                    win_rate = perf.get('win_rate', 50) / 100
                    avg_win = abs(perf.get('avg_win', 0.02))
                    avg_loss = abs(perf.get('avg_loss', 0.01))
        
        # Calculate Kelly percentage
        if avg_loss > 0:
            edge = win_rate - (1 - win_rate) / (avg_win / avg_loss)
            kelly_pct = max(0, edge)  # Kelly never suggests negative position size
            
            # Typically use half-Kelly for more conservative sizing
            half_kelly = kelly_pct * 0.5
            
            # Cap at reasonable maximum
            kelly_pct = min(self.max_position_size * 1.5, half_kelly)
            
            # Apply to available capital
            kelly_size = self.balance.get('ZUSD', 0) * kelly_pct
            
            return kelly_size
        else:
            return 0

    def calculate_tiered_position_size(self, symbol, signal):
        """Calculate position size using a tiered approach based on signal confidence."""
        available_capital = self.balance.get('ZUSD', 0)
        confidence = signal.get('confidence', 0.5)
        
        # Base position size tier
        if confidence >= 0.60:  # Very strong signal
            base_position_pct = self.max_position_size * 1.3
        elif confidence >= 0.55:  # Strong signal
            base_position_pct = self.max_position_size * 1.0
        elif confidence >= 0.53:  # Moderate signal
            base_position_pct = self.max_position_size * 0.7
        else:  # Weak signal
            base_position_pct = self.max_position_size * 0.5
        
        # Apply allocation weighting from symbols dictionary
        symbol_weight = self.symbols.get(symbol, 0.1)
        position_pct = base_position_pct * symbol_weight * 2  # Reduce multiplier from 4 to 2
        
        # Apply risk management but with a minimum factor
        if hasattr(self, 'risk_management') and 'risk_reduction_factor' in self.risk_management:
            risk_factor = max(0.75, self.risk_management['risk_reduction_factor'])
            position_pct *= risk_factor
        
        # Calculate actual dollar amount with a reasonable minimum
        position_size = available_capital * position_pct
        min_size = max(5000, self.initial_capital * 0.001)  # At least 0.1% of initial capital
        
        if position_size < min_size and confidence > 0.53:
            position_size = min_size  # Ensure minimum size for decent signals
        
        return min(position_size, available_capital * 0.95)  # Cap at 95% of available

    def _initialize_risk_management(self):
        """Initialize advanced risk management module."""
        # Set up portfolio-wide risk parameters
        self.risk_management = {
            'max_portfolio_risk': 0.04,        # Maximum 4% portfolio at risk at any time
            'max_single_trade_risk': 0.015,    # Maximum 1.5% risk on any single trade
            'max_symbol_allocation': 0.25,     # Maximum 25% in any single coin
            'max_correlated_allocation': 0.40, # Maximum 40% in correlated coins
            'drawdown_protection': {
                'active': True,
                'tier1_drawdown': 0.08,        # First tier protection at 8% drawdown
                'tier1_risk_reduction': 0.25,  # Reduce position sizes by 25%
                'tier2_drawdown': 0.15,        # Second tier protection at 15% drawdown
                'tier2_risk_reduction': 0.50,  # Reduce position sizes by 50%
                'max_drawdown': 0.15,          # Updated from 0.20 to 0.15
                'recovery_threshold': 0.05,    # Recovery threshold when drawdown < 5%
            },
            'current_drawdown': 0.0,           # Current maximum drawdown
            'portfolio_at_risk': 0.0,          # Current amount of portfolio at risk
            'active_risk_protection': False,   # Whether drawdown protection is active
            'risk_reduction_factor': 1.0,      # Current risk reduction factor (1.0 = no reduction)
            'correlation_matrix': {},          # Store correlation matrix of traded coins
            'adaptive_stops': {
                'enabled': True,
                'volatility_multiplier': 0.8,  # Lower = tighter stops in volatile markets
                'min_stop_distance': 0.006,    # Minimum 0.6% stop distance
                'max_stop_distance': 0.018,    # Maximum 1.8% stop distance
                'atr_periods': 14,             # ATR calculation periods
                'atr_multiplier': 1.5          # Multiplier for ATR-based stops
            }
        }
        
        # Asset correlation groups (for managing exposure to correlated assets)
        self.correlation_groups = {
            'large_caps': ['XBTUSD', 'ETHUSD'],
            'alt_layer1': ['SOLUSD', 'AVAXUSD'],
            'payment_tokens': ['XRPUSD'],
            'meme_coins': ['XDGUSD']
        }

        # Symbol-specific risk parameters
        self.symbol_risk_params = {
            'XBTUSD': {
                'stop_loss_pct': 0.012,         # Updated from 0.016
                'take_profit_pct': 0.030,
                'trailing_stop_pct': 0.010      # Updated from 0.014
            },
            'ETHUSD': {
                'stop_loss_pct': 0.012,         # Updated from 0.017
                'take_profit_pct': 0.028,
                'trailing_stop_pct': 0.010      # Updated from 0.013
            },
            'SOLUSD': {
                'stop_loss_pct': 0.010,         # Significantly tighter from 0.022
                'take_profit_pct': 0.025,       # Lower than before
                'trailing_stop_pct': 0.008      # Significantly tighter from 0.018
            },
            'AVAXUSD': {
                'stop_loss_pct': 0.012,         # Updated from 0.015
                'take_profit_pct': 0.028,
                'trailing_stop_pct': 0.010      # Updated from 0.014
            },
            'XRPUSD': {
                'stop_loss_pct': 0.012,         # Updated from 0.014
                'take_profit_pct': 0.026,
                'trailing_stop_pct': 0.010      # Updated from 0.012
            },
            'XDGUSD': {
                'stop_loss_pct': 0.012,         # Updated from 0.020
                'take_profit_pct': 0.028,
                'trailing_stop_pct': 0.010      # Updated from 0.016
            }
        }
    
    def _initialize_pattern_recognition_cache(self):
        """Initialize cache for technical pattern recognition."""
        self.pattern_recognition = {
            'common_patterns': {
                'head_and_shoulders': {'lookback': 40, 'reliability': 0.65},
                'double_top': {'lookback': 30, 'reliability': 0.62},
                'double_bottom': {'lookback': 30, 'reliability': 0.64},
                'triangle': {'lookback': 25, 'reliability': 0.58},
                'wedge': {'lookback': 25, 'reliability': 0.56},
                'cup_and_handle': {'lookback': 60, 'reliability': 0.66},
                'engulfing': {'lookback': 5, 'reliability': 0.59}
            },
            'detection_history': {},  # Store detected patterns for analytics
            'pattern_success_rate': {}  # Track success rate of pattern-based trades
        }
    
    def _check_api_keys(self):
        """Validate API keys and disable those that aren't working."""
        # Check NewsAPI key
        if hasattr(self, 'newsapi_key') and self.newsapi_key:
            try:
                url = f"https://newsapi.org/v2/everything?q=bitcoin&apiKey={self.newsapi_key}"
                response = requests.get(url, timeout=10)
                if response.status_code == 401:  # Unauthorized
                    self.logger.error("NewsAPI key is invalid - disabling NewsAPI")
                    self.newsapi_available = False
                    self.newsapi_key = None
                elif response.status_code == 429:  # Too many requests
                    self.logger.warning("NewsAPI rate limit reached - will try again later")
                elif response.status_code == 200:
                    self.logger.info("NewsAPI key validated successfully")
            except Exception as e:
                self.logger.warning(f"Error validating NewsAPI key: {str(e)}")

        # Check CryptoCompare API key if available
        if hasattr(self, 'cryptocompare_api_key') and self.cryptocompare_api_key:
            try:
                url = f"https://min-api.cryptocompare.com/data/v2/news/?api_key={self.cryptocompare_api_key}"
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    self.logger.warning(f"CryptoCompare API returned status {response.status_code}")
                    if response.status_code == 401:
                        self.logger.error("CryptoCompare API key is invalid - disabling CryptoCompare")
                        self.cryptocompare_available = False
                        self.cryptocompare_api_key = None
                else:
                    self.logger.info("CryptoCompare API key validated successfully")
            except Exception as e:
                self.logger.warning(f"Error validating CryptoCompare API key: {str(e)}")
                
        # Check CoinMarketCap API key if available
        if hasattr(self, 'coinmarketcap_key') and self.coinmarketcap_key:
            try:
                url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
                headers = {
                    'Accepts': 'application/json',
                    'X-CMC_PRO_API_KEY': self.coinmarketcap_key,
                }
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code != 200:
                    self.logger.warning(f"CoinMarketCap API returned status {response.status_code}")
                    if response.status_code == 401 or response.status_code == 403:
                        self.logger.error("CoinMarketCap API key is invalid - disabling CoinMarketCap")
                        self.coinmarketcap_available = False
                        self.coinmarketcap_key = None
                else:
                    self.logger.info("CoinMarketCap API key validated successfully")
            except Exception as e:
                self.logger.warning(f"Error validating CoinMarketCap API key: {str(e)}")

    def _setup_logging(self):
        """Set up logging with proper rotation and formatting."""
        logger = logging.getLogger("KryptosTradingBot")
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        
        # File handler with rotation
        log_file = f'logs/kryptos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Configure logger
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.propagate = False  # Don't propagate to root logger
        
        return logger
    
    def _load_config(self, config_path):
        """Load configuration from JSON file or use defaults."""
        default_config = {
            "initial_capital": 1000000.0,
            "database": {
                "name": "kryptos_trading.db"
            },
            "symbols": {
                "XBTUSD": 0.25,  # Bitcoin
                "ETHUSD": 0.20,  # Ethereum
                "SOLUSD": 0.15,  # Solana
                "AVAXUSD": 0.15,  # Avalanche
                "XRPUSD": 0.15,   # XRP
                "XDGUSD": 0.10    # Dogecoin
            },
            "thresholds": {
                "buy": {
                    "XBTUSD": 0.51,  # Lowered thresholds
                    "ETHUSD": 0.51,
                    "SOLUSD": 0.52,
                    "AVAXUSD": 0.52,
                    "XRPUSD": 0.53,
                    "XDGUSD": 0.54
                }
            },
            "risk": {
                "max_drawdown": 0.20,
                "trailing_stop_pct": 0.008,
                "max_trades_per_hour": 4,  # Increased
                "trade_cooldown": 180,     # Reduced from 360
                "max_position_size": 0.08,
                "min_position_value": 1000.0,
                "max_total_risk": 0.25,
                "stop_loss_pct": 0.007,
                "take_profit_pct": 0.018
            },
            "technical": {
                "sma_short": 20,
                "sma_long": 50,
                "rsi_period": 14,
                "rsi_oversold": 35,
                "rsi_overbought": 65,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9
            },
            "dca": {
                "enabled": True,
                "parts": 3,            # Increased from 2
                "time_between": 30     # Reduced from 40
            },
            "api": {
                "timeframe": 5,
                "retry_delay": 1.0,
                "max_retry_delay": 60
            },
            "paths": {
                "models": "models",
                "data": "data",
                "cache": "cache",
                "backtest": "backtest_results"
            },
            "signal_weights": {
                "ml": 0.25,
                "ai": 0.25, 
                "basic": 0.25,         # New: basic technical signals
                "sentiment": 0.25,
                "pattern": 0.0          # Will be dynamically adjusted when patterns detected
            }
        }
        
        # Try to load from file
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    # Use json.load instead of directly evaluating the file
                    config = json.load(f)
                self.logger.info(f"Loaded configuration from {config_path}")
                return config
            except Exception as e:
                self.logger.error(f"Error loading config from {config_path}: {str(e)}")
                self.logger.info("Using default configuration")
                return default_config
        else:
            self.logger.info("Using default configuration")
            return default_config
    
    def _create_directories(self):
        """Create necessary directories for the bot."""
        dirs = [
            'data',
            'models',
            'cache',
            'cache/sentiment',
            'cache/historical',
            'cache/patterns',  # New directory for pattern analysis
            'logs',
            'backtest_results',
            'risk_analytics'   # New directory for risk analysis
        ]
        
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
        
        self.logger.info(f"Created {len(dirs)} directories")
    
    def init_database(self):
        """Initialize SQLite database with proper schema."""
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            
            # Create tables with clear schema
            c.execute('''CREATE TABLE IF NOT EXISTS balance
                    (currency TEXT PRIMARY KEY, amount REAL)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS positions
                    (symbol TEXT PRIMARY KEY, 
                     volume REAL, 
                     entry_price REAL, 
                     entry_time TEXT, 
                     high_price REAL, 
                     stop_loss REAL, 
                     take_profit REAL,
                     trailing_stop_pct REAL,
                     first_tier_executed BOOLEAN,
                     second_tier_executed BOOLEAN,
                     third_tier_executed BOOLEAN)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS trade_history
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     timestamp TEXT, 
                     symbol TEXT, 
                     type TEXT, 
                     price REAL, 
                     quantity REAL, 
                     value REAL, 
                     balance_after REAL, 
                     profit_loss REAL, 
                     pnl_percentage REAL,
                     signal_confidence REAL,
                     market_cycle TEXT,
                     signals_used TEXT,
                     exit_reason TEXT)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS portfolio_history
                    (timestamp TEXT PRIMARY KEY, 
                     balance REAL, 
                     equity REAL,
                     drawdown REAL,
                     portfolio_risk REAL)''')  # Added portfolio_risk
            
            c.execute('''CREATE TABLE IF NOT EXISTS market_data
                        (timestamp TEXT,
                        symbol TEXT, 
                        close REAL, 
                        volume REAL,
                        rsi REAL, 
                        macd REAL, 
                        trend TEXT, 
                        volatility REAL,
                        market_regime TEXT,
                        PRIMARY KEY (timestamp, symbol))''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS sentiment_data
                    (timestamp TEXT,
                     symbol TEXT,
                     twitter_score REAL,
                     reddit_score REAL,
                     news_score REAL,
                     combined_score REAL,
                     PRIMARY KEY (timestamp, symbol))''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS backtest_results
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     start_date TEXT,
                     end_date TEXT,
                     initial_capital REAL,
                     final_equity REAL,
                     total_return REAL,
                     sharpe_ratio REAL,
                     max_drawdown REAL,
                     win_rate REAL,
                     profit_factor REAL,
                     parameters TEXT,
                     timestamp TEXT)''')
                     
            # New tables for enhanced features
            c.execute('''CREATE TABLE IF NOT EXISTS detected_patterns
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     timestamp TEXT,
                     symbol TEXT,
                     pattern_type TEXT,
                     confidence REAL,
                     price_at_detection REAL,
                     predicted_direction TEXT,
                     success BOOLEAN,
                     price_after REAL)''')
                     
            c.execute('''CREATE TABLE IF NOT EXISTS signal_performance
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     timestamp TEXT,
                     symbol TEXT,
                     signal_type TEXT,
                     signal_value REAL,
                     prediction TEXT,
                     success BOOLEAN,
                     profit_loss REAL)''')
                     
            c.execute('''CREATE TABLE IF NOT EXISTS market_regimes
                    (timestamp TEXT,
                     symbol TEXT,
                     regime TEXT,
                     volatility REAL,
                     trend_strength REAL,
                     PRIMARY KEY (timestamp, symbol))''')
            
            # Initialize balance if empty
            c.execute('SELECT COUNT(*) FROM balance')
            if c.fetchone()[0] == 0:
                c.execute('INSERT INTO balance VALUES (?, ?)',
                         ('ZUSD', self.balance['ZUSD']))
            
            conn.commit()
            conn.close()
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            traceback.print_exc()
    
    def _initialize_api(self):
        """Initialize connection to trading API."""
        try:
            # Initialize Kraken API
            self.api = krakenex.API()
            self.kraken = KrakenAPI(self.api, retry=0.5)
            
            # Set up API credentials if available
            api_key = os.environ.get('KRAKEN_API_KEY')
            api_secret = os.environ.get('KRAKEN_PRIVATE_KEY')
            
            if api_key and api_secret:
                self.api.key = api_key
                self.api.secret = api_secret
                self.logger.info("API credentials loaded")
                
                # Test authenticated connection
                try:
                    balance = self.api.query_private('Balance')
                    if 'error' in balance and balance['error']:
                        self.logger.warning(f"API authentication error: {balance['error']}")
                    else:
                        self.logger.info("API authentication successful")
                except Exception as auth_err:
                    self.logger.warning(f"API authentication error: {str(auth_err)}")
            else:
                self.logger.info("No API credentials found - running in public-only mode")
            
            # Set API parameters
            self.timeframe = self.config.get('api', {}).get('timeframe', 5)
            self.api_retry_delay = self.config.get('api', {}).get('retry_delay', 1.0)
            self.max_retry_delay = self.config.get('api', {}).get('max_retry_delay', 60)
            
            # Initialize rate limiting
            self.rate_limit_timestamps = {}
            self.rate_limit_failures = {}  # Track recent failures for adaptive backoff
            self.min_request_interval = 1.0  # seconds
            
        except Exception as e:
            self.logger.error(f"Error initializing API: {str(e)}")
            traceback.print_exc()
    
    def _configure_trading_parameters(self):
        """Set up trading parameters from configuration."""
        # Symbols and allocations
        self.symbols = self.config.get('symbols', {
            "XBTUSD": 0.25,
            "ETHUSD": 0.20,
            "SOLUSD": 0.15,
            "AVAXUSD": 0.15,
            "XRPUSD": 0.15,
            "XDGUSD": 0.10
        })
        
        # Buy thresholds
        self.buy_thresholds = self.config.get('thresholds', {}).get('buy', {
            'XBTUSD': 0.51,  # Lowered thresholds
            'ETHUSD': 0.51,
            'SOLUSD': 0.52,
            'AVAXUSD': 0.52,
            'XRPUSD': 0.53,
            'XDGUSD': 0.54
        })
        
        # Risk parameters
        risk_config = self.config.get('risk', {})
        self.max_drawdown = risk_config.get('max_drawdown', 0.20)
        self.trailing_stop_pct = risk_config.get('trailing_stop_pct', 0.008)
        self.max_trades_per_hour = risk_config.get('max_trades_per_hour', 4)  # Increased
        self.trade_cooldown = risk_config.get('trade_cooldown', 180)  # Reduced
        self.max_position_size = risk_config.get('max_position_size', 0.15)
        self.min_position_value = risk_config.get('min_position_value', 1000.0)
        self.max_total_risk = risk_config.get('max_total_risk', 0.25)
        self.stop_loss_pct = risk_config.get('stop_loss_pct', 0.007)
        self.take_profit_pct = risk_config.get('take_profit_pct', 0.018)
        
        # Initialize tracking variables
        self.last_trade_time = {}
        
        # Technical parameters
        tech_config = self.config.get('technical', {})
        self.sma_short = tech_config.get('sma_short', 20)
        self.sma_long = tech_config.get('sma_long', 50)
        self.rsi_period = tech_config.get('rsi_period', 14)
        self.rsi_oversold = tech_config.get('rsi_oversold', 35)
        self.rsi_overbought = tech_config.get('rsi_overbought', 65)
        self.macd_fast = tech_config.get('macd_fast', 12)
        self.macd_slow = tech_config.get('macd_slow', 26)
        self.macd_signal = tech_config.get('macd_signal', 9)
        
        # Signal weights
        self.signal_weights = self.config.get('signal_weights', {
            'ml': 0.25,
            'ai': 0.25,
            'basic': 0.25,  # New basic signal weight
            'sentiment': 0.25,
            'pattern': 0.0   # Will be adjusted when patterns detected
        })
        
        # DCA settings
        dca_config = self.config.get('dca', {})
        self.dca_enabled = dca_config.get('enabled', True)
        self.dca_parts = dca_config.get('parts', 3)  # Increased from 2
        self.dca_time_between = dca_config.get('time_between', 30)  # Reduced from 40
        self.dca_plans = []
        
        # Cache settings
        self.price_cache = {}
        self.data_cache = {}
        
        self.logger.info("Trading parameters configured")
    
    def _initialize_components(self):
        """Initialize the ML, AI and sentiment analysis components."""
        # Import component modules with correct paths
        try:
            # Fix the import paths to use relative imports
            from core.models.ml_model import MLModelManager
            from core.models.ai_model import AITradingEnhancer
            from core.sentiment.twitter_analyzer import TwitterSentimentAnalyzer
            from core.sentiment.reddit_analyzer import RedditSentimentAnalyzer
            from core.sentiment.news_analyzer import EnhancedNewsAnalyzer
            
            # Initialize models
            self.model_manager = MLModelManager()
            self.ai_enhancer = AITradingEnhancer()
            
            # Initialize sentiment analyzers
            self.twitter_analyzer = TwitterSentimentAnalyzer()
            self.reddit_analyzer = RedditSentimentAnalyzer()
            self.news_analyzer = EnhancedNewsAnalyzer()
            
            # Provide cross-references if needed
            if hasattr(self.news_analyzer, 'set_twitter_analyzer'):
                self.news_analyzer.set_twitter_analyzer(self.twitter_analyzer)
            
            # Initialize pattern recognition
            self._initialize_pattern_recognition()
            
            self.logger.info("Components initialized")
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            # Initialize with empty objects to prevent further errors
            self.model_manager = None
            self.ai_enhancer = None
            self.twitter_analyzer = None
            self.reddit_analyzer = None
            self.news_analyzer = None
            raise  # Re-raise the exception for proper debugging
    
    def _initialize_pattern_recognition(self):
        """Initialize the pattern recognition component."""
        try:
            # Create a simple pattern recognition object
            self.pattern_recognition_system = {
                'patterns': {
                    'head_and_shoulders': self._detect_head_and_shoulders,
                    'double_top': self._detect_double_top,
                    'double_bottom': self._detect_double_bottom,
                    'ascending_triangle': self._detect_ascending_triangle,
                    'descending_triangle': self._detect_descending_triangle,
                    'bull_flag': self._detect_bull_flag,
                    'bear_flag': self._detect_bear_flag,
                    'engulfing_bullish': self._detect_engulfing_bullish,
                    'engulfing_bearish': self._detect_engulfing_bearish
                },
                'pattern_history': {},
                'success_rate': {}
            }
            
            # Load success rates if available
            try:
                pattern_stats_path = 'cache/patterns/pattern_stats.json'
                if os.path.exists(pattern_stats_path):
                    with open(pattern_stats_path, 'r') as f:
                        pattern_stats = json.load(f)
                        self.pattern_recognition_system['success_rate'] = pattern_stats
            except Exception as e:
                self.logger.warning(f"Could not load pattern stats: {str(e)}")
                
            self.logger.info(f"Pattern recognition system initialized with {len(self.pattern_recognition_system['patterns'])} patterns")
            
        except Exception as e:
            self.logger.error(f"Error initializing pattern recognition: {str(e)}")
    
    def _load_disk_cache(self):
        """Load cached data from disk."""
        try:
            # Sentiment data cache
            sentiment_cache_path = os.path.join('cache/sentiment', 'sentiment_cache.json')
            if os.path.exists(sentiment_cache_path):
                with open(sentiment_cache_path, 'r') as f:
                    self.sentiment_cache = json.load(f)
                self.logger.info(f"Loaded sentiment cache with {len(self.sentiment_cache)} entries")
            else:
                self.sentiment_cache = {}
            
            # Pattern recognition cache
            pattern_cache_path = os.path.join('cache/patterns', 'pattern_cache.json')
            if os.path.exists(pattern_cache_path):
                with open(pattern_cache_path, 'r') as f:
                    self.pattern_cache = json.load(f)
                self.logger.info(f"Loaded pattern cache with {len(self.pattern_cache)} entries")
            else:
                self.pattern_cache = {}
                
            # Market regime cache
            regime_cache_path = os.path.join('cache', 'market_regimes.json')
            if os.path.exists(regime_cache_path):
                with open(regime_cache_path, 'r') as f:
                    regime_data = json.load(f)
                    self.market_regimes = regime_data
                self.logger.info(f"Loaded market regime cache")
            
        except Exception as e:
            self.logger.warning(f"Error loading disk cache: {str(e)}")
    
    def _save_disk_cache(self):
        """Save cache data to disk."""
        try:
            # Ensure directories exist
            os.makedirs('cache/sentiment', exist_ok=True)
            os.makedirs('cache/patterns', exist_ok=True)
            
            # Save sentiment cache
            if hasattr(self, 'sentiment_cache'):
                with open(os.path.join('cache/sentiment', 'sentiment_cache.json'), 'w') as f:
                    json.dump(self.sentiment_cache, f)
            
            # Save pattern cache
            if hasattr(self, 'pattern_cache'):
                with open(os.path.join('cache/patterns', 'pattern_cache.json'), 'w') as f:
                    json.dump(self.pattern_cache, f)
                    
            # Save pattern stats
            if hasattr(self, 'pattern_recognition_system') and 'success_rate' in self.pattern_recognition_system:
                with open(os.path.join('cache/patterns', 'pattern_stats.json'), 'w') as f:
                    json.dump(self.pattern_recognition_system['success_rate'], f)
                    
            # Save market regimes
            if hasattr(self, 'market_regimes'):
                with open(os.path.join('cache', 'market_regimes.json'), 'w') as f:
                    json.dump(self.market_regimes, f)
            
            self.logger.info("Cache data saved to disk")
            
        except Exception as e:
            self.logger.warning(f"Error saving disk cache: {str(e)}")
        
    def _load_market_regimes_from_db(self):
        """Load market regimes from database."""
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            
            # Check if the table exists before querying
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_regimes'")
            if not c.fetchone():
                self.logger.warning("Market regimes table does not exist, using defaults")
                conn.close()
                return False
                
            # Initialize symbols dictionary if needed
            if 'symbols' not in self.market_regimes:
                self.market_regimes['symbols'] = {}
                
            # Load regimes for each symbol with a single query instead of multiple
            c.execute('''
                SELECT symbol, regime, volatility, trend_strength, timestamp 
                FROM market_regimes 
                WHERE symbol IN ({}) 
                GROUP BY symbol
                HAVING MAX(timestamp)
            '''.format(','.join(['?']*len(self.symbols))), list(self.symbols.keys()))
            
            regime_data = c.fetchall()
            for row in regime_data:
                symbol, regime, volatility, trend_strength, timestamp = row
                self.market_regimes['symbols'][symbol] = {
                    'regime': regime,
                    'volatility': float(volatility) if volatility else 0.0,
                    'trend_strength': float(trend_strength) if trend_strength else 0.0,
                    'timestamp': timestamp
                }
                self.logger.info(f"Loaded market regime for {symbol}: {regime}")
                
            conn.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading market regimes from database: {str(e)}")
            return False

    def _load_models(self):
        """Load ML and AI models with more aggressive retry/train behavior."""
        try:
            # First try to load the models
            ml_loaded = self.model_manager.load_model() if self.model_manager else False
            ai_loaded = self.ai_enhancer.load_models() if self.ai_enhancer else False
            
            if ml_loaded and ai_loaded:
                self.is_models_loaded = True
                self.logger.info("All models loaded successfully")
            else:
                self.is_models_loaded = False
                self.logger.warning("Some models failed to load - will train on first run")
                
                # Initiate immediate model training on startup for better performance
                if not self.is_backtesting:
                    asyncio.create_task(self._immediate_model_training())
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            self.is_models_loaded = False
    
    async def _immediate_model_training(self):
        """Immediately start training models on startup for better performance."""
        try:
            self.logger.info("Starting immediate model training on startup")
            
            # Wait a short time to allow system to finish initializing
            await asyncio.sleep(5)
            
            # Collect training data
            training_data = await self.collect_training_data(days=60)  # Use last 60 days
            
            if training_data is None or len(training_data) < 1000:
                self.logger.warning("Insufficient data for initial training")
                return False
            
            # Train models
            ml_success = await self.train_ml_model(training_data)
            ai_success = await self.train_ai_model(training_data)
            
            if ml_success and ai_success:
                self.is_models_loaded = True
                self.logger.info("Initial training completed successfully")
                return True
            else:
                self.logger.error("Initial training failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in immediate model training: {str(e)}")
            return False
    
    def _load_saved_state(self):
        """Load saved trading state from database with improved balance handling."""
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            
            # Load balance - keep the initial value in a separate variable for reference
            original_initial_capital = self.initial_capital
            
            c.execute('SELECT * FROM balance')
            balance_data = c.fetchall()
            if balance_data:
                self.balance = {row[0]: row[1] for row in balance_data}
                if 'ZUSD' in self.balance and self.balance['ZUSD'] > 0:
                    self.logger.info(f"Loaded balance from database: ${self.balance['ZUSD']:.2f}")
                else:
                    # If balance is missing or invalid, use initial capital
                    self.balance = {'ZUSD': self.initial_capital}
                    self.logger.warning(f"Invalid balance in database, using initial capital: ${self.initial_capital:.2f}")
            else:
                # No balance found, use initial capital
                self.logger.warning(f"No balance found in database, using initial capital: ${self.initial_capital:.2f}")
            
            # Load positions
            c.execute('SELECT * FROM positions')
            position_data = c.fetchall()
            if position_data:
                self.positions = {}
                for row in position_data:
                    try:
                        # Handle potential issues with timestamp parsing
                        entry_time = datetime.fromisoformat(row[3]) if isinstance(row[3], str) else row[3]
                        
                        self.positions[row[0]] = {
                            'volume': row[1],
                            'entry_price': row[2],
                            'entry_time': entry_time,
                            'high_price': row[4],
                            'stop_loss': row[5] if len(row) > 5 else None,
                            'take_profit': row[6] if len(row) > 6 else None,
                            'trailing_stop_pct': row[7] if len(row) > 7 else self.trailing_stop_pct,
                            'first_tier_executed': bool(row[8]) if len(row) > 8 else False,
                            'second_tier_executed': bool(row[9]) if len(row) > 9 else False,
                            'third_tier_executed': bool(row[10]) if len(row) > 10 else False
                        }
                    except Exception as pos_err:
                        self.logger.error(f"Error loading position {row[0]}: {str(pos_err)}")
            
            # Load recent trade history (100 most recent)
            c.execute('SELECT * FROM trade_history ORDER BY timestamp DESC LIMIT 100')
            trade_data = c.fetchall()
            if trade_data:
                self.trade_history = []
                for row in trade_data:
                    try:
                        # Handle potential issues with timestamp parsing
                        trade_time = datetime.fromisoformat(row[1]) if isinstance(row[1], str) else row[1]
                        
                        # Add type checks to handle corrupted data
                        def safe_float(value, default=0.0):
                            if isinstance(value, (int, float)):
                                return float(value)
                            elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                                return float(value)
                            else:
                                return default
                        
                        trade = {
                            'id': row[0],
                            'timestamp': trade_time,
                            'symbol': row[2],
                            'type': row[3],
                            'price': safe_float(row[4]),
                            'quantity': safe_float(row[5]),
                            'value': safe_float(row[6]),
                            'balance_after': safe_float(row[7]),
                            'profit_loss': safe_float(row[8], 0.0),
                            'pnl_percentage': safe_float(row[9], 0.0),
                            'signal_confidence': safe_float(row[10], 0.5),
                            'market_cycle': row[11] if row[11] is not None else 'unknown',
                            'signals_used': row[12] if len(row) > 12 and row[12] is not None else '',
                            'exit_reason': row[13] if len(row) > 13 and row[13] is not None else None
                        }
                        self.trade_history.append(trade)
                    except Exception as trade_err:
                        self.logger.error(f"Error loading trade {row[0]}: {str(trade_err)}")
                
                # Reverse to get chronological order
                self.trade_history.reverse()
            
            # Load portfolio history (100 most recent)
            c.execute('SELECT * FROM portfolio_history ORDER BY timestamp DESC LIMIT 100')
            portfolio_data = c.fetchall()
            if portfolio_data:
                self.portfolio_history = []
                for row in portfolio_data:
                    try:
                        # Handle potential issues with timestamp parsing
                        portfolio_time = datetime.fromisoformat(row[0]) if isinstance(row[0], str) else row[0]
                        
                        entry = {
                            'timestamp': portfolio_time,
                            'balance': float(row[1]),
                            'equity': float(row[2]),
                            'drawdown': float(row[3]) if len(row) > 3 and row[3] is not None else 0.0,
                            'portfolio_risk': float(row[4]) if len(row) > 4 and row[4] is not None else 0.0
                        }
                        self.portfolio_history.append(entry)
                    except Exception as portfolio_err:
                        self.logger.error(f"Error loading portfolio history entry: {str(portfolio_err)}")
                
                # Reverse to get chronological order
                self.portfolio_history.reverse()
                
                # Update risk management with current drawdown - MOVED to after risk management initialization
                
            # Load market regimes - MOVED to after initialization
            
            conn.close()
            
            # Validate state
            if self.balance.get('ZUSD', 0) <= 0:
                self.balance = {'ZUSD': self.initial_capital}
                self.logger.warning(f"Invalid balance loaded, reset to ${self.initial_capital:.2f}")
            
            # Don't calculate total equity yet - we need to wait for API initialization
            
            self.logger.info(f"Trading state loaded: Balance=${self.balance.get('ZUSD', 0):.2f}, "
                        f"Positions={len(self.positions)}, "
                        f"Trades={len(self.trade_history)}")
            
        except Exception as e:
            self.logger.error(f"Error loading trading state: {str(e)}")
            traceback.print_exc()
            # Initialize with defaults if loading fails
            self.balance = {'ZUSD': self.initial_capital}
    
    def _start_background_tasks(self):
        """Start background tasks for data collection, monitoring, etc."""
        try:
            # Create tasks
            self.tasks = []
            
            # Create event loop if needed
            try:
                self.event_loop = asyncio.get_event_loop()
            except RuntimeError:
                self.event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.event_loop)
            
            # Sentiment collection task
            self.tasks.append(
                asyncio.create_task(self._sentiment_collection_task())
            )
            
            # Portfolio monitoring task
            self.tasks.append(
                asyncio.create_task(self._portfolio_monitoring_task())
            )
            
            # Model retraining task
            self.tasks.append(
                asyncio.create_task(self._model_retraining_task())
            )
            
            # Data cleanup task
            self.tasks.append(
                asyncio.create_task(self._data_cleanup_task())
            )
            
            # Market cycle detection task
            self.tasks.append(
                asyncio.create_task(self._market_regime_detection_task())
            )
            
            # Performance analytics task
            self.tasks.append(
                asyncio.create_task(self._performance_analytics_task())
            )
            
            # Risk management analysis task
            self.tasks.append(
                asyncio.create_task(self._risk_management_task())
            )
            
            self.logger.info(f"Started {len(self.tasks)} background tasks")
        except Exception as e:
            self.logger.error(f"Error starting background tasks: {str(e)}")
            self.tasks = []

    async def _sentiment_collection_task(self):
        """Periodically collect sentiment data."""
        while self.is_running:
            try:
                if self.is_backtesting:
                    await asyncio.sleep(60)
                    continue
                    
                self.logger.info("Collecting sentiment data...")
                
                # Process just 1 symbol per cycle to minimize API usage
                random_symbol = random.choice(list(self.symbols.keys()))
                self.logger.info(f"Processing sentiment for: {random_symbol}")
                
                try:
                    # Twitter sentiment - simplified with neutral fallback
                    twitter_sentiment = self.twitter_analyzer.get_twitter_sentiment(random_symbol)
                    self.logger.info(f"Twitter sentiment for {random_symbol}: {twitter_sentiment['score']:.2f} "
                                f"(source: {twitter_sentiment['source']})")
                    
                    await asyncio.sleep(5)  # Small delay
                    
                    # Reddit sentiment
                    reddit_sentiment = await self.reddit_analyzer.get_reddit_sentiment(random_symbol)
                    self.logger.info(f"Reddit sentiment for {random_symbol}: {reddit_sentiment['score']:.2f}")
                    
                    await asyncio.sleep(5)
                    
                    # News sentiment
                    news_impact = self.news_analyzer.get_news_impact(random_symbol)
                    self.logger.info(f"News impact for {random_symbol}: {news_impact['impact_score']:.2f}")
                    
                    # Combine sentiment scores
                    combined_score = self._calculate_combined_sentiment(
                        twitter_sentiment.get('score', 0),
                        reddit_sentiment.get('score', 0),
                        news_impact.get('impact_score', 0)
                    )
                    
                    # Store in database
                    self._store_sentiment_data(random_symbol, {
                        'twitter_score': twitter_sentiment.get('score', 0),
                        'reddit_score': reddit_sentiment.get('score', 0),
                        'news_score': news_impact.get('impact_score', 0),
                        'combined_score': combined_score
                    })
                    
                    # Store in cache for quicker access
                    if not hasattr(self, 'sentiment_cache'):
                        self.sentiment_cache = {}
                        
                    self.sentiment_cache[random_symbol] = {
                        'twitter_score': twitter_sentiment.get('score', 0),
                        'reddit_score': reddit_sentiment.get('score', 0),
                        'news_score': news_impact.get('impact_score', 0),
                        'combined_score': combined_score,
                        'timestamp': time.time()
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error collecting sentiment for {random_symbol}: {str(e)}")
                
                # Save sentiment cache periodically
                self._save_disk_cache()
                
                # Wait much longer between collection cycles
                self.logger.info("Sentiment collection complete - next update in 120 minutes")
                await asyncio.sleep(7200)  # 120 minutes (2 hours)
                
            except Exception as e:
                self.logger.error(f"Error in sentiment collection task: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    def _adjust_risk_for_market_events(self):
        """Adjust risk parameters when significant market events are detected."""
        # Temporarily reduce position sizes
        self.risk_management['risk_reduction_factor'] = max(
            self.risk_management['risk_reduction_factor'] * 0.8,  # 20% reduction
            0.5  # Don't go below 50% of normal size
        )
        
        # Tighten stops on existing positions
        for symbol, position in self.positions.items():
            # Only tighten stops for positions in profit
            if 'entry_price' in position and 'high_price' in position:
                entry_price = position['entry_price']
                high_price = position['high_price']
                current_price = high_price  # Use high price as conservative estimate
                
                if current_price > entry_price:
                    # Position is in profit, tighten trailing stop
                    original_stop_pct = position.get('trailing_stop_pct', self.trailing_stop_pct)
                    tighter_stop_pct = original_stop_pct * 0.8  # 20% tighter
                    position['trailing_stop_pct'] = tighter_stop_pct
                    
                    self.logger.info(f"Tightened trailing stop for {symbol} due to market events: {original_stop_pct:.3f} -> {tighter_stop_pct:.3f}")
        
        # Flag for temporary caution
        self.risk_management['market_event_caution'] = True
        self.risk_management['market_event_caution_until'] = time.time() + 3600  # 1 hour of caution
    
    def _calculate_combined_sentiment(self, twitter_score, reddit_score, news_score):
        """Calculate a weighted combined sentiment score."""
        # Define weights (can be adjusted based on performance)
        twitter_weight = 0.35
        reddit_weight = 0.25
        news_weight = 0.40
        
        # Calculate weighted average
        weighted_sum = (
            twitter_score * twitter_weight +
            reddit_score * reddit_weight +
            news_score * news_weight
        )
        
        return weighted_sum
    
    def _store_sentiment_data(self, symbol, sentiment_data):
        """Store sentiment data in database."""
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            
            current_time = datetime.now().isoformat()
            
            c.execute('''
                INSERT OR REPLACE INTO sentiment_data
                (timestamp, symbol, twitter_score, reddit_score, news_score, combined_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                current_time,
                symbol,
                sentiment_data.get('twitter_score', 0),
                sentiment_data.get('reddit_score', 0),
                sentiment_data.get('news_score', 0),
                sentiment_data.get('combined_score', 0)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing sentiment data: {str(e)}")
    
    async def _portfolio_monitoring_task(self):
        """Monitor portfolio, positions, and execute DCA plans with accurate P&L reporting."""
        while self.is_running:
            try:
                if self.is_backtesting:
                    await asyncio.sleep(60)
                    continue
                
                # Allow startup time
                await asyncio.sleep(30)
                
                # Monitor positions
                if self.positions:
                    self.logger.info(f"Monitoring {len(self.positions)} active positions")
                    await self.monitor_positions()
                
                # Check DCA plans
                if self.dca_enabled and self.dca_plans:
                    self.logger.info(f"Checking {len(self.dca_plans)} active DCA plans")
                    await self.check_dca_plans()
                
                # Update portfolio value and metrics
                total_equity = self.calculate_total_equity()
                max_equity = max([entry['equity'] for entry in self.portfolio_history], default=self.initial_capital)
                current_drawdown = (max_equity - total_equity) / max_equity if max_equity > 0 else 0
                
                # Calculate portfolio at risk
                portfolio_at_risk = self._calculate_portfolio_at_risk()
                
                # Update risk management system
                self.risk_management['current_drawdown'] = current_drawdown
                self.risk_management['portfolio_at_risk'] = portfolio_at_risk
                
                self.portfolio_history.append({
                    'timestamp': datetime.now(),
                    'balance': self.balance['ZUSD'],
                    'equity': total_equity,
                    'drawdown': current_drawdown,
                    'portfolio_risk': portfolio_at_risk
                })
                
                # Store in database
                self._store_portfolio_update(total_equity, current_drawdown, portfolio_at_risk)
                
                # Calculate and log metrics
                metrics = self.get_performance_metrics()
                
                # Use accurate P&L reporting that matches the analysis script
                total_pnl = metrics['total_pnl']
                pnl_percentage = metrics['pnl_percentage']
                
                self.logger.info(f"Portfolio value: ${metrics['current_equity']:.2f}, "
                            f"P&L: ${total_pnl:.2f} ({pnl_percentage:.2f}%)")
                
                if metrics['total_trades'] > 0:
                    self.logger.info(f"Win rate: {metrics['win_rate']:.1f}%, "
                                f"Profit factor: {metrics['profit_factor']:.2f}, "
                                f"Drawdown: {metrics['drawdown']:.2f}%")
                
                # Check drawdown protection
                self._check_drawdown_protection(current_drawdown)
                
                # Save state occasionally (25% chance each cycle)
                if random.random() < 0.25:
                    self.save_state()
                
                # Wait for next check (2 minutes)
                await asyncio.sleep(120)
                
            except Exception as e:
                self.logger.error(f"Error in portfolio monitoring task: {str(e)}")
                await asyncio.sleep(60)

    def _check_drawdown_protection(self, current_drawdown):
        """Apply drawdown protection rules."""
        drawdown_settings = self.risk_management['drawdown_protection']
        
        # Check if drawdown exceeds thresholds
        if current_drawdown >= drawdown_settings['max_drawdown']:
            # Catastrophic drawdown - halt trading
            self.logger.warning(f"CRITICAL DRAWDOWN ALERT: {current_drawdown*100:.2f}% exceeds maximum allowed {drawdown_settings['max_drawdown']*100:.2f}%")
            self.logger.warning("Trading halted until drawdown recovers")
            
            # Set emergency risk reduction
            self.risk_management['active_risk_protection'] = True
            self.risk_management['risk_reduction_factor'] = 0.0  # Complete halt (0% of normal size)
            
        elif current_drawdown >= drawdown_settings['tier2_drawdown']:
            # Severe drawdown - apply tier 2 protection
            if not self.risk_management['active_risk_protection'] or self.risk_management['risk_reduction_factor'] > (1.0 - drawdown_settings['tier2_risk_reduction']):
                self.logger.warning(f"SEVERE DRAWDOWN ALERT: {current_drawdown*100:.2f}% exceeds tier 2 threshold {drawdown_settings['tier2_drawdown']*100:.2f}%")
                self.logger.warning(f"Applying {drawdown_settings['tier2_risk_reduction']*100:.0f}% risk reduction")
                
                self.risk_management['active_risk_protection'] = True
                self.risk_management['risk_reduction_factor'] = 1.0 - drawdown_settings['tier2_risk_reduction']
                
        elif current_drawdown >= drawdown_settings['tier1_drawdown']:
            # Moderate drawdown - apply tier 1 protection
            if not self.risk_management['active_risk_protection']:
                self.logger.warning(f"DRAWDOWN ALERT: {current_drawdown*100:.2f}% exceeds tier 1 threshold {drawdown_settings['tier1_drawdown']*100:.2f}%")
                self.logger.warning(f"Applying {drawdown_settings['tier1_risk_reduction']*100:.0f}% risk reduction")
                
                self.risk_management['active_risk_protection'] = True
                self.risk_management['risk_reduction_factor'] = 1.0 - drawdown_settings['tier1_risk_reduction']
                
        elif self.risk_management['active_risk_protection'] and current_drawdown < drawdown_settings['recovery_threshold']:
            # Recovery - remove protection
            self.logger.info(f"DRAWDOWN RECOVERED: {current_drawdown*100:.2f}% below recovery threshold {drawdown_settings['recovery_threshold']*100:.2f}%")
            self.logger.info("Removing risk protection")
            
            self.risk_management['active_risk_protection'] = False
            self.risk_management['risk_reduction_factor'] = 1.0  # Back to normal sizing
    
    def _calculate_portfolio_at_risk(self):
        """Calculate the percentage of portfolio currently at risk."""
        try:
            # Get total equity
            total_equity = self.calculate_total_equity()
            if total_equity <= 0:
                return 0.0
            
            total_at_risk = 0.0
            
            # Calculate risk for each position
            for symbol, position in self.positions.items():
                if 'volume' not in position or 'entry_price' not in position or 'stop_loss' not in position:
                    continue
                
                # Get current price
                current_price = None
                if hasattr(self, 'price_cache') and f"price_{symbol}" in self.price_cache:
                    current_price = self.price_cache[f"price_{symbol}"]["price"]
                
                if not current_price:
                    try:
                        # Try to get price from API
                        current_price = self.api.query_public('Ticker', {'pair': symbol})
                        if 'result' in current_price and symbol in current_price['result']:
                            current_price = float(current_price['result'][symbol]['c'][0])
                        else:
                            current_price = position['entry_price']  # Fallback
                    except:
                        current_price = position['entry_price']  # Fallback
                
                # Calculate position risk
                position_value = position['volume'] * current_price
                stop_level = position['stop_loss']
                
                # If price moved up significantly, stop may be higher than entry
                if stop_level <= 0:
                    stop_level = position['entry_price'] * (1 - self.stop_loss_pct)
                
                # Calculate money at risk
                if current_price > stop_level:
                    money_at_risk = position_value * ((current_price - stop_level) / current_price)
                    total_at_risk += money_at_risk
            
            # Convert to percentage of portfolio
            portfolio_at_risk = total_at_risk / total_equity
            
            return portfolio_at_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio at risk: {str(e)}")
            return 0.0
    
    def _store_portfolio_update(self, total_equity, drawdown, portfolio_risk=0.0):
        """Store portfolio update in database."""
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            
            current_time = datetime.now().isoformat()
            
            c.execute('''
                INSERT OR REPLACE INTO portfolio_history
                (timestamp, balance, equity, drawdown, portfolio_risk)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                current_time,
                self.balance['ZUSD'],
                total_equity,
                drawdown,
                portfolio_risk
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing portfolio update: {str(e)}")
    
    async def _model_retraining_task(self):
            """Periodically retrain models to adapt to market changes."""
            while self.is_running:
                try:
                    if self.is_backtesting:
                        await asyncio.sleep(60)
                        continue
                    
                    # Wait for initial startup period
                    await asyncio.sleep(1800)  # 30 minutes
                    
                    # Check if we need initial training
                    if not self.is_models_loaded:
                        self.logger.info("Performing initial model training...")
                        
                        # Collect training data
                        training_data = await self.collect_training_data()
                        
                        if training_data is None or len(training_data) < 1000:
                            self.logger.warning("Insufficient data for initial training")
                            await asyncio.sleep(1800)  # Try again in 30 minutes
                            continue
                        
                        # Train models
                        ml_success = await self.train_ml_model(training_data)
                        ai_success = await self.train_ai_model(training_data)
                        
                        if ml_success and ai_success:
                            self.is_models_loaded = True
                            self.logger.info("Initial training completed successfully")
                        else:
                            self.logger.error("Initial training failed")
                            await asyncio.sleep(3600)  # Try again in 1 hour
                            continue
                    
                    # Regular retraining (every 24 hours)
                    self.logger.info("Starting scheduled model retraining")
                    
                    # Collect updated training data
                    training_data = await self.collect_training_data(days=60)  # Use 60 days of data
                    
                    if training_data is None or len(training_data) < 1000:
                        self.logger.warning("Insufficient data for retraining")
                        await asyncio.sleep(21600)  # Try again in 6 hours
                        continue
                    
                    # Retrain models
                    await self.train_ml_model(training_data)
                    await self.train_ai_model(training_data)
                    
                    # Perform correlation analysis to update risk management
                    await self._perform_correlation_analysis()
                    
                    # Perform feature importance analysis
                    await self._analyze_feature_importance()
                    
                    self.logger.info("Model retraining complete, next retraining in 24 hours")
                    await asyncio.sleep(86400)  # 24 hours
                    
                except Exception as e:
                    self.logger.error(f"Error in model retraining task: {str(e)}")
                    traceback.print_exc()
                    await asyncio.sleep(7200)  # 2 hours
        
    async def _perform_correlation_analysis(self):
        """Analyze asset correlations to improve risk management."""
        try:
            self.logger.info("Performing asset correlation analysis...")
            
            # Get historical data for all symbols
            price_data = {}
            for symbol in self.symbols:
                df = await self.get_historical_data(symbol, lookback_days=30)
                if df is not None and not df.empty and 'close' in df.columns:
                    # Resample to daily returns for better correlation analysis
                    daily_returns = df['close'].resample('D').last().pct_change().dropna()
                    price_data[symbol] = daily_returns
            
            if len(price_data) < 2:
                self.logger.warning("Insufficient data for correlation analysis")
                return
            
            # Create DataFrame with all returns
            returns_df = pd.DataFrame(price_data)
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            
            # Update correlation matrix in risk management
            self.risk_management['correlation_matrix'] = correlation_matrix.to_dict()
            
            # Log high correlations
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    coin1 = correlation_matrix.columns[i]
                    coin2 = correlation_matrix.columns[j]
                    corr = correlation_matrix.iloc[i, j]
                    
                    if abs(corr) > 0.7:  # High correlation threshold
                        high_correlations.append((coin1, coin2, corr))
            
            if high_correlations:
                self.logger.info("High correlations detected:")
                for coin1, coin2, corr in high_correlations:
                    self.logger.info(f"  {coin1} - {coin2}: {corr:.3f}")
                    
                    # Update correlation groups dynamically
                    self._update_correlation_groups(coin1, coin2, corr)
            
            self.logger.info("Correlation analysis completed")
            
        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {str(e)}")
    
    def _update_correlation_groups(self, coin1, coin2, correlation):
        """Update correlation groups based on observed correlations."""
        # If correlation is very high and positive
        if correlation > 0.7:
            # Find existing groups containing either coin
            found_group = None
            for group_name, coins in self.correlation_groups.items():
                if coin1 in coins or coin2 in coins:
                    found_group = group_name
                    break
            
            if found_group:
                # Add both coins to the group if not already there
                if coin1 not in self.correlation_groups[found_group]:
                    self.correlation_groups[found_group].append(coin1)
                if coin2 not in self.correlation_groups[found_group]:
                    self.correlation_groups[found_group].append(coin2)
            else:
                # Create a new group
                group_name = f"correlated_group_{len(self.correlation_groups) + 1}"
                self.correlation_groups[group_name] = [coin1, coin2]
    
    async def _analyze_feature_importance(self):
        """Analyze feature importance from ML models to improve trading signals."""
        try:
            if not hasattr(self.model_manager, 'feature_importance') or not self.model_manager.feature_importance:
                self.logger.warning("No feature importance data available")
                return
            
            # Get feature importance
            feature_importance = self.model_manager.feature_importance
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Log top features
            self.logger.info("Top important features:")
            for feature, importance in sorted_features[:5]:
                self.logger.info(f"  {feature}: {importance:.4f}")
            
            # Adjust signal weights based on important features
            self._adjust_signal_weights_from_features(sorted_features)
            
        except Exception as e:
            self.logger.error(f"Error analyzing feature importance: {str(e)}")
    
    def _adjust_signal_weights_from_features(self, sorted_features):
        """Adjust trading signal weights based on ML feature importance."""
        try:
            # Group features by category
            feature_categories = {
                'technical': ['rsi', 'macd', 'bb_', 'sma_', 'ema_', 'adx', 'atr', 'cci'],
                'price': ['returns', 'close', 'high', 'low', 'open'],
                'volume': ['volume'],
                'volatility': ['std', 'rolling_std']
            }
            
            # Calculate importance by category
            category_importance = {cat: 0.0 for cat in feature_categories}
            
            for feature, importance in sorted_features:
                for category, patterns in feature_categories.items():
                    if any(pattern in feature for pattern in patterns):
                        category_importance[category] += importance
                        break
            
            # Normalize importance values
            total_importance = sum(category_importance.values())
            if total_importance > 0:
                normalized_importance = {
                    cat: val / total_importance for cat, val in category_importance.items()
                }
                
                # Adjust trading signal weights
                # If technical indicators are highly important
                if normalized_importance.get('technical', 0) > 0.5:
                    # Boost basic technical signals
                    self.signal_weights['basic'] = min(0.35, self.signal_weights['basic'] * 1.2)
                    self.logger.info(f"Increased basic technical signal weight to {self.signal_weights['basic']:.2f}")
                
                # If volatility measures are important
                if normalized_importance.get('volatility', 0) > 0.2:
                    # Reduce basic signals and increase AI signals (better at handling volatility)
                    self.signal_weights['basic'] = max(0.15, self.signal_weights['basic'] * 0.9)
                    self.signal_weights['ai'] = min(0.35, self.signal_weights['ai'] * 1.1)
                    self.logger.info("Adjusted signal weights for volatility patterns")
                
                # Re-normalize weights to sum to 1.0 (excluding pattern weight)
                pattern_weight = self.signal_weights.get('pattern', 0)
                total_other_weights = sum(v for k, v in self.signal_weights.items() if k != 'pattern')
                
                if total_other_weights > 0:
                    scaling_factor = (1.0 - pattern_weight) / total_other_weights
                    for k in self.signal_weights:
                        if k != 'pattern':
                            self.signal_weights[k] *= scaling_factor
                
                self.logger.info(f"Adjusted signal weights based on feature importance: {self.signal_weights}")
        
        except Exception as e:
            self.logger.error(f"Error adjusting signal weights: {str(e)}")
    
    async def _market_regime_detection_task(self):
        """Periodically detect market regimes to adapt trading strategies."""
        while self.is_running:
            try:
                if self.is_backtesting:
                    await asyncio.sleep(60)
                    continue
                
                # Wait for initial period
                await asyncio.sleep(1200)  # 20 minutes after startup
                
                current_time = datetime.now()
                time_since_last = (current_time - self.last_regime_check).total_seconds()
                
                # Check market regimes every 8 hours or at the start
                if time_since_last < 28800 and len(self.market_regimes['regime_history']) > 0:
                    await asyncio.sleep(1800)  # Check again in 30 minutes
                    continue
                
                self.logger.info("Analyzing market regimes...")
                self.last_regime_check = current_time
                
                # Get regime data for each symbol
                for symbol in self.symbols:
                    try:
                        # Get recent data
                        df = await self.get_historical_data(symbol, lookback_days=10)
                        if df is None or len(df) < 20:
                            continue
                        
                        # Process data
                        df = self.calculate_indicators(df)
                        
                        # Detect market regime
                        regime = self.detect_market_cycle(df)
                        
                        # Calculate other regime metrics
                        volatility = self._calculate_volatility(df)
                        trend_strength = self._calculate_trend_strength(df)
                        
                        # Store regime
                        self.market_regimes['symbols'][symbol] = {
                            'regime': regime,
                            'volatility': volatility,
                            'trend_strength': trend_strength,
                            'timestamp': current_time.isoformat()
                        }
                        
                        # Store in database
                        self._store_market_regime(symbol, regime, volatility, trend_strength)
                        
                        self.logger.info(f"Market regime for {symbol}: {regime} (volatility: {volatility:.3f}, trend: {trend_strength:.3f})")
                        
                        # If regime has changed, adapt parameters
                        self._adapt_parameters_to_regime(symbol, regime)
                        
                        await asyncio.sleep(2)  # Small delay between symbols
                        
                    except Exception as e:
                        self.logger.error(f"Error detecting regime for {symbol}: {str(e)}")
                
                # Analyze global regime
                await self._detect_global_market_regime()
                
                # Save regime data to disk
                self._save_disk_cache()
                
                # Standard check interval (8 hours)
                await asyncio.sleep(28800)
                
            except Exception as e:
                self.logger.error(f"Error in market regime detection: {str(e)}")
                await asyncio.sleep(3600)  # 1 hour on error
    
    def _calculate_volatility(self, df):
        """Calculate market volatility."""
        if 'close' not in df.columns:
            return 0.0
        
        # Calculate returns
        returns = df['close'].pct_change().dropna()
        
        # Calculate annualized volatility
        if len(returns) >= 5:  # Need at least a few data points
            volatility = returns.std() * np.sqrt(365 * 24 / self.timeframe)
            return volatility
        
        return 0.0
    
    def _calculate_trend_strength(self, df):
        """Calculate the strength of the current trend."""
        if 'close' not in df.columns or len(df) < 20:
            return 0.0
        
        # Use ADX if available
        if 'adx' in df.columns:
            adx = df['adx'].iloc[-1]
            return adx / 100.0  # Normalize to 0-1 range
        
        # Alternative method using moving averages
        try:
            sma20 = df['close'].rolling(window=20).mean()
            sma50 = df['close'].rolling(window=50).mean() if len(df) >= 50 else df['close'].rolling(window=20).mean()
            
            # Check if shorter MA is above longer MA (uptrend) or below (downtrend)
            is_uptrend = sma20.iloc[-1] > sma50.iloc[-1]
            
            # Calculate the percentage difference
            pct_diff = abs(sma20.iloc[-1] - sma50.iloc[-1]) / sma50.iloc[-1]
            
            # Scale to 0-1 range and apply sign based on trend direction
            trend_strength = min(1.0, pct_diff * 10)  # Scale factor of 10 to make it more sensitive
            
            if not is_uptrend:
                trend_strength = -trend_strength
                
            return trend_strength
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {str(e)}")
            return 0.0
    
    def _store_market_regime(self, symbol, regime, volatility, trend_strength):
        """Store market regime in database."""
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            
            current_time = datetime.now().isoformat()
            
            c.execute('''
                INSERT OR REPLACE INTO market_regimes
                (timestamp, symbol, regime, volatility, trend_strength)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                current_time,
                symbol,
                regime,
                volatility,
                trend_strength
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing market regime: {str(e)}")
    
    async def _detect_global_market_regime(self):
        """Detect the global market regime based on individual symbols."""
        try:
            # Count regimes
            regime_counts = {}
            
            # Consider only recent regimes
            recent_cutoff = datetime.now() - timedelta(hours=12)
            
            for symbol, data in self.market_regimes['symbols'].items():
                # Skip outdated regime data
                regime_time = datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp']
                if regime_time < recent_cutoff:
                    continue
                
                regime = data['regime']
                if regime not in regime_counts:
                    regime_counts[regime] = 0
                regime_counts[regime] += 1
            
            if not regime_counts:
                self.logger.warning("No recent regime data available")
                return
            
            # Find most common regime
            global_regime = max(regime_counts.items(), key=lambda x: x[1])[0]
            
            # Check if regime has changed
            previous_regime = self.market_regimes['global']
            if global_regime != previous_regime:
                self.logger.info(f"Global market regime changed: {previous_regime} -> {global_regime}")
                
                # Record regime change
                self.market_regimes['regime_history'].append({
                    'from': previous_regime,
                    'to': global_regime,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Keep history limited in size
                if len(self.market_regimes['regime_history']) > 20:
                    self.market_regimes['regime_history'] = self.market_regimes['regime_history'][-20:]
                
                # Update global regime
                self.market_regimes['global'] = global_regime
                
                # Adapt global parameters to new regime
                self._adapt_global_parameters(global_regime)
            
        except Exception as e:
            self.logger.error(f"Error detecting global market regime: {str(e)}")
    
    def _adapt_parameters_to_regime(self, symbol, regime):
        """Adapt trading parameters for a symbol based on its market regime."""
        try:
            # Skip if regime hasn't changed
            if symbol in self.market_regimes['symbols'] and self.market_regimes['symbols'][symbol].get('regime') == regime:
                return
                
            self.logger.info(f"Adapting parameters for {symbol} to {regime} regime")
            
            # Adjust buy threshold based on regime
            if regime in ['bull_trend', 'breakout']:
                # More aggressive in bullish regimes
                original_threshold = self.buy_thresholds.get(symbol, 0.52)
                self.buy_thresholds[symbol] = original_threshold * 0.96  # Lower threshold
                self.logger.info(f"Adjusted buy threshold for {symbol}: {original_threshold:.3f} -> {self.buy_thresholds[symbol]:.3f}")
                
            elif regime in ['bear_trend', 'breakdown']:
                # More conservative in bearish regimes
                original_threshold = self.buy_thresholds.get(symbol, 0.52)
                self.buy_thresholds[symbol] = original_threshold * 1.06  # Higher threshold
                self.logger.info(f"Adjusted buy threshold for {symbol}: {original_threshold:.3f} -> {self.buy_thresholds[symbol]:.3f}")
                
            # Symbol-specific risk parameters stored for reference
            if not hasattr(self, 'symbol_risk_params'):
                self.symbol_risk_params = {}
                
            # Update parameters based on regime
            self.symbol_risk_params[symbol] = {
                'regime': regime,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct,
                'trailing_stop_pct': self.trailing_stop_pct
            }
            
            # Apply regime-specific adjustments (these will be used dynamically during trading)
            if regime in self.regime_parameters:
                self.symbol_risk_params[symbol].update(self.regime_parameters[regime])
            
        except Exception as e:
            self.logger.error(f"Error adapting parameters for {symbol}: {str(e)}")
    
    def _adapt_global_parameters(self, global_regime):
        """Adapt global trading parameters based on global market regime."""
        try:
            self.logger.info(f"Adapting global parameters to {global_regime} regime")
            
            # Apply regime-specific parameters
            if global_regime in self.regime_parameters:
                params = self.regime_parameters[global_regime]
                
                # Adjust stop loss parameters
                self.stop_loss_pct = params.get('STOP_LOSS_PCT', self.stop_loss_pct)
                self.take_profit_pct = params.get('TAKE_PROFIT_PCT', self.take_profit_pct)
                self.trailing_stop_pct = params.get('TRAILING_STOP_PCT', self.trailing_stop_pct)
                
                # Adjust position sizing
                self.max_position_size = params.get('MAX_POSITION_SIZE', self.max_position_size)
                
                # Adjust DCA parameters
                if 'DCA_PARTS' in params:
                    self.dca_parts = params['DCA_PARTS']
                
                self.logger.info(f"Updated global parameters: stop_loss_pct={self.stop_loss_pct:.3f}, "
                            f"take_profit_pct={self.take_profit_pct:.3f}, "
                            f"max_position_size={self.max_position_size:.3f}")
                            
                # Adjust signal weights based on regime
                if global_regime in ['bull_trend', 'breakout']:
                    # In bull markets, favor AI and sentiment more
                    self.signal_weights['ml'] = 0.22
                    self.signal_weights['ai'] = 0.28
                    self.signal_weights['basic'] = 0.22
                    self.signal_weights['sentiment'] = 0.28
                    self.signal_weights['pattern'] = 0.0  # Adjusted dynamically when patterns detected
                    
                elif global_regime in ['bear_trend', 'breakdown']:
                    # In bear markets, favor ML and basic technical more
                    self.signal_weights['ml'] = 0.30
                    self.signal_weights['ai'] = 0.20
                    self.signal_weights['basic'] = 0.30
                    self.signal_weights['sentiment'] = 0.20
                    self.signal_weights['pattern'] = 0.0
                    
                else:
                    # Balanced weights for ranging markets
                    self.signal_weights['ml'] = 0.25
                    self.signal_weights['ai'] = 0.25
                    self.signal_weights['basic'] = 0.25
                    self.signal_weights['sentiment'] = 0.25
                    self.signal_weights['pattern'] = 0.0
            
        except Exception as e:
            self.logger.error(f"Error adapting global parameters: {str(e)}")
    
    async def _performance_analytics_task(self):
        """Analyze trading performance to improve strategies."""
        while self.is_running:
            try:
                if self.is_backtesting:
                    await asyncio.sleep(60)
                    continue
                
                # Wait for initial period
                await asyncio.sleep(3600)  # 1 hour after startup
                
                self.logger.info("Analyzing trading performance...")
                
                # Analyze trades by signal type
                self._analyze_signal_performance()
                
                # Analyze trade performance by time of day
                self._analyze_time_patterns()
                
                # Analyze position sizing effectiveness
                self._analyze_position_sizing()
                
                # Analyze pattern detection success rate
                await self._analyze_pattern_success()
                
                # Generate performance report
                self._generate_performance_report()
                
                # Wait for next analysis cycle (6 hours)
                self.logger.info("Performance analysis complete, next analysis in 6 hours")
                await asyncio.sleep(21600)  # 6 hours
                
            except Exception as e:
                self.logger.error(f"Error in performance analytics task: {str(e)}")
                await asyncio.sleep(7200)  # 2 hours on error
    
    def _analyze_signal_performance(self):
        """Analyze performance of different trading signals."""
        try:
            # Get recent trades (last 100)
            recent_trades = self.trade_history[-100:] if len(self.trade_history) >= 100 else self.trade_history
            
            if not recent_trades:
                return
            
            # Group trades by signal type
            signal_performance = {
                'ml': {'wins': 0, 'losses': 0, 'total_pnl': 0.0},
                'ai': {'wins': 0, 'losses': 0, 'total_pnl': 0.0},
                'basic': {'wins': 0, 'losses': 0, 'total_pnl': 0.0},
                'sentiment': {'wins': 0, 'losses': 0, 'total_pnl': 0.0},
                'pattern': {'wins': 0, 'losses': 0, 'total_pnl': 0.0},
                'combined': {'wins': 0, 'losses': 0, 'total_pnl': 0.0}
            }
            
            for trade in recent_trades:
                # Skip if not a completed trade or no signals info
                if trade.get('type') not in ['sell', 'partial_sell'] or 'profit_loss' not in trade:
                    continue
                
                # Get signals used
                signals_used = trade.get('signals_used', '')
                
                # If no specific signals recorded, count as combined
                if not signals_used:
                    signal = 'combined'
                else:
                    # Use the primary signal
                    signal = signals_used.split(',')[0].strip()
                    
                    # Map to main categories
                    if signal not in signal_performance:
                        signal = 'combined'
                
                # Record win/loss and PnL
                pnl = trade.get('profit_loss', 0)
                if pnl > 0:
                    signal_performance[signal]['wins'] += 1
                else:
                    signal_performance[signal]['losses'] += 1
                    
                signal_performance[signal]['total_pnl'] += pnl
            
            # Calculate win rates and return metrics
            results = {}
            for signal, data in signal_performance.items():
                total_trades = data['wins'] + data['losses']
                if total_trades > 0:
                    win_rate = data['wins'] / total_trades * 100
                    avg_pnl = data['total_pnl'] / total_trades
                    
                    results[signal] = {
                        'win_rate': win_rate,
                        'total_trades': total_trades,
                        'total_pnl': data['total_pnl'],
                        'avg_pnl': avg_pnl
                    }
                    
                    self.logger.info(f"Signal performance - {signal}: {win_rate:.1f}% win rate over {total_trades} trades, avg PnL: ${avg_pnl:.2f}")
            
            # Store for future reference
            self.performance_analytics['signal_performance'] = results
            
            # Adjust weights based on performance
            self._adjust_signal_weights_from_performance(results)
            
        except Exception as e:
            self.logger.error(f"Error analyzing signal performance: {str(e)}")
    
    def _adjust_signal_weights_from_performance(self, signal_performance):
        """Adjust signal weights based on historical performance."""
        try:
            # Minimum data threshold
            min_trades = 5
            
            # Copy current weights
            original_weights = self.signal_weights.copy()
            
            # Track signals with enough data
            valid_signals = {}
            
            # Calculate performance scores for each signal
            for signal, data in signal_performance.items():
                if signal != 'combined' and data.get('total_trades', 0) >= min_trades:
                    # Calculate weighted score: win rate * avg PnL
                    win_rate = data.get('win_rate', 50) / 100  # Convert to 0-1
                    avg_pnl = data.get('avg_pnl', 0)
                    
                    # If avg_pnl is negative, use a reduced score
                    if avg_pnl < 0:
                        avg_pnl = avg_pnl / 10  # Reduce negative impact
                    
                    # Calculate performance score
                    score = win_rate * (1 + avg_pnl / 1000)  # Scale PnL effect
                    
                    valid_signals[signal] = score
            
            # Skip if not enough data
            if len(valid_signals) < 2:
                return
            
            # Calculate total score for normalization
            total_score = sum(valid_signals.values())
            
            if total_score <= 0:
                return
                
            # Calculate target weights (reserve 30% for basic signal)
            pattern_weight = self.signal_weights.get('pattern', 0)
            distributable_weight = 1.0 - pattern_weight
            
            # Calculate new weights based on performance
            new_weights = {}
            for signal in self.signal_weights:
                if signal == 'pattern':
                    # Keep pattern weight unchanged
                    new_weights[signal] = pattern_weight
                elif signal in valid_signals:
                    # Calculate performance-based weight
                    new_weights[signal] = (valid_signals[signal] / total_score) * distributable_weight
                else:
                    # Keep original weight for signals without data
                    new_weights[signal] = self.signal_weights[signal]
            
            # Ensure minimum weights and normalize
            min_weight = 0.15  # Minimum 15% weight for any signal
            for signal in new_weights:
                if signal != 'pattern':  # Don't apply min to pattern weight
                    new_weights[signal] = max(min_weight, new_weights[signal])
            
            # Re-normalize to sum to 1.0
            total_non_pattern = sum(w for s, w in new_weights.items() if s != 'pattern')
            scaling_factor = (1.0 - pattern_weight) / total_non_pattern
            
            for signal in new_weights:
                if signal != 'pattern':
                    new_weights[signal] *= scaling_factor
            
            # Update weights if they've changed significantly
            weight_changed = False
            for signal, weight in new_weights.items():
                if abs(weight - self.signal_weights.get(signal, 0)) > 0.03:
                    weight_changed = True
                    break
                    
            if weight_changed:
                self.signal_weights = new_weights
                self.logger.info(f"Adjusted signal weights based on performance: {self.signal_weights}")
                
                # Log the changes
                for signal in self.signal_weights:
                    if signal in original_weights:
                        change = self.signal_weights[signal] - original_weights[signal]
                        self.logger.info(f"  {signal}: {original_weights[signal]:.2f} -> {self.signal_weights[signal]:.2f} ({change:+.2f})")
            
        except Exception as e:
            self.logger.error(f"Error adjusting signal weights: {str(e)}")
    
    def _analyze_time_patterns(self):
        """Analyze trading performance by time of day and day of week."""
        try:
            # Get recent trades with timestamps
            recent_trades = [t for t in self.trade_history if 'timestamp' in t]
            
            if len(recent_trades) < 20:
                return
            
            # Group by hour of day
            hour_performance = {}
            for i in range(24):
                hour_performance[i] = {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0}
            
            # Group by day of week
            day_performance = {}
            for i in range(7):
                day_performance[i] = {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0}
            
            # Analyze trades
            for trade in recent_trades:
                if trade.get('type') not in ['sell', 'partial_sell']:
                    continue
                
                timestamp = trade.get('timestamp')
                pnl = trade.get('profit_loss', 0)
                
                # Hour analysis
                hour = timestamp.hour
                hour_performance[hour]['trades'] += 1
                hour_performance[hour]['pnl'] += pnl
                
                if pnl > 0:
                    hour_performance[hour]['wins'] += 1
                else:
                    hour_performance[hour]['losses'] += 1
                
                # Day analysis
                day = timestamp.weekday()
                day_performance[day]['trades'] += 1
                day_performance[day]['pnl'] += pnl
                
                if pnl > 0:
                    day_performance[day]['wins'] += 1
                else:
                    day_performance[day]['losses'] += 1
            
            # Find best and worst times
            best_hour = max(hour_performance.items(), key=lambda x: x[1]['pnl'] if x[1]['trades'] > 0 else -float('inf'))
            worst_hour = min(hour_performance.items(), key=lambda x: x[1]['pnl'] if x[1]['trades'] > 0 else float('inf'))
            
            best_day = max(day_performance.items(), key=lambda x: x[1]['pnl'] if x[1]['trades'] > 0 else -float('inf'))
            worst_day = min(day_performance.items(), key=lambda x: x[1]['pnl'] if x[1]['trades'] > 0 else float('inf'))
            
            # Find hours with enough trades for analysis
            hour_win_rates = {}
            for hour, data in hour_performance.items():
                if data['trades'] >= 5:  # At least 5 trades to consider
                    win_rate = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
                    hour_win_rates[hour] = {
                        'win_rate': win_rate, 
                        'trades': data['trades'],
                        'pnl': data['pnl']
                    }
            
            # Log results
            self.logger.info("Time pattern analysis:")
            
            if best_hour[1]['trades'] > 0:
                best_hr = best_hour[0]
                best_data = best_hour[1]
                best_win_rate = best_data['wins'] / best_data['trades'] * 100 if best_data['trades'] > 0 else 0
                self.logger.info(f"  Best hour: {best_hr:02d}:00 - {best_win_rate:.1f}% win rate, ${best_data['pnl']:.2f} PnL over {best_data['trades']} trades")
            
            if worst_hour[1]['trades'] > 0:
                worst_hr = worst_hour[0]
                worst_data = worst_hour[1]
                worst_win_rate = worst_data['wins'] / worst_data['trades'] * 100 if worst_data['trades'] > 0 else 0
                self.logger.info(f"  Worst hour: {worst_hr:02d}:00 - {worst_win_rate:.1f}% win rate, ${worst_data['pnl']:.2f} PnL over {worst_data['trades']} trades")
            
            # Store analytics
            self.performance_analytics['time_patterns'] = {
                'hour_performance': hour_performance,
                'day_performance': day_performance,
                'best_hour': best_hour[0],
                'worst_hour': worst_hour[0],
                'best_day': best_day[0],
                'worst_day': worst_day[0]
            }
            
            # Adjust trading behavior based on time patterns
            self._adjust_for_time_patterns()
            
        except Exception as e:
            self.logger.error(f"Error analyzing time patterns: {str(e)}")
    
    def _adjust_for_time_patterns(self):
        """Adjust trading behavior based on discovered time patterns."""
        try:
            time_patterns = self.performance_analytics.get('time_patterns', {})
            if not time_patterns:
                return
            
            # Check if we have hour performance data
            hour_performance = time_patterns.get('hour_performance', {})
            if not hour_performance:
                return
            
            # Get current hour
            current_hour = datetime.now().hour
            
            # Get data for current hour
            hour_data = hour_performance.get(current_hour, {})
            trades = hour_data.get('trades', 0)
            
            if trades < 5:  # Not enough data for this hour
                return
                
            win_rate = hour_data.get('wins', 0) / trades * 100 if trades > 0 else 0
            
            # Create time-based adjustments
            if not hasattr(self, 'time_adjustments'):
                self.time_adjustments = {}
            
            # Set adjustments based on win rate
            if win_rate < 40:  # Poor performance hour
                self.time_adjustments[current_hour] = {
                    'buy_threshold_mod': 1.08,  # More conservative
                    'position_size_mod': 0.7   # Smaller positions
                }
                self.logger.info(f"Applied conservative adjustments for low-performance hour {current_hour}")
                
            elif win_rate > 60:  # Good performance hour
                self.time_adjustments[current_hour] = {
                    'buy_threshold_mod': 0.95,  # More aggressive
                    'position_size_mod': 1.2   # Larger positions
                }
                self.logger.info(f"Applied aggressive adjustments for high-performance hour {current_hour}")
                
            else:  # Neutral performance
                self.time_adjustments[current_hour] = {
                    'buy_threshold_mod': 1.0,
                    'position_size_mod': 1.0
                }
            
        except Exception as e:
            self.logger.error(f"Error adjusting for time patterns: {str(e)}")
    
    def _analyze_position_sizing(self):
        """Analyze effectiveness of position sizing strategy."""
        try:
            # Get recent trades
            recent_trades = [t for t in self.trade_history 
                            if t.get('type') in ['sell', 'partial_sell'] 
                            and 'value' in t and 'profit_loss' in t]
            
            if len(recent_trades) < 10:
                return
            
            # Group by position size buckets
            size_buckets = {
                'small': {'trades': 0, 'wins': 0, 'total_pnl': 0, 'avg_return': 0},
                'medium': {'trades': 0, 'wins': 0, 'total_pnl': 0, 'avg_return': 0},
                'large': {'trades': 0, 'wins': 0, 'total_pnl': 0, 'avg_return': 0}
            }
            
            # Calculate quartiles for position sizes
            position_values = [t.get('value', 0) for t in recent_trades]
            small_threshold = np.percentile(position_values, 33)
            large_threshold = np.percentile(position_values, 66)
            
            # Analyze trades
            for trade in recent_trades:
                position_value = trade.get('value', 0)
                pnl = trade.get('profit_loss', 0)
                pnl_pct = trade.get('pnl_percentage', 0)
                
                # Determine size bucket
                if position_value <= small_threshold:
                    bucket = 'small'
                elif position_value <= large_threshold:
                    bucket = 'medium'
                else:
                    bucket = 'large'
                
                # Add to bucket
                size_buckets[bucket]['trades'] += 1
                size_buckets[bucket]['total_pnl'] += pnl
                
                if pnl > 0:
                    size_buckets[bucket]['wins'] += 1
            
            # Calculate metrics
            for bucket, data in size_buckets.items():
                if data['trades'] > 0:
                    data['win_rate'] = data['wins'] / data['trades'] * 100
                    data['avg_pnl'] = data['total_pnl'] / data['trades']
                    
                    self.logger.info(f"Position size '{bucket}': {data['win_rate']:.1f}% win rate, "
                                f"${data['avg_pnl']:.2f} avg PnL, {data['trades']} trades")
            
            # Store analytics
            self.performance_analytics['position_sizing'] = size_buckets
            
            # Adjust position sizing based on results
            self._optimize_position_sizing(size_buckets)
            
        except Exception as e:
            self.logger.error(f"Error analyzing position sizing: {str(e)}")
    
    def _optimize_position_sizing(self, size_buckets):
        """Optimize position sizing based on historical performance."""
        try:
            # Find best performing bucket
            best_bucket = max(size_buckets.items(), key=lambda x: x[1].get('win_rate', 0) if x[1].get('trades', 0) > 5 else 0)
            best_name = best_bucket[0]
            best_data = best_bucket[1]
            
            # Only adjust if we have enough data
            if best_data.get('trades', 0) < 5:
                return
            
            # Create position size modifiers
            if not hasattr(self, 'size_optimization'):
                self.size_optimization = {
                    'base_multiplier': 1.0,
                    'best_bucket': None
                }
            
            # Log the optimal size
            self.logger.info(f"Optimal position size bucket: '{best_name}' with {best_data.get('win_rate', 0):.1f}% win rate")
            
            # Adjust position sizing multiplier
            if best_name == 'small':
                self.size_optimization['base_multiplier'] = .08  # Reduce position sizes
            elif best_name == 'large':
                self.size_optimization['base_multiplier'] = 1.5  # Increase position sizes
            else:
                self.size_optimization['base_multiplier'] = 1.1  # Keep medium sizes
                
            self.size_optimization['best_bucket'] = best_name
            
            self.logger.info(f"Adjusted position size multiplier to {self.size_optimization['base_multiplier']}")
            
        except Exception as e:
            self.logger.error(f"Error optimizing position sizing: {str(e)}")
    
    async def _analyze_pattern_success(self):
        """Analyze success rate of detected chart patterns."""
        try:
            # Get pattern detection history from database
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            
            c.execute('''
                SELECT pattern_type, success, COUNT(*) 
                FROM detected_patterns 
                GROUP BY pattern_type, success
            ''')
            
            pattern_stats = c.fetchall()
            conn.close()
            
            if not pattern_stats:
                return
                
            # Calculate success rates
            success_rates = {}
            for pattern_type, success, count in pattern_stats:
                if pattern_type not in success_rates:
                    success_rates[pattern_type] = {'success': 0, 'fail': 0}
                
                if success:
                    success_rates[pattern_type]['success'] += count
                else:
                    success_rates[pattern_type]['fail'] += count
            
            # Calculate percentages
            for pattern, data in success_rates.items():
                total = data['success'] + data['fail']
                if total > 0:
                    rate = data['success'] / total * 100
                    success_rates[pattern]['rate'] = rate
                    success_rates[pattern]['total'] = total
                    
                    self.logger.info(f"Pattern '{pattern}': {rate:.1f}% success rate from {total} instances")
            
            # Store results
            self.pattern_recognition_system['success_rate'] = success_rates
            
            # Update pattern weight based on overall success
            total_patterns = sum(data.get('total', 0) for data in success_rates.values())
            total_success = sum(data.get('success', 0) for data in success_rates.values())
            
            if total_patterns >= 10:  # Need minimum sample size
                overall_success_rate = total_success / total_patterns if total_patterns > 0 else 0
                
                # Adjust pattern weight dynamically
                if overall_success_rate > 60:  # Good success rate
                    # Calculate weight based on success rate and sample size
                    pattern_influence = min(0.25, 0.1 + (overall_success_rate - 60) * 0.005)
                    
                    # Update weight
                    old_pattern_weight = self.signal_weights.get('pattern', 0)
                    self.signal_weights['pattern'] = pattern_influence
                    
                    # Normalize other weights
                    if pattern_influence > 0:
                        remaining_weight = 1.0 - pattern_influence
                        weight_sum = sum(w for k, w in self.signal_weights.items() if k != 'pattern')
                        
                        if weight_sum > 0:
                            scaling_factor = remaining_weight / weight_sum
                            for k in self.signal_weights:
                                if k != 'pattern':
                                    self.signal_weights[k] *= scaling_factor
                    
                    self.logger.info(f"Adjusted pattern weight from {old_pattern_weight:.2f} to {pattern_influence:.2f} "
                                f"based on {overall_success_rate:.1f}% success rate")
                else:
                    # Low success rate, reduce pattern influence
                    self.signal_weights['pattern'] = 0.0
                    self.logger.info(f"Set pattern weight to 0 due to low {overall_success_rate:.1f}% success rate")
            
        except Exception as e:
            self.logger.error(f"Error analyzing pattern success: {str(e)}")
    
    def check_service_config(self):
        """Check and recommend systemd service file configuration for rate limiting."""
        try:
            # Check if we're running as a systemd service
            service_file_path = '/etc/systemd/system/kryptos-bot.service'
            if os.path.exists(service_file_path):
                self.logger.info("Detected systemd service file")
                
                # Read service file content
                with open(service_file_path, 'r') as f:
                    service_content = f.read()
                    
                # Check for rate limiting options
                if 'Restart=on-failure' in service_content and 'RestartSec=60' not in service_content:
                    self.logger.warning("Service file may need rate limit settings. Consider adding: RestartSec=60")
                    
                # Check for memory limits
                if 'MemoryLimit=' not in service_content:
                    self.logger.info("Consider adding memory limits to service file: MemoryLimit=2G")
            
        except Exception as e:
            self.logger.debug(f"Error checking service config: {str(e)}")

    def _generate_performance_report(self):
        """Generate a comprehensive performance report."""
        try:
            # Get overall metrics
            metrics = self.get_performance_metrics()
            
            # Build a report
            report = {
                'timestamp': datetime.now().isoformat(),
                'overall_metrics': metrics,
                'signal_performance': self.performance_analytics.get('signal_performance', {}),
                'time_patterns': self.performance_analytics.get('time_patterns', {}),
                'position_sizing': self.performance_analytics.get('position_sizing', {}),
                'pattern_performance': self.pattern_recognition_system.get('success_rate', {}),
                'drawdown_history': [
                    {'timestamp': entry.get('timestamp').isoformat(), 'drawdown': entry.get('drawdown', 0)}
                    for entry in self.portfolio_history[-50:] if 'drawdown' in entry
                ]
            }
            
            # Save report to disk
            report_path = os.path.join('risk_analytics', f'performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            self.logger.info(f"Performance report saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {str(e)}")
    
    async def _risk_management_task(self):
        """Periodically assess and manage portfolio risk."""
        while self.is_running:
            try:
                if self.is_backtesting:
                    await asyncio.sleep(60)
                    continue
                
                # Wait for initial period
                await asyncio.sleep(1500)  # 25 minutes after startup
                
                self.logger.info("Performing risk management analysis...")
                
                # Calculate overall portfolio risk
                portfolio_at_risk = self._calculate_portfolio_at_risk()
                self.risk_management['portfolio_at_risk'] = portfolio_at_risk
                
                # Calculate maximum drawdown
                max_drawdown = 0
                if self.portfolio_history and len(self.portfolio_history) > 1:
                    equity_values = [entry['equity'] for entry in self.portfolio_history]
                    running_max = pd.Series(equity_values).cummax()
                    drawdowns = (pd.Series(equity_values) / running_max - 1) * 100
                    max_drawdown = abs(drawdowns.min()) / 100  # Convert back to decimal
                
                self.risk_management['current_drawdown'] = max_drawdown
                
                # Check risk levels and apply protections
                if portfolio_at_risk > self.risk_management['max_portfolio_risk']:
                    self.logger.warning(f"RISK ALERT: Portfolio at risk ({portfolio_at_risk*100:.1f}%) exceeds maximum ({self.risk_management['max_portfolio_risk']*100:.1f}%)")
                    await self._reduce_portfolio_risk()
                
                # Check correlation exposure
                await self._check_correlation_exposure()
                
                # Check open positions for optimal position management
                await self._optimize_open_positions()
                
                # Reset risk protection if recovered
                if (max_drawdown < self.risk_management['drawdown_protection']['recovery_threshold'] and
                    self.risk_management['active_risk_protection']):
                    self.logger.info(f"Drawdown recovered to {max_drawdown*100:.1f}%, removing risk protection")
                    self.risk_management['active_risk_protection'] = False
                    self.risk_management['risk_reduction_factor'] = 1.0
                
                # Clear market event caution if expired
                if hasattr(self.risk_management, 'market_event_caution') and self.risk_management.get('market_event_caution'):
                    caution_until = self.risk_management.get('market_event_caution_until', 0)
                    if time.time() > caution_until:
                        self.risk_management['market_event_caution'] = False
                        self.logger.info("Market event caution period expired")
                
                # Wait for next check cycle (30 minutes)
                self.logger.info("Risk management complete, next check in 30 minutes")
                await asyncio.sleep(1800)
                
            except Exception as e:
                self.logger.error(f"Error in risk management task: {str(e)}")
                await asyncio.sleep(900)  # 15 minutes on error
    
    async def _reduce_portfolio_risk(self):
        """Reduce portfolio risk by closing or reducing riskier positions."""
        try:
            if not self.positions:
                return
                
            # Calculate risk per position
            position_risks = []
            for symbol, position in self.positions.items():
                # Get current price
                current_price = await self.get_latest_price(symbol)
                if not current_price:
                    continue
                
                # Get stop loss
                stop_loss = position.get('stop_loss')
                if not stop_loss or stop_loss <= 0:
                    stop_loss = position['entry_price'] * (1 - self.stop_loss_pct)
                
                # Calculate risk amount
                value = position['volume'] * current_price
                at_risk = value * ((current_price - stop_loss) / current_price) if current_price > stop_loss else 0
                
                # Calculate unrealized P&L
                entry_price = position['entry_price']
                unrealized_pnl = (current_price - entry_price) / entry_price * 100
                
                # Calculate position age in hours
                age_hours = (datetime.now() - position['entry_time']).total_seconds() / 3600
                
                position_risks.append({
                    'symbol': symbol,
                    'value': value,
                    'at_risk': at_risk,
                    'unrealized_pnl': unrealized_pnl,
                    'age_hours': age_hours
                })
            
            if not position_risks:
                return
                
            # Sort positions by risk criteria
            # Priority: 1. Loss positions, 2. High risk positions, 3. Oldest positions
            position_risks.sort(key=lambda x: (
                x['unrealized_pnl'] < 0,  # First, loss positions
                -x['unrealized_pnl'],     # Then by highest loss first
                x['at_risk'],             # Then by amount at risk
                x['age_hours']            # Then by age
            ), reverse=True)
            
            # Close or reduce highest risk positions
            positions_modified = 0
            
            for position_risk in position_risks:
                symbol = position_risk['symbol']
                
                # Skip if already at safe levels
                if self.risk_management['portfolio_at_risk'] <= self.risk_management['max_portfolio_risk']:
                    break
                
                # For positions in loss, completely close
                if position_risk['unrealized_pnl'] < -3:  # More than 3% loss
                    self.logger.warning(f"Closing losing position {symbol} to reduce portfolio risk")
                    
                    # Close position
                    await self.close_position(symbol, await self.get_latest_price(symbol), 'risk_management')
                    positions_modified += 1
                    
                # For positions in profit, take partial profits
                elif position_risk['unrealized_pnl'] > 1:  # In profit
                    self.logger.info(f"Taking partial profits on {symbol} to reduce portfolio risk")
                    
                    # Close 50% of position
                    await self.close_partial_position(symbol, await self.get_latest_price(symbol), 0.5, 'risk_management')
                    positions_modified += 1
                
                # Limit operations to avoid excess trading
                if positions_modified >= 3:
                    break
            
            # Recalculate risk after adjustments
            if positions_modified > 0:
                self.risk_management['portfolio_at_risk'] = self._calculate_portfolio_at_risk()
                self.logger.info(f"Portfolio risk reduced to {self.risk_management['portfolio_at_risk']*100:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Error reducing portfolio risk: {str(e)}")
    
    async def _check_correlation_exposure(self):
        """Check and manage exposure to correlated assets."""
        try:
            # Skip if no correlation data
            if not self.risk_management.get('correlation_matrix'):
                return
            
            # Calculate exposure by correlation group
            if not self.positions:
                return
                
            # Get total equity
            total_equity = self.calculate_total_equity()
            
            # Initialize group exposures
            group_exposures = {}
            for group in self.correlation_groups:
                group_exposures[group] = 0.0
            
            # Calculate exposures
            for symbol, position in self.positions.items():
                # Get current price
                current_price = await self.get_latest_price(symbol)
                if not current_price:
                    continue
                
                # Calculate position value
                position_value = position['volume'] * current_price
                position_pct = position_value / total_equity if total_equity > 0 else 0
                
                # Add to each group the position belongs to
                for group, symbols in self.correlation_groups.items():
                    if symbol in symbols:
                        group_exposures[group] += position_pct
            
            # Check for excessive group exposure
            for group, exposure in group_exposures.items():
                if exposure > self.risk_management['max_correlated_allocation']:
                    self.logger.warning(f"Excessive exposure to correlation group '{group}': {exposure*100:.1f}% > {self.risk_management['max_correlated_allocation']*100:.1f}%")
                    
                    # Reduce exposure if needed
                    if exposure > self.risk_management['max_correlated_allocation'] * 1.2:  # 20% over limit
                        await self._reduce_group_exposure(group)
            
        except Exception as e:
            self.logger.error(f"Error checking correlation exposure: {str(e)}")
    
    async def _reduce_group_exposure(self, group):
        """Reduce exposure to a correlated asset group."""
        try:
            # Get symbols in the group
            symbols = self.correlation_groups.get(group, [])
            if not symbols:
                return
                
            # Find positions in this group
            group_positions = []
            for symbol in symbols:
                if symbol in self.positions:
                    # Get current price
                    current_price = await self.get_latest_price(symbol)
                    if not current_price:
                        continue
                    
                    # Calculate unrealized P&L
                    position = self.positions[symbol]
                    entry_price = position['entry_price']
                    unrealized_pnl = (current_price - entry_price) / entry_price * 100
                    
                    # Calculate position value
                    position_value = position['volume'] * current_price
                    
                    group_positions.append({
                        'symbol': symbol,
                        'unrealized_pnl': unrealized_pnl,
                        'value': position_value,
                        'entry_time': position['entry_time']
                    })
            
            if not group_positions:
                return
                
            # Sort positions (prioritize taking profits on profitable positions)
            group_positions.sort(key=lambda x: (-x['unrealized_pnl'], -(datetime.now() - x['entry_time']).total_seconds()))
            
            # Reduce exposure (take profits first, then close old positions if needed)
            positions_modified = 0
            for position in group_positions:
                symbol = position['symbol']
                
                # For profitable positions, take partial profits
                if position['unrealized_pnl'] > 1:
                    self.logger.info(f"Taking partial profits on {symbol} to reduce correlation exposure")
                    
                    # Close 50% of position
                    await self.close_partial_position(symbol, await self.get_latest_price(symbol), 0.5, 'correlation_management')
                    positions_modified += 1
                    
                # For loss positions that are old, consider closing
                elif (datetime.now() - position['entry_time']).total_seconds() > 86400:  # Older than 24 hours
                    self.logger.warning(f"Closing old position {symbol} to reduce correlation exposure")
                    
                    # Close position
                    await self.close_position(symbol, await self.get_latest_price(symbol), 'correlation_management')
                    positions_modified += 1
                
                # Limit operations
                if positions_modified >= 2:
                    break
            
        except Exception as e:
            self.logger.error(f"Error reducing group exposure: {str(e)}")
    
    async def _optimize_open_positions(self):
        """Optimize management of open positions based on current conditions."""
        try:
            # Skip if no positions
            if not self.positions:
                return
            
            # Get global market conditions
            global_regime = self.market_regimes.get('global', 'unknown')
            
            for symbol, position in self.positions.items():
                # Get current price
                current_price = await self.get_latest_price(symbol)
                if not current_price:
                    continue
                
                # Calculate unrealized P&L
                entry_price = position['entry_price']
                unrealized_pnl_pct = (current_price - entry_price) / entry_price * 100
                
                # Calculate position age in hours
                age_hours = (datetime.now() - position['entry_time']).total_seconds() / 3600
                
                # Check tier execution flags
                first_tier_executed = position.get('first_tier_executed', False)
                second_tier_executed = position.get('second_tier_executed', False)
                third_tier_executed = position.get('third_tier_executed', False)
                
                # Get symbol-specific regime if available
                symbol_regime = (self.market_regimes.get('symbols', {}).get(symbol, {}).get('regime', global_regime) 
                               if hasattr(self.market_regimes, 'symbols') else global_regime)
                
                # In strong bull markets, let profits run longer
                if symbol_regime in ['bull_trend', 'breakout']:
                    # Add third profit tier if in strong bull market and not already executed
                    if unrealized_pnl_pct >= 4.5 and not third_tier_executed:
                        self.logger.info(f"Third profit target for {symbol} in bull market: +4.5%, taking 25% profits")
                        success = await self.close_partial_position(symbol, current_price, 0.25, 'third_tier_profit')
                        if success:
                            position['third_tier_executed'] = True
                
                # In bear markets, take profits faster
                elif symbol_regime in ['bear_trend', 'breakdown']:
                    # Take any profit above 1.5% in bear markets for older positions
                    if unrealized_pnl_pct >= 1.5 and age_hours > 12 and not first_tier_executed:
                        self.logger.info(f"Taking early profits for {symbol} in bear market: +{unrealized_pnl_pct:.1f}%, taking 50% profits")
                        success = await self.close_partial_position(symbol, current_price, 0.5, 'bear_market_profit')
                        if success:
                            position['first_tier_executed'] = True
                
                # For aged positions not showing good progress
                if age_hours > 24 and abs(unrealized_pnl_pct) < 1.0:
                    # Cut positions going nowhere after 24 hours
                    self.logger.info(f"Closing stagnant position {symbol} after {age_hours:.1f} hours with minimal movement ({unrealized_pnl_pct:.2f}%)")
                    await self.close_position(symbol, current_price, 'time_exit')
                    
                # Tighten stops for profitable positions as they age
                elif unrealized_pnl_pct > 2.0 and age_hours > 12:
                    # Calculate current trailing stop
                    trailing_stop_pct = position.get('trailing_stop_pct', self.trailing_stop_pct)
                    
                    # Tighten by 20% for aged positions
                    new_trailing_stop_pct = trailing_stop_pct * 0.8
                    
                    if new_trailing_stop_pct != trailing_stop_pct:
                        position['trailing_stop_pct'] = new_trailing_stop_pct
                        self.logger.info(f"Tightened trailing stop for {symbol}: {trailing_stop_pct:.3f} -> {new_trailing_stop_pct:.3f}")
                
                # Apply tiered take-profit if not already done
                # First tier - take small partial profits at 1.5% gain
                if unrealized_pnl_pct >= 1.5 and not first_tier_executed:
                    # Take 30% off at first profit target
                    self.logger.info(f"First profit target reached for {symbol}: +1.5%, taking 30% profits")
                    success = await self.close_partial_position(symbol, current_price, 0.3, 'first_tier_profit')
                    if success:
                        position['first_tier_executed'] = True
                
                # Second tier - take more profits at 3.0% gain
                if unrealized_pnl_pct >= 3.0 and not second_tier_executed and first_tier_executed:
                    # Take another 40% off (leaves 30% to run)
                    self.logger.info(f"Second profit target reached for {symbol}: +3.0%, taking 40% profits")
                    success = await self.close_partial_position(symbol, current_price, 0.4, 'second_tier_profit')
                    if success:
                        position['second_tier_executed'] = True
                        
                        # Widen trailing stop for remainder to let it run
                        position['trailing_stop_pct'] = position.get('trailing_stop_pct', self.trailing_stop_pct) * 1.5
                        
            # Save state after position management
            self.save_state()
                
        except Exception as e:
            self.logger.error(f"Error optimizing open positions: {str(e)}")
    
    async def _data_cleanup_task(self):
        """Periodically clean up old data to manage storage."""
        while self.is_running:
            try:
                if self.is_backtesting:
                    await asyncio.sleep(60)
                    continue
                
                # Wait before first cleanup
                await asyncio.sleep(3600)  # 1 hour
                
                self.logger.info("Starting data cleanup")
                
                # Clean database
                self._cleanup_database()
                
                # Trim in-memory data
                if len(self.portfolio_history) > 1000:
                    self.portfolio_history = self.portfolio_history[-1000:]
                
                if len(self.trade_history) > 500:
                    self.trade_history = self.trade_history[-500:]
                
                # Trim pattern history
                if hasattr(self, 'pattern_recognition_system') and 'pattern_history' in self.pattern_recognition_system:
                    pattern_history = self.pattern_recognition_system['pattern_history']
                    for symbol in pattern_history:
                        if len(pattern_history[symbol]) > 100:
                            pattern_history[symbol] = pattern_history[symbol][-100:]
                
                # Trim market regime history
                if hasattr(self, 'market_regimes') and 'regime_history' in self.market_regimes:
                    if len(self.market_regimes['regime_history']) > 20:
                        self.market_regimes['regime_history'] = self.market_regimes['regime_history'][-20:]
                
                # Clear caches
                self._clear_old_cache_entries()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                self.logger.info("Data cleanup completed")
                
                # Wait for next cleanup (3 hours)
                await asyncio.sleep(10800)
                
            except Exception as e:
                self.logger.error(f"Error in data cleanup task: {str(e)}")
                await asyncio.sleep(3600)
    
    def _cleanup_database(self):
        """Clean up old data from database."""
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            
            # Keep only last 7 days of market data
            c.execute("DELETE FROM market_data WHERE timestamp < datetime('now', '-7 days')")
            
            # Keep only last 7 days of sentiment data
            c.execute("DELETE FROM sentiment_data WHERE timestamp < datetime('now', '-7 days')")
            
            # Keep only last 30 days of market regimes
            c.execute("DELETE FROM market_regimes WHERE timestamp < datetime('now', '-30 days')")
            
            # Limit portfolio history
            c.execute('''
                DELETE FROM portfolio_history 
                WHERE timestamp NOT IN (
                    SELECT timestamp FROM portfolio_history
                    ORDER BY timestamp DESC
                    LIMIT 10000
                )
            ''')
            
            # Limit trade history but keep all profitable trades
            c.execute('''
                DELETE FROM trade_history
                WHERE id NOT IN (
                    SELECT id FROM trade_history
                    WHERE profit_loss > 0
                    UNION
                    SELECT id FROM trade_history
                    ORDER BY timestamp DESC
                    LIMIT 1000
                )
            ''')
            
            # Limit pattern detection history
            c.execute('''
                DELETE FROM detected_patterns
                WHERE id NOT IN (
                    SELECT id FROM detected_patterns
                    ORDER BY timestamp DESC
                    LIMIT 1000
                )
            ''')
            
            # Optimize database
            c.execute("VACUUM")
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up database: {str(e)}")
    
    def _clear_old_cache_entries(self):
        """Clear old cache entries."""
        try:
            # Clear price cache entries older than 5 minutes
            current_time = time.time()
            cache_timeout = 300  # 5 minutes
            
            # Clean price cache
            if hasattr(self, 'price_cache'):
                keys_to_remove = []
                for key, entry in self.price_cache.items():
                    if current_time - entry.get('timestamp', 0) > cache_timeout:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self.price_cache[key]
                
                self.logger.debug(f"Cleared {len(keys_to_remove)} old price cache entries")
            
            # Clean data cache
            if hasattr(self, 'data_cache'):
                keys_to_remove = []
                data_cache_timeout = 3600  # 1 hour
                
                for key, entry in self.data_cache.items():
                    if current_time - entry.get('timestamp', 0) > data_cache_timeout:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self.data_cache[key]
                    
                self.logger.debug(f"Cleared {len(keys_to_remove)} old data cache entries")
            
            # Save reduced caches to disk occasionally
            if random.random() < 0.2:  # 20% chance
                self._save_disk_cache()
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
    
    def save_state(self):
        """Save current trading state to database with improved balance handling."""
        try:
            conn = sqlite3.connect(self.db_name)
            conn.execute("BEGIN TRANSACTION")
            c = conn.cursor()
            
            # Save balance - CRITICAL for tracking profit/loss
            c.execute('DELETE FROM balance')
            for currency, amount in self.balance.items():
                c.execute('INSERT INTO balance VALUES (?, ?)', (currency, amount))
                self.logger.debug(f"Saved balance for {currency}: {amount:.2f}")
            
            # Save positions
            c.execute('DELETE FROM positions')
            for symbol, pos in self.positions.items():
                c.execute('''
                    INSERT INTO positions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    pos['volume'],
                    pos['entry_price'],
                    pos['entry_time'].isoformat(),
                    pos.get('high_price', pos['entry_price']),
                    pos.get('stop_loss'),
                    pos.get('take_profit'),
                    pos.get('trailing_stop_pct', self.trailing_stop_pct),
                    pos.get('first_tier_executed', False),
                    pos.get('second_tier_executed', False),
                    pos.get('third_tier_executed', False)
                ))
            
            # Save trade history - don't delete, just add new trades
            for trade in self.trade_history:
                # Skip trades that are already in the database
                if 'id' in trade and trade['id'] is not None:
                    continue
                    
                c.execute('''
                    INSERT INTO trade_history 
                    (timestamp, symbol, type, price, quantity, value, 
                    balance_after, profit_loss, pnl_percentage, 
                    signal_confidence, market_cycle, signals_used, exit_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade['timestamp'].isoformat(),
                    trade.get('symbol', 'unknown'),
                    trade.get('type', 'unknown'),
                    trade.get('price', 0),
                    trade.get('quantity', 0),
                    trade.get('value', 0),
                    trade.get('balance_after', 0),
                    trade.get('profit_loss', 0),
                    trade.get('pnl_percentage', 0),
                    trade.get('signal_confidence', 0.5),
                    trade.get('market_cycle', 'unknown'),
                    trade.get('signals_used', ''),
                    trade.get('exit_reason')
                ))
                
                # Get the last inserted ID
                c.execute("SELECT last_insert_rowid()")
                trade['id'] = c.fetchone()[0]
            
            # Save current portfolio state if we have new data
            if self.portfolio_history:
                latest_portfolio = self.portfolio_history[-1]
                c.execute('''
                    INSERT OR REPLACE INTO portfolio_history (timestamp, balance, equity, drawdown, portfolio_risk)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    latest_portfolio['timestamp'].isoformat(),
                    latest_portfolio['balance'],
                    latest_portfolio['equity'],
                    latest_portfolio.get('drawdown', 0),
                    latest_portfolio.get('portfolio_risk', 0)
                ))
            
            conn.commit()
            conn.close()
            
            total_equity = self.calculate_total_equity()
            self.logger.debug(f"Trading state saved - Balance: ${self.balance.get('ZUSD', 0):.2f}, Total Equity: ${total_equity:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving trading state: {str(e)}")
            traceback.print_exc()
            
            # Try to rollback
            try:
                if conn:
                    conn.rollback()
                    conn.close()
            except:
                pass
                
            return False

    def _load_csv_data(self, csv_path):
        """Helper method to properly load CSV files."""
        try:
            # Read CSV file without trying to parse dates automatically
            df = pd.read_csv(csv_path)
            
            # Convert timestamp column to datetime manually if it exists
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df.set_index('timestamp', inplace=True)
            
            # Convert columns to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by index
            df = df.sort_index()
            
            return df
        except Exception as e:
            self.logger.error(f"Error reading CSV: {str(e)}")
            return None

    async def get_historical_data(self, symbol, interval=None, lookback_days=7):
        """Get historical price data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'XBTUSD')
            interval: Time interval in minutes (default: use self.timeframe)
            lookback_days: Days of historical data to retrieve
            
        Returns:
            DataFrame with historical data or None if not available
        """
        try:
            # For backtesting, check if dates are specified
            if hasattr(self, 'is_backtesting') and self.is_backtesting:
                if hasattr(self, 'backtest_start_date') and hasattr(self, 'backtest_end_date'):
                    # Calculate a longer lookback period to ensure we have enough data for indicators
                    backtest_start = datetime.strptime(self.backtest_start_date, '%Y-%m-%d')
                    lookback_start = backtest_start - timedelta(days=30)  # Extra data for indicators
                    
                    # Calculate actual lookback days
                    lookback_days = (datetime.now() - lookback_start).days
                    self.logger.info(f"Using {lookback_days} days lookback for {symbol} to cover backtest period")
            
            # Check cache first (but bypass for backtesting to ensure fresh data)
            cache_key = f"{symbol}_{lookback_days}_{interval or self.timeframe}"
            if not hasattr(self, 'is_backtesting') or not self.is_backtesting:
                if cache_key in self.data_cache:
                    cache_entry = self.data_cache[cache_key]
                    cache_age = time.time() - cache_entry.get('timestamp', 0)
                    
                    # Use cache if it's less than 30 minutes old
                    if cache_age < 1800:
                        self.logger.info(f"Using cached data for {symbol}")
                        return cache_entry.get('data')
            
            # Try loading from CSV first
            csv_path = f"cache/historical/{symbol}_{interval or self.timeframe}.csv"
            if os.path.exists(csv_path):
                self.logger.info(f"Loading {symbol} data from CSV: {csv_path}")
                df = self._load_csv_data(csv_path)
                
                if df is not None and len(df) > 100:
                    self.logger.info(f"Successfully loaded {len(df)} rows for {symbol} from CSV")
                    
                    # Filter for backtest date range if needed
                    if hasattr(self, 'is_backtesting') and self.is_backtesting:
                        if hasattr(self, 'backtest_start_date') and hasattr(self, 'backtest_end_date'):
                            start_date = datetime.strptime(self.backtest_start_date, '%Y-%m-%d')
                            # Add one day to end_date to include the full end day
                            end_date = datetime.strptime(self.backtest_end_date, '%Y-%m-%d') + timedelta(days=1)
                            
                            # Log what we're filtering
                            self.logger.info(f"Filtering data from {start_date} to {end_date}")
                            
                            # First add data from before start_date for indicators
                            start_with_buffer = start_date - timedelta(days=30)
                            
                            if isinstance(df.index, pd.DatetimeIndex):
                                mask = (df.index >= start_with_buffer) & (df.index < end_date)
                                filtered_df = df.loc[mask]
                                
                                if not filtered_df.empty:
                                    self.logger.info(f"Found {len(filtered_df)} rows within filtered range")
                                    df = filtered_df
                                else:
                                    self.logger.warning(f"No data found in the specified date range")
                                    return None
                            else:
                                self.logger.warning("Index is not DatetimeIndex, can't filter by date")
                    
                    # Cache the data
                    self.data_cache[cache_key] = {
                        'data': df,
                        'timestamp': time.time()
                    }
                    
                    return df
            
            # Sleep to respect rate limits
            await self._respect_rate_limit(f"ohlc_{symbol}")
            
            # Calculate unix timestamp for lookback
            since_timestamp = int(time.time() - (lookback_days * 24 * 60 * 60))
            
            # Make API call with retry logic
            max_retries = 3
            ohlc = None
            
            for retry in range(max_retries):
                try:
                    self.logger.info(f"Fetching OHLC data for {symbol} from Kraken API (attempt {retry+1}/{max_retries})")
                    ohlc, last = self.kraken.get_ohlc_data(
                        symbol, 
                        interval=interval or self.timeframe,
                        since=since_timestamp
                    )
                    
                    # Handle 'T' column rename if present 
                    if 'T' in ohlc.columns:
                        ohlc.rename(columns={'T': 'mT'}, inplace=True)
                    
                    self.logger.info(f"API returned {len(ohlc)} rows for {symbol}")
                    
                    # Success, reset failure counter
                    if f"ohlc_{symbol}" in self.rate_limit_failures:
                        self.rate_limit_failures[f"ohlc_{symbol}"] = 0
                        
                    break  # Success, exit retry loop
                except Exception as api_err:
                    self.logger.warning(f"API error getting OHLC data for {symbol} (attempt {retry+1}): {str(api_err)}")
                    
                    # Increment failure counter
                    if f"ohlc_{symbol}" not in self.rate_limit_failures:
                        self.rate_limit_failures[f"ohlc_{symbol}"] = 0
                    self.rate_limit_failures[f"ohlc_{symbol}"] += 1
                    
                    if retry < max_retries - 1:
                        # Exponential backoff
                        wait_time = (2 ** retry) * 2
                        self.logger.info(f"Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        # All retries failed
                        self.logger.error(f"All retries failed for {symbol}")
                        
                        # Try to use cached data even if expired
                        if cache_key in self.data_cache:
                            self.logger.info(f"Using expired cached data for {symbol}")
                            return self.data_cache[cache_key].get('data')
                            
                        return None
            
            # Process data
            if ohlc is not None and not ohlc.empty:
                # Ensure index is DatetimeIndex
                if not isinstance(ohlc.index, pd.DatetimeIndex):
                    try:
                        # First save original index if we need to recover
                        original_index = ohlc.index.copy()
                        
                        # Try to convert to datetime index
                        ohlc.index = pd.to_datetime(ohlc.index)
                    except Exception as datetime_err:
                        self.logger.warning(f"Error converting index to datetime: {str(datetime_err)}")
                        # Recover original index if conversion fails
                        ohlc.index = original_index
                
                # Convert columns to float
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in ohlc.columns:
                        ohlc[col] = pd.to_numeric(ohlc[col], errors='coerce')
                
                # Remove NaN values
                ohlc = ohlc.dropna(subset=['close', 'volume'])
                
                # Ensure positive values
                ohlc = ohlc[ohlc['close'] > 0]
                ohlc = ohlc[ohlc['volume'] > 0]
                
                # Sort index to ensure chronological order
                ohlc = ohlc.sort_index()
                
                # For backtesting, filter to the specific date range
                if hasattr(self, 'is_backtesting') and self.is_backtesting:
                    if hasattr(self, 'backtest_start_date') and hasattr(self, 'backtest_end_date'):
                        start_date = datetime.strptime(self.backtest_start_date, '%Y-%m-%d')
                        # Add one day to end_date to include the full end day
                        end_date = datetime.strptime(self.backtest_end_date, '%Y-%m-%d') + timedelta(days=1)
                        
                        # Add buffer period for indicators
                        start_with_buffer = start_date - timedelta(days=30)
                        
                        # Log what we're filtering
                        self.logger.info(f"Filtering data from {start_with_buffer} to {end_date}")
                        
                        if isinstance(ohlc.index, pd.DatetimeIndex):
                            mask = (ohlc.index >= start_with_buffer) & (ohlc.index < end_date)
                            filtered_df = ohlc.loc[mask]
                            
                            if not filtered_df.empty:
                                self.logger.info(f"Found {len(filtered_df)} rows within filtered range")
                                ohlc = filtered_df
                            else:
                                self.logger.warning(f"No data found in the specified date range")
                                return None
                
                # Save to CSV for future use
                try:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                    
                    # Create a copy with timestamp as a column for saving
                    df_to_save = ohlc.reset_index()
                    df_to_save.rename(columns={'index': 'timestamp'}, inplace=True)
                    
                    # Save to CSV
                    df_to_save.to_csv(csv_path, index=False)
                    self.logger.info(f"Saved {len(ohlc)} rows to {csv_path}")
                except Exception as save_err:
                    self.logger.error(f"Error saving CSV data: {str(save_err)}")
                
                # Cache data (except for backtesting)
                if not hasattr(self, 'is_backtesting') or not self.is_backtesting:
                    self.data_cache[cache_key] = {
                        'data': ohlc,
                        'timestamp': time.time()
                    }
                
                self.logger.info(f"Successfully processed {len(ohlc)} rows of data for {symbol}")
                return ohlc
            
            self.logger.warning(f"No data returned for {symbol}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {str(e)}")
            traceback.print_exc()
            return None

    async def cleanup(self):
        """Properly clean up background tasks and resources, saving state."""
        try:
            self.logger.info("Cleaning up resources...")
            
            # Set running flag to false to stop background tasks
            self.is_running = False
            
            # Wait for tasks to finish
            if hasattr(self, 'tasks'):
                for task in self.tasks:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        except Exception as e:
                            self.logger.error(f"Error cancelling task: {str(e)}")
            
            # Close Reddit analyzer session
            if hasattr(self, 'reddit_analyzer') and self.reddit_analyzer is not None:
                if hasattr(self.reddit_analyzer, '_close_session') and callable(self.reddit_analyzer._close_session):
                    try:
                        await self.reddit_analyzer._close_session()
                        self.logger.info("Closed Reddit analyzer session")
                    except Exception as e:
                        self.logger.error(f"Error closing Reddit session: {str(e)}")
            
            # Close Twitter analyzer session
            if hasattr(self, 'twitter_analyzer') and self.twitter_analyzer is not None:
                if hasattr(self.twitter_analyzer, 'client') and self.twitter_analyzer.client is not None:
                    if hasattr(self.twitter_analyzer.client, 'close') and callable(self.twitter_analyzer.client.close):
                        try:
                            self.twitter_analyzer.client.close()
                            self.logger.info("Closed Twitter analyzer session")
                        except Exception as e:
                            self.logger.error(f"Error closing Twitter session: {str(e)}")
            
            # Close news analyzer session
            if hasattr(self, 'news_analyzer') and self.news_analyzer is not None:
                if hasattr(self.news_analyzer, 'session') and self.news_analyzer.session is not None:
                    if hasattr(self.news_analyzer.session, 'close') and callable(self.news_analyzer.session.close):
                        try:
                            if asyncio.iscoroutinefunction(self.news_analyzer.session.close):
                                await self.news_analyzer.session.close()
                            else:
                                self.news_analyzer.session.close()
                            self.logger.info("Closed news analyzer session")
                        except Exception as e:
                            self.logger.error(f"Error closing news session: {str(e)}")
            
            # Close any other aiohttp sessions
            import sys
            import gc
            from aiohttp import ClientSession
            
            # Find and close any other client sessions that might exist
            for obj in gc.get_objects():
                if isinstance(obj, ClientSession) and not obj.closed:
                    try:
                        if not obj._closed:
                            await obj.close()
                            self.logger.info("Closed orphaned client session")
                    except Exception as e:
                        self.logger.error(f"Error closing orphaned session: {str(e)}")
            
            # Calculate final equity for logging
            total_equity = self.calculate_total_equity()
            self.logger.info(f"Final equity before shutdown: ${total_equity:.2f}")
            
            # Save state before exiting
            self.save_state()
            
            # Save disk cache
            self._save_disk_cache()
            
            self.logger.info("Cleanup completed successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            traceback.print_exc()

    async def download_historical_data(self, symbol, start_date, end_date, interval=None):
        """
        Download historical data for a specific period and save to CSV.
        
        Args:
            symbol: Trading symbol (e.g., 'XBTUSD')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Time interval in minutes (default: use self.timeframe)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Downloading historical data for {symbol} from {start_date} to {end_date}")
            
            # Parse dates
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Calculate lookback days
            days = (datetime.now() - start_dt).days + 30  # extra buffer
            
            # Get data
            df = await self.get_historical_data(symbol, interval=interval or self.timeframe, lookback_days=days)
            
            if df is None or len(df) < 10:
                self.logger.error(f"Failed to download data for {symbol}")
                return False
            
            # Save to CSV
            csv_path = f"cache/historical/{symbol}_{interval or self.timeframe}.csv"
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            
            # Create a copy with timestamp as a column for saving
            df_to_save = df.reset_index()
            df_to_save.rename(columns={'index': 'timestamp'}, inplace=True)
            
            # Save to CSV
            df_to_save.to_csv(csv_path, index=False)
            self.logger.info(f"Saved {len(df)} rows to {csv_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading historical data: {str(e)}")
            traceback.print_exc()
            return False

    async def _respect_rate_limit(self, endpoint):
        """Respect API rate limits by waiting if needed with adaptive backoff."""
        current_time = time.time()
        
        # Check if we need to wait
        if endpoint in self.rate_limit_timestamps:
            last_request_time = self.rate_limit_timestamps[endpoint]
            time_since_last = current_time - last_request_time
            
            # Use adaptive wait time based on recent failures
            wait_time = self.min_request_interval
            if hasattr(self, 'rate_limit_failures') and endpoint in self.rate_limit_failures:
                failures = self.rate_limit_failures[endpoint]
                if failures > 0:
                    # Exponential backoff: double wait time for each recent failure (up to max)
                    wait_time = min(self.max_retry_delay, wait_time * (2 ** failures))
                    # Add a small random jitter to avoid synchronized requests
                    wait_time += random.uniform(0.1, 0.5)
                    self.logger.debug(f"Using adaptive wait time for {endpoint}: {wait_time:.1f}s (failures: {failures})")
            
            # Enforce a minimum wait time for all requests (higher than default)
            min_wait = 1.2  # Increase from 1.0 to 1.2 seconds
            
            # For certain high-volume endpoints, use even higher minimum
            if endpoint.startswith('ohlc_'):
                min_wait = 2.0  # 2 seconds for OHLC data requests
            
            wait_time = max(wait_time, min_wait)
            
            if time_since_last < wait_time:
                actual_wait = wait_time - time_since_last
                self.logger.debug(f"Rate limiting {endpoint}: waiting {actual_wait:.1f}s")
                await asyncio.sleep(actual_wait)
        
        # Update timestamp
        self.rate_limit_timestamps[endpoint] = time.time()
    
    def _initialize_api(self):
        """Initialize connection to trading API."""
        try:
            # Initialize Kraken API
            self.api = krakenex.API()
            self.kraken = KrakenAPI(self.api, retry=0.5)
            
            # Set up API credentials if available
            api_key = os.environ.get('KRAKEN_API_KEY')
            api_secret = os.environ.get('KRAKEN_PRIVATE_KEY')
            
            if api_key and api_secret:
                self.api.key = api_key
                self.api.secret = api_secret
                self.logger.info("API credentials loaded")
                
                # Test authenticated connection
                try:
                    balance = self.api.query_private('Balance')
                    if 'error' in balance and balance['error']:
                        self.logger.warning(f"API authentication error: {balance['error']}")
                    else:
                        self.logger.info("API authentication successful")
                except Exception as auth_err:
                    self.logger.warning(f"API authentication error: {str(auth_err)}")
            else:
                self.logger.info("No API credentials found - running in public-only mode")
            
            # Set API parameters
            self.timeframe = self.config.get('api', {}).get('timeframe', 5)
            self.api_retry_delay = self.config.get('api', {}).get('retry_delay', 1.0)
            self.max_retry_delay = self.config.get('api', {}).get('max_retry_delay', 60)
            
            # Initialize rate limiting with failure tracking
            self.rate_limit_timestamps = {}
            self.rate_limit_failures = {}  # Track recent failures for adaptive backoff
            self.min_request_interval = 1.0  # seconds
            
        except Exception as e:
            self.logger.error(f"Error initializing API: {str(e)}")
            traceback.print_exc()

    async def get_latest_price(self, symbol):
        """Get the latest price for a symbol with improved error handling."""
        try:
            # Check cache first
            cache_key = f"price_{symbol}"
            if cache_key in self.price_cache:
                cache_entry = self.price_cache[cache_key]
                cache_age = time.time() - cache_entry.get('timestamp', 0)
                
                # Use cache if it's less than 30 seconds old (reduced from 10s)
                if cache_age < 30:
                    return cache_entry.get('price')
            
            # Respect rate limit with longer wait times
            await self._respect_rate_limit(f"ticker_{symbol}")
            
            # Try Ticker API with better error handling
            try:
                ticker = self.api.query_public('Ticker', {'pair': symbol})
                if ticker and 'result' in ticker and symbol in ticker['result']:
                    price = float(ticker['result'][symbol]['c'][0])
                    
                    if price > 0:
                        # Cache the price
                        self.price_cache[cache_key] = {
                            'price': price,
                            'timestamp': time.time()
                        }
                        self.logger.debug(f"Got price for {symbol}: ${price:.2f}")
                        return price
                else:
                    self.logger.warning(f"Invalid API response for {symbol}: {ticker}")
            except Exception as e:
                self.logger.debug(f"Ticker API error for {symbol}: {str(e)}")
            
            # Add a short delay before fallback method
            await asyncio.sleep(1)
            
            # Fallback: Try OHLC data
            try:
                # Use smaller interval (1) for more current price
                ohlc, _ = self.kraken.get_ohlc_data(symbol, interval=1, count=10)
                if ohlc is not None and not ohlc.empty:
                    price = float(ohlc.iloc[-1]['close'])
                    
                    if price > 0:
                        # Cache the price
                        self.price_cache[cache_key] = {
                            'price': price,
                            'timestamp': time.time()
                        }
                        self.logger.info(f"Got price for {symbol} via OHLC: ${price:.2f}")
                        return price
            except Exception as e:
                self.logger.debug(f"OHLC API error for {symbol}: {str(e)}")
            
            # If all fails, use cached price if available
            if cache_key in self.price_cache:
                price = self.price_cache[cache_key].get('price')
                self.logger.warning(f"Using expired cached price for {symbol}: ${price:.2f}")
                return price
            
            # Last resort: Check if we have a position with an entry price
            if symbol in self.positions:
                price = self.positions[symbol]['entry_price']
                self.logger.warning(f"Using position entry price for {symbol}: ${price:.2f}")
                return price
            
            self.logger.error(f"Unable to get price for {symbol}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting latest price: {str(e)}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators for trading decisions."""
        try:
            df = df.copy()
            
            # Basic price indicators
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log1p(df['returns'])
            df['rolling_std_20'] = df['returns'].rolling(window=20, min_periods=1).std()
            
            # Moving averages
            df['sma_short'] = df['close'].rolling(window=self.sma_short, min_periods=1).mean()
            df['sma_long'] = df['close'].rolling(window=self.sma_long, min_periods=1).mean()
            df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
            df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
            df['sma_20_50_ratio'] = df['sma_20'] / df['sma_50']
            
            # Exponential moving averages
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            
            # Volume indicators
            if 'volume' in df.columns:
                vol_ma = df['volume'].rolling(window=20, min_periods=1).mean()
                df['volume_ma_ratio'] = df['volume'] / vol_ma
                df['volume_std'] = df['volume'].rolling(window=20, min_periods=1).std()
            else:
                df['volume_ma_ratio'] = 1.0
                df['volume_std'] = 0.0
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi'] = df['rsi'].fillna(50)
            df['rsi_divergence'] = df['rsi'].diff()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_diff'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20, min_periods=1).mean()
            bb_std = df['close'].rolling(window=20, min_periods=1).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14, min_periods=1).mean()
            
            # ADX
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff(-1)
            plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
            minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
            tr = true_range
            plus_di = 100 * (plus_dm.rolling(window=14, min_periods=1).mean() / tr.rolling(window=14, min_periods=1).mean())
            minus_di = 100 * (minus_dm.rolling(window=14, min_periods=1).mean() / tr.rolling(window=14, min_periods=1).mean())
            df['adx'] = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).rolling(window=14, min_periods=1).mean()
            df['adx_pos'] = plus_di
            df['adx_neg'] = minus_di
            
            # CCI
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            mean_deviation = abs(typical_price - typical_price.rolling(window=20, min_periods=1).mean())
            df['cci'] = (typical_price - typical_price.rolling(window=20, min_periods=1).mean()) / (0.015 * mean_deviation.rolling(window=20, min_periods=1).mean())
            
            # Momentum indicators
            df['momentum_1d'] = df['close'] / df['close'].shift(1) - 1
            df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
            
            # Stochastic oscillator
            high_14 = df['high'].rolling(window=14, min_periods=1).max()
            low_14 = df['low'].rolling(window=14, min_periods=1).min()
            df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
            df['stoch_d'] = df['stoch_k'].rolling(window=3, min_periods=1).mean()
            
            # Clean up NaN and infinity values
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill().bfill().fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            # Return original dataframe if error
            return df
    
    def detect_market_cycle(self, df):
        """Determine the current market cycle with improved detection of ranging markets."""
        try:
            if df is None or len(df) < 50:
                return "unknown"
            
            # Calculate key metrics
            returns = df['close'].pct_change().dropna()
            volatility = returns.rolling(20).std() * np.sqrt(252)
            recent_volatility = volatility.iloc[-20:].mean() if len(volatility) >= 20 else volatility.mean()
            
            # Trend detection
            sma20 = df['close'].rolling(window=20).mean()
            sma50 = df['close'].rolling(window=50).mean()
            sma200 = df['close'].rolling(window=200).mean() if len(df) >= 200 else None
            
            # Short term momentum
            momentum_short = df['close'].pct_change(5).iloc[-1] if len(df) > 5 else 0
            
            # Medium term momentum
            momentum_medium = df['close'].pct_change(20).iloc[-1] if len(df) > 20 else 0
            
            # Check if price is above major moving averages
            current_price = df['close'].iloc[-1]
            above_sma20 = current_price > sma20.iloc[-1] if len(sma20) > 0 else False
            above_sma50 = current_price > sma50.iloc[-1] if len(sma50) > 0 else False
            above_sma200 = current_price > sma200.iloc[-1] if sma200 is not None and len(sma200) > 0 else False
            
            # Calculate trend strength more accurately
            if len(df) >= 50:
                trend_strength = abs(sma20.iloc[-1] / sma50.iloc[-1] - 1)
            else:
                trend_strength = 0
            
            # RSI trend
            rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
            
            # MACD trend
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                macd = df['macd'].iloc[-1]
                macd_signal = df['macd_signal'].iloc[-1]
                macd_trend = macd > macd_signal
            else:
                macd_trend = None
            
            # Check volume trend
            if 'volume' in df.columns:
                volume_ratio = df['volume'].iloc[-5:].mean() / df['volume'].iloc[-20:-5].mean() if len(df) >= 20 else 1.0
                high_volume = volume_ratio > 1.2
            else:
                high_volume = False
            
            # Improved ranging market detection with multiple sub-types
            if 0.0025 < trend_strength < 0.01 and abs(momentum_medium) < 0.05:
                # Check for price swings within a range
                consecutive_crossovers = 0
                for i in range(-20, -1):
                    if i+1 >= 0:
                        break
                    if (sma20.iloc[i] > sma50.iloc[i] and sma20.iloc[i+1] < sma50.iloc[i+1]) or \
                    (sma20.iloc[i] < sma50.iloc[i] and sma20.iloc[i+1] > sma50.iloc[i+1]):
                        consecutive_crossovers += 1
                
                # More accurate ranging detection
                if consecutive_crossovers >= 2:
                    if recent_volatility > 0.35:
                        return "ranging_volatile"  # High volatility ranging
                    elif trend_strength < 0.005:
                        return "tight_ranging"     # Very tight range
                    else:
                        return "ranging"          # Standard ranging
            
            # Check for strong momentum changes - breakouts/breakdowns
            if momentum_short > 0.05 and momentum_medium > 0.1 and above_sma20 and above_sma50 and rsi > 65:
                if high_volume:
                    return "breakout"
                return "bull_trend"
            
            if momentum_short < -0.05 and momentum_medium < -0.1 and not above_sma20 and not above_sma50 and rsi < 35:
                if high_volume:
                    return "breakdown"
                return "bear_trend"
            
            # Determine market cycle
            if recent_volatility > 0.80:
                return "high_volatility"
            
            # Strong bull trend
            if above_sma20 and above_sma50 and above_sma200 and momentum_short > 0.02 and momentum_medium > 0.05:
                return "bull_trend"
            
            # Strong bear trend
            if not above_sma20 and not above_sma50 and momentum_short < -0.02 and momentum_medium < -0.05:
                return "bear_trend"
            
            # Ranging market with bullish bias
            if above_sma20 and above_sma50 and abs(momentum_short) < 0.01:
                if rsi > 50 and (macd_trend is True or macd_trend is None):
                    return "ranging_bullish"
                else:
                    return "ranging"
            
            # Ranging market with bearish bias
            if not above_sma20 and not above_sma50 and abs(momentum_short) < 0.01:
                if rsi < 50 and (macd_trend is False or macd_trend is None):
                    return "ranging_bearish"
                else:
                    return "ranging"
            
            # Low volatility period
            if recent_volatility < 0.20:
                return "low_volatility"
            
            # Recovery phase
            if above_sma20 and not above_sma50 and momentum_short > 0:
                return "recovery"
            
            # Mixed signals
            return "mixed"
            
        except Exception as e:
            self.logger.error(f"Error detecting market cycle: {str(e)}")
            return "unknown"
    
    def detect_early_reversal(self, df):
        """Detect early signs of market reversal."""
        try:
            if df is None or len(df) < 10:
                return False
            
            # Get recent data
            recent_data = df.tail(10)
            reversal_signals = 0
            
            # 1. RSI oversold with bullish divergence
            if 'rsi' in recent_data.columns:
                rsi_current = recent_data['rsi'].iloc[-1]
                rsi_prev = recent_data['rsi'].iloc[-2]
                
                if rsi_current < 35:  # Oversold
                    reversal_signals += 1
                
                if rsi_current > rsi_prev and rsi_prev < 35:  # Turning up from oversold
                    reversal_signals += 1
            
            # 2. MACD bullish divergence
            if all(col in recent_data.columns for col in ['close', 'macd']):
                if (recent_data['close'].iloc[-1] < recent_data['close'].iloc[-3]) and \
                   (recent_data['macd'].iloc[-1] > recent_data['macd'].iloc[-3]):
                    reversal_signals += 1
            
            # 3. Volume spike
            if 'volume' in recent_data.columns:
                vol_avg = recent_data['volume'].iloc[:-1].mean()
                recent_vol = recent_data['volume'].iloc[-1]
                
                if recent_vol > vol_avg * 1.5:
                    reversal_signals += 1
            
            # 4. Bollinger Band bounce
            if all(col in recent_data.columns for col in ['close', 'bb_lower']):
                lower_band_touch = False
                for i in range(-3, 0):
                    if recent_data['close'].iloc[i] <= recent_data['bb_lower'].iloc[i] * 1.01:
                        lower_band_touch = True
                
                if lower_band_touch and recent_data['close'].iloc[-1] > recent_data['close'].iloc[-2]:
                    reversal_signals += 1
            
            # 5. Candlestick patterns
            if all(col in recent_data.columns for col in ['open', 'close', 'high', 'low']):
                latest = recent_data.iloc[-1]
                body_size = abs(latest['open'] - latest['close'])
                lower_wick = min(latest['open'], latest['close']) - latest['low']
                
                if body_size > 0 and lower_wick > body_size * 2:
                    reversal_signals += 1
            
            # Return True if we have enough signals (increased from 3 to 2)
            return reversal_signals >= 2
            
        except Exception as e:
            self.logger.error(f"Error in reversal detection: {str(e)}")
            return False
    
    # Pattern detection methods
    def _detect_head_and_shoulders(self, df):
        """Detect head and shoulders pattern."""
        try:
            if len(df) < 80:
                return {'detected': False}
                
            # Get recent data for pattern detection
            data = df.tail(60)
            
            # Moving average to smooth price
            ma = data['close'].rolling(window=3).mean()
            
            # Find peaks and troughs
            peaks = []
            troughs = []
            
            for i in range(2, len(ma)-2):
                # Peak: higher than both neighbors
                if ma.iloc[i] > ma.iloc[i-1] and ma.iloc[i] > ma.iloc[i-2] and \
                   ma.iloc[i] > ma.iloc[i+1] and ma.iloc[i] > ma.iloc[i+2]:
                    peaks.append((i, ma.iloc[i]))
                    
                # Trough: lower than both neighbors
                if ma.iloc[i] < ma.iloc[i-1] and ma.iloc[i] < ma.iloc[i-2] and \
                   ma.iloc[i] < ma.iloc[i+1] and ma.iloc[i] < ma.iloc[i+2]:
                    troughs.append((i, ma.iloc[i]))
            
            if len(peaks) < 3 or len(troughs) < 2:
                return {'detected': False}
                
            # Look for head and shoulders pattern (3 peaks with middle one higher)
            for i in range(len(peaks)-2):
                left = peaks[i]
                head = peaks[i+1]
                right = peaks[i+2]
                
                # Check if head is higher than shoulders
                if head[1] > left[1] and head[1] > right[1]:
                    # Check if shoulders are at similar heights (within 10%)
                    shoulder_diff = abs(left[1] - right[1]) / ((left[1] + right[1])/2)
                    if shoulder_diff < 0.1:
                        # Find neckline (trough between shoulders)
                        neckline = None
                        for trough in troughs:
                            if left[0] < trough[0] < right[0]:
                                if neckline is None or trough[1] < neckline[1]:
                                    neckline = trough
                                    
                        if neckline:
                            price = data['close'].iloc[-1]
                            neckline_price = neckline[1]
                            
                            # Check if price is near or below neckline
                            if price <= neckline_price * 1.02:
                                return {
                                    'detected': True,
                                    'pattern': 'head_and_shoulders',
                                    'direction': 'bearish',
                                    'confidence': 0.7,
                                    'price_target': neckline_price * 0.94  # Target 6% below neckline
                                }
            
            return {'detected': False}
            
        except Exception as e:
            self.logger.error(f"Error detecting head and shoulders: {str(e)}")
            return {'detected': False}
    
    def _detect_double_top(self, df):
        """Detect double top pattern."""
        try:
            if len(df) < 40:
                return {'detected': False}
                
            # Get recent data for pattern detection
            data = df.tail(40)
            
            # Moving average to smooth price
            ma = data['close'].rolling(window=3).mean()
            
            # Find peaks
            peaks = []
            
            for i in range(2, len(ma)-2):
                # Peak: higher than both neighbors
                if ma.iloc[i] > ma.iloc[i-1] and ma.iloc[i] > ma.iloc[i-2] and \
                   ma.iloc[i] > ma.iloc[i+1] and ma.iloc[i] > ma.iloc[i+2]:
                    peaks.append((i, ma.iloc[i]))
            
            if len(peaks) < 2:
                return {'detected': False}
                
            # Check last two peaks
            if len(peaks) >= 2:
                peak1 = peaks[-2]
                peak2 = peaks[-1]
                
                # Peaks should be similar height (within 3%)
                peak_diff = abs(peak1[1] - peak2[1]) / peak1[1]
                
                if peak_diff < 0.03:
                    # Check if enough space between peaks
                    if peak2[0] - peak1[0] >= 5:
                        # Check if current price is below peaks
                        current_price = data['close'].iloc[-1]
                        
                        if current_price < min(peak1[1], peak2[1]) * 0.98:
                            # Find trough between peaks
                            trough_idx = None
                            trough_value = float('inf')
                            
                            for j in range(peak1[0], peak2[0]):
                                if ma.iloc[j] < trough_value:
                                    trough_idx = j
                                    trough_value = ma.iloc[j]
                            
                            if trough_idx is not None:
                                # Calculate target (mirror of the pattern height)
                                height = min(peak1[1], peak2[1]) - trough_value
                                target = trough_value - height
                                
                                return {
                                    'detected': True,
                                    'pattern': 'double_top',
                                    'direction': 'bearish',
                                    'confidence': 0.65,
                                    'price_target': target
                                }
            
            return {'detected': False}
            
        except Exception as e:
            self.logger.error(f"Error detecting double top: {str(e)}")
            return {'detected': False}
    
    def _detect_double_bottom(self, df):
        """Detect double bottom pattern."""
        try:
            if len(df) < 40:
                return {'detected': False}
                
            # Get recent data for pattern detection
            data = df.tail(40)
            
            # Moving average to smooth price
            ma = data['close'].rolling(window=3).mean()
            
            # Find troughs
            troughs = []
            
            for i in range(2, len(ma)-2):
                # Trough: lower than both neighbors
                if ma.iloc[i] < ma.iloc[i-1] and ma.iloc[i] < ma.iloc[i-2] and \
                   ma.iloc[i] < ma.iloc[i+1] and ma.iloc[i] < ma.iloc[i+2]:
                    troughs.append((i, ma.iloc[i]))
            
            if len(troughs) < 2:
                return {'detected': False}
                
            # Check last two troughs
            if len(troughs) >= 2:
                trough1 = troughs[-2]
                trough2 = troughs[-1]
                
                # Troughs should be similar height (within 3%)
                trough_diff = abs(trough1[1] - trough2[1]) / trough1[1]
                
                if trough_diff < 0.03:
                    # Check if enough space between troughs
                    if trough2[0] - trough1[0] >= 5:
                        # Check if current price is above troughs
                        current_price = data['close'].iloc[-1]
                        
                        if current_price > max(trough1[1], trough2[1]) * 1.02:
                            # Find peak between troughs
                            peak_idx = None
                            peak_value = -float('inf')
                            
                            for j in range(trough1[0], trough2[0]):
                                if ma.iloc[j] > peak_value:
                                    peak_idx = j
                                    peak_value = ma.iloc[j]
                            
                            if peak_idx is not None:
                                # Calculate target (mirror of the pattern height)
                                height = peak_value - max(trough1[1], trough2[1])
                                target = peak_value + height
                                
                                return {
                                    'detected': True,
                                    'pattern': 'double_bottom',
                                    'direction': 'bullish',
                                    'confidence': 0.65,
                                    'price_target': target
                                }
            
            return {'detected': False}
            
        except Exception as e:
            self.logger.error(f"Error detecting double bottom: {str(e)}")
            return {'detected': False}
    
    def _detect_ascending_triangle(self, df):
        """Detect ascending triangle pattern."""
        # Implementation for ascending triangle detection
        return {'detected': False}
    
    def _detect_descending_triangle(self, df):
        """Detect descending triangle pattern."""
        # Implementation for descending triangle detection
        return {'detected': False}
    
    def _detect_bull_flag(self, df):
        """Detect bull flag pattern."""
        # Implementation for bull flag detection
        return {'detected': False}
    
    def _detect_bear_flag(self, df):
        """Detect bear flag pattern."""
        # Implementation for bear flag detection
        return {'detected': False}
    
    def _detect_engulfing_bullish(self, df):
        """Detect bullish engulfing candlestick pattern."""
        try:
            if len(df) < 10:
                return {'detected': False}
                
            # Get recent data for pattern detection
            data = df.tail(10)
            
            # Need at least 2 candlesticks
            if len(data) < 2:
                return {'detected': False}
                
            # Get last two candlesticks
            prev = data.iloc[-2]
            curr = data.iloc[-1]
            
            # Check for bullish engulfing pattern
            # 1. Previous candlestick is bearish (close < open)
            # 2. Current candlestick is bullish (close > open)
            # 3. Current candlestick completely engulfs previous one
            if prev['close'] < prev['open'] and curr['close'] > curr['open'] and \
               curr['open'] <= prev['close'] and curr['close'] >= prev['open']:
                
                return {
                    'detected': True,
                    'pattern': 'engulfing_bullish',
                    'direction': 'bullish',
                    'confidence': 0.6,
                    'price_target': curr['close'] * 1.02  # Target 2% above current close
                }
            
            return {'detected': False}
            
        except Exception as e:
            self.logger.error(f"Error detecting bullish engulfing: {str(e)}")
            return {'detected': False}
    
    def _detect_engulfing_bearish(self, df):
        """Detect bearish engulfing candlestick pattern."""
        try:
            if len(df) < 10:
                return {'detected': False}
                
            # Get recent data for pattern detection
            data = df.tail(10)
            
            # Need at least 2 candlesticks
            if len(data) < 2:
                return {'detected': False}
                
            # Get last two candlesticks
            prev = data.iloc[-2]
            curr = data.iloc[-1]
            
            # Check for bearish engulfing pattern
            # 1. Previous candlestick is bullish (close > open)
            # 2. Current candlestick is bearish (close < open)
            # 3. Current candlestick completely engulfs previous one
            if prev['close'] > prev['open'] and curr['close'] < curr['open'] and \
               curr['open'] >= prev['close'] and curr['close'] <= prev['open']:
                
                return {
                    'detected': True,
                    'pattern': 'engulfing_bearish',
                    'direction': 'bearish',
                    'confidence': 0.6,
                    'price_target': curr['close'] * 0.98  # Target 2% below current close
                }
            
            return {'detected': False}
            
        except Exception as e:
            self.logger.error(f"Error detecting bearish engulfing: {str(e)}")
            return {'detected': False}
    
    async def _train_models_and_wait(self):
        """Train models synchronously and wait for completion before allowing trading."""
        try:
            self.logger.info("Starting model training...")
            
            # Collect training data
            training_data = await self.collect_training_data(days=60)  # Use last 60 days
            
            if training_data is None or len(training_data) < 1000:
                self.logger.error("Insufficient data for model training. Cannot proceed.")
                return False
            
            # Train ML model
            self.logger.info("Training ML model...")
            ml_success = await self.train_ml_model(training_data)
            if not ml_success:
                self.logger.error("ML model training failed.")
                return False
            
            # Try to train AI model
            self.logger.info("Training AI model...")
            ai_success = False
            
            # Try up to 3 times with different architectures
            for attempt in range(3):
                try:
                    self.logger.info(f"AI model training attempt {attempt+1}/3")
                    ai_success = await self.train_ai_model(training_data)
                    if ai_success:
                        break
                    else:
                        # Try with simpler LSTM-only model on last attempt
                        if attempt == 2:
                            self.logger.info("Fallback to LSTM-only model")
                            # Create only LSTM model and save it
                            from core.models.ai_model import AITradingEnhancer
                            self.ai_enhancer = AITradingEnhancer()
                            X, y = self.ai_enhancer.prepare_sequence_data(training_data)
                            if X is not None and len(X) > 0:
                                lstm_model = self.ai_enhancer.build_lstm_model()
                                lstm_model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
                                lstm_model.save(os.path.join("models", "lstm_model.keras"))
                                # Save feature scaler
                                joblib.dump(self.ai_enhancer.feature_scaler, 
                                        os.path.join("models", "ai_feature_scaler.joblib"))
                                ai_success = True
                except Exception as e:
                    self.logger.error(f"Error in AI training attempt {attempt+1}: {str(e)}")
                    await asyncio.sleep(2)  # Brief pause before retry
            
            if not ai_success:
                self.logger.warning("AI model training failed, but will proceed with ML model only.")
            
            # Set models as loaded if at least ML model is trained
            if ml_success:
                self.is_models_loaded = True
                self.logger.info("Model training completed. ML model ready for use.")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            traceback.print_exc()
            return False

    async def generate_basic_trading_signal(self, symbol, df=None):
        """Generate basic technical trading signal."""
        try:
            # Get historical data if not provided
            if df is None:
                df = await self.get_historical_data(symbol)
                
            if df is None or len(df) < 20:
                return {'action': 'hold', 'confidence': 0.5, 'reason': 'insufficient_data'}
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Get latest values
            latest = df.iloc[-1]
            
            # Initialize confidence scores for each indicator
            indicator_scores = {}
            reasons = []
            
            # 1. RSI
            rsi = latest.get('rsi', 50)
            if rsi < 30:
                indicator_scores['rsi'] = 0.75  # Strong buy
                reasons.append(f"RSI oversold at {rsi:.1f}")
            elif rsi < 40:
                indicator_scores['rsi'] = 0.65  # Moderate buy
                reasons.append(f"RSI approaching oversold at {rsi:.1f}")
            elif rsi > 70:
                indicator_scores['rsi'] = 0.25  # Strong sell
                reasons.append(f"RSI overbought at {rsi:.1f}")
            elif rsi > 60:
                indicator_scores['rsi'] = 0.35  # Moderate sell
                reasons.append(f"RSI approaching overbought at {rsi:.1f}")
            else:
                indicator_scores['rsi'] = 0.5  # Neutral
            
            # 2. MACD
            if 'macd' in latest and 'macd_signal' in latest:
                macd = latest['macd']
                macd_signal = latest['macd_signal']
                macd_diff = macd - macd_signal
                
                # Check for crossovers
                if len(df) > 2:
                    prev_macd = df.iloc[-2]['macd']
                    prev_signal = df.iloc[-2]['macd_signal']
                    prev_diff = prev_macd - prev_signal
                    
                    # Bullish crossover (MACD crosses above signal)
                    if prev_diff <= 0 and macd_diff > 0:
                        indicator_scores['macd'] = 0.75  # Strong buy
                        reasons.append("Bullish MACD crossover")
                    # Bearish crossover (MACD crosses below signal)
                    elif prev_diff >= 0 and macd_diff < 0:
                        indicator_scores['macd'] = 0.25  # Strong sell
                        reasons.append("Bearish MACD crossover")
                    # MACD above signal (bullish)
                    elif macd_diff > 0:
                        indicator_scores['macd'] = 0.60  # Moderate buy
                        reasons.append("MACD above signal line")
                    # MACD below signal (bearish)
                    elif macd_diff < 0:
                        indicator_scores['macd'] = 0.40  # Moderate sell
                        reasons.append("MACD below signal line")
                    else:
                        indicator_scores['macd'] = 0.5  # Neutral
                else:
                    indicator_scores['macd'] = 0.5  # Neutral
            else:
                indicator_scores['macd'] = 0.5  # Neutral
            
            # 3. Moving Averages
            if all(k in latest for k in ['sma_20', 'sma_50']):
                sma20 = latest['sma_20']
                sma50 = latest['sma_50']
                close = latest['close']
                
                # Check both price vs MA and MA crossovers
                if close > sma20 and close > sma50 and sma20 > sma50:
                    # Strong uptrend
                    indicator_scores['moving_avg'] = 0.70  # Strong buy
                    reasons.append("Strong uptrend (price > SMA20 > SMA50)")
                elif close < sma20 and close < sma50 and sma20 < sma50:
                    # Strong downtrend
                    indicator_scores['moving_avg'] = 0.30  # Strong sell
                    reasons.append("Strong downtrend (price < SMA20 < SMA50)")
                elif close > sma20 and sma20 < sma50:
                    # Potential trend reversal (bullish)
                    indicator_scores['moving_avg'] = 0.65  # Moderate buy
                    reasons.append("Potential bullish reversal (price > SMA20)")
                elif close < sma20 and sma20 > sma50:
                    # Potential trend reversal (bearish)
                    indicator_scores['moving_avg'] = 0.35  # Moderate sell
                    reasons.append("Potential bearish reversal (price < SMA20)")
                else:
                    indicator_scores['moving_avg'] = 0.5  # Neutral
            else:
                indicator_scores['moving_avg'] = 0.5  # Neutral
            
            # 4. Bollinger Bands
            if all(k in latest for k in ['bb_lower', 'bb_upper', 'bb_middle']):
                bb_lower = latest['bb_lower']
                bb_upper = latest['bb_upper']
                bb_middle = latest['bb_middle']
                close = latest['close']
                
                # Price near or below lower band (potential buy)
                if close <= bb_lower * 1.01:
                    indicator_scores['bollinger'] = 0.75  # Strong buy
                    reasons.append("Price at lower Bollinger Band")
                # Price near or above upper band (potential sell)
                elif close >= bb_upper * 0.99:
                    indicator_scores['bollinger'] = 0.25  # Strong sell
                    reasons.append("Price at upper Bollinger Band")
                # Price below middle band (slight bearish)
                elif close < bb_middle:
                    indicator_scores['bollinger'] = 0.45  # Slight sell
                # Price above middle band (slight bullish)
                elif close > bb_middle:
                    indicator_scores['bollinger'] = 0.55  # Slight buy
                else:
                    indicator_scores['bollinger'] = 0.5  # Neutral
            else:
                indicator_scores['bollinger'] = 0.5  # Neutral
            
            # 5. Stochastic Oscillator
            if all(k in latest for k in ['stoch_k', 'stoch_d']):
                stoch_k = latest['stoch_k']
                stoch_d = latest['stoch_d']
                
                # Oversold with bullish crossover
                if stoch_k < 20 and stoch_d < 20 and stoch_k > stoch_d:
                    indicator_scores['stochastic'] = 0.75  # Strong buy
                    reasons.append("Stochastic oversold with bullish crossover")
                # Overbought with bearish crossover
                elif stoch_k > 80 and stoch_d > 80 and stoch_k < stoch_d:
                    indicator_scores['stochastic'] = 0.25  # Strong sell
                    reasons.append("Stochastic overbought with bearish crossover")
                # Oversold territory
                elif stoch_k < 20 and stoch_d < 20:
                    indicator_scores['stochastic'] = 0.65  # Moderate buy
                    reasons.append("Stochastic in oversold territory")
                # Overbought territory
                elif stoch_k > 80 and stoch_d > 80:
                    indicator_scores['stochastic'] = 0.35  # Moderate sell
                    reasons.append("Stochastic in overbought territory")
                else:
                    indicator_scores['stochastic'] = 0.5  # Neutral
            else:
                indicator_scores['stochastic'] = 0.5  # Neutral
            
            # 6. Volume analysis
            if 'volume' in latest and 'volume_ma_ratio' in latest:
                vol_ratio = latest['volume_ma_ratio']
                
                # Check price movement with volume confirmation
                if vol_ratio > 1.5 and df['close'].iloc[-1] > df['close'].iloc[-2]:
                    indicator_scores['volume'] = 0.65  # Moderate buy
                    reasons.append(f"Strong volume ({vol_ratio:.1f}x) with price increase")
                elif vol_ratio > 1.5 and df['close'].iloc[-1] < df['close'].iloc[-2]:
                    indicator_scores['volume'] = 0.35  # Moderate sell
                    reasons.append(f"Strong volume ({vol_ratio:.1f}x) with price decrease")
                elif vol_ratio > 1.2:
                    indicator_scores['volume'] = 0.55  # Slight buy
                    reasons.append(f"Above average volume ({vol_ratio:.1f}x)")
                else:
                    indicator_scores['volume'] = 0.5  # Neutral
            else:
                indicator_scores['volume'] = 0.5  # Neutral
            
            # 7. ADX (trend strength)
            if 'adx' in latest and 'adx_pos' in latest and 'adx_neg' in latest:
                adx = latest['adx']
                adx_pos = latest['adx_pos']
                adx_neg = latest['adx_neg']
                
                # Strong trend with directional bias
                if adx > 25:
                    if adx_pos > adx_neg:
                        indicator_scores['adx'] = 0.65  # Moderate buy
                        reasons.append(f"Strong trend (ADX={adx:.1f}) with bullish bias")
                    else:
                        indicator_scores['adx'] = 0.35  # Moderate sell
                        reasons.append(f"Strong trend (ADX={adx:.1f}) with bearish bias")
                else:
                    indicator_scores['adx'] = 0.5  # Neutral - weak trend
            else:
                indicator_scores['adx'] = 0.5  # Neutral
            
            # Calculate final signal
            weights = {
                'rsi': 0.20,
                'macd': 0.20,
                'moving_avg': 0.20,
                'bollinger': 0.15,
                'stochastic': 0.10,
                'volume': 0.10,
                'adx': 0.05
            }
            
            # Calculate weighted average
            weighted_sum = 0
            total_weight = 0
            
            for indicator, score in indicator_scores.items():
                weight = weights.get(indicator, 0)
                weighted_sum += score * weight
                total_weight += weight
            
            if total_weight > 0:
                confidence = weighted_sum / total_weight
            else:
                confidence = 0.5  # Neutral if no indicators available
            
            # Determine action based on confidence
            if confidence > 0.58:
                action = 'buy'
            elif confidence < 0.42:
                action = 'sell'
            else:
                action = 'hold'
            
            # Format reasons into a more readable string
            reason_str = ', '.join(reasons[:3])  # Limit to top 3 reasons
            
            return {
                'action': action,
                'confidence': confidence,
                'reasons': reason_str,
                'indicators': indicator_scores
            }
            
        except Exception as e:
            self.logger.error(f"Error generating basic signal: {str(e)}")
            traceback.print_exc()
            return {'action': 'hold', 'confidence': 0.5, 'reason': f'error: {str(e)}'}
    
    async def detect_chart_patterns(self, symbol, df=None):
        """Detect chart patterns for advanced trading signals."""
        try:
            # Get historical data if not provided
            if df is None:
                df = await self.get_historical_data(symbol)
                
            if df is None or len(df) < 40:
                return None
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Run pattern detection algorithms
            patterns = []
            
            # Run each pattern detector
            for pattern_name, pattern_func in self.pattern_recognition_system['patterns'].items():
                result = pattern_func(df)
                if result.get('detected', False):
                    patterns.append(result)
                    
                    # Log pattern detection
                    self.logger.info(f"Detected {pattern_name} pattern for {symbol} ({result.get('direction', 'unknown')})")
                    
                    # Store in pattern history
                    if 'pattern_history' not in self.pattern_recognition_system:
                        self.pattern_recognition_system['pattern_history'] = {}
                    
                    if symbol not in self.pattern_recognition_system['pattern_history']:
                        self.pattern_recognition_system['pattern_history'][symbol] = []
                    
                    # Add pattern to history
                    self.pattern_recognition_system['pattern_history'][symbol].append({
                        'pattern': pattern_name,
                        'timestamp': datetime.now().isoformat(),
                        'price': df['close'].iloc[-1],
                        'direction': result.get('direction', 'unknown'),
                        'confidence': result.get('confidence', 0.5),
                        'target': result.get('price_target')
                    })
                    
                    # Store in database
                    self._store_pattern_detection(symbol, pattern_name, result)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting chart patterns: {str(e)}")
            return None
    
    def _store_pattern_detection(self, symbol, pattern_type, result):
        """Store detected pattern in the database for tracking."""
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            
            # Add to detected_patterns table
            c.execute('''
                INSERT INTO detected_patterns
                (timestamp, symbol, pattern_type, confidence, price_at_detection, predicted_direction)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                symbol,
                pattern_type,
                result.get('confidence', 0.5),
                result.get('price', 0),
                result.get('direction', 'unknown')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing pattern detection: {str(e)}")
    
    async def generate_trading_signal(self, symbol, df=None):
        """Generate comprehensive trading signal combining ML, AI, basic signals, and sentiment analysis."""
        try:
            # Get historical data if not provided
            if df is None:
                df = await self.get_historical_data(symbol)
                
            if df is None or len(df) < 20:
                return {'action': 'hold', 'confidence': 0.5, 'reason': 'insufficient_data'}
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Detect market cycle
            market_cycle = self.detect_market_cycle(df)
            
            # Check for early reversal patterns
            reversal_detected = self.detect_early_reversal(df)
            if reversal_detected:
                self.logger.info(f"Early reversal signals detected for {symbol}")
            
            # Get chart patterns
            patterns = await self.detect_chart_patterns(symbol, df)
            pattern_signal = None
            if patterns:
                # Use the most confident pattern
                best_pattern = max(patterns, key=lambda x: x.get('confidence', 0))
                pattern_signal = {
                    'action': 'buy' if best_pattern.get('direction') == 'bullish' else 'sell',
                    'confidence': best_pattern.get('confidence', 0.5),
                    'pattern': best_pattern.get('pattern')
                }
                self.logger.info(f"Using pattern signal for {symbol}: {pattern_signal['action']} with {pattern_signal['confidence']:.2f} confidence ({pattern_signal['pattern']})")
            
            # Get ML model prediction
            try:
                ml_features = self.model_manager.prepare_features(df, use_available_only=True)
                ml_signal = self.model_manager.predict(ml_features)
                ml_confidence = ml_signal.get('confidence', 0.5)
                ml_action = ml_signal.get('action', 'hold')
                self.logger.info(f"ML signal for {symbol}: {ml_action} with {ml_confidence:.2f} confidence")
            except Exception as e:
                self.logger.error(f"ML model prediction error: {str(e)}")
                ml_confidence = 0.5  # Neutral on error
                ml_action = 'hold'
            
            # Get AI model prediction
            try:
                ai_signal = self.ai_enhancer.predict_next_movement(df)
                ai_confidence = ai_signal.get('confidence', 0.5)
                ai_action = ai_signal.get('direction', 'hold')
                self.logger.info(f"AI signal for {symbol}: {ai_action} with {ai_confidence:.2f} confidence")
            except Exception as e:
                self.logger.error(f"AI model prediction error: {str(e)}")
                ai_confidence = 0.5  # Neutral on error
                ai_action = 'hold'
            
            # Get basic technical signals
            try:
                basic_signal = await self.generate_basic_trading_signal(symbol, df)
                basic_confidence = basic_signal.get('confidence', 0.5)
                basic_action = basic_signal.get('action', 'hold')
                basic_reasons = basic_signal.get('reasons', '')
                self.logger.info(f"Basic signal for {symbol}: {basic_action} with {basic_confidence:.2f} confidence ({basic_reasons})")
            except Exception as e:
                self.logger.error(f"Basic signal error: {str(e)}")
                basic_confidence = 0.5  # Neutral on error
                basic_action = 'hold'
                basic_reasons = ''
            
            # Get sentiment analysis
            try:
                # Get sentiment from database (most recent entries)
                conn = sqlite3.connect(self.db_name)
                c = conn.cursor()
                c.execute('''
                    SELECT twitter_score, reddit_score, news_score, combined_score 
                    FROM sentiment_data 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                ''', (symbol,))
                sentiment_data = c.fetchone()
                conn.close()
                
                if sentiment_data:
                    twitter_score = sentiment_data[0]
                    reddit_score = sentiment_data[1]
                    news_score = sentiment_data[2]
                    sentiment_confidence = 0.5 + sentiment_data[3] * 0.1  # Scale to confidence
                    self.logger.info(f"Sentiment for {symbol}: {sentiment_confidence:.2f} confidence (Twitter: {twitter_score:.2f}, Reddit: {reddit_score:.2f}, News: {news_score:.2f})")
                else:
                    # If no data in DB, try to get fresh data
                    twitter_sentiment = self.twitter_analyzer.get_twitter_sentiment(symbol)
                    reddit_sentiment = await self.reddit_analyzer.get_reddit_sentiment(symbol)
                    news_impact = self.news_analyzer.get_news_impact(symbol)
                    
                    twitter_score = twitter_sentiment.get('score', 0)
                    reddit_score = reddit_sentiment.get('score', 0)
                    news_score = news_impact.get('impact_score', 0)
                    
                    # Calculate weighted sentiment
                    twitter_weight = 0.35
                    reddit_weight = 0.25
                    news_weight = 0.40
                    
                    combined_sentiment = (
                        twitter_score * twitter_weight +
                        reddit_score * reddit_weight +
                        news_score * news_weight
                    )
                    
                    # Convert to confidence (0.4-0.6 range)
                    sentiment_confidence = 0.5 + combined_sentiment * 0.1
                    self.logger.info(f"Fresh sentiment for {symbol}: {sentiment_confidence:.2f} confidence")
            except Exception as e:
                self.logger.error(f"Error processing sentiment: {str(e)}")
                twitter_score = 0
                reddit_score = 0
                news_score = 0
                sentiment_confidence = 0.5  # Neutral on error
            
            # Apply dynamic weights based on market conditions
            weights = dict(self.signal_weights)  # Start with default weights
            
            # Adjust for market cycle
            if market_cycle in ['bull_trend', 'breakout']:
                # In strong uptrends, favor AI and sentiment more
                weights['ml'] *= 0.9
                weights['ai'] *= 1.1
                weights['sentiment'] *= 1.1
                weights['basic'] *= 0.9
            elif market_cycle in ['bear_trend', 'breakdown']:
                # In downtrends, favor ML and basic signals more
                weights['ml'] *= 1.1
                weights['ai'] *= 0.9
                weights['sentiment'] *= 0.9
                weights['basic'] *= 1.1
            
            # Adjust for pattern detection
            if pattern_signal:
                # Temporary pattern weight (only when pattern detected)
                pattern_weight = min(0.2, pattern_signal['confidence'] * 0.25)
                
                # Reduce other weights proportionally
                total_other_weight = sum(weights.values())
                for k in weights:
                    weights[k] *= (1.0 - pattern_weight) / total_other_weight
                
                # Add pattern weight
                weights['pattern'] = pattern_weight
            else:
                weights['pattern'] = 0.0
            
            # Adjust for reversal detection
            if reversal_detected:
                # Boost basic signals and AI in reversals
                weights['basic'] *= 1.2
                weights['ai'] *= 1.1
                weights['ml'] *= 0.9
                
                # Re-normalize
                total_weight = sum(weights.values())
                for k in weights:
                    weights[k] /= total_weight
            
            # Apply time-based adjustments if available
            current_hour = datetime.now().hour
            if hasattr(self, 'time_adjustments') and current_hour in self.time_adjustments:
                time_adj = self.time_adjustments[current_hour]
                time_modifier = time_adj.get('buy_threshold_mod', 1.0)
                self.logger.info(f"Applied time-based adjustment for hour {current_hour}: modifier {time_modifier}")
            else:
                time_modifier = 1.0
            
            # Get symbol-specific buy threshold
            buy_threshold = self.buy_thresholds.get(symbol, 0.53) * time_modifier
            
            # Map signal actions to confidence values for weighting
            signal_values = {
                'ml': ml_confidence if ml_action == 'buy' else (1 - ml_confidence if ml_action == 'sell' else 0.5),
                'ai': ai_confidence if ai_action == 'buy' else (1 - ai_confidence if ai_action == 'sell' else 0.5),
                'basic': basic_confidence if basic_action == 'buy' else (1 - basic_confidence if basic_action == 'sell' else 0.5),
                'sentiment': sentiment_confidence
            }
            
            # Add pattern signal if available
            if pattern_signal:
                signal_values['pattern'] = pattern_signal['confidence'] if pattern_signal['action'] == 'buy' else (1 - pattern_signal['confidence'])
            
            # Calculate combined confidence
            weighted_sum = 0
            total_weight = 0
            signals_used = []
            
            for signal_type, value in signal_values.items():
                weight = weights.get(signal_type, 0)
                if weight > 0:
                    weighted_sum += value * weight
                    total_weight += weight
                    signals_used.append(signal_type)
            
            if total_weight > 0:
                combined_confidence = weighted_sum / total_weight
            else:
                combined_confidence = 0.5  # Neutral if no signals available
            
            # Apply market cycle boosts
            if market_cycle in ['bull_trend', 'breakout'] and combined_confidence > 0.5:
                boost_factor = 1.05  # 5% boost
                combined_confidence = min(0.70, combined_confidence * boost_factor)
                self.logger.info(f"Applied bullish market cycle boost: {boost_factor}x")
            
            # Apply reversal boost if detected and close to threshold
            if reversal_detected and 0.45 <= combined_confidence <= 0.52:
                reversal_boost = 0.05  # Increased from 0.03
                original_confidence = combined_confidence
                combined_confidence += reversal_boost
                self.logger.info(f"Applied reversal boost: {original_confidence:.3f} -> {combined_confidence:.3f}")
            
            # Apply risk management adjustment
            if hasattr(self, 'risk_management') and 'risk_reduction_factor' in self.risk_management:
                risk_factor = self.risk_management['risk_reduction_factor']
                if risk_factor < 1.0:
                    # Makes it harder to trigger buy signals
                    if combined_confidence > 0.5:  # Only apply to buy signals
                        original_confidence = combined_confidence
                        confidence_above_neutral = combined_confidence - 0.5
                        reduced_confidence = 0.5 + (confidence_above_neutral * risk_factor)
                        combined_confidence = reduced_confidence
                        self.logger.info(f"Applied risk reduction: {original_confidence:.3f} -> {combined_confidence:.3f} (factor: {risk_factor:.2f})")
            
            # Determine action based on combined confidence
            if combined_confidence > buy_threshold:
                action = 'buy'
            elif combined_confidence < 0.44:  # More eager to take profits
                action = 'sell'
            else:
                action = 'hold'
            
            # Safety override for extreme market conditions
            if action == 'buy' and market_cycle in ['breakdown'] and combined_confidence < 0.58:
                action = 'hold'
                self.logger.info(f"Buy suppressed due to {market_cycle} market cycle")
            
            # For reversals, be more willing to buy
            if action == 'hold' and reversal_detected and combined_confidence > buy_threshold * 0.95:
                action = 'buy'
                self.logger.info(f"Buy triggered due to reversal detection despite lower confidence")
            
            # Log signal details
            self.logger.info(f"Signal for {symbol}: {action.upper()} (confidence: {combined_confidence:.3f})")
            self.logger.info(f"Market cycle: {market_cycle}")
            self.logger.info(f"ML confidence: {ml_confidence:.3f}, AI confidence: {ai_confidence:.3f}, Basic: {basic_confidence:.3f}, Sentiment: {sentiment_confidence:.3f}")
            
            # Create final signal with all details
            signal = {
                'action': action,
                'confidence': combined_confidence,
                'market_cycle': market_cycle,
                'ml_confidence': ml_confidence,
                'ai_confidence': ai_confidence,
                'basic_confidence': basic_confidence,
                'basic_reasons': basic_reasons,
                'sentiment': {
                    'twitter': twitter_score,
                    'reddit': reddit_score,
                    'news': news_score,
                    'combined': sentiment_confidence
                },
                'pattern': pattern_signal['pattern'] if pattern_signal else None,
                'buy_threshold': buy_threshold,
                'weights': weights,
                'signals_used': ','.join(signals_used),
                'reversal_detected': reversal_detected,
                'timestamp': datetime.now()
            }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating trading signal: {str(e)}")
            traceback.print_exc()
            return {'action': 'hold', 'confidence': 0.5, 'reason': f'error: {str(e)}'}

    async def collect_training_data(self, days=30):
        """Collect historical data for all symbols for training."""
        try:
            all_data = []
            
            for symbol in self.symbols:
                self.logger.info(f"Collecting training data for {symbol}...")
                
                # Get historical data
                df = await self.get_historical_data(symbol, lookback_days=days)
                
                if df is not None and not df.empty:
                    # Calculate indicators
                    df = self.calculate_indicators(df)
                    
                    # Add symbol column
                    df['symbol'] = symbol
                    
                    all_data.append(df)
                    self.logger.info(f"Collected {len(df)} rows for {symbol}")
                else:
                    self.logger.warning(f"No data available for {symbol}")
                
                # Delay between symbols
                await asyncio.sleep(2)
            
            if not all_data:
                self.logger.warning("No data collected for training")
                return None
            
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"Combined dataset: {len(combined_df)} rows")
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Error collecting training data: {str(e)}")
            return None
    
    async def train_ml_model(self, training_data):
        """Train the ML model with the provided data."""
        try:
            self.logger.info("Training ML model...")
            
            # Prepare features and labels
            features_df = self.model_manager.prepare_features(training_data)
            labels = self.model_manager.create_labels(training_data)
            
            # Check data
            if features_df.empty or len(labels) == 0:
                self.logger.warning("No valid features or labels for ML training")
                return False
            
            # Train model
            success = self.model_manager.train_model(features_df, labels)
            
            if success:
                self.logger.info("ML model trained successfully")
                return True
            else:
                self.logger.error("ML model training failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error training ML model: {str(e)}")
            traceback.print_exc()
            return False
    
    async def train_ai_model(self, training_data):
        """Train the AI model with the provided data."""
        try:
            self.logger.info("Training AI model...")
            
            # Prepare sequence data
            X, y = self.ai_enhancer.prepare_sequence_data(training_data)
            
            # Check data
            if X is None or y is None or len(X) < 100:
                self.logger.warning("No valid sequence data for AI training")
                return False
            
            # Train model
            success = self.ai_enhancer.train_models(X, y)
            
            if success:
                self.logger.info("AI model trained successfully")
                return True
            else:
                self.logger.error("AI model training failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error training AI model: {str(e)}")
            traceback.print_exc()
            return False

    async def execute_trade(self, symbol, signal):
        """Execute a trade based on the generated signal."""
        try:
            # Validate signal
            if not isinstance(signal, dict) or 'action' not in signal:
                self.logger.error(f"Invalid signal format: {signal}")
                return None
            
            # Get action
            action = signal.get('action')
            
            # Skip if no action needed
            if action not in ['buy', 'sell']:
                self.logger.info(f"No trade needed for {symbol} - action: {action}")
                return None
            
            # Get current price
            price = await self.get_latest_price(symbol)
            if not price or price <= 0:
                self.logger.error(f"Invalid price for {symbol}: {price}")
                return None
            
            # Handle buy order
            if action == 'buy':
                # Check if already have position
                if symbol in self.positions:
                    self.logger.info(f"Already have position for {symbol}, skipping buy")
                    return None
                
                # Calculate position size
                position_size = await self.calculate_position_size(symbol, signal)
                if position_size <= 0:
                    self.logger.info(f"Invalid position size for {symbol}: {position_size}")
                    return None
                
                # Handle buy order
                if action == 'buy':
                    # [Keep existing code for position checking and size calculation]
                    
                    # Calculate quantity
                    quantity = position_size / price
                    
                    # Calculate adaptive stop loss
                    df = await self.get_historical_data(symbol)
                    stop_loss = await self.calculate_adaptive_stop_loss(symbol, price, df)
                    
                    # Calculate take profit based on risk-reward ratio
                    risk = price - stop_loss
                    take_profit = price + (risk * 2.5)  # 2.5:1 reward-to-risk ratio
                    
                    # Ensure take profit is at least the minimum percentage
                    min_take_profit = price * (1 + self.take_profit_pct)
                    take_profit = max(take_profit, min_take_profit)
                    
                    # Get trailing stop percentage
                    if symbol in self.symbol_risk_params:
                        trailing_stop_pct = self.symbol_risk_params[symbol].get('trailing_stop_pct', self.trailing_stop_pct)
                    else:
                        trailing_stop_pct = self.trailing_stop_pct
                    
                    # Update balance
                    self.balance['ZUSD'] -= position_size
                    
                    # Record position
                    self.positions[symbol] = {
                        'volume': quantity,
                        'entry_price': price,
                        'entry_time': datetime.now(),
                        'high_price': price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'trailing_stop_pct': trailing_stop_pct,
                        'signal_confidence': signal.get('confidence', 0.5),
                        'market_cycle': signal.get('market_cycle', 'unknown'),
                        'first_tier_executed': False,
                        'second_tier_executed': False,
                        'third_tier_executed': False
                    }
                    
                
                # Record trade
                trade = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'type': 'buy',
                    'price': price,
                    'quantity': quantity,
                    'value': position_size,
                    'balance_after': self.balance['ZUSD'],
                    'profit_loss': 0.0,  # 0 for buy orders
                    'signal_confidence': signal.get('confidence', 0.5),
                    'market_cycle': signal.get('market_cycle', 'unknown'),
                    'signals_used': signal.get('signals_used', '')
                }
                
                self.trade_history.append(trade)
                
                # Update last trade time
                self.last_trade_time[symbol] = datetime.now()
                
                # Update portfolio history
                total_equity = self.calculate_total_equity()
                max_equity = max([entry['equity'] for entry in self.portfolio_history], default=self.initial_capital)
                drawdown = (max_equity - total_equity) / max_equity if max_equity > 0 else 0
                
                self.portfolio_history.append({
                    'timestamp': datetime.now(),
                    'balance': self.balance['ZUSD'],
                    'equity': total_equity,
                    'drawdown': drawdown
                })
                
                # Log trade
                self.logger.info(f"BUY {quantity:.8f} {symbol} @ ${price:.2f} "
                            f"(${position_size:.2f}) "
                            f"Stop: ${levels['stop_loss']:.2f}, "
                            f"Target: ${levels['take_profit']:.2f}")
                
                # Save state
                self.save_state()
                
                return trade
                
            # Handle sell order
            elif action == 'sell':
                # Check if we have a position
                if symbol not in self.positions:
                    self.logger.warning(f"No position for {symbol}, skipping sell")
                    return None
                
                # Get position details
                position = self.positions[symbol]
                entry_price = position['entry_price']
                quantity = position['volume']
                
                # Calculate values
                sale_value = price * quantity
                entry_value = entry_price * quantity
                profit_loss = sale_value - entry_value
                pnl_percentage = (profit_loss / entry_value) * 100 if entry_value > 0 else 0
                
                # Update balance
                self.balance['ZUSD'] += sale_value
                
                # Record trade
                trade = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'type': 'sell',
                    'price': price,
                    'quantity': quantity,
                    'value': sale_value,
                    'balance_after': self.balance['ZUSD'],
                    'entry_price': entry_price,
                    'profit_loss': profit_loss,
                    'pnl_percentage': pnl_percentage,
                    'market_cycle': signal.get('market_cycle', position.get('market_cycle', 'unknown')),
                    'signal_confidence': signal.get('confidence', 0.5),
                    'signals_used': signal.get('signals_used', ''),
                    'exit_reason': 'signal'
                }
                
                self.trade_history.append(trade)
                
                # Remove position
                del self.positions[symbol]
                
                # Update last trade time
                self.last_trade_time[symbol] = datetime.now()
                
                # Update portfolio history
                total_equity = self.calculate_total_equity()
                max_equity = max([entry['equity'] for entry in self.portfolio_history], default=self.initial_capital)
                drawdown = (max_equity - total_equity) / max_equity if max_equity > 0 else 0
                
                self.portfolio_history.append({
                    'timestamp': datetime.now(),
                    'balance': self.balance['ZUSD'],
                    'equity': total_equity,
                    'drawdown': drawdown
                })
                
                # Log trade
                self.logger.info(f"SELL {quantity:.8f} {symbol} @ ${price:.2f} "
                            f"P&L: ${profit_loss:.2f} ({pnl_percentage:.2f}%)")
                
                # Save state
                self.save_state()
                
                return trade
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            traceback.print_exc()
            return None

    async def calculate_position_size(self, symbol, signal):
        """Calculate optimal position size based on signal confidence and risk parameters."""
        try:
            # Get available capital
            available_capital = self.balance.get('ZUSD', 0)
            
            # Check minimum balance requirement (0.05% of initial capital)
            min_balance = self.initial_capital * 0.0005
            if available_capital < min_balance:
                self.logger.warning(f"Insufficient balance for trading: ${available_capital:.2f}")
                return 0
            
            # Try Kelly Criterion sizing first if we have enough historical data
            kelly_size = self.calculate_kelly_position_size(symbol, signal)
            
            # Try tiered position sizing as alternative
            tiered_size = self.calculate_tiered_position_size(symbol, signal)
            
            # Use the more aggressive of the two methods, but with a cap
            position_size = max(kelly_size, tiered_size)
            
            # Set a reasonable minimum position size (0.1% of initial capital)
            min_position = self.initial_capital * 0.001
            if position_size < min_position:
                if signal.get('confidence', 0) > 0.53:  # Only force minimum for decent signals
                    position_size = min_position
                else:
                    self.logger.info(f"Position size ${position_size:.2f} below minimum ${min_position:.2f}")
                    return 0
            
            # Set a reasonable maximum position size (5% of capital per trade)
            max_position = available_capital * 0.05
            position_size = min(position_size, max_position)
            
            self.logger.info(f"Position size for {symbol}: ${position_size:.2f} "
                        f"({position_size/available_capital*100:.2f}% of available capital)")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0

    async def optimize_entry_exit_levels(self, symbol, entry_price, market_cycle, volatility=None):
        """Optimize stop loss and take profit levels based on volatility and market cycle."""
        try:
            # Use symbol-specific parameters if available
            if hasattr(self, 'symbol_risk_params') and symbol in self.symbol_risk_params:
                params = self.symbol_risk_params[symbol]
                
                # Adjust parameters based on market cycle
                if market_cycle in ['bull_trend', 'breakout']:
                    stop_loss_pct = params.get('stop_loss_pct', self.stop_loss_pct) * 1.1  # Wider in bull market
                    take_profit_pct = params.get('take_profit_pct', self.take_profit_pct) * 1.2  # Higher target
                    trailing_stop_pct = params.get('trailing_stop_pct', self.trailing_stop_pct) * 1.1
                elif market_cycle in ['bear_trend', 'breakdown']:
                    stop_loss_pct = params.get('stop_loss_pct', self.stop_loss_pct) * 0.9  # Tighter in bear market
                    take_profit_pct = params.get('take_profit_pct', self.take_profit_pct) * 0.8  # Lower target
                    trailing_stop_pct = params.get('trailing_stop_pct', self.trailing_stop_pct) * 0.9
                else:
                    stop_loss_pct = params.get('stop_loss_pct', self.stop_loss_pct)
                    take_profit_pct = params.get('take_profit_pct', self.take_profit_pct)
                    trailing_stop_pct = params.get('trailing_stop_pct', self.trailing_stop_pct)
                
                # Calculate levels
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
                
                self.logger.info(f"Using symbol-specific parameters for {symbol}: stop={stop_loss_pct:.3f}, target={take_profit_pct:.3f}")
                
                return {
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'trailing_stop_pct': trailing_stop_pct
                }
            
            # Optimize stop loss based on market cycle
            if market_cycle in ['ranging', 'low_volatility']:
                atr_multiplier = 3.0  # Wider stops in ranging markets
            elif market_cycle in ['bull_trend', 'breakout']:
                atr_multiplier = 2.5  # Standard stops in trending markets
            elif market_cycle in ['bear_trend', 'breakdown', 'high_volatility']:
                atr_multiplier = 2.0  # Tighter stops in bearish/volatile markets
            else:
                atr_multiplier = 2.5  # Default
            
            # Calculate stop distance
            stop_distance = atr_relative * atr_multiplier
            
            # Apply asset-specific adjustments
            if symbol == 'XBTUSD':
                # Bitcoin typically requires wider stops
                stop_distance *= 1.2
            elif symbol in ['XDGUSD', 'XRPUSD']:
                # More volatile assets need tighter stops
                stop_distance *= 0.8
            
            # Calculate stop loss price
            stop_loss = entry_price * (1 - stop_distance)
            
            # Calculate risk (entry to stop loss)
            risk = entry_price - stop_loss
            
            # Determine risk-reward ratio based on market regime
            if market_cycle in ['bull_trend', 'breakout']:
                rr_ratio = 3.0  # Higher targets in strong trends
            elif market_cycle in ['ranging', 'ranging_bullish']:
                rr_ratio = 1.5  # More conservative in ranges
            elif market_cycle in ['bear_trend', 'breakdown', 'high_volatility']:
                rr_ratio = 2.0  # Standard in bearish conditions
            else:
                rr_ratio = 2.0  # Default
            
            # Calculate take profit
            take_profit = entry_price + (risk * rr_ratio)
            
            # Calculate trailing stop percentage
            trailing_stop_pct = stop_distance * 0.7  # Slightly tighter than initial stop
            
            # Apply reasonable bounds
            trailing_stop_pct = max(0.005, min(0.05, trailing_stop_pct))  # 0.5% to 5%
            
            self.logger.info(f"Optimized levels for {symbol} - "
                        f"Stop: ${stop_loss:.2f} ({stop_distance*100:.2f}%), "
                        f"Target: ${take_profit:.2f}, "
                        f"R:R: {rr_ratio:.1f}, "
                        f"Trailing: {trailing_stop_pct*100:.2f}%")
            
            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'trailing_stop_pct': trailing_stop_pct
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing entry/exit levels: {str(e)}")
            
            # Return default values on error
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
            
            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'trailing_stop_pct': self.trailing_stop_pct
            }

    async def execute_dca_trade(self, symbol, signal):
        """Execute a buy with dollar-cost averaging strategy with improved market adaptivity."""
        try:
            # Use configuration parameters
            dca_config = self.config.get('dca', {})
            dca_parts = dca_config.get('parts', self.dca_parts)
            dca_time_between = dca_config.get('time_between', self.dca_time_between)
            bull_first_part_pct = dca_config.get('bull_first_part_pct', 0.60)
            bear_first_part_pct = dca_config.get('bear_first_part_pct', 0.30)
            
            # Get current price
            price = await self.get_latest_price(symbol)
            if not price or price <= 0:
                self.logger.error(f"Invalid price for {symbol}: {price}")
                return None
            
            # Get market cycle
            market_cycle = signal.get('market_cycle', 'unknown')
            
            # Adjust DCA strategy based on market cycle
            if market_cycle in ['bull_trend', 'breakout']:
                # In strong uptrends, use faster entries with more upfront
                first_part_pct = bull_first_part_pct
                remaining_pct = (1.0 - first_part_pct) / (dca_parts - 1) if dca_parts > 1 else 0
                dca_time_between = max(15, int(dca_time_between * 0.5))  # Faster entries
                self.logger.info(f"Using bull market DCA strategy: {first_part_pct*100:.0f}% initial, {dca_time_between}min intervals")
            elif market_cycle in ['bear_trend', 'breakdown']:
                # In downtrends, more cautious with smaller initial entry
                first_part_pct = bear_first_part_pct
                remaining_pct = (1.0 - first_part_pct) / (dca_parts - 1) if dca_parts > 1 else 0
                dca_time_between = min(120, int(dca_time_between * 2))  # Slower entries
                self.logger.info(f"Using bear market DCA strategy: {first_part_pct*100:.0f}% initial, {dca_time_between}min intervals")
            elif market_cycle in ['ranging', 'ranging_volatile', 'tight_ranging']:
                # More balanced approach for ranging markets
                first_part_pct = 0.40
                remaining_pct = (1.0 - first_part_pct) / (dca_parts - 1) if dca_parts > 1 else 0
                # Use standard timing for ranging markets
                self.logger.info(f"Using ranging market DCA strategy: {first_part_pct*100:.0f}% initial, {dca_time_between}min intervals")
            else:
                # Standard approach for neutral markets
                first_part_pct = 0.45  # 45% in first entry
                remaining_pct = (1.0 - first_part_pct) / (dca_parts - 1) if dca_parts > 1 else 0
                self.logger.info(f"Using standard DCA strategy: {first_part_pct*100:.0f}% initial, {dca_time_between}min intervals")
            
            # Apply risk management adjustment if active
            if hasattr(self, 'risk_management') and self.risk_management.get('active_risk_protection', False):
                risk_factor = self.risk_management.get('risk_reduction_factor', 1.0)
                first_part_pct = min(0.75, first_part_pct * (2 - risk_factor))  # Increase first part when risk protected
                self.logger.info(f"Risk protection active: Adjusted first part to {first_part_pct*100:.0f}%")
            
            # Calculate total position size
            total_position_size = await self.calculate_position_size(symbol, signal)
            if total_position_size <= 0:
                self.logger.info(f"Invalid total position size for {symbol}: {total_position_size}")
                return None
            
            # Calculate part sizes
            first_part_size = total_position_size * first_part_pct
            remaining_part_size = total_position_size * remaining_pct if dca_parts > 1 else 0
            
            self.logger.info(f"DCA Strategy: Executing part 1/{dca_parts} for {symbol} "
                        f"({first_part_pct*100:.0f}%, ${first_part_size:.2f})")
            
            # Execute first trade with part size
            first_trade = await self._execute_dca_part(
                symbol, 
                signal, 
                first_part_size, 
                price, 
                is_first_part=True
            )
            
            # Create plan for remaining parts
            if dca_parts > 1 and first_trade:
                # Calculate target prices with adaptive strategy
                target_prices = self._calculate_dca_target_prices(
                    price,
                    dca_parts - 1,  # Remaining parts
                    market_cycle
                )
                
                dca_plan = {
                    'symbol': symbol,
                    'signal': signal,
                    'total_parts': dca_parts,
                    'completed_parts': 1,
                    'part_size': remaining_part_size,
                    'first_entry_price': price,
                    'first_entry_time': datetime.now(),
                    'time_between_parts': dca_time_between * 60,  # Convert minutes to seconds
                    'market_cycle': market_cycle,
                    'target_prices': target_prices
                }
                
                self.dca_plans.append(dca_plan)
                self.logger.info(f"Created DCA plan with {dca_parts} parts, "
                            f"next execution in {dca_time_between} minutes")
                
                # Log target prices for clarity
                if target_prices:
                    targets_str = ", ".join([f"${p:.2f}" for p in target_prices])
                    self.logger.info(f"DCA target prices: {targets_str}")
            
            return first_trade
            
        except Exception as e:
            self.logger.error(f"Error in DCA trade: {str(e)}")
            traceback.print_exc()
            return None

    async def _execute_dca_part(self, symbol, signal, position_size, price, is_first_part=False):
        """Execute a single DCA part."""
        try:
            # Calculate quantity
            quantity = position_size / price
            
            # Optimize entry/exit levels
            market_cycle = signal.get('market_cycle', 'unknown')
            
            # If this is first part, calculate stop loss and take profit
            if is_first_part:
                # Use adaptive stop loss for first part
                df = await self.get_historical_data(symbol)
                stop_loss = await self.calculate_adaptive_stop_loss(symbol, price, df)
                
                # Calculate take profit with better risk-reward ratio
                risk = price - stop_loss
                take_profit = price + (risk * 2.5)  # 2.5:1 reward-to-risk
                
                # Ensure minimum take profit
                min_take_profit = price * (1 + self.take_profit_pct)
                take_profit = max(take_profit, min_take_profit)
                
                # Get trailing stop from symbol params or default
                if symbol in self.symbol_risk_params:
                    trailing_stop_pct = self.symbol_risk_params[symbol].get('trailing_stop_pct', self.trailing_stop_pct)
                else:
                    trailing_stop_pct = self.trailing_stop_pct
            else:
                # For additional parts, use existing levels if available
                if symbol in self.positions:
                    stop_loss = self.positions[symbol].get('stop_loss')
                    take_profit = self.positions[symbol].get('take_profit')
                    trailing_stop_pct = self.positions[symbol].get('trailing_stop_pct', self.trailing_stop_pct)
                else:
                    # Shouldn't happen, but just in case
                    df = await self.get_historical_data(symbol)
                    stop_loss = await self.calculate_adaptive_stop_loss(symbol, price, df)
                    take_profit = price * (1 + self.take_profit_pct)
                    if symbol in self.symbol_risk_params:
                        trailing_stop_pct = self.symbol_risk_params[symbol].get('trailing_stop_pct', self.trailing_stop_pct)
                    else:
                        trailing_stop_pct = self.trailing_stop_pct
            
            # Update balance
            self.balance['ZUSD'] -= position_size
            
            # Update or create position
            if symbol in self.positions:
                # Add to existing position
                existing_position = self.positions[symbol]
                
                # Calculate new weighted position
                total_quantity = existing_position['volume'] + quantity
                weighted_entry = (existing_position['entry_price'] * existing_position['volume'] + 
                                price * quantity) / total_quantity
                
                # Update position
                self.positions[symbol]['volume'] = total_quantity
                self.positions[symbol]['entry_price'] = weighted_entry
                self.positions[symbol]['high_price'] = max(price, existing_position.get('high_price', price))
                
                self.logger.info(f"Added {quantity:.8f} {symbol} @ ${price:.2f} to position "
                            f"New position: {total_quantity:.8f} @ ${weighted_entry:.2f} (weighted)")
            else:
                # Create new position
                self.positions[symbol] = {
                    'volume': quantity,
                    'entry_price': price,
                    'entry_time': datetime.now(),
                    'high_price': price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'trailing_stop_pct': trailing_stop_pct,
                    'signal_confidence': signal.get('confidence', 0.5),
                    'market_cycle': market_cycle,
                    'first_tier_executed': False,
                    'second_tier_executed': False,
                    'third_tier_executed': False
                }
                
                self.logger.info(f"New position: {quantity:.8f} {symbol} @ ${price:.2f} "
                            f"Stop: ${stop_loss:.2f}, Target: ${take_profit:.2f}")
            
            # Record trade
            trade_type = 'buy' if is_first_part else 'dca_buy'
            
            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'type': trade_type,
                'price': price,
                'quantity': quantity,
                'value': position_size,
                'balance_after': self.balance['ZUSD'],
                'signal_confidence': signal.get('confidence', 0.5),
                'market_cycle': market_cycle,
                'signals_used': signal.get('signals_used', '')
            }
            
            self.trade_history.append(trade)
            
            # Update last trade time
            self.last_trade_time[symbol] = datetime.now()
            
            # Update portfolio history
            total_equity = self.calculate_total_equity()
            max_equity = max([entry['equity'] for entry in self.portfolio_history], default=self.initial_capital)
            drawdown = (max_equity - total_equity) / max_equity if max_equity > 0 else 0
            
            self.portfolio_history.append({
                'timestamp': datetime.now(),
                'balance': self.balance['ZUSD'],
                'equity': total_equity,
                'drawdown': drawdown
            })
            
            # Save state
            self.save_state()
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Error executing DCA part: {str(e)}")
            traceback.print_exc()
            return None

    def _calculate_dca_target_prices(self, entry_price, remaining_parts, market_cycle):
        """Calculate target prices for DCA parts based on market conditions."""
        targets = []
        
        # No targets needed if only 1 part
        if remaining_parts <= 0:
            return targets
        
        # Adjust drop percentages based on market cycle
        if market_cycle in ['bull_trend', 'breakout']:
            # In bull trends, use smaller drops and tighter spacing
            base_drop = 0.005  # 0.5% base drop
            step_factor = 1.2   # Each step is 1.2x larger
        elif market_cycle in ['bear_trend', 'breakdown']:
            # In bear trends, use larger drops with wider spacing
            base_drop = 0.01   # 1% base drop
            step_factor = 1.5  # Each step is 1.5x larger
        else:
            # Standard approach
            base_drop = 0.0075  # 0.75% base drop
            step_factor = 1.3   # Each step is 1.3x larger
        
        # Calculate targets with progressive drops
        for i in range(1, remaining_parts + 1):
            # Progressive drops: -0.75%, then -0.98%, then -1.27%, etc. (with step factor 1.3)
            drop_pct = -base_drop * (step_factor ** (i-1))
            target_price = entry_price * (1 + drop_pct)
            targets.append(target_price)
        
        return targets

    async def check_dca_plans(self):
        """Process any active DCA plans."""
        if not self.dca_plans:
            return
        
        current_time = datetime.now()
        plans_to_remove = []
        
        for i, plan in enumerate(self.dca_plans):
            # Skip completed plans
            if plan['completed_parts'] >= plan['total_parts']:
                plans_to_remove.append(i)
                continue
            
            # Check if it's time for next part
            time_since_first = (current_time - plan['first_entry_time']).total_seconds()
            parts_due = min(plan['total_parts'], int(time_since_first / plan['time_between_parts']) + 1)
            parts_to_execute = parts_due - plan['completed_parts']
            
            if parts_to_execute <= 0:
                continue
            
            # Get current price
            symbol = plan['symbol']
            current_price = await self.get_latest_price(symbol)
            
            if not current_price or current_price <= 0:
                self.logger.warning(f"Invalid price for DCA part: {current_price}")
                continue
            
            # Check if we're at a target price (if available)
            target_prices = plan.get('target_prices', [])
            current_part = plan['completed_parts']
            
            # If we have a target price for this part, check if we're below it
            price_triggered = False
            if current_part < len(target_prices) + 1:  # +1 since we're 0-indexed
                target_price = target_prices[current_part - 1]  # -1 to account for 0-indexing
                if current_price <= target_price:
                    price_triggered = True
                    self.logger.info(f"DCA price target triggered: {current_price:.2f} <= {target_price:.2f}")
            
            # Dynamic price threshold based on time elapsed
            first_price = plan['first_entry_price']
            hours_elapsed = time_since_first / 3600  # Convert seconds to hours
            
            # Gradually increase acceptable price threshold over time
            price_adjustment = min(0.02, 0.005 + (0.001 * hours_elapsed))  # Maximum 2% adjustment
            max_price_threshold = first_price * (1 + price_adjustment)
            
            # Skip if price is too high and not triggered by target
            if current_price > max_price_threshold and not price_triggered:
                # Force execution if final part and we've waited too long (24+ hours)
                if hours_elapsed > 24 and plan['completed_parts'] == plan['total_parts'] - 1:
                    self.logger.info(f"Forcing final DCA execution after {hours_elapsed:.1f} hours despite price")
                else:
                    self.logger.info(f"Skipping DCA part: price {current_price:.2f} > threshold {max_price_threshold:.2f}")
                    continue
            
            # Execute due parts
            for j in range(parts_to_execute):
                part_number = plan['completed_parts'] + 1
                self.logger.info(f"DCA Plan: Executing part {part_number}/{plan['total_parts']} for {symbol}")
                
                # Execute this part
                success = await self._execute_dca_part(
                    symbol,
                    plan['signal'],
                    plan['part_size'],
                    current_price,
                    is_first_part=False
                )
                
                if success:
                    plan['completed_parts'] += 1
                    self.logger.info(f"DCA part {part_number} executed successfully")
                else:
                    self.logger.warning(f"DCA part {part_number} execution failed")
                    break
            
            # Check if plan is complete
            if plan['completed_parts'] >= plan['total_parts']:
                self.logger.info(f"DCA plan for {symbol} completed")
                plans_to_remove.append(i)
        
        # Remove completed plans
        for i in sorted(plans_to_remove, reverse=True):
            del self.dca_plans[i]

    async def calculate_adaptive_stop_loss(self, symbol, entry_price, current_data=None):
        """Calculate stop loss based on market volatility and price action."""
        try:
            # Get historical data if not provided
            if current_data is None:
                current_data = await self.get_historical_data(symbol, lookback_days=3)
                if current_data is None or len(current_data) < 20:
                    # Fall back to default stop if no data
                    return entry_price * (1 - self.stop_loss_pct)
            
            # Calculate ATR (Average True Range) for volatility
            high_low = current_data['high'] - current_data['low']
            high_close = abs(current_data['high'] - current_data['close'].shift())
            low_close = abs(current_data['low'] - current_data['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=self.risk_management['adaptive_stops']['atr_periods']).mean().iloc[-1]
            
            # Normalize ATR as percentage of price
            atr_pct = atr / entry_price
            
            # Get market cycle to adjust stop distance
            market_cycle = self.detect_market_cycle(current_data)
            
            # Adjust multiplier based on market cycle
            cycle_multiplier = 1.0  # Default
            if market_cycle in ['bull_trend', 'breakout']:
                cycle_multiplier = 0.9  # Tighter in strong uptrend
            elif market_cycle in ['bear_trend', 'breakdown']:
                cycle_multiplier = 0.8  # Even tighter in downtrends
            elif market_cycle in ['ranging']:
                cycle_multiplier = 1.1  # Slightly wider in ranges to avoid whipsaws
            
            # Calculate stop distance as percentage
            stop_distance = min(
                self.risk_management['adaptive_stops']['max_stop_distance'],
                max(
                    self.risk_management['adaptive_stops']['min_stop_distance'],
                    atr_pct * self.risk_management['adaptive_stops']['atr_multiplier'] * cycle_multiplier
                )
            )
            
            # Calculate actual stop price
            stop_loss = entry_price * (1 - stop_distance)
            
            self.logger.info(f"Adaptive stop loss for {symbol}: {stop_distance*100:.2f}% from entry "
                        f"(ATR: {atr_pct*100:.2f}%, cycle: {market_cycle})")
            
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptive stop loss: {str(e)}")
            # Fall back to default stop if calculation fails
            return entry_price * (1 - self.stop_loss_pct)

    async def monitor_positions(self):
        """Monitor active positions for take profit, stop loss, or trailing stop events."""
        try:
            # Skip if no positions
            if not self.positions:
                return
            
            current_time = datetime.now()
            positions_to_close = []
            
            # Check each position
            for symbol, position in self.positions.items():
                # Get current price
                current_price = await self.get_latest_price(symbol)
                if not current_price:
                    self.logger.warning(f"Could not get current price for {symbol}, skipping monitoring")
                    continue
                
                # Calculate metrics
                entry_price = position['entry_price']
                quantity = position['volume']
                entry_time = position['entry_time']
                position_age = (current_time - entry_time).total_seconds() / 3600  # Hours
                
                unrealized_pnl = (current_price - entry_price) * quantity
                pnl_percentage = ((current_price - entry_price) / entry_price) * 100
                
                # Update high price if we have a new high
                if current_price > position.get('high_price', entry_price):
                    position['high_price'] = current_price
                    self.logger.info(f"New high price for {symbol}: ${current_price:.2f}")
                
                # Get market data for current cycle detection
                df = await self.get_historical_data(symbol, lookback_days=1)
                market_cycle = "unknown"
                
                if df is not None and not df.empty:
                    df = self.calculate_indicators(df)
                    market_cycle = self.detect_market_cycle(df)
                
                # Get stop and target levels
                stop_loss = position.get('stop_loss', entry_price * (1 - self.stop_loss_pct))
                take_profit = position.get('take_profit', entry_price * (1 + self.take_profit_pct))
                
                # Check if market is experiencing high volatility
                high_volatility = False
                if df is not None and 'bb_width' in df.columns and df['bb_width'].iloc[-1] > 0.05:
                    high_volatility = True
                
                # Adjust stop loss when trade becomes profitable
                if pnl_percentage > 1.0 and position.get('stop_loss', 0) < position['entry_price']:
                    # Move stop loss to breakeven (entry price)
                    position['stop_loss'] = position['entry_price']
                    self.logger.info(f"Moving stop loss to breakeven for {symbol} at +{pnl_percentage:.2f}% profit")
                
                # If very profitable but not yet at first tier, tighten stop loss further
                if pnl_percentage > 2.0 and not position.get('first_tier_executed', False):
                    # Calculate a tighter trailing stop specifically for this scenario
                    position['stop_loss'] = max(
                        position['stop_loss'],  # Don't move stop down
                        current_price * (1 - position.get('trailing_stop_pct', self.trailing_stop_pct) * 0.7)
                    )
                    self.logger.info(f"Tightened stop loss for highly profitable {symbol} position")
                
                # Check time-based stop loss adjustment for aged positions
                if position_age > 48:  # If position held for more than 48 hours
                    if pnl_percentage < 0:
                        # For old unprofitable positions, tighten stop loss by 20%
                        old_stop = position['stop_loss']
                        new_stop = entry_price * (1 - self.stop_loss_pct * 0.8)
                        position['stop_loss'] = max(old_stop, new_stop)  # Don't move stop down
                        self.logger.info(f"Tightened stop loss for aged unprofitable {symbol} position")
                    elif pnl_percentage < 0.5:
                        # For barely profitable old positions, set stop to breakeven
                        position['stop_loss'] = max(position['stop_loss'], entry_price)
                        self.logger.info(f"Moving stop to breakeven for barely profitable aged {symbol} position")
                
                # Add special condition for SOLUSD to limit losses
                if symbol == 'SOLUSD' and pnl_percentage < -1.5:
                    positions_to_close.append({
                        'symbol': symbol,
                        'reason': 'emergency_exit_solana',
                        'price': current_price
                    })
                    self.logger.info(f"Emergency exit for {symbol} due to quick loss of {pnl_percentage:.2f}%")
                    continue
                
                # Dynamic trailing stop based on profit and market cycle
                if pnl_percentage > 0:
                    # Get base trailing stop
                    trailing_stop_pct = position.get('trailing_stop_pct', self.trailing_stop_pct)
                    
                    # Scale trailing stop based on profit level more aggressively
                    if pnl_percentage < 2.0:  # Lower threshold
                        trailing_pct = trailing_stop_pct * 0.9  # Tighter initially
                    elif pnl_percentage < 5.0:
                        trailing_pct = trailing_stop_pct * 1.1
                    else:
                        trailing_pct = trailing_stop_pct * 1.3  # Less wide for big winners
                else:
                    trailing_pct = position.get('trailing_stop_pct', self.trailing_stop_pct)
                
                # Calculate trailing stop level
                highest_price = position['high_price']
                trailing_stop = highest_price * (1 - trailing_pct)
                
                # TIERED TAKE-PROFIT STRATEGY
                # First tier - take small partial profits at 1.2% gain (earlier than before)
                if current_price >= entry_price * 1.012 and pnl_percentage > 0 and not position.get('first_tier_executed', False):
                    # Take 25% off at first profit target (less than before)
                    self.logger.info(f"First profit target reached for {symbol}: +1.2%, taking 25% profits")
                    success = await self.close_partial_position(symbol, current_price, 0.25, 'first_tier_profit')
                    if success:
                        position['first_tier_executed'] = True
                
                # Second tier - take more profits at 2.5% gain (earlier than before)
                if current_price >= entry_price * 1.025 and not position.get('second_tier_executed', False) and position.get('first_tier_executed', False):
                    # Take another 35% off
                    self.logger.info(f"Second profit target reached for {symbol}: +2.5%, taking 35% profits")
                    success = await self.close_partial_position(symbol, current_price, 0.35, 'second_tier_profit')
                    if success:
                        position['second_tier_executed'] = True
                        
                        # Widen trailing stop for remainder to let it run
                        position['trailing_stop_pct'] = trailing_pct * 1.3
                
                # Check exit conditions
                # 1. Stop loss hit - with volatility check
                if ((current_price <= stop_loss and not high_volatility) or 
                    current_price <= (stop_loss * 0.9)):  # Force exit if 10% below stop
                    positions_to_close.append({
                        'symbol': symbol,
                        'reason': 'stop_loss',
                        'price': current_price
                    })
                    self.logger.info(f"Stop loss triggered for {symbol} at ${current_price:.2f}")
                    continue
                
                # 2. Take profit hit
                if current_price >= take_profit:
                    positions_to_close.append({
                        'symbol': symbol,
                        'reason': 'take_profit',
                        'price': current_price
                    })
                    self.logger.info(f"Take profit target reached for {symbol} at ${current_price:.2f}")
                    continue
                
                # 3. Trailing stop hit (only for positions in profit)
                if pnl_percentage > 0 and current_price < trailing_stop:
                    positions_to_close.append({
                        'symbol': symbol,
                        'reason': 'trailing_stop',
                        'price': current_price
                    })
                    self.logger.info(f"Trailing stop triggered for {symbol} at ${current_price:.2f}")
                    continue
                
                # Time-based exit strategy adapted to market cycle
                max_hold_times = {
                    'bull_trend': 120,    # 5 days in bull trend
                    'breakout': 144,      # 6 days in breakout
                    'ranging_bullish': 96, # 4 days in bullish range
                    'ranging': 72,         # 3 days in range
                    'bear_trend': 36,      # 1.5 days in bear trend
                    'breakdown': 24,       # 1 day in breakdown
                    'recovery': 84,        # 3.5 days in recovery
                    'low_volatility': 96,  # 4 days in low volatility
                    'high_volatility': 36, # 1.5 days in high volatility
                    'mixed': 72,           # 3 days in mixed
                    'unknown': 72          # 3 days default
                }
                
                max_hold_time = max_hold_times.get(market_cycle, 72)
                
                # Exit unprofitable trades faster
                if pnl_percentage < -2.0 and position_age > 12:
                    positions_to_close.append({
                        'symbol': symbol,
                        'reason': 'unprofitable_aged',
                        'price': current_price
                    })
                    self.logger.info(f"Closing unprofitable position for {symbol} after {position_age:.1f} hours")
                elif position_age > max_hold_time:
                    positions_to_close.append({
                        'symbol': symbol,
                        'reason': 'max_hold_time',
                        'price': current_price
                    })
                    self.logger.info(f"Maximum hold time reached for {symbol}: {position_age:.1f} > {max_hold_time:.1f} hours")
            
            # Close positions that need to be closed
            for close_info in positions_to_close:
                symbol = close_info['symbol']
                reason = close_info['reason']
                price = close_info['price']
                
                await self.close_position(symbol, price, reason)
            
            # Save state if we closed any positions
            if positions_to_close:
                self.save_state()
                
        except Exception as e:
            self.logger.error(f"Error monitoring positions: {str(e)}")
            traceback.print_exc()

    async def close_position(self, symbol, price, reason='manual'):
        """Close a position completely."""
        try:
            if symbol not in self.positions:
                self.logger.warning(f"No position found for {symbol}")
                return False
            
            position = self.positions[symbol]
            quantity = position['volume']
            entry_price = position['entry_price']
            
            # Validate quantity and price
            if quantity <= 0 or price <= 0:
                self.logger.error(f"Invalid quantity ({quantity}) or price ({price}) for {symbol}")
                return False
                
            # Calculate values
            sale_value = price * quantity
            entry_value = entry_price * quantity
            profit_loss = sale_value - entry_value
            pnl_percentage = (profit_loss / entry_value) * 100 if entry_value > 0 else 0
            
            # Update balance
            self.balance['ZUSD'] += sale_value
            
            # Create trade record
            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'type': 'sell',
                'price': price,
                'quantity': quantity,
                'value': sale_value,
                'balance_after': self.balance['ZUSD'],
                'entry_price': entry_price,
                'profit_loss': profit_loss,
                'pnl_percentage': pnl_percentage,
                'market_cycle': position.get('market_cycle', 'unknown'),
                'signals_used': position.get('signals_used', ''),
                'exit_reason': reason
            }
            
            # Add to trade history
            self.trade_history.append(trade)
            
            # Remove position
            del self.positions[symbol]
            
            # Update last trade time
            self.last_trade_time[symbol] = datetime.now()
            
            # Update portfolio history
            total_equity = self.calculate_total_equity()
            max_equity = max([entry['equity'] for entry in self.portfolio_history], default=self.initial_capital)
            drawdown = (max_equity - total_equity) / max_equity if max_equity > 0 else 0
            
            self.portfolio_history.append({
                'timestamp': datetime.now(),
                'balance': self.balance['ZUSD'],
                'equity': total_equity,
                'drawdown': drawdown
            })
            
            # Log the trade
            self.logger.info(f"CLOSED {symbol} position: {quantity:.8f} @ ${price:.2f} "
                        f"P&L: ${profit_loss:.2f} ({pnl_percentage:.2f}%) "
                        f"Reason: {reason}")
            
            # Check performance stats
            if profit_loss > 0:
                self.logger.info(f"WINNING TRADE: {symbol} made ${profit_loss:.2f} profit")
            else:
                self.logger.info(f"LOSING TRADE: {symbol} lost ${abs(profit_loss):.2f}")
            
            # Update strategy performance metrics if available
            if hasattr(self, 'performance_analytics') and 'strategy_performance' in self.performance_analytics:
                signal_types = trade.get('signals_used', '').split(',')
                for signal_type in signal_types:
                    if signal_type.strip():
                        if signal_type not in self.performance_analytics['strategy_performance']:
                            self.performance_analytics['strategy_performance'][signal_type] = {
                                'wins': 0, 'losses': 0, 'total_pnl': 0
                            }
                        
                        stats = self.performance_analytics['strategy_performance'][signal_type]
                        if profit_loss > 0:
                            stats['wins'] += 1
                        else:
                            stats['losses'] += 1
                        stats['total_pnl'] += profit_loss
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            return False

    async def close_partial_position(self, symbol, price, position_fraction=0.5, reason='partial'):
        """Close a fraction of the position to lock in profits."""
        try:
            if symbol not in self.positions:
                self.logger.warning(f"No position found for {symbol}")
                return False
            
            position = self.positions[symbol]
            entry_price = position['entry_price']
            total_quantity = position['volume']
            close_quantity = total_quantity * position_fraction
            
            # Validate inputs
            if total_quantity <= 0 or close_quantity <= 0 or price <= 0:
                self.logger.error(f"Invalid quantities or price for partial close: {symbol}")
                return False
            
            # Calculate values
            close_value = close_quantity * price
            pnl = (price - entry_price) * close_quantity
            pnl_percentage = ((price / entry_price) - 1) * 100
            
            # Update position
            new_quantity = total_quantity - close_quantity
            self.positions[symbol]['volume'] = new_quantity
            
            # Update balance
            self.balance['ZUSD'] += close_value
            
            # Record trade
            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'type': 'partial_sell',
                'price': price,
                'quantity': close_quantity,
                'value': close_value,
                'entry_price': entry_price,
                'profit_loss': pnl,
                'pnl_percentage': pnl_percentage,
                'balance_after': self.balance['ZUSD'],
                'exit_reason': reason,
                'remaining_position': new_quantity,
                'market_cycle': position.get('market_cycle', 'unknown'),
                'signals_used': position.get('signals_used', '')
            }
            
            self.trade_history.append(trade)
            
            self.logger.info(f"PARTIAL CLOSE {position_fraction*100:.0f}% of {symbol} position at ${price:.2f}, "
                        f"P&L: ${pnl:.2f} ({pnl_percentage:.2f}%)")
            self.logger.info(f"Remaining position: {new_quantity:.8f} {symbol}")
            
            # Update portfolio history
            total_equity = self.calculate_total_equity()
            max_equity = max([entry['equity'] for entry in self.portfolio_history], default=self.initial_capital)
            drawdown = (max_equity - total_equity) / max_equity if max_equity > 0 else 0
            
            self.portfolio_history.append({
                'timestamp': datetime.now(),
                'balance': self.balance['ZUSD'],
                'equity': total_equity,
                'drawdown': drawdown
            })
            
            # Save state
            self.save_state()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in partial position close: {str(e)}")
            return False

    def calculate_total_equity(self):
        """Calculate total portfolio value including all positions."""
        try:
            equity = self.balance.get('ZUSD', 0)
            
            # Add the value of all positions
            for symbol, position in self.positions.items():
                # Get current price from cache if available
                if hasattr(self, 'price_cache') and f"price_{symbol}" in self.price_cache:
                    current_price = self.price_cache[f"price_{symbol}"]["price"]
                else:
                    # Try to get from sync endpoint
                    try:
                        ticker = self.api.query_public('Ticker', {'pair': symbol})
                        if ticker and 'result' in ticker and symbol in ticker['result']:
                            current_price = float(ticker['result'][symbol]['c'][0])
                        else:
                            # Skip if we can't get a price
                            self.logger.warning(f"Could not get price for {symbol}, using entry price for equity calculation")
                            current_price = position['entry_price']
                    except Exception as e:
                        self.logger.warning(f"Error getting price for {symbol}: {str(e)}")
                        current_price = position['entry_price']  # Use entry price as fallback
                
                # Skip if price is invalid
                if not current_price or current_price <= 0:
                    self.logger.warning(f"Invalid price for {symbol}: {current_price}, using entry price")
                    current_price = position['entry_price']
                
                # Calculate position value
                volume = position['volume']
                value = volume * current_price
                equity += value
            
            return equity
            
        except Exception as e:
            self.logger.error(f"Error calculating total equity: {str(e)}")
            return self.balance.get('ZUSD', 0)

    def get_performance_metrics(self):
        """Get detailed performance metrics for the portfolio with accurate P&L calculation."""
        try:
            # Calculate current equity
            current_equity = self.calculate_total_equity()
            
            # Get trade history P&L
            trade_pnl = 0.0
            win_count = 0
            loss_count = 0
            
            # Calculate P&L from completed trades rather than using initial capital
            for trade in self.trade_history:
                if trade.get('type') in ['sell', 'partial_sell'] and 'profit_loss' in trade:
                    profit_loss = trade.get('profit_loss', 0)
                    trade_pnl += profit_loss
                    
                    if profit_loss > 0:
                        win_count += 1
                    elif profit_loss < 0:
                        loss_count += 1
            
            # Get initial capital - for reference only, not for P&L calculation
            initial_capital = self.initial_capital
            
            # For percentage calculation, use appropriate base value
            # If trade P&L is positive, calculate percentage based on (initial_capital - trade_pnl)
            # This gives an accurate percentage that matches the analysis script
            if trade_pnl != 0:
                pnl_percentage = (trade_pnl / (max(current_equity - trade_pnl, initial_capital * 0.1))) * 100
            else:
                pnl_percentage = 0.0
            
            # Calculate additional metrics
            max_equity = max([entry['equity'] for entry in self.portfolio_history]) if self.portfolio_history else initial_capital
            drawdown = (max_equity - current_equity) / max_equity * 100 if max_equity > 0 else 0
            
            # Calculate win rate
            total_trades = win_count + loss_count
            win_rate = win_count / total_trades * 100 if total_trades > 0 else 0
            
            # Calculate average profit and loss
            winning_trades = [t for t in self.trade_history if t.get('type') in ['sell', 'partial_sell'] and t.get('profit_loss', 0) > 0]
            losing_trades = [t for t in self.trade_history if t.get('type') in ['sell', 'partial_sell'] and t.get('profit_loss', 0) < 0]
            
            avg_profit = sum(t.get('profit_loss', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(abs(t.get('profit_loss', 0)) for t in losing_trades) / len(losing_trades) if losing_trades else 0
            
            # Calculate profit factor
            profit_sum = sum(t.get('profit_loss', 0) for t in winning_trades)
            loss_sum = sum(abs(t.get('profit_loss', 0)) for t in losing_trades)
            profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')
            
            # Calculate Sharpe ratio (if we have enough data)
            sharpe_ratio = 0
            if len(self.portfolio_history) > 30:
                try:
                    # Get daily returns
                    equity_values = [entry['equity'] for entry in self.portfolio_history]
                    returns = np.diff(equity_values) / np.array(equity_values[:-1])
                    
                    # Calculate annualized Sharpe
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    risk_free_rate = 0.02 / 365  # Assume 2% annual risk-free rate
                    
                    if std_return > 0:
                        sharpe_ratio = (avg_return - risk_free_rate) / std_return * np.sqrt(365)
                except Exception as e:
                    self.logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            
            # Calculate max consecutive losses
            consecutive_losses = 0
            max_consecutive_losses = 0
            
            for trade in self.trade_history:
                if trade.get('type') in ['sell', 'partial_sell']:
                    if trade.get('profit_loss', 0) < 0:
                        consecutive_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    else:
                        consecutive_losses = 0
            
            # Portfolio composition
            positions_value = current_equity - self.balance.get('ZUSD', 0)
            cash_percentage = self.balance.get('ZUSD', 0) / current_equity * 100 if current_equity > 0 else 100
            
            # Position details
            position_details = {}
            for symbol, position in self.positions.items():
                # Get current price
                if hasattr(self, 'price_cache') and f"price_{symbol}" in self.price_cache:
                    current_price = self.price_cache[f"price_{symbol}"]["price"]
                else:
                    try:
                        ticker = self.api.query_public('Ticker', {'pair': symbol})
                        if ticker and 'result' in ticker and symbol in ticker['result']:
                            current_price = float(ticker['result'][symbol]['c'][0])
                        else:
                            current_price = position['entry_price']  # Use entry price as fallback
                    except:
                        current_price = position['entry_price']  # Use entry price as fallback
                
                volume = position['volume']
                value = volume * current_price
                entry_value = volume * position['entry_price']
                unrealized_pnl = value - entry_value
                unrealized_pnl_pct = (unrealized_pnl / entry_value) * 100 if entry_value > 0 else 0
                
                position_details[symbol] = {
                    'volume': volume,
                    'entry_price': position['entry_price'],
                    'current_price': current_price,
                    'value': value,
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_pct': unrealized_pnl_pct,
                    'age_hours': (datetime.now() - position['entry_time']).total_seconds() / 3600,
                    'stop_loss': position.get('stop_loss'),
                    'take_profit': position.get('take_profit')
                }
            
            # Compile all metrics - using trade_pnl instead of (current_equity - initial_capital)
            metrics = {
                'current_equity': current_equity,
                'initial_capital': initial_capital,
                'total_pnl': trade_pnl,  # This is the key change - using trade-based P&L
                'pnl_percentage': pnl_percentage,
                'max_equity': max_equity,
                'drawdown': drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'winning_trades': win_count,
                'losing_trades': loss_count,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_consecutive_losses': max_consecutive_losses,
                'cash_balance': self.balance.get('ZUSD', 0),
                'cash_percentage': cash_percentage,
                'positions_value': positions_value,
                'position_count': len(self.positions),
                'positions': position_details
            }
            
            # Log the P&L calculation method
            self.logger.debug(f"P&L calculation: trade_pnl={trade_pnl:.2f}, current_equity={current_equity:.2f}, initial_capital={initial_capital:.2f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            traceback.print_exc()
            
            # Return basic metrics on error
            return {
                'current_equity': self.balance.get('ZUSD', 0),
                'initial_capital': self.initial_capital,
                'total_pnl': 0.0,
                'pnl_percentage': 0.0,
                'position_count': len(self.positions)
            }

    async def run(self):
        """Main trading loop with improved error handling, cycle timing, and state persistence."""
        try:
            self.logger.info("\n=== KRYPTOS TRADING BOT STARTED ===")
            
            # Calculate current equity
            total_equity = self.calculate_total_equity()
            
            self.logger.info(f"Initial balance: ${self.balance.get('ZUSD', 0):.2f}")
            self.logger.info(f"Initial equity: ${total_equity:.2f}")
            self.logger.info(f"Trading pairs: {list(self.symbols.keys())}")
            
            # Wait for model training to complete before starting trading
            if not self.is_models_loaded:
                self.logger.info("Models not trained yet. Starting initial training before trading.")
                training_success = await self._train_models_and_wait()
                if not training_success:
                    self.logger.error("Model training failed. Bot will not start trading without models.")
                    self.logger.info("Please fix model training issues and restart the bot.")
                    return
            
            # Verify components are working
            self._verify_components_working()
            
            # Main trading loop
            cycle_count = 0
            last_metrics_time = time.time()
            last_state_save_time = time.time()
            
            while self.is_running:
                try:
                    # Record cycle start time for accurate sleep calculation
                    cycle_start = time.time()
                    
                    cycle_count += 1
                    self.logger.info(f"\n=== Trading Cycle {cycle_count} ===")
                    self.logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Log metrics periodically (every 10 cycles)
                    if cycle_count % 10 == 0 or time.time() - last_metrics_time > 1800:
                        metrics = self.get_performance_metrics()
                        self.logger.info(f"Current Equity: ${metrics['current_equity']:.2f}")
                        self.logger.info(f"P&L: ${metrics['total_pnl']:.2f} ({metrics['pnl_percentage']:.2f}%)")
                        
                        if metrics.get('total_trades', 0) > 0:
                            self.logger.info(f"Win Rate: {metrics.get('win_rate', 0):.1f}% "
                                        f"({int(metrics.get('win_rate', 0)*metrics.get('total_trades', 0)/100)}/"
                                        f"{metrics.get('total_trades', 0)} trades)")
                            
                        last_metrics_time = time.time()
                    
                    # Process each trading pair
                    for symbol in self.symbols:
                        try:
                            self.logger.info(f"\nAnalyzing {symbol}...")
                            
                            # Skip if we already have a position and aren't looking to add
                            if symbol in self.positions and not self.dca_enabled:
                                self.logger.info(f"Already have a position for {symbol}, monitoring only")
                                continue
                            
                            # Get historical data
                            df = await self.get_historical_data(symbol)
                            if df is None or len(df) < 20:
                                self.logger.warning(f"No data available for {symbol}, skipping")
                                continue
                            
                            # Generate trading signal
                            signal = await self.generate_trading_signal(symbol, df)
                            
                            # Check if we should trade now
                            if signal['action'] == 'buy' and symbol not in self.positions:
                                # Check trade timing (cooldown period)
                                current_time = datetime.now()
                                last_trade = self.last_trade_time.get(symbol)
                                
                                if last_trade and (current_time - last_trade).total_seconds() < self.trade_cooldown:
                                    cooldown_seconds = self.trade_cooldown
                                    elapsed = (current_time - last_trade).total_seconds()
                                    self.logger.info(f"Trade cooldown active: {elapsed:.0f}/{cooldown_seconds}s elapsed")
                                    continue
                                
                                # Count recent trades (last hour) to avoid overtrading
                                hour_ago = current_time - timedelta(hours=1)
                                recent_trades = [t for t in self.trade_history 
                                            if t.get('timestamp', datetime.now()) > hour_ago]
                                
                                if len(recent_trades) >= self.max_trades_per_hour:
                                    self.logger.info(f"Max trades per hour reached ({self.max_trades_per_hour})")
                                    continue
                            
                            # Execute trades based on signal
                            if signal['action'] == 'buy' and symbol not in self.positions:
                                if self.dca_enabled:
                                    self.logger.info(f"Using DCA strategy for {symbol}")
                                    trade_result = await self.execute_dca_trade(symbol, signal)
                                else:
                                    trade_result = await self.execute_trade(symbol, signal)
                                
                                if trade_result:
                                    self.logger.info(f"Trade executed successfully for {symbol}")
                                    # Save state after each trade for extra safety
                                    self.save_state()
                                else:
                                    self.logger.info(f"No trade executed for {symbol}")
                                    
                            elif signal['action'] == 'sell' and symbol in self.positions:
                                trade_result = await self.close_position(
                                    symbol, 
                                    await self.get_latest_price(symbol), 
                                    'signal'
                                )
                                
                                if trade_result:
                                    self.logger.info(f"Position closed for {symbol} based on signal")
                                    # Save state after each trade for extra safety
                                    self.save_state()
                                else:
                                    self.logger.info(f"Failed to close position for {symbol}")
                            else:
                                self.logger.info(f"No action needed for {symbol}")
                            
                            # Delay between symbols to prevent rate limiting
                            await asyncio.sleep(2)
                            
                        except Exception as symbol_error:
                            self.logger.error(f"Error processing {symbol}: {str(symbol_error)}")
                            traceback.print_exc()
                            await asyncio.sleep(1)
                    
                    # Monitor positions
                    self.logger.info("\nMonitoring positions...")
                    await self.monitor_positions()
                    
                    # Check DCA plans
                    if self.dca_enabled and self.dca_plans:
                        self.logger.info(f"Checking {len(self.dca_plans)} active DCA plans")
                        await self.check_dca_plans()
                    
                    # Save state every 5 minutes at minimum
                    current_time = time.time()
                    if cycle_count % 5 == 0 or current_time - last_state_save_time > 300:
                        self.save_state()
                        self.logger.info("Trading state saved")
                        last_state_save_time = current_time
                    
                    # Calculate time spent in cycle
                    cycle_duration = time.time() - cycle_start
                    
                    # Determine wait time for next cycle (target 120 seconds per cycle)
                    wait_time = max(5, 120 - cycle_duration)  # At least 5 seconds
                    
                    self.logger.info(f"Waiting {wait_time:.0f} seconds for next cycle...")
                    await asyncio.sleep(wait_time)
                    
                except Exception as cycle_error:
                    self.logger.error(f"Error in trading cycle: {str(cycle_error)}")
                    traceback.print_exc()
                    
                    # Use progressive backoff on errors
                    wait_time = min(300, 30 * (cycle_count % 5 + 1))  # 30 to 300 seconds
                    self.logger.info(f"Waiting {wait_time:.0f} seconds before next cycle...")
                    await asyncio.sleep(wait_time)
            
            self.logger.info("Trading bot stopped gracefully")
            await self.cleanup()  # Ensure proper cleanup
            
        except Exception as e:
            self.logger.error(f"Fatal error in trading bot: {str(e)}")
            traceback.print_exc()
            await self.cleanup()  # Try to clean up even after error

    def _verify_components_working(self):
        """Verify that all critical components are working properly."""
        try:
            self.logger.info("Verifying critical components...")
            
            # Check API connection
            try:
                ticker = self.api.query_public('Ticker', {'pair': list(self.symbols.keys())[0]})
                if 'error' in ticker and ticker['error']:
                    self.logger.warning(f"API connection warning: {ticker['error']}")
                else:
                    self.logger.info("API connection verified")
            except Exception as e:
                self.logger.error(f"API connection error: {str(e)}")
            
            # Check database connection
            try:
                conn = sqlite3.connect(self.db_name)
                c = conn.cursor()
                c.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = c.fetchall()
                conn.close()
                self.logger.info(f"Database connection verified, {len(tables)} tables found")
            except Exception as e:
                self.logger.error(f"Database connection error: {str(e)}")
            
            # Check model status
            if self.is_models_loaded:
                self.logger.info("ML/AI models loaded successfully")
            else:
                self.logger.warning("ML/AI models not loaded, will train on first run")
            
            # Check required directories
            for dir_name in ['data', 'models', 'cache', 'logs', 'backtest_results']:
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name, exist_ok=True)
                    self.logger.warning(f"Created missing directory: {dir_name}")
                else:
                    self.logger.info(f"Directory verified: {dir_name}")
            
            # Check risk management configuration
            if hasattr(self, 'risk_management'):
                self.logger.info("Risk management module configured")
                if self.risk_management.get('active_risk_protection', False):
                    self.logger.warning("Risk protection is active with factor: " + 
                                    str(self.risk_management.get('risk_reduction_factor', 1.0)))
            else:
                self.logger.warning("Risk management not properly initialized")
            
            # Check performance analytics
            if hasattr(self, 'performance_analytics'):
                self.logger.info("Performance analytics tracking enabled")
            
            # Verify all symbols have correct thresholds
            missing_thresholds = [s for s in self.symbols if s not in self.buy_thresholds]
            if missing_thresholds:
                self.logger.warning(f"Missing buy thresholds for symbols: {missing_thresholds}")
                # Set defaults for missing thresholds
                for s in missing_thresholds:
                    self.buy_thresholds[s] = 0.52
                    self.logger.info(f"Set default buy threshold 0.52 for {s}")
            
            self.logger.info("Component verification completed")
            
        except Exception as e:
            self.logger.error(f"Error verifying components: {str(e)}")

    async def backtest(self, start_date, end_date, parameters=None):
        """Run backtesting for the strategy between given dates.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            parameters: Optional dictionary of parameters to override defaults
            
        Returns:
            Dictionary with backtest results
        """
        try:
            self.logger.info(f"Starting backtest from {start_date} to {end_date}")
            
            # Enable backtesting mode
            self.is_backtesting = True
            
            # Store date range for filtering in get_historical_data
            self.backtest_start_date = start_date
            self.backtest_end_date = end_date
            
            # Save original parameters to restore later
            original_params = {
                'balance': self.balance.copy(),
                'positions': self.positions.copy(),
                'trade_history': self.trade_history.copy(),
                'portfolio_history': self.portfolio_history.copy(),
                'buy_thresholds': {k: v for k, v in self.buy_thresholds.items()},
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct,
                'trailing_stop_pct': self.trailing_stop_pct,
                'max_position_size': self.max_position_size,
                'dca_enabled': self.dca_enabled,
                'dca_parts': self.dca_parts,
                'dca_time_between': self.dca_time_between,
                'signal_weights': self.signal_weights.copy() if hasattr(self, 'signal_weights') else None
            }
            
            # Apply custom parameters if provided
            if parameters:
                self.logger.info(f"Applying custom parameters: {parameters}")
                
                if 'BUY_THRESHOLD_COEF' in parameters:
                    for symbol in self.buy_thresholds:
                        # Apply the coefficient to current thresholds
                        self.buy_thresholds[symbol] *= parameters['BUY_THRESHOLD_COEF']
                
                if 'STOP_LOSS_PCT' in parameters:
                    self.stop_loss_pct = parameters['STOP_LOSS_PCT']
                    
                if 'TAKE_PROFIT_PCT' in parameters:
                    self.take_profit_pct = parameters['TAKE_PROFIT_PCT']
                    
                if 'TRAILING_STOP_PCT' in parameters:
                    self.trailing_stop_pct = parameters['TRAILING_STOP_PCT']
                    
                if 'MAX_POSITION_SIZE' in parameters:
                    self.max_position_size = parameters['MAX_POSITION_SIZE']
                    
                if 'DCA_ENABLED' in parameters:
                    self.dca_enabled = parameters['DCA_ENABLED']
                    
                if 'DCA_PARTS' in parameters:
                    self.dca_parts = parameters['DCA_PARTS']
                    
                if 'DCA_TIME_BETWEEN' in parameters:
                    self.dca_time_between = parameters['DCA_TIME_BETWEEN']
                    
                # Handle signal weights if provided
                if 'SIGNAL_WEIGHTS' in parameters and isinstance(parameters['SIGNAL_WEIGHTS'], dict):
                    for signal_type, weight in parameters['SIGNAL_WEIGHTS'].items():
                        if signal_type in self.signal_weights:
                            self.signal_weights[signal_type] = weight
                    
                    # Normalize weights to sum to 1.0
                    pattern_weight = self.signal_weights.get('pattern', 0)
                    total_other_weights = sum(v for k, v in self.signal_weights.items() if k != 'pattern')
                    
                    if total_other_weights > 0:
                        scaling_factor = (1.0 - pattern_weight) / total_other_weights
                        for k in self.signal_weights:
                            if k != 'pattern':
                                self.signal_weights[k] *= scaling_factor
            
            # Reset trading state
            self.balance = {'ZUSD': self.initial_capital}
            self.positions = {}
            self.trade_history = []
            self.portfolio_history = [{
                'timestamp': datetime.strptime(start_date, '%Y-%m-%d'),
                'balance': self.initial_capital,
                'equity': self.initial_capital,
                'drawdown': 0.0
            }]
            self.dca_plans = []
            self.last_trade_time = {}
            
            # Parse dates
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Calculate number of days
            days = (end_dt - start_dt).days
            if days <= 0:
                raise ValueError("End date must be after start date")
                
            self.logger.info(f"Backtesting period: {days} days")
            
            # Create date range for testing
            current_dt = start_dt
            
            # Load historical data for all symbols
            historical_data = {}
            
            for symbol in self.symbols:
                self.logger.info(f"Loading historical data for {symbol}")
                
                # Get historical data with sufficient lookback for indicators
                df = await self.get_historical_data(symbol, interval=self.timeframe, lookback_days=days+30)
                
                if df is not None and len(df) > 2:
                    # Check if we have enough data points
                    if isinstance(df.index, pd.DatetimeIndex):
                        # Filter for backtest period to check data availability
                        mask = (df.index.date >= start_dt.date()) & (df.index.date <= end_dt.date())
                        filtered_df = df.loc[mask]
                        
                        if len(filtered_df) > 0:
                            # Calculate indicators
                            df = self.calculate_indicators(df)
                            
                            # Add symbol column
                            df['symbol'] = symbol
                            
                            historical_data[symbol] = df
                            self.logger.info(f"Loaded {len(filtered_df)} data points for {symbol} in backtest range")
                        else:
                            self.logger.warning(f"No data available for {symbol} in the specified date range")
                    else:
                        self.logger.warning(f"DataFrame index is not DatetimeIndex for {symbol}")
                else:
                    self.logger.warning(f"Insufficient data for {symbol}, skipping")
                
                await asyncio.sleep(1)  # Prevent rate limiting
            
            # Check if we have data for at least one symbol
            if not historical_data:
                raise ValueError("No historical data available for any symbol")
            
            # Track backtest progress
            total_days = days
            days_completed = 0
            last_progress_print = 0
            
            # Iterate through each day
            while current_dt <= end_dt:
                day_str = current_dt.strftime('%Y-%m-%d')
                
                # Print progress every 5%
                days_completed += 1
                progress = days_completed / total_days * 100
                if progress - last_progress_print >= 5:
                    self.logger.info(f"Backtest progress: {progress:.1f}% completed")
                    last_progress_print = progress
                
                # Process each symbol
                for symbol, data in historical_data.items():
                    # Get data for current day - improved method
                    if isinstance(data.index, pd.DatetimeIndex):
                        day_data = data[data.index.date == current_dt.date()]
                    else:
                        day_data = data[data.index.strftime('%Y-%m-%d') == day_str]
                    
                    if day_data.empty:
                        continue
                    
                    # Log symbol processing for debugging
                    self.logger.debug(f"Processing {symbol} for {day_str} - {len(day_data)} data points")
                    
                    # Use last price of the day for simplicity
                    last_price = day_data['close'].iloc[-1]
                    
                    # Cache price for the day
                    self.price_cache[f"price_{symbol}"] = {
                        'price': last_price,
                        'timestamp': time.time()
                    }
                    
                    # Generate trading signal - use data up to current day only
                    # Use proper data slicing to avoid future data leakage
                    current_day_end = day_data.index[-1]
                    historical_data_until_day = data[data.index <= current_day_end]
                    
                    signal = await self.generate_trading_signal(symbol, historical_data_until_day)
                    
                    # Execute trades based on signal
                    if signal['action'] == 'buy' and symbol not in self.positions:
                        if self.dca_enabled:
                            await self.execute_dca_trade(symbol, signal)
                        else:
                            await self.execute_trade(symbol, signal)
                            
                    elif signal['action'] == 'sell' and symbol in self.positions:
                        await self.close_position(symbol, last_price, 'signal')
                
                # Monitor positions for the day
                await self.monitor_positions()
                
                # Check DCA plans
                if self.dca_enabled and self.dca_plans:
                    await self.check_dca_plans()
                
                # Update portfolio value for the day
                current_equity = self.calculate_total_equity()
                max_equity = max([entry['equity'] for entry in self.portfolio_history], default=self.initial_capital)
                drawdown = (max_equity - current_equity) / max_equity if max_equity > 0 else 0
                
                self.portfolio_history.append({
                    'timestamp': current_dt,
                    'balance': self.balance['ZUSD'],
                    'equity': current_equity,
                    'drawdown': drawdown
                })
                
                # Move to next day
                current_dt += timedelta(days=1)
            
            # Calculate backtest results
            results = self._calculate_backtest_results(start_date, end_date, parameters)
            
            # Restore original parameters
            self.balance = original_params['balance']
            self.positions = original_params['positions']
            self.trade_history = original_params['trade_history']
            self.portfolio_history = original_params['portfolio_history']
            self.buy_thresholds = original_params['buy_thresholds']
            self.stop_loss_pct = original_params['stop_loss_pct']
            self.take_profit_pct = original_params['take_profit_pct']
            self.trailing_stop_pct = original_params['trailing_stop_pct']
            self.max_position_size = original_params['max_position_size']
            self.dca_enabled = original_params['dca_enabled']
            self.dca_parts = original_params['dca_parts']
            self.dca_time_between = original_params['dca_time_between']
            if original_params['signal_weights']:
                self.signal_weights = original_params['signal_weights']
            
            # Disable backtesting mode
            self.is_backtesting = False
            
            self.logger.info(f"Backtest completed: {results['total_return']:.2f}% return, Sharpe: {results['sharpe_ratio']:.2f}")
            
            # Save results to database
            self._save_backtest_results(results)
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error in backtesting: {str(e)}")
            traceback.print_exc()
            
            # Disable backtesting mode
            self.is_backtesting = False
            
            # Restore original parameters if defined
            if 'original_params' in locals():
                self.balance = original_params['balance']
                self.positions = original_params['positions']
                self.trade_history = original_params['trade_history']
                self.portfolio_history = original_params['portfolio_history']
                self.buy_thresholds = original_params['buy_thresholds']
                self.stop_loss_pct = original_params['stop_loss_pct']
                self.take_profit_pct = original_params['take_profit_pct']
                self.trailing_stop_pct = original_params['trailing_stop_pct']
                self.max_position_size = original_params['max_position_size']
                self.dca_enabled = original_params['dca_enabled']
                self.dca_parts = original_params['dca_parts']
                self.dca_time_between = original_params['dca_time_between']
                if original_params['signal_weights']:
                    self.signal_weights = original_params['signal_weights']
            
            return {
                'error': str(e),
                'success': False,
                'initial_capital': self.initial_capital,
                'final_equity': self.initial_capital,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'profit_factor': 0.0
            }

    def _calculate_backtest_results(self, start_date, end_date, parameters):
        """Calculate performance metrics from backtest."""
        try:
            # Get initial and final equity
            initial_equity = self.initial_capital
            final_equity = self.calculate_total_equity()
            
            # Calculate overall return
            total_return = ((final_equity - initial_equity) / initial_equity) * 100
            
            # Calculate daily returns
            equity_series = pd.Series(
                [entry['equity'] for entry in self.portfolio_history],
                index=[entry['timestamp'] for entry in self.portfolio_history]
            )
            
            daily_returns = equity_series.pct_change().dropna()
            
            # Calculate Sharpe ratio
            if len(daily_returns) > 1:
                avg_daily_return = daily_returns.mean()
                daily_std = daily_returns.std()
                risk_free_rate = 0.02 / 365  # Assume 2% annual risk-free rate
                
                if daily_std > 0:
                    sharpe_ratio = (avg_daily_return - risk_free_rate) / daily_std * np.sqrt(252)
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns / running_max - 1) * 100
            max_drawdown = abs(drawdown.min())
            
            # Calculate trade metrics
            all_trades = [t for t in self.trade_history if t.get('type') in ['sell', 'partial_sell']]
            wins = sum(1 for t in all_trades if t.get('profit_loss', 0) > 0)
            losses = sum(1 for t in all_trades if t.get('profit_loss', 0) < 0)
            total_trades = len(all_trades)
            
            win_rate = wins / total_trades * 100 if total_trades > 0 else 0
            
            # Calculate profit factor
            profit_sum = sum(t.get('profit_loss', 0) for t in all_trades if t.get('profit_loss', 0) > 0)
            loss_sum = sum(abs(t.get('profit_loss', 0)) for t in all_trades if t.get('profit_loss', 0) < 0)
            profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')
            
            # Calculate strategy metrics by symbol
            symbol_metrics = {}
            for symbol in self.symbols:
                symbol_trades = [t for t in all_trades if t.get('symbol') == symbol]
                if symbol_trades:
                    symbol_wins = sum(1 for t in symbol_trades if t.get('profit_loss', 0) > 0)
                    symbol_losses = sum(1 for t in symbol_trades if t.get('profit_loss', 0) < 0)
                    symbol_win_rate = symbol_wins / len(symbol_trades) * 100 if len(symbol_trades) > 0 else 0
                    symbol_pnl = sum(t.get('profit_loss', 0) for t in symbol_trades)
                    
                    symbol_metrics[symbol] = {
                        'trades': len(symbol_trades),
                        'wins': symbol_wins,
                        'losses': symbol_losses,
                        'win_rate': symbol_win_rate,
                        'total_pnl': symbol_pnl
                    }
            
            # Compile results
            results = {
                'start_date': start_date,
                'end_date': end_date,
                'days': len(equity_series.index.unique()),
                'initial_capital': initial_equity,
                'final_equity': final_equity,
                'total_return': total_return,
                'annualized_return': total_return / len(equity_series.index.unique()) * 365,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'winning_trades': wins,
                'losing_trades': losses,
                'profit_factor': profit_factor,
                'symbol_metrics': symbol_metrics,
                'parameters': parameters,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating backtest results: {str(e)}")
            traceback.print_exc()
            
            return {
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': self.initial_capital,
                'final_equity': self.calculate_total_equity(),
                'total_return': ((self.calculate_total_equity() - self.initial_capital) / self.initial_capital) * 100,
                'error': str(e)
            }

    def _save_backtest_results(self, results):
        """Save backtest results to database."""
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            
            # Convert parameters to JSON
            if results.get('parameters'):
                params_json = json.dumps(results['parameters'])
            else:
                params_json = '{}'
            
            # Insert backtest results
            c.execute('''
                INSERT INTO backtest_results
                (start_date, end_date, initial_capital, final_equity, total_return,
                sharpe_ratio, max_drawdown, win_rate, profit_factor, parameters, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                results.get('start_date', ''),
                results.get('end_date', ''),
                results.get('initial_capital', 0),
                results.get('final_equity', 0),
                results.get('total_return', 0),
                results.get('sharpe_ratio', 0),
                results.get('max_drawdown', 0),
                results.get('win_rate', 0),
                results.get('profit_factor', 0),
                params_json,
                results.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            ))
            
            conn.commit()
            
            # Save detailed backtest results to JSON file
            results_dir = 'backtest_results'
            os.makedirs(results_dir, exist_ok=True)
            results_file = os.path.join(results_dir, f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            conn.close()
            
            self.logger.info(f"Backtest results saved to database and {results_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving backtest results: {str(e)}")

    async def optimize_parameters(self, start_date, end_date, iterations=20):
        """Optimize trading parameters using grid search."""
        try:
            self.logger.info(f"Starting parameter optimization with {iterations} iterations")
            
            # Define parameter ranges for optimization
            param_ranges = {
                'BUY_THRESHOLD_COEF': [0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04],
                'STOP_LOSS_PCT': [0.005, 0.007, 0.01, 0.015],
                'TAKE_PROFIT_PCT': [0.015, 0.02, 0.025, 0.03, 0.04],
                'TRAILING_STOP_PCT': [0.005, 0.008, 0.01, 0.015],
                'MAX_POSITION_SIZE': [0.05, 0.08, 0.1, 0.15],
                'DCA_ENABLED': [True, False],
                'DCA_PARTS': [2, 3],
                'DCA_TIME_BETWEEN': [30, 60, 120],
                'SIGNAL_WEIGHTS': [
                    {'ml': 0.30, 'ai': 0.20, 'basic': 0.30, 'sentiment': 0.20, 'pattern': 0.0},
                    {'ml': 0.25, 'ai': 0.25, 'basic': 0.25, 'sentiment': 0.25, 'pattern': 0.0},
                    {'ml': 0.20, 'ai': 0.30, 'basic': 0.20, 'sentiment': 0.30, 'pattern': 0.0}
                ]
            }
            
            # Track best results
            best_return = -float('inf')
            best_sharpe = -float('inf')
            best_params_return = None
            best_params_sharpe = None
            
            # Store all results for analysis
            all_results = []
            
            # Use random search
            for i in range(iterations):
                # Randomly select parameters
                params = {
                    'BUY_THRESHOLD_COEF': random.choice(param_ranges['BUY_THRESHOLD_COEF']),
                    'STOP_LOSS_PCT': random.choice(param_ranges['STOP_LOSS_PCT']),
                    'TAKE_PROFIT_PCT': random.choice(param_ranges['TAKE_PROFIT_PCT']),
                    'TRAILING_STOP_PCT': random.choice(param_ranges['TRAILING_STOP_PCT']),
                    'MAX_POSITION_SIZE': random.choice(param_ranges['MAX_POSITION_SIZE']),
                    'DCA_ENABLED': random.choice(param_ranges['DCA_ENABLED'])
                }
                
                # Only add DCA params if enabled
                if params['DCA_ENABLED']:
                    params['DCA_PARTS'] = random.choice(param_ranges['DCA_PARTS'])
                    params['DCA_TIME_BETWEEN'] = random.choice(param_ranges['DCA_TIME_BETWEEN'])
                
                # Add signal weights occasionally (33% of iterations)
                if random.random() < 0.33:
                    params['SIGNAL_WEIGHTS'] = random.choice(param_ranges['SIGNAL_WEIGHTS'])
                
                self.logger.info(f"Optimization iteration {i+1}/{iterations}")
                self.logger.info(f"Testing parameters: {params}")
                
                # Run backtest with these parameters
                results = await self.backtest(start_date, end_date, params)
                
                # Add to all results
                all_results.append({
                    'parameters': params,
                    'results': results
                })
                
                # Check if this is the best result
                if results.get('total_return', -float('inf')) > best_return:
                    best_return = results.get('total_return', -float('inf'))
                    best_params_return = params.copy()
                    self.logger.info(f"New best return: {best_return:.2f}%")
                
                if results.get('sharpe_ratio', -float('inf')) > best_sharpe:
                    best_sharpe = results.get('sharpe_ratio', -float('inf'))
                    best_params_sharpe = params.copy()
                    self.logger.info(f"New best Sharpe ratio: {best_sharpe:.2f}")
                
                # Save intermediate results
                self._save_optimization_results(all_results, best_params_return, best_params_sharpe)
            
            self.logger.info(f"Parameter optimization completed")
            self.logger.info(f"Best return parameters: {best_params_return}")
            self.logger.info(f"Best return: {best_return:.2f}%")
            self.logger.info(f"Best Sharpe parameters: {best_params_sharpe}")
            self.logger.info(f"Best Sharpe: {best_sharpe:.2f}")
            
            # Return both sets of optimal parameters
            return {
                'best_return': {
                    'parameters': best_params_return,
                    'return': best_return
                },
                'best_sharpe': {
                    'parameters': best_params_sharpe,
                    'sharpe': best_sharpe
                },
                'all_results': all_results
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {str(e)}")
            traceback.print_exc()
            
            return {
                'error': str(e),
                'success': False
            }

    def _save_optimization_results(self, all_results, best_return_params, best_sharpe_params):
        """Save optimization results to disk."""
        try:
            results_dir = 'backtest_results'
            os.makedirs(results_dir, exist_ok=True)
            results_file = os.path.join(results_dir, f"optimization_{datetime.now().strftime('%Y%m%d')}.json")
            
            # Prepare data for serialization
            serializable_results = []
            for result in all_results:
                # Convert any non-serializable data
                serializable_result = {
                    'parameters': result['parameters'],
                    'results': {k: v for k, v in result['results'].items() if k != 'symbol_metrics'}
                }
                serializable_results.append(serializable_result)
            
            # Save results
            with open(results_file, 'w') as f:
                json.dump({
                    'all_results': serializable_results,
                    'best_return_params': best_return_params,
                    'best_sharpe_params': best_sharpe_params,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }, f, indent=2, default=str)
                
            self.logger.info(f"Optimization results saved to {results_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving optimization results: {str(e)}")

    async def apply_optimized_parameters(self, params):
        """Apply optimized parameters from backtesting."""
        try:
            self.logger.info("Applying optimized parameters...")
            
            # Apply threshold coefficient to all symbols
            if 'BUY_THRESHOLD_COEF' in params:
                coef = params['BUY_THRESHOLD_COEF']
                for symbol in self.buy_thresholds:
                    original = self.buy_thresholds[symbol]
                    self.buy_thresholds[symbol] = original * coef
                    self.logger.info(f"Updated {symbol} threshold: {original:.3f} -> {self.buy_thresholds[symbol]:.3f}")
            
            # Apply risk parameters
            if 'STOP_LOSS_PCT' in params:
                self.stop_loss_pct = params['STOP_LOSS_PCT']
                self.logger.info(f"Updated stop loss to {self.stop_loss_pct*100:.2f}%")
                
            if 'TAKE_PROFIT_PCT' in params:
                self.take_profit_pct = params['TAKE_PROFIT_PCT']
                self.logger.info(f"Updated take profit to {self.take_profit_pct*100:.2f}%")
                
            if 'TRAILING_STOP_PCT' in params:
                self.trailing_stop_pct = params['TRAILING_STOP_PCT']
                self.logger.info(f"Updated trailing stop to {self.trailing_stop_pct*100:.2f}%")
            
            # Update position sizing
            if 'MAX_POSITION_SIZE' in params:
                self.max_position_size = params['MAX_POSITION_SIZE']
                self.logger.info(f"Updated max position size to {self.max_position_size*100:.2f}%")
            
            # Update DCA settings
            if 'DCA_ENABLED' in params:
                self.dca_enabled = params['DCA_ENABLED']
                self.logger.info(f"DCA {'enabled' if self.dca_enabled else 'disabled'}")
                
            if 'DCA_PARTS' in params and self.dca_enabled:
                self.dca_parts = params['DCA_PARTS']
                self.logger.info(f"Updated DCA parts to {self.dca_parts}")
                
            if 'DCA_TIME_BETWEEN' in params and self.dca_enabled:
                self.dca_time_between = params['DCA_TIME_BETWEEN']
                self.logger.info(f"Updated DCA time between parts to {self.dca_time_between} minutes")
            
            # Update signal weights if provided
            if 'SIGNAL_WEIGHTS' in params and isinstance(params['SIGNAL_WEIGHTS'], dict):
                for signal_type, weight in params['SIGNAL_WEIGHTS'].items():
                    if signal_type in self.signal_weights:
                        old_weight = self.signal_weights[signal_type]
                        self.signal_weights[signal_type] = weight
                        self.logger.info(f"Updated {signal_type} signal weight: {old_weight:.2f} -> {weight:.2f}")
                
                # Normalize weights to sum to 1.0
                pattern_weight = self.signal_weights.get('pattern', 0)
                total_other_weights = sum(v for k, v in self.signal_weights.items() if k != 'pattern')
                
                if total_other_weights > 0:
                    scaling_factor = (1.0 - pattern_weight) / total_other_weights
                    for k in self.signal_weights:
                        if k != 'pattern':
                            self.signal_weights[k] *= scaling_factor
            
            # Save updated config
            self._save_config()
            
            # Save state
            self.save_state()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying optimized parameters: {str(e)}")
            return False

    def _save_config(self):
        """Save current configuration to file."""
        try:
            config = {
                "initial_capital": self.initial_capital,
                "symbols": self.symbols,
                "thresholds": {
                    "buy": self.buy_thresholds
                },
                "risk": {
                    "max_drawdown": self.max_drawdown,
                    "trailing_stop_pct": self.trailing_stop_pct,
                    "max_trades_per_hour": self.max_trades_per_hour,
                    "trade_cooldown": self.trade_cooldown,
                    "max_position_size": self.max_position_size,
                    "min_position_value": self.min_position_value,
                    "max_total_risk": self.max_total_risk,
                    "stop_loss_pct": self.stop_loss_pct,
                    "take_profit_pct": self.take_profit_pct
                },
                "technical": {
                    "sma_short": self.sma_short,
                    "sma_long": self.sma_long,
                    "rsi_period": self.rsi_period,
                    "rsi_oversold": self.rsi_oversold,
                    "rsi_overbought": self.rsi_overbought,
                    "macd_fast": self.macd_fast,
                    "macd_slow": self.macd_slow,
                    "macd_signal": self.macd_signal
                },
                "dca": {
                    "enabled": self.dca_enabled,
                    "parts": self.dca_parts,
                    "time_between": self.dca_time_between
                },
                "api": {
                    "timeframe": self.timeframe,
                    "retry_delay": self.api_retry_delay,
                    "max_retry_delay": self.max_retry_delay
                },
                "signal_weights": self.signal_weights
            }
            
            # Ensure directory exists
            os.makedirs('config', exist_ok=True)
            
            # Save to file
            with open('config/kryptos_config.json', 'w') as f:
                json.dump(config, f, indent=4, default=str)
                
            self.logger.info("Configuration saved to config/kryptos_config.json")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")

    def get_dashboard_data(self):
        """Get data for the trading dashboard."""
        try:
            # Get performance metrics
            metrics = self.get_performance_metrics()
            
            # Get position details
            position_data = []
            for symbol, position in self.positions.items():
                # Get current price
                if hasattr(self, 'price_cache') and f"price_{symbol}" in self.price_cache:
                    current_price = self.price_cache[f"price_{symbol}"]["price"]
                else:
                    # Try to get latest price
                    try:
                        ticker = self.api.query_public('Ticker', {'pair': symbol})
                        if ticker and 'result' in ticker and symbol in ticker['result']:
                            current_price = float(ticker['result'][symbol]['c'][0])
                        else:
                            current_price = position['entry_price']
                    except:
                        current_price = position['entry_price']
                
                # Calculate values
                volume = position['volume']
                value = volume * current_price
                entry_value = volume * position['entry_price']
                unrealized_pnl = value - entry_value
                unrealized_pnl_pct = unrealized_pnl / entry_value * 100 if entry_value > 0 else 0
                
                # Format time held
                time_held = datetime.now() - position['entry_time']
                hours = time_held.total_seconds() / 3600
                
                if hours < 24:
                    time_held_str = f"{hours:.1f} hours"
                else:
                    days = hours / 24
                    time_held_str = f"{days:.1f} days"
                
                position_data.append({
                    'symbol': symbol,
                    'volume': volume,
                    'entry_price': position['entry_price'],
                    'current_price': current_price,
                    'value': value,
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_pct': unrealized_pnl_pct,
                    'time_held': time_held_str,
                    'stop_loss': position.get('stop_loss'),
                    'take_profit': position.get('take_profit'),
                    'profit_tiers': {
                        'first_tier_executed': position.get('first_tier_executed', False),
                        'second_tier_executed': position.get('second_tier_executed', False),
                        'third_tier_executed': position.get('third_tier_executed', False)
                    }
                })
            
            # Get recent trades
            recent_trades = self.trade_history[-20:] if len(self.trade_history) >= 20 else self.trade_history
            
            # Format trades for display
            trade_data = []
            for trade in recent_trades:
                # Convert timestamp to string
                timestamp_str = trade['timestamp'].strftime('%Y-%m-%d %H:%M')
                
                # Format trade details
                trade_dict = {
                    'timestamp': timestamp_str,
                    'symbol': trade['symbol'],
                    'type': trade['type'],
                    'price': trade['price'],
                    'quantity': trade['quantity'],
                    'value': trade['value']
                }
                
                # Add profit/loss if available
                if 'profit_loss' in trade:
                    trade_dict['profit_loss'] = trade['profit_loss']
                    trade_dict['pnl_percentage'] = trade.get('pnl_percentage', 0)
                    trade_dict['exit_reason'] = trade.get('exit_reason', '')
                
                trade_data.append(trade_dict)
            
            # Get equity history for chart
            equity_history = []
            for entry in self.portfolio_history:
                timestamp_str = entry['timestamp'].strftime('%Y-%m-%d %H:%M')
                equity_history.append({
                    'timestamp': timestamp_str,
                    'equity': entry['equity'],
                    'balance': entry['balance'],
                    'drawdown': entry.get('drawdown', 0) * 100  # Convert to percentage
                })
            
            # Get sentiment data
            sentiment_data = {}
            
            try:
                conn = sqlite3.connect(self.db_name)
                c = conn.cursor()
                
                for symbol in self.symbols:
                    # Get latest sentiment
                    c.execute('''
                        SELECT twitter_score, reddit_score, news_score, combined_score, timestamp
                        FROM sentiment_data
                        WHERE symbol = ?
                        ORDER BY timestamp DESC
                        LIMIT 1
                    ''', (symbol,))
                    
                    row = c.fetchone()
                    
                    if row:
                        sentiment_data[symbol] = {
                            'twitter': row[0],
                            'reddit': row[1],
                            'news': row[2],
                            'combined': row[3],
                            'timestamp': row[4]
                        }
                
                conn.close()
            except Exception as e:
                self.logger.error(f"Error getting sentiment data: {str(e)}")
            
            # Get market regimes
            market_regimes = {}
            if hasattr(self, 'market_regimes') and 'symbols' in self.market_regimes:
                for symbol, data in self.market_regimes['symbols'].items():
                    market_regimes[symbol] = {
                        'regime': data.get('regime', 'unknown'),
                        'volatility': data.get('volatility', 0),
                        'timestamp': data.get('timestamp', '')
                    }
            
            # Get active DCA plans
            dca_plans_data = []
            for plan in self.dca_plans:
                dca_plans_data.append({
                    'symbol': plan.get('symbol', ''),
                    'completed_parts': plan.get('completed_parts', 0),
                    'total_parts': plan.get('total_parts', 0),
                    'first_entry_time': plan.get('first_entry_time', datetime.now()).strftime('%Y-%m-%d %H:%M'),
                    'first_entry_price': plan.get('first_entry_price', 0),
                    'next_part_due': (plan.get('first_entry_time', datetime.now()) + 
                                    timedelta(seconds=plan.get('time_between_parts', 0) * 
                                            plan.get('completed_parts', 0))).strftime('%Y-%m-%d %H:%M'),
                    'part_size': plan.get('part_size', 0)
                })
            
            # Compile dashboard data
            dashboard = {
                'metrics': metrics,
                'positions': position_data,
                'trades': trade_data,
                'equity_history': equity_history,
                'sentiment': sentiment_data,
                'market_regimes': market_regimes,
                'dca_plans': dca_plans_data,
                'signal_weights': self.signal_weights if hasattr(self, 'signal_weights') else {},
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Error generating dashboard data: {str(e)}")
            traceback.print_exc()
            
            # Return minimal data on error
            return {
                'error': str(e),
                'metrics': {
                    'current_equity': self.balance.get('ZUSD', 0),
                    'cash_balance': self.balance.get('ZUSD', 0)
                },
                'positions': [],
                'trades': [],
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    async def generate_performance_report(self, detailed=False):
        """Generate a comprehensive performance report for analysis."""
        try:
            self.logger.info("Generating performance report...")
            
            # Get performance metrics
            metrics = self.get_performance_metrics()
            
            # Get trade statistics
            all_trades = [t for t in self.trade_history if t.get('type') in ['sell', 'partial_sell']]
            
            # Analyze trades by symbol
            symbol_performance = {}
            for symbol in self.symbols:
                symbol_trades = [t for t in all_trades if t.get('symbol') == symbol]
                if symbol_trades:
                    wins = sum(1 for t in symbol_trades if t.get('profit_loss', 0) > 0)
                    losses = sum(1 for t in symbol_trades if t.get('profit_loss', 0) < 0)
                    total_pnl = sum(t.get('profit_loss', 0) for t in symbol_trades)
                    win_rate = wins / len(symbol_trades) * 100 if len(symbol_trades) > 0 else 0
                    
                    symbol_performance[symbol] = {
                        'trades': len(symbol_trades),
                        'wins': wins,
                        'losses': losses,
                        'win_rate': win_rate,
                        'total_pnl': total_pnl,
                        'avg_pnl': total_pnl / len(symbol_trades) if len(symbol_trades) > 0 else 0
                    }
            
            # Analyze trades by market cycle
            cycle_performance = {}
            for trade in all_trades:
                cycle = trade.get('market_cycle', 'unknown')
                if cycle not in cycle_performance:
                    cycle_performance[cycle] = {
                        'trades': 0,
                        'wins': 0,
                        'losses': 0,
                        'total_pnl': 0
                    }
                
                cycle_performance[cycle]['trades'] += 1
                if trade.get('profit_loss', 0) > 0:
                    cycle_performance[cycle]['wins'] += 1
                else:
                    cycle_performance[cycle]['losses'] += 1
                cycle_performance[cycle]['total_pnl'] += trade.get('profit_loss', 0)
            
            # Calculate win rates for market cycles
            for cycle, data in cycle_performance.items():
                data['win_rate'] = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
                data['avg_pnl'] = data['total_pnl'] / data['trades'] if data['trades'] > 0 else 0
            
            # Analyze by exit reason
            exit_performance = {}
            for trade in all_trades:
                reason = trade.get('exit_reason', 'unknown')
                if reason not in exit_performance:
                    exit_performance[reason] = {
                        'trades': 0,
                        'wins': 0,
                        'losses': 0,
                        'total_pnl': 0
                    }
                
                exit_performance[reason]['trades'] += 1
                if trade.get('profit_loss', 0) > 0:
                    exit_performance[reason]['wins'] += 1
                else:
                    exit_performance[reason]['losses'] += 1
                exit_performance[reason]['total_pnl'] += trade.get('profit_loss', 0)
            
            # Calculate win rates for exit reasons
            for reason, data in exit_performance.items():
                data['win_rate'] = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
                data['avg_pnl'] = data['total_pnl'] / data['trades'] if data['trades'] > 0 else 0
            
            # Calculate monthly returns if we have enough data
            monthly_returns = {}
            if len(self.portfolio_history) > 30:
                equity_series = pd.Series(
                    [entry['equity'] for entry in self.portfolio_history],
                    index=[entry['timestamp'] for entry in self.portfolio_history]
                )
                
                # Resample to monthly returns
                monthly_equity = equity_series.resample('M').last()
                monthly_returns_series = monthly_equity.pct_change()
                
                # Convert to dictionary
                for date, value in monthly_returns_series.items():
                    if not pd.isna(value):
                        month_str = date.strftime('%Y-%m')
                        monthly_returns[month_str] = value * 100  # Convert to percentage
            
            # Prepare detailed report
            report = {
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'overall_metrics': metrics,
                'symbol_performance': symbol_performance,
                'market_cycle_performance': cycle_performance,
                'exit_reason_performance': exit_performance,
                'monthly_returns': monthly_returns,
                'current_parameters': {
                    'buy_thresholds': self.buy_thresholds,
                    'stop_loss_pct': self.stop_loss_pct,
                    'take_profit_pct': self.take_profit_pct,
                    'trailing_stop_pct': self.trailing_stop_pct,
                    'max_position_size': self.max_position_size,
                    'signal_weights': self.signal_weights if hasattr(self, 'signal_weights') else {}
                }
            }
            
            # Add detailed trade history if requested
            if detailed:
                formatted_trades = []
                for trade in self.trade_history:
                    formatted_trade = {k: str(v) if isinstance(v, datetime) else v for k, v in trade.items()}
                    if 'id' in formatted_trade:
                        del formatted_trade['id']  # Remove database ID
                    formatted_trades.append(formatted_trade)
                
                report['detailed_trade_history'] = formatted_trades
            
            # Save report to disk
            report_dir = 'risk_analytics'
            os.makedirs(report_dir, exist_ok=True)
            report_file = os.path.join(report_dir, f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Performance report saved to {report_file}")
            
            return report
        
        except Exception as e:
            self.logger.error(f"Error generating performance report: {str(e)}")
            traceback.print_exc()
            return {"error": str(e)}

    def _process_export_data(self, data):
        """Process data for export to external systems."""
        # Helper function to make data safe for export by converting objects to strings
        if isinstance(data, dict):
            return {k: self._process_export_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._process_export_data(item) for item in data]
        elif isinstance(data, (datetime, np.datetime64)):
            return data.isoformat() if hasattr(data, 'isoformat') else str(data)
        elif isinstance(data, (np.int64, np.float64)):
            return int(data) if isinstance(data, np.int64) else float(data)
        else:
            return data

    async def export_data(self, format_type='json', path=None):
        """Export trading data for external analysis."""
        try:
            self.logger.info(f"Exporting data in {format_type} format")
            
            # Determine export path
            if path is None:
                export_dir = 'exports'
                os.makedirs(export_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                path = os.path.join(export_dir, f"kryptos_export_{timestamp}.{format_type}")
            
            # Prepare export data
            export_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'bot_version': '1.0',
                    'initial_capital': self.initial_capital
                },
                'current_state': {
                    'balance': self.balance,
                    'positions': self._process_export_data(self.positions),
                    'portfolio_value': self.calculate_total_equity()
                },
                'trade_history': self._process_export_data(self.trade_history),
                'portfolio_history': self._process_export_data(self.portfolio_history),
                'performance_metrics': self.get_performance_metrics()
            }
            
            # Export based on format
            if format_type.lower() == 'json':
                with open(path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                    
            elif format_type.lower() == 'csv':
                # Export multiple CSV files
                export_dir = os.path.dirname(path)
                base_name = os.path.splitext(os.path.basename(path))[0]
                
                # Export trade history
                trades_df = pd.DataFrame(self._process_export_data(self.trade_history))
                trades_path = os.path.join(export_dir, f"{base_name}_trades.csv")
                trades_df.to_csv(trades_path, index=False)
                
                # Export portfolio history
                portfolio_df = pd.DataFrame(self._process_export_data(self.portfolio_history))
                portfolio_path = os.path.join(export_dir, f"{base_name}_portfolio.csv")
                portfolio_df.to_csv(portfolio_path, index=False)
                
                # Create metadata file
                with open(os.path.join(export_dir, f"{base_name}_metadata.json"), 'w') as f:
                    json.dump({
                        'metadata': export_data['metadata'],
                        'current_state': export_data['current_state'],
                        'performance_metrics': export_data['performance_metrics']
                    }, f, indent=2, default=str)
                    
                path = export_dir  # Return directory instead of file
                
            else:
                self.logger.error(f"Unsupported export format: {format_type}")
                return None
            
            self.logger.info(f"Data successfully exported to {path}")
            return path
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {str(e)}")
            traceback.print_exc()
            return None

    async def import_data(self, file_path):
        """Import trading data from a previous export."""
        try:
            self.logger.info(f"Importing data from {file_path}")
            
            # Determine file type
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.json':
                # Import JSON file
                with open(file_path, 'r') as f:
                    import_data = json.load(f)
                
                # Validate imported data
                if not all(key in import_data for key in ['metadata', 'current_state', 'trade_history', 'portfolio_history']):
                    self.logger.error("Invalid import file format: missing required keys")
                    return False
                
                # Convert ISO dates back to datetime objects
                self._convert_dates_in_data(import_data)
                
                # Import state
                self.balance = import_data['current_state']['balance']
                self.positions = import_data['current_state']['positions']
                self.trade_history = import_data['trade_history']
                self.portfolio_history = import_data['portfolio_history']
                
                # Save imported state to database
                self.save_state()
                
                self.logger.info(f"Data successfully imported from {file_path}")
                return True
                
            elif file_ext == '.csv':
                # TODO: Implement CSV import if needed
                self.logger.error("CSV import not yet implemented")
                return False
                
            else:
                self.logger.error(f"Unsupported import file type: {file_ext}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error importing data: {str(e)}")
            traceback.print_exc()
            return False
        
    def _convert_dates_in_data(self, data):
        """Recursively convert ISO date strings to datetime objects."""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 10:
                    try:
                        # Try to parse as datetime
                        data[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        # Not a datetime string
                        pass
                elif isinstance(value, (dict, list)):
                    self._convert_dates_in_data(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    self._convert_dates_in_data(item)

    def get_log_tail(self, lines=100):
        """Get the most recent lines from the log file."""
        try:
            # Find the most recent log file
            log_dir = 'logs'
            log_files = [f for f in os.listdir(log_dir) if f.startswith('kryptos_')]
            
            if not log_files:
                return "No log files found."
                
            # Sort by modification time (newest first)
            newest_log = sorted(log_files, key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)[0]
            log_path = os.path.join(log_dir, newest_log)
            
            # Read the tail of the log file
            with open(log_path, 'r') as f:
                log_content = f.readlines()
            
            # Return the last N lines
            return ''.join(log_content[-lines:])
            
        except Exception as e:
            return f"Error reading log file: {str(e)}"

    def health_check(self):
        """Perform a health check of the trading bot."""
        try:
            health_status = {
                'status': 'ok',
                'timestamp': datetime.now().isoformat(),
                'components': {},
                'issues': []
            }
            
            # Check API connection
            try:
                ticker = self.api.query_public('Time')
                if 'error' in ticker and ticker['error']:
                    health_status['components']['api'] = 'warning'
                    health_status['issues'].append(f"API warning: {ticker['error']}")
                else:
                    health_status['components']['api'] = 'ok'
            except Exception as e:
                health_status['components']['api'] = 'error'
                health_status['issues'].append(f"API error: {str(e)}")
                health_status['status'] = 'warning'
            
            # Check database
            try:
                conn = sqlite3.connect(self.db_name)
                c = conn.cursor()
                c.execute("SELECT count(*) FROM sqlite_master")
                c.fetchone()
                conn.close()
                health_status['components']['database'] = 'ok'
            except Exception as e:
                health_status['components']['database'] = 'error'
                health_status['issues'].append(f"Database error: {str(e)}")
                health_status['status'] = 'error'
            
            # Check models
            if self.is_models_loaded:
                health_status['components']['models'] = 'ok'
            else:
                health_status['components']['models'] = 'warning'
                health_status['issues'].append("ML/AI models not loaded")
                health_status['status'] = 'warning'
            
            # Check background tasks
            running_tasks = sum(1 for task in self.tasks if not task.done())
            if running_tasks == len(self.tasks):
                health_status['components']['background_tasks'] = 'ok'
            else:
                health_status['components']['background_tasks'] = 'warning'
                health_status['issues'].append(f"Some background tasks not running ({running_tasks}/{len(self.tasks)})")
                health_status['status'] = 'warning'
            
            # Check balance
            if self.balance.get('ZUSD', 0) > 0:
                health_status['components']['balance'] = 'ok'
            else:
                health_status['components']['balance'] = 'error'
                health_status['issues'].append("Zero or negative balance")
                health_status['status'] = 'error'
            
            # Get trade activity info
            recent_time = datetime.now() - timedelta(hours=24)
            recent_trades = [t for t in self.trade_history if t.get('timestamp', datetime.now()) > recent_time]
            health_status['recent_activity'] = {
                'trades_24h': len(recent_trades),
                'active_positions': len(self.positions),
                'active_dca_plans': len(self.dca_plans) if hasattr(self, 'dca_plans') else 0
            }
            
            # System usage stats
            import psutil
            health_status['system'] = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent
            }
            
            # Check for critical resource usage
            if health_status['system']['memory_percent'] > 90:
                health_status['issues'].append(f"High memory usage: {health_status['system']['memory_percent']}%")
                health_status['status'] = 'warning'
                
            if health_status['system']['disk_percent'] > 90:
                health_status['issues'].append(f"High disk usage: {health_status['system']['disk_percent']}%")
                health_status['status'] = 'warning'
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    def _register_signal_handlers(self):
        """Register signal handlers for clean shutdown."""
        import signal
        
        def signal_handler(sig, frame):
            """Handle termination signals by saving state before exit."""
            self.logger.info(f"Received signal {sig}, saving state before shutdown...")
            
            # Set running flag to false
            self.is_running = False
            
            # Save current state
            self.save_state()
            
            # Save disk cache
            self._save_disk_cache()
            
            # Calculate current equity for logging
            total_equity = self.calculate_total_equity()
            self.logger.info(f"Final state saved - Balance: ${self.balance.get('ZUSD', 0):.2f}, "
                        f"Total Equity: ${total_equity:.2f}")
            
            # Exit with success code
            self.logger.info("Graceful shutdown complete")
            sys.exit(0)
        
        # Register handlers for SIGTERM (systemd stop) and SIGINT (Ctrl+C)
        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            self.logger.info("Signal handlers registered for clean shutdown")
        except Exception as e:
            self.logger.error(f"Error registering signal handlers: {str(e)}")