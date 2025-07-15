#!/usr/bin/env python
"""
KryptosTradingBot - Advanced Cryptocurrency Trading System

A streamlined, modular crypto trading bot with ML/AI prediction, sentiment analysis,
and comprehensive backtesting capabilities.
"""

import argparse
import asyncio
import os
import json
import sys
import logging
from datetime import datetime, timedelta

# Ensure modules directory is in the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("KryptosTradingBot")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="KryptosTradingBot - Advanced Crypto Trading System")
    
    # Main operation modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="Run the trading bot")
    group.add_argument("--backtest", action="store_true", help="Run backtesting")
    group.add_argument("--optimize", action="store_true", help="Optimize trading parameters")
    group.add_argument("--dashboard", action="store_true", help="Start the dashboard server")
    
    # Configuration
    parser.add_argument("--config", type=str, default="config/config.json", help="Path to configuration file")
    
    # Backtest and optimization parameters
    parser.add_argument("--start-date", type=str, help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for backtest (YYYY-MM-DD)")
    parser.add_argument("--iterations", type=int, default=20, help="Number of iterations for optimization")
    
    # Dashboard parameters
    parser.add_argument("--port", type=int, default=8080, help="Port for dashboard server")
    
    return parser.parse_args()

async def main():
    """Main entry point for the trading bot."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Import trading bot here to avoid circular imports
    from core.bot import KryptosTradingBot
    
    # Load configuration
    config_path = args.config
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Creating default configuration file...")
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Create default configuration
        default_config = {
            "initial_capital": 1000000.0,
            "symbols": {
                "XBTUSD": 0.25,
                "ETHUSD": 0.20,
                "SOLUSD": 0.15,
                "AVAXUSD": 0.15,
                "XRPUSD": 0.15,
                "XDGUSD": 0.10
            },
            "risk": {
                "max_drawdown": 0.20,
                "trailing_stop_pct": 0.008,
                "max_trades_per_hour": 3,
                "trade_cooldown": 360,
                "max_position_size": 0.08,
                "min_position_value": 1000.0,
                "max_total_risk": 0.25,
                "stop_loss_pct": 0.007,
                "take_profit_pct": 0.018
            },
            "api": {
                "timeframe": 5,
                "retry_delay": 1.0,
                "max_retry_delay": 60
            }
        }
        
        # Save default configuration
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
            
        logger.info(f"Default configuration saved to {config_path}")
    
    # Initialize the trading bot
    trading_bot = KryptosTradingBot(config_path)
    
    try:
        # Run in selected mode
        if args.run:
            logger.info("Starting trading bot in live mode...")
            await trading_bot.run()
            
        elif args.backtest:
            # Validate backtesting parameters
            if not args.start_date or not args.end_date:
                logger.error("Both --start-date and --end-date are required for backtesting")
                return
                
            logger.info(f"Starting backtesting from {args.start_date} to {args.end_date}...")
            
            # Give background tasks time to start up properly
            await asyncio.sleep(2)
            
            # Run backtest
            results = await trading_bot.backtest(args.start_date, args.end_date)
            
            # Display backtest results
            logger.info("Backtest Results:")
            logger.info(f"Initial Capital: ${results['initial_capital']:.2f}")
            logger.info(f"Final Equity: ${results['final_equity']:.2f}")
            logger.info(f"Total Return: {results['total_return']:.2f}%")
            logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            logger.info(f"Max Drawdown: {results['max_drawdown']:.2f}%")
            logger.info(f"Win Rate: {results['win_rate']:.2f}%")
            logger.info(f"Total Trades: {results['total_trades']}")
            logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
            
            # Properly clean up background tasks when backtesting completes
            await trading_bot.cleanup()
            
        elif args.optimize:
            # Validate optimization parameters
            if not args.start_date or not args.end_date:
                logger.error("Both --start-date and --end-date are required for optimization")
                return
                
            logger.info(f"Starting parameter optimization from {args.start_date} to {args.end_date} "
                      f"with {args.iterations} iterations...")
            
            # Give background tasks time to start up properly
            await asyncio.sleep(2)
            
            results = await trading_bot.optimize_parameters(
                args.start_date,
                args.end_date,
                args.iterations
            )
            
            # Display optimization results
            logger.info("Optimization Results:")
            
            # Best return parameters
            logger.info("\nBest Return Parameters:")
            logger.info(f"Return: {results['best_return']['return']:.2f}%")
            for param, value in results['best_return']['parameters'].items():
                logger.info(f"{param}: {value}")
            
            # Best Sharpe parameters
            logger.info("\nBest Sharpe Parameters:")
            logger.info(f"Sharpe: {results['best_sharpe']['sharpe']:.2f}")
            for param, value in results['best_sharpe']['parameters'].items():
                logger.info(f"{param}: {value}")
                
            # Ask user if they want to apply the optimized parameters
            apply_params = input("Apply optimized parameters? (best_return/best_sharpe/no): ").lower()
            
            if apply_params == "best_return":
                await trading_bot.apply_optimized_parameters(results['best_return']['parameters'])
                logger.info("Applied best return parameters")
            elif apply_params == "best_sharpe":
                await trading_bot.apply_optimized_parameters(results['best_sharpe']['parameters'])
                logger.info("Applied best Sharpe parameters")
            else:
                logger.info("Optimized parameters not applied")
            
            # Properly clean up background tasks
            await trading_bot.cleanup()
            
        elif args.dashboard:
            # Start the dashboard server
            try:
                # Try importing here to catch import errors early
                try:
                    import flask
                    import flask_socketio
                except ImportError:
                    logger.error("Dashboard dependencies not installed. "
                               "Please install with: pip install flask flask-socketio")
                    logger.info("Installing missing dependencies...")
                    
                    import subprocess
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "flask-socketio", "eventlet"])
                    logger.info("Dependencies installed successfully")
                
                # Now import the dashboard module
                from dashboard.server import start_dashboard_server
                
                logger.info(f"Starting dashboard server on port {args.port}...")
                
                # Get initial dashboard data
                dashboard_data = trading_bot.get_dashboard_data()
                
                # Start the server
                await start_dashboard_server(args.port, trading_bot, dashboard_data)
                
            except ImportError as e:
                logger.error(f"Dashboard import error: {str(e)}")
                logger.error("Make sure dashboard files are in the correct location")
                return
    
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        await trading_bot.cleanup()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.exception("Exception details:")
        # Try to clean up
        if 'trading_bot' in locals():
            await trading_bot.cleanup()

if __name__ == "__main__":
    try:
        # Run the main async function
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.exception("Exception details:")