"""
EnhancedNewsAnalyzer - Crypto news sentiment analysis

This module provides news sentiment analysis optimized for cryptocurrency trading.
"""

import os
import json
import logging
import random
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import numpy as np
import requests

class EnhancedNewsAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger("EnhancedNewsAnalyzer")
        
        # API configurations
        self.newsapi_key = os.environ.get("NEWSAPI_KEY")
        self.cryptocompare_api_key = os.environ.get("CRYPTOCOMPARE_API_KEY")
        self.coinmarketcap_key = os.environ.get("COINMARKETCAP_API_KEY")
        
        # Check API availability
        self.newsapi_available = bool(self.newsapi_key)
        self.cryptocompare_available = bool(self.cryptocompare_api_key)
        self.coinmarketcap_available = bool(self.coinmarketcap_key)
        
        if not self.newsapi_available:
            self.logger.warning("NewsAPI key not configured")
        if not self.cryptocompare_available:
            self.logger.warning("CryptoCompare API key not configured")
        if not self.coinmarketcap_available:
            self.logger.warning("CoinMarketCap API key not configured")
        
        # Setup caching with adaptive durations
        self.cache = {}
        self.disk_cache_path = 'cache/news'
        os.makedirs(self.disk_cache_path, exist_ok=True)
        self.base_cache_duration = 7200  # 2 hours
        
        # Different cache durations for different market conditions
        self.market_volatility = 'normal'  # Can be 'low', 'normal', 'high'
        self.volatility_cache_adjustments = {
            'low': 1.5,      # Cache 50% longer in low volatility
            'normal': 1.0,   # Standard cache duration
            'high': 0.5      # Cache 50% less in high volatility
        }
        
        # Market events detection cache
        self.market_events_cache = {'data': None, 'timestamp': 0}
        self.market_events_duration = 1800  # 30 minutes
        
        # Keywords that make news more impactful
        self.high_impact_keywords = [
            'regulation', 'sec', 'lawsuit', 'hack', 'security breach',
            'major announcement', 'partnership', 'listing', 'etf',
            'approval', 'adoption', 'institutional', 'ban', 'restriction',
            'investigation', 'fraud', 'scam', 'exit scam', 'pump and dump',
            'whale', 'manipulation', 'central bank', 'federal reserve',
            'bill', 'legislation', 'tax', 'treasury', 'framework'
        ]
        
        # Initialize NLP model
        self.nlp_model = self._initialize_nlp_model()
        
        # API rate limiting
        self.api_rate_limits = {
            'newsapi': {'last_request': 0, 'min_interval': 600},  # 10 minutes
            'cryptocompare': {'last_request': 0, 'min_interval': 300},  # 5 minutes
            'coinmarketcap': {'last_request': 0, 'min_interval': 60}  # 1 minute
        }
        
        # Initialize historical news data for fallback
        self.historical_news = self._initialize_historical_data()
        
        # Load cached data
        self._load_disk_cache()
        
        # Twitter analyzer for reference (optional)
        self.twitter_analyzer = None
        
        self.logger.info("EnhancedNewsAnalyzer initialized")
        
    def set_twitter_analyzer(self, twitter_analyzer):
        """Set reference to Twitter analyzer for enhanced correlation."""
        self.twitter_analyzer = twitter_analyzer
    
    def _initialize_nlp_model(self):
        """Initialize VADER NLP model for sentiment analysis."""
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            return SentimentIntensityAnalyzer()
        except ImportError:
            self.logger.error("NLTK not installed for news sentiment analysis")
            return None
    
    def _initialize_historical_data(self):
        """Initialize historical news data for when APIs are unavailable."""
        current_time = time.time()
        
        # Base data that can be updated manually when important news occurs
        historical_data = {
            'XBTUSD': {'impact_score': 0.2, 'volume': 10, 'latest_headline': 'Bitcoin continues secular bull market'},
            'ETHUSD': {'impact_score': 0.1, 'volume': 8, 'latest_headline': 'Ethereum ecosystem continues to grow'},
            'SOLUSD': {'impact_score': 0.05, 'volume': 6, 'latest_headline': 'Solana gains institutional adoption'},
            'AVAXUSD': {'impact_score': 0.0, 'volume': 5, 'latest_headline': 'Avalanche focuses on new partnerships'},
            'XRPUSD': {'impact_score': 0.0, 'volume': 5, 'latest_headline': 'XRP maintains market position'},
            'XDGUSD': {'impact_score': 0.0, 'volume': 4, 'latest_headline': 'Dogecoin community remains active'}
        }
        
        # Add timestamps and metadata
        for symbol, data in historical_data.items():
            data['timestamp'] = current_time
            data['api_source'] = 'historical'
        
        return historical_data
    
    def _load_disk_cache(self):
        """Load cached news data from disk."""
        try:
            cache_file = os.path.join(self.disk_cache_path, 'news_cache.json')
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    
                # Process cache data
                for symbol, data in cache_data.items():
                    # Convert timestamp strings back to float
                    if 'timestamp' in data:
                        data['timestamp'] = float(data['timestamp'])
                    self.cache[symbol] = data
                    
                self.logger.info(f"Loaded {len(cache_data)} items from news cache")
        except Exception as e:
            self.logger.error(f"Error loading news cache: {str(e)}")
    
    def _save_to_disk_cache(self):
        """Save current cache to disk."""
        try:
            cache_file = os.path.join(self.disk_cache_path, 'news_cache.json')
            with open(cache_file, 'w') as f:
                json.dump(self.cache, f)
                
            self.logger.info(f"Saved {len(self.cache)} items to news cache")
        except Exception as e:
            self.logger.error(f"Error saving news cache: {str(e)}")
    
    def update_market_volatility(self, volatility_level):
        """Update the market volatility level to adjust caching duration."""
        if volatility_level in self.volatility_cache_adjustments:
            self.market_volatility = volatility_level
            self.logger.info(f"Market volatility set to {volatility_level}")
            
            # Clear cache when volatility increases to ensure fresh data
            if volatility_level == 'high':
                self.cache = {}
                self.logger.info("Cache cleared due to high market volatility")
    
    def get_effective_cache_duration(self, symbol):
        """Get the effective cache duration based on symbol and market conditions."""
        # Base duration
        base_duration = self.base_cache_duration
        
        # Adjust based on market volatility
        volatility_factor = self.volatility_cache_adjustments.get(self.market_volatility, 1.0)
        
        # Different durations for different coins
        symbol_factors = {
            'XBTUSD': 0.8,    # Shorter cache for BTC (higher importance)
            'ETHUSD': 0.8,    # Shorter cache for ETH (higher importance)
            'SOLUSD': 1.0,    # Standard cache for others
            'AVAXUSD': 1.0,
            'XRPUSD': 1.2,    # Longer cache for less volatile assets
            'XDGUSD': 1.5     # Much longer cache for DOGE
        }
        
        symbol_factor = symbol_factors.get(symbol, 1.0)
        
        # Calculate final duration
        effective_duration = base_duration * volatility_factor * symbol_factor
        
        return effective_duration
    
    def _get_cryptocompare_news(self, coin, lookback_hours=24):
        """Get news from CryptoCompare API with optimized parameters."""
        try:
            if not self.cryptocompare_api_key:
                return None
                
            # Check rate limits
            current_time = time.time()
            rate_limit = self.api_rate_limits['cryptocompare']
            if current_time - rate_limit['last_request'] < rate_limit['min_interval']:
                self.logger.info(f"Skipping CryptoCompare API request - rate limiting")
                return None
                
            # Update last request time
            self.api_rate_limits['cryptocompare']['last_request'] = current_time
            
            # Make API request with increased timeout
            url = f"https://min-api.cryptocompare.com/data/v2/news/?categories={coin}&api_key={self.cryptocompare_api_key}"
            
            try:
                response = requests.get(url, timeout=20)
            except requests.exceptions.Timeout:
                self.logger.warning(f"CryptoCompare API timeout for {coin}, using cached or default data")
                return None
            except requests.exceptions.RequestException as req_err:
                self.logger.warning(f"CryptoCompare API request failed: {str(req_err)}")
                return None
            
            if response.status_code != 200:
                self.logger.error(f"CryptoCompare API error: {response.status_code}")
                return None
                
            data = response.json()
            
            if 'Data' not in data or not data['Data']:
                return None
                
            # Filter by time
            cutoff_time = time.time() - (lookback_hours * 3600)
            recent_news = [
                item for item in data['Data']
                if item['published_on'] >= cutoff_time
            ]
            
            if not recent_news:
                return None
                
            # Process the news for sentiment
            impact_score = self._calculate_news_impact(recent_news)
            
            result = {
                'impact_score': impact_score,
                'volume': len(recent_news),
                'latest_headline': recent_news[0]['title'] if recent_news else None,
                'timestamp': time.time(),
                'api_source': 'cryptocompare'
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting CryptoCompare news: {str(e)}")
            return None
            
    def _calculate_news_impact(self, news_items):
        """Calculate the impact score of news items from CryptoCompare."""
        if not news_items:
            return 0
            
        total_score = 0
        
        for item in news_items:
            # Base score
            item_score = 0
            
            # Check title for high impact keywords
            title = item['title'].lower()
            body = item.get('body', '').lower()
            
            # Check for high impact keywords
            for keyword in self.high_impact_keywords:
                if keyword.lower() in title:
                    item_score += 0.2  # Higher weight for keywords in title
                elif keyword.lower() in body:
                    item_score += 0.1  # Lower weight for keywords in body
            
            # Check categories for importance
            categories = item.get('categories', '').lower()
            if 'regulation' in categories:
                item_score += 0.2
            if 'exchange' in categories:
                item_score += 0.1
            if 'mining' in categories:
                item_score += 0.1
            
            # Check source reputation (could be expanded)
            source = item.get('source', '').lower()
            if source in ['coindesk', 'cointelegraph', 'bloomberg', 'forbes']:
                item_score += 0.1
            
            # Recency factor - more recent news has higher impact
            hours_old = (time.time() - item['published_on']) / 3600
            recency_factor = max(0.5, 1.0 - (hours_old / 48))  # Taper off over 48 hours
            
            # Apply recency factor
            item_score *= recency_factor
            
            # Add to total
            total_score += item_score
        
        # Normalize score to -1 to 1 range
        normalized_score = min(1.0, max(-1.0, total_score / len(news_items) * 2))
        
        return normalized_score
    
    def _get_newsapi_news(self, coin, lookback_hours=24):
        """Get news from NewsAPI.org with improved error handling and fallbacks."""
        try:
            if not self.newsapi_key:
                self.logger.warning("No NewsAPI key configured - using default data")
                return None
                    
            # Check rate limits
            current_time = time.time()
            rate_limit = self.api_rate_limits['newsapi']
            if current_time - rate_limit['last_request'] < rate_limit['min_interval']:
                self.logger.info(f"Skipping NewsAPI request - rate limiting")
                return None
                    
            # Update last request time
            self.api_rate_limits['newsapi']['last_request'] = current_time
            
            # Calculate date range
            from_date = datetime.now() - timedelta(hours=lookback_hours)
            from_date_str = from_date.strftime('%Y-%m-%d')
            
            # Define search queries for cryptocurrency
            search_term = f"{coin} cryptocurrency"
            
            # Make API request with better error handling
            url = f"https://newsapi.org/v2/everything?q={search_term}&from={from_date_str}&sortBy=publishedAt&apiKey={self.newsapi_key}"
            
            try:
                response = requests.get(url, timeout=15)
                
                # Handle different error codes with specific messages
                if response.status_code == 401:
                    self.logger.error(f"NewsAPI authentication error (401): Invalid API key")
                    self.newsapi_available = False  # Disable API for this session
                    return None
                elif response.status_code == 429:
                    self.logger.error(f"NewsAPI rate limit exceeded (429): Too many requests")
                    # Increase backoff for rate limiting
                    self.api_rate_limits['newsapi']['min_interval'] *= 1.5
                    return None
                elif response.status_code != 200:
                    self.logger.error(f"NewsAPI error: {response.status_code}")
                    return None
                    
                data = response.json()
                
                if 'articles' not in data or not data['articles']:
                    self.logger.info(f"No news articles found for {coin}")
                    return None
                    
                # Process articles
                news_items = []
                
                for article in data['articles'][:20]:  # Limit to 20 articles
                    if article['title'] and article['description']:
                        # Convert to format similar to CryptoCompare
                        try:
                            published_time = datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ").timestamp()
                        except:
                            published_time = time.time()  # Default to current time if parsing fails
                        
                        item = {
                            'title': article['title'],
                            'body': article['description'],
                            'published_on': published_time,
                            'source': article.get('source', {}).get('name', 'unknown'),
                            'url': article['url']
                        }
                        news_items.append(item)
                
                if not news_items:
                    self.logger.info(f"No valid news items found for {coin}")
                    return None
                    
                # Calculate impact score 
                impact_score = self._calculate_news_impact(news_items)
                
                result = {
                    'impact_score': impact_score,
                    'volume': len(news_items),
                    'latest_headline': news_items[0]['title'] if news_items else None,
                    'timestamp': time.time(),
                    'api_source': 'newsapi'
                }
                
                self.logger.info(f"NewsAPI returned {len(news_items)} articles for {coin}")
                return result
                
            except requests.exceptions.Timeout:
                self.logger.warning(f"NewsAPI timeout for {coin}, using cached or default data")
                return None
            except requests.exceptions.RequestException as req_err:
                self.logger.warning(f"NewsAPI request failed: {str(req_err)}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error getting NewsAPI news: {str(e)}")
            traceback.print_exc()
            return None
    
    def _get_coinmarketcap_news(self, coin, lookback_hours=24):
        """Get market data using CoinMarketCap API."""
        try:
            if not self.coinmarketcap_key:
                return None
                
            # Check rate limits
            current_time = time.time()
            rate_limit = self.api_rate_limits['coinmarketcap']
            if current_time - rate_limit['last_request'] < rate_limit['min_interval']:
                self.logger.info(f"Skipping CoinMarketCap API request - rate limiting")
                return None
                
            # Update last request time
            self.api_rate_limits['coinmarketcap']['last_request'] = current_time
            
            # Get coin symbol without USD suffix
            coin_symbol = coin.replace('USD', '')
            
            # API endpoint for cryptocurrency info
            url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
            
            headers = {
                'Accepts': 'application/json',
                'X-CMC_PRO_API_KEY': self.coinmarketcap_key,
            }
            
            parameters = {
                'start': '1',
                'limit': '100',  # Reduced to improve response time
                'convert': 'USD'
            }
            
            self.logger.info(f"Requesting market data from CoinMarketCap API for {coin}")
            
            # Create a session 
            session = requests.Session()
            session.headers.update(headers)
            
            response = session.get(url, params=parameters, timeout=15)
            
            if response.status_code != 200:
                self.logger.error(f"CoinMarketCap API error: {response.status_code}")
                
                # Log detailed error information
                try:
                    error_data = json.loads(response.text)
                    self.logger.error(f"API Error details: {error_data}")
                except:
                    self.logger.error(f"Raw API Response: {response.text[:500]}...")
                    
                return None
                
            data = json.loads(response.text)
            
            if 'data' not in data or not data['data']:
                self.logger.warning("No data returned from CoinMarketCap API")
                return None
                
            # Try to find our target coin in the results
            target_coin_data = None
            related_coins_data = []
            
            for crypto in data['data']:
                symbol = crypto.get('symbol', '')
                if symbol == coin_symbol:
                    target_coin_data = crypto
                elif symbol in ['BTC', 'ETH']:  # Always include major coins
                    related_coins_data.append(crypto)
                    
            if not target_coin_data and not related_coins_data:
                self.logger.warning(f"Could not find data for {coin_symbol} or related coins")
                return None
                
            # Calculate market sentiment from price changes
            sentiment_score = 0
            volume = 0
            latest_headline = None
            
            if target_coin_data:
                # Get percent changes
                percent_change_24h = target_coin_data.get('quote', {}).get('USD', {}).get('percent_change_24h', 0)
                percent_change_7d = target_coin_data.get('quote', {}).get('USD', {}).get('percent_change_7d', 0)
                
                # Translate price changes to sentiment
                sentiment_score = (percent_change_24h * 0.7 + percent_change_7d * 0.3) / 100
                # Normalize to -1 to 1 range
                sentiment_score = max(-1.0, min(1.0, sentiment_score))
                
                # Get volume data
                volume = int(target_coin_data.get('quote', {}).get('USD', {}).get('volume_24h', 0) / 1000000)  # In millions
                
                # Create a headline based on the price movement
                price = target_coin_data.get('quote', {}).get('USD', {}).get('price', 0)
                
                if percent_change_24h > 5:
                    latest_headline = f"{coin_symbol} surges {percent_change_24h:.1f}% in 24 hours, trading at ${price:.2f}"
                elif percent_change_24h > 2:
                    latest_headline = f"{coin_symbol} rises {percent_change_24h:.1f}% in 24 hours, now at ${price:.2f}"
                elif percent_change_24h < -5:
                    latest_headline = f"{coin_symbol} drops {abs(percent_change_24h):.1f}% in 24 hours, trading at ${price:.2f}"
                elif percent_change_24h < -2:
                    latest_headline = f"{coin_symbol} dips {abs(percent_change_24h):.1f}% in 24 hours, now at ${price:.2f}"
                else:
                    latest_headline = f"{coin_symbol} stable at ${price:.2f}, {percent_change_24h:.1f}% change in 24 hours"
            
            # If we didn't find the target coin, use related coins data
            if not target_coin_data and related_coins_data:
                # Calculate average sentiment from related coins
                total_sentiment = 0
                for related_coin in related_coins_data:
                    percent_change = related_coin.get('quote', {}).get('USD', {}).get('percent_change_24h', 0)
                    total_sentiment += percent_change
                    
                avg_sentiment = total_sentiment / len(related_coins_data) / 100
                sentiment_score = max(-1.0, min(1.0, avg_sentiment))
                
                volume = len(related_coins_data)
                latest_headline = f"Market overview: {related_coins_data[0]['symbol']} and other coins showing {total_sentiment/len(related_coins_data):.1f}% average change"
                
            # Create result in the expected format
            result = {
                'impact_score': sentiment_score,
                'volume': volume if volume > 0 else 10,  # Ensure we have some volume
                'latest_headline': latest_headline,
                'timestamp': time.time(),
                'api_source': 'coinmarketcap'
            }
            
            self.logger.info(f"Generated market data for {coin}: sentiment={sentiment_score:.2f}, headline='{latest_headline}'")
            return result
                
        except Exception as e:
            self.logger.error(f"Error getting market data from CoinMarketCap: {str(e)}")
            traceback.print_exc()
            
            # Provide fallback data
            fallback_result = {
                'impact_score': 0,
                'volume': 10,
                'latest_headline': f"Market data for {coin.replace('USD', '')} currently unavailable",
                'timestamp': time.time(),
                'api_source': 'fallback'
            }
            return fallback_result
    
    def get_news_impact(self, symbol: str, lookback_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive news sentiment analysis for a cryptocurrency symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'XBTUSD')
            lookback_hours: How many hours back to analyze news
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Standardize symbol format
            symbol = symbol.upper()
            
            # Check cache first
            current_time = time.time()
            if symbol in self.cache:
                cache_data = self.cache[symbol]
                cache_age = current_time - cache_data.get('timestamp', 0)
                cache_duration = self.get_effective_cache_duration(symbol)
                
                # Return cached data if still valid
                if cache_age < cache_duration:
                    self.logger.info(f"Using cached news data for {symbol}, age: {cache_age/60:.1f} minutes")
                    return cache_data
            
            # Get coin symbol (without USD suffix)
            coin = symbol.replace('USD', '')
            
            # Try each data source with fallbacks
            result = None
            
            # 1. First try CryptoCompare (specialized crypto news)
            if self.cryptocompare_available:
                result = self._get_cryptocompare_news(coin, lookback_hours)
                
                if result:
                    self.logger.info(f"Using CryptoCompare news data for {symbol}")
            
            # 2. If no result, try NewsAPI
            if not result and self.newsapi_available:
                result = self._get_newsapi_news(coin, lookback_hours)
                
                if result:
                    self.logger.info(f"Using NewsAPI data for {symbol}")
            
            # 3. If still no result, try CoinMarketCap
            if not result and self.coinmarketcap_available:
                result = self._get_coinmarketcap_news(coin, lookback_hours)
                
                if result:
                    self.logger.info(f"Using CoinMarketCap data for {symbol}")
            
            # 4. Final fallback to historical/default data
            if not result:
                self.logger.warning(f"Using historical fallback data for {symbol}")
                
                if symbol in self.historical_news:
                    result = self.historical_news[symbol].copy()
                    result['timestamp'] = current_time  # Update timestamp
                else:
                    # Create default neutral sentiment if nothing else is available
                    result = {
                        'impact_score': 0.0,
                        'volume': 5,
                        'latest_headline': f"No recent news available for {coin}",
                        'timestamp': current_time,
                        'api_source': 'default'
                    }
            
            # Enrich with additional information
            if self.twitter_analyzer and hasattr(self.twitter_analyzer, 'get_twitter_sentiment'):
                try:
                    twitter_data = self.twitter_analyzer.get_twitter_sentiment(symbol)
                    result['twitter_correlation'] = twitter_data.get('score', 0)
                except Exception as e:
                    self.logger.error(f"Error getting Twitter correlation: {str(e)}")
            
            # Cache the result
            self.cache[symbol] = result
            
            # Periodically save cache to disk (10% chance each call)
            if random.random() < 0.1:
                self._save_to_disk_cache()
            
            return result
                
        except Exception as e:
            self.logger.error(f"Error getting news impact for {symbol}: {str(e)}")
            traceback.print_exc()
            
            # Return neutral sentiment on error
            return {
                'impact_score': 0.0,
                'volume': 0,
                'latest_headline': f"Error retrieving news for {symbol}",
                'timestamp': time.time(),
                'api_source': 'error',
                'error': str(e)
            }
    
    def detect_market_moving_events(self) -> Dict[str, Any]:
        """Detect significant market-moving events across cryptocurrencies.
        
        Returns:
            Dictionary with market event detection results
        """
        try:
            # Check cache first
            current_time = time.time()
            if (self.market_events_cache['data'] and 
                current_time - self.market_events_cache['timestamp'] < self.market_events_duration):
                return self.market_events_cache['data']
            
            # List of major cryptocurrencies to check
            major_coins = ['BTC', 'ETH', 'SOL', 'XRP', 'AVAX']
            
            # API endpoint for common news
            if not self.cryptocompare_api_key:
                self.logger.warning("No CryptoCompare API key for market event detection")
                return {'detected': False, 'count': 0, 'events': []}
            
            # Check rate limits
            rate_limit = self.api_rate_limits['cryptocompare']
            if current_time - rate_limit['last_request'] < rate_limit['min_interval']:
                self.logger.info("Skipping market event detection - rate limiting")
                return {'detected': False, 'count': 0, 'events': []}
            
            # Update last request time
            self.api_rate_limits['cryptocompare']['last_request'] = current_time
            
            # Get latest general crypto news
            url = f"https://min-api.cryptocompare.com/data/v2/news/?categories=General&api_key={self.cryptocompare_api_key}"
            
            try:
                response = requests.get(url, timeout=15)
            except:
                self.logger.warning("Error getting market events data")
                return {'detected': False, 'count': 0, 'events': []}
            
            if response.status_code != 200:
                self.logger.error(f"API error in market event detection: {response.status_code}")
                return {'detected': False, 'count': 0, 'events': []}
            
            data = response.json()
            
            if 'Data' not in data or not data['Data']:
                return {'detected': False, 'count': 0, 'events': []}
            
            # Look for high-impact news in the last 6 hours
            cutoff_time = time.time() - (6 * 3600)
            recent_news = [
                item for item in data['Data']
                if item['published_on'] >= cutoff_time
            ]
            
            # For each news item, check if it contains critical keywords
            critical_keywords = [
                'crash', 'collapse', 'ban', 'regulation', 'sec',
                'lawsuit', 'hack', 'exploit', 'breach', 'scandal',
                'emergency', 'crisis', 'halt', 'suspend', 'fraud',
                'investigation', 'major announcement', 'critical'
            ]
            
            potential_events = []
            
            for item in recent_news:
                title = item['title'].lower()
                body = item.get('body', '').lower()
                
                # Count how many critical keywords are found
                keyword_count = sum(1 for keyword in critical_keywords 
                                  if keyword in title or keyword in body)
                
                # Check if the title contains any coin names
                mentions_major_coin = any(coin.lower() in title for coin in major_coins)
                
                # Score the potential impact
                impact_score = keyword_count * 0.1
                
                # Major coins get higher scores
                if mentions_major_coin:
                    impact_score += 0.2
                
                # High source reputation boosts score
                source = item.get('source', '').lower()
                if source in ['coindesk', 'cointelegraph', 'bloomberg', 'forbes', 'reuters']:
                    impact_score += 0.1
                
                # If it seems significant, add to potential events
                if impact_score > 0.2 or keyword_count >= 2:
                    potential_events.append({
                        'headline': item['title'],
                        'source': item.get('source', 'Unknown'),
                        'url': item.get('url', ''),
                        'published_on': item['published_on'],
                        'impact_score': impact_score
                    })
            
            # Sort by impact score
            potential_events.sort(key=lambda x: x['impact_score'], reverse=True)
            
            # Prepare result
            result = {
                'detected': len(potential_events) > 0,
                'count': len(potential_events),
                'events': potential_events[:5],  # Return top 5 events
                'timestamp': current_time
            }
            
            # Cache the result
            self.market_events_cache = {
                'data': result,
                'timestamp': current_time
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error detecting market events: {str(e)}")
            return {'detected': False, 'count': 0, 'events': []}
    
    def analyze_sentiment_correlation(self, symbol: str) -> Dict[str, Any]:
        """Analyze correlation between different sentiment sources for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'XBTUSD')
            
        Returns:
            Dictionary with correlation analysis
        """
        try:
            # Get news sentiment
            news_impact = self.get_news_impact(symbol)
            news_score = news_impact.get('impact_score', 0)
            
            # Attempt to get Twitter sentiment if available
            twitter_score = 0
            
            if self.twitter_analyzer and hasattr(self.twitter_analyzer, 'get_twitter_sentiment'):
                try:
                    twitter_data = self.twitter_analyzer.get_twitter_sentiment(symbol)
                    twitter_score = twitter_data.get('score', 0)
                except Exception as e:
                    self.logger.warning(f"Error getting Twitter data: {str(e)}")
            
            # Calculate correlation (simple for now)
            correlation = 'mixed'
            if abs(news_score - twitter_score) < 0.2:
                correlation = 'aligned'
            elif news_score * twitter_score > 0:  # Same direction
                correlation = 'similar'
            elif abs(news_score) < 0.1 or abs(twitter_score) < 0.1:
                correlation = 'neutral'
            else:
                correlation = 'divergent'
                
            # Summarize results
            combined_score = (news_score * 0.6) + (twitter_score * 0.4)
            
            return {
                'news_score': news_score,
                'twitter_score': twitter_score,
                'combined_score': combined_score,
                'correlation': correlation,
                'headline': news_impact.get('latest_headline', 'No headline available')
            }
                
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment correlation: {str(e)}")
            return {
                'news_score': 0,
                'twitter_score': 0,
                'combined_score': 0,
                'correlation': 'error',
                'headline': f"Error: {str(e)}"
            }
    
    def get_global_market_sentiment(self) -> Dict[str, Any]:
        """Get overall crypto market sentiment from multiple sources."""
        try:
            # Key cryptocurrencies to analyze
            major_coins = ['XBTUSD', 'ETHUSD']
            secondary_coins = ['SOLUSD', 'AVAXUSD', 'XRPUSD']
            
            # Get sentiment for each coin
            major_sentiments = []
            for coin in major_coins:
                sentiment = self.get_news_impact(coin)
                major_sentiments.append(sentiment.get('impact_score', 0))
            
            secondary_sentiments = []
            for coin in secondary_coins:
                sentiment = self.get_news_impact(coin)
                secondary_sentiments.append(sentiment.get('impact_score', 0))
            
            # Calculate weighted average
            if major_sentiments:
                major_avg = sum(major_sentiments) / len(major_sentiments)
            else:
                major_avg = 0
                
            if secondary_sentiments:
                secondary_avg = sum(secondary_sentiments) / len(secondary_sentiments)
            else:
                secondary_avg = 0
            
            # Weight major coins more heavily
            global_sentiment = (major_avg * 0.7) + (secondary_avg * 0.3)
            
            # Detect major events
            events_data = self.detect_market_moving_events()
            
            # Classify market sentiment
            if global_sentiment > 0.3:
                sentiment_label = 'very bullish'
            elif global_sentiment > 0.1:
                sentiment_label = 'bullish'
            elif global_sentiment > -0.1:
                sentiment_label = 'neutral'
            elif global_sentiment > -0.3:
                sentiment_label = 'bearish'
            else:
                sentiment_label = 'very bearish'
                
            # Add volatility indicator
            if events_data.get('detected', False) and events_data.get('count', 0) > 2:
                volatility = 'high'
            elif abs(global_sentiment) > 0.2:
                volatility = 'moderate'
            else:
                volatility = 'low'
            
            return {
                'global_sentiment': global_sentiment,
                'sentiment_label': sentiment_label,
                'volatility': volatility,
                'bitcoin_sentiment': major_sentiments[0] if major_sentiments else 0,
                'ethereum_sentiment': major_sentiments[1] if len(major_sentiments) > 1 else 0,
                'events_detected': events_data.get('count', 0),
                'timestamp': time.time()
            }
                
        except Exception as e:
            self.logger.error(f"Error getting global market sentiment: {str(e)}")
            return {
                'global_sentiment': 0,
                'sentiment_label': 'unknown',
                'volatility': 'unknown',
                'bitcoin_sentiment': 0,
                'ethereum_sentiment': 0,
                'events_detected': 0,
                'timestamp': time.time(),
                'error': str(e)
            }