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

class TwitterSentimentAnalyzer:
    """Advanced Twitter sentiment analyzer optimized for cryptocurrency market intelligence."""
    def __init__(self):
        self.logger = logging.getLogger("TwitterSentimentAnalyzer")
        
        # Load API credentials
        self.api_key = os.environ.get('TWITTER_API_KEY')
        self.api_secret = os.environ.get('TWITTER_API_SECRET')
        self.bearer_token = os.environ.get('TWITTER_BEARER_TOKEN')
        
        # Initialize NLP models
        self.nlp_model = self._initialize_nlp_model()
        
        # Setup caching with adaptive durations - INCREASED CACHE DURATION
        self.cache = {}
        self.disk_cache_path = 'cache/twitter_sentiment'
        os.makedirs(self.disk_cache_path, exist_ok=True)
        self.base_cache_duration = 14400  # 4 hours base cache (increased from 7200/2 hours)
        
        # Different cache durations for different cryptocurrencies - INCREASED
        self.symbol_cache_durations = {
            'XBTUSD': 7200,      # 2 hours for BTC (increased from 3600/1 hour) 
            'ETHUSD': 7200,      # 2 hours for ETH (increased from 3600/1 hour)
            'SOLUSD': 14400,     # 4 hours (increased from 7200/2 hours)
            'AVAXUSD': 14400,    # 4 hours (increased from 7200/2 hours)
            'XRPUSD': 21600,     # 6 hours (increased from 10800/3 hours)
            'XDGUSD': 28800      # 8 hours (increased from 14400/4 hours)
        }
        
        # API usage optimization - IMPROVED RATE LIMITING
        self.daily_request_limit = 300  # More conservative limit (reduced from 450)
        self.daily_requests_used = 0
        self.request_reset_date = datetime.now().date()
        self.rate_limited_until = None  # Time until rate limit expires
        self.last_request_time = 0  # For enforcing minimum intervals between requests
        self.min_request_interval = 60  # Minimum 60 seconds between requests (new)
        
        # Default sentiment values for fallback
        self.default_sentiment = self._initialize_default_sentiment()
        
        # Add additional influencer tracking for better signal quality
        self.crypto_influencers = [
            'cz_binance', 'elonmusk', 'VitalikButerin', 'SBF_FTX', 
            'CoinDesk', 'coinbase', 'binance', 'krakenfx', 'Gemini',
            'michaeljburry', 'PeterSchiff', 'APompliano', 'cryptohayes',
            'CoinMarketCap', 'GaryGensler', 'SEC_Enforcement'
        ]
        
        # Load cached data from disk
        self._load_disk_cache()
        
        # Initialize Twitter client on first use
        self.twitter_client = None
        
        self.logger.info("TwitterSentimentAnalyzer initialized")
    
    def is_rate_limited(self):
        """Check if we're currently under a rate limit restriction."""
        return self.rate_limited_until is not None and time.time() < self.rate_limited_until

    def reset_rate_limits(self):
        """Reset rate limit counters when a new day begins."""
        current_date = datetime.now().date()
        if current_date > self.request_reset_date:
            self.daily_requests_used = 0
            self.request_reset_date = current_date
            self.rate_limited_until = None
            self.logger.info("Twitter API rate limits reset for new day")
            return True
        return False
    
    def get_cache_stats(self):
        """Get statistics about the cache usage."""
        total_items = len(self.cache)
        current_time = time.time()
        
        # Count items by age
        items_under_1h = sum(1 for item in self.cache.values() 
                           if current_time - item.get('timestamp', 0) < 3600)
        items_under_4h = sum(1 for item in self.cache.values() 
                           if current_time - item.get('timestamp', 0) < 14400)
        items_older = total_items - items_under_4h
        
        return {
            'total_items': total_items,
            'under_1h': items_under_1h,
            'under_4h': items_under_4h,
            'older_items': items_older,
            'api_requests_today': self.daily_requests_used,
            'rate_limited': self.is_rate_limited()
        }
    
    def _initialize_nlp_model(self):
        """Initialize VADER NLP model for sentiment analysis."""
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            return SentimentIntensityAnalyzer()
        except ImportError:
            self.logger.error("NLTK not installed. Run 'pip install nltk' and 'python -m nltk.downloader vader_lexicon'")
            return None
    
    def _initialize_twitter_client(self):
        """Initialize Twitter API client with proper error handling."""
        try:
            import tweepy
            
            if not self.bearer_token:
                self.logger.error("Missing Twitter bearer token")
                return None
                
            # Clean and decode token if needed
            bearer_token = self.bearer_token.strip()
            if '%' in bearer_token:
                import urllib.parse
                bearer_token = urllib.parse.unquote(bearer_token)
                
            # Create client with wait_on_rate_limit=False to avoid tweepy waiting
            client = tweepy.Client(
                bearer_token=bearer_token,
                wait_on_rate_limit=False,
                return_type=tweepy.Response
            )
            
            self.logger.info("Twitter API client initialized")
            return client
                
        except ImportError:
            self.logger.error("Tweepy not installed. Run 'pip install tweepy>=4.12.0'")
            return None
        except Exception as e:
            self.logger.error(f"Twitter client initialization error: {str(e)}")
            return None
    
    def _initialize_default_sentiment(self):
        """Create reasonable default sentiment values based on historical patterns."""
        defaults = {
            'XBTUSD': {'score': 0.1, 'volume': 20, 'bullish_ratio': 0.55, 'latest_tweet': None},
            'ETHUSD': {'score': 0.05, 'volume': 15, 'bullish_ratio': 0.52, 'latest_tweet': None},
            'SOLUSD': {'score': 0.02, 'volume': 10, 'bullish_ratio': 0.51, 'latest_tweet': None},
            'AVAXUSD': {'score': 0.0, 'volume': 10, 'bullish_ratio': 0.5, 'latest_tweet': None},
            'XRPUSD': {'score': 0.0, 'volume': 10, 'bullish_ratio': 0.5, 'latest_tweet': None},
            'XDGUSD': {'score': 0.0, 'volume': 10, 'bullish_ratio': 0.5, 'latest_tweet': None}
        }
        
        return defaults
    
    def _load_disk_cache(self):
        """Load cached sentiment data from disk."""
        try:
            cache_file = os.path.join(self.disk_cache_path, 'sentiment_cache.json')
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    
                # Process cache data
                for symbol, data in cache_data.items():
                    # Convert timestamp strings back to float
                    if 'timestamp' in data:
                        data['timestamp'] = float(data['timestamp'])
                    self.cache[symbol] = data
                    
                self.logger.info(f"Loaded {len(cache_data)} items from sentiment cache")
        except Exception as e:
            self.logger.error(f"Error loading sentiment cache: {str(e)}")
    
    def _save_to_disk_cache(self):
        """Save current cache to disk."""
        try:
            cache_file = os.path.join(self.disk_cache_path, 'sentiment_cache.json')
            with open(cache_file, 'w') as f:
                json.dump(self.cache, f)
                
            self.logger.info(f"Saved {len(self.cache)} items to sentiment cache")
        except Exception as e:
            self.logger.error(f"Error saving sentiment cache: {str(e)}")

    def get_twitter_sentiment(self, symbol, lookback_hours=24):
        """Get Twitter sentiment for a cryptocurrency symbol with optimized API usage."""
        current_time = time.time()
        
        # Check cache first
        if symbol in self.cache:
            cache_duration = self.symbol_cache_durations.get(symbol, self.base_cache_duration)
            if (current_time - self.cache[symbol]['timestamp']) < cache_duration:
                self.logger.info(f"Using cached Twitter data for {symbol} (age: {(current_time - self.cache[symbol]['timestamp'])/60:.1f} minutes)")
                return self.cache[symbol]
        
        # Check API rate limits reset
        current_date = datetime.now().date()
        if current_date > self.request_reset_date:
            self.daily_requests_used = 0
            self.request_reset_date = current_date
        
        # Initialize Twitter client if needed
        if not self.twitter_client:
            self.twitter_client = self._initialize_twitter_client()
        
        # If Twitter client is unavailable, return neutral score
        if not self.twitter_client:
            return self._get_neutral_sentiment(symbol)
        
        try:
            # Format search queries for better results
            search_term = symbol.replace('USD', '')
            
            # Use different queries based on symbol
            if search_term == 'XBT':
                search_term = 'Bitcoin OR BTC'
            elif search_term == 'XDG':
                search_term = 'Dogecoin OR DOGE'
            elif search_term == 'XRP':
                search_term = 'Ripple OR XRP'
                
            # Build effective query with crypto-specific terms
            query = f"({search_term}) (crypto OR price)"
            
            self.logger.info(f"Searching Twitter for: {query}")
            
            # Try to get tweets
            try:
                response = self.twitter_client.search_recent_tweets(
                    query=query,
                    max_results=20,
                    tweet_fields=['created_at', 'public_metrics']
                )
                
                if response and hasattr(response, 'data') and response.data:
                    search_tweets = response.data
                    self.logger.info(f"Found {len(search_tweets)} tweets for query: {query}")
                    
                    # Process tweets as before
                    sentiment_scores = []
                    
                    for tweet in search_tweets:
                        if self.nlp_model:
                            sentiment = self.nlp_model.polarity_scores(tweet.text)
                            sentiment_scores.append(sentiment['compound'])
                    
                    # Calculate average sentiment
                    if sentiment_scores:
                        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                        normalized_sentiment = max(-1, min(1, avg_sentiment))
                    else:
                        normalized_sentiment = 0  # Neutral if no scores
                    
                    # Create result
                    result = {
                        'score': normalized_sentiment,
                        'volume': len(search_tweets),
                        'latest_tweet': search_tweets[0].text if search_tweets else None,
                        'timestamp': current_time,
                        'source': 'twitter_api'
                    }
                    
                    # Cache the result
                    self.cache[symbol] = result
                    return result
                else:
                    return self._get_neutral_sentiment(symbol)
                    
            except Exception as e:
                # If rate limit exceeded, return neutral sentiment
                self.logger.warning(f"Twitter API error - returning neutral sentiment: {str(e)}")
                return self._get_neutral_sentiment(symbol)
                
        except Exception as e:
            self.logger.error(f"Error in Twitter sentiment analysis: {str(e)}")
            return self._get_neutral_sentiment(symbol)

    def _get_neutral_sentiment(self, symbol):
        """Return a neutral sentiment score for a symbol."""
        result = {
            'score': 0.0,  # Perfectly neutral
            'volume': 0,
            'latest_tweet': None,
            'timestamp': time.time(),
            'source': 'neutral_fallback'
        }
        
        self.logger.info(f"Using neutral sentiment score for {symbol} due to rate limiting")
        return result
    
    def _get_default_sentiment(self, symbol, reason):
        """Get default sentiment with proper tracking."""
        default_data = self.default_sentiment.get(symbol, {
            'score': 0, 
            'volume': 0, 
            'bullish_ratio': 0.5,
            'latest_tweet': None
        })
        
        # Add metadata
        result = default_data.copy()
        result['timestamp'] = time.time()
        result['source'] = f'default_{reason}'
        
        return result

    def _generate_simulated_data(self, symbol):
        """Generate simulated sentiment data when API access is limited."""
        # Get default values for this symbol
        default_data = self.default_sentiment.get(symbol, {
            'score': 0, 
            'volume': 0, 
            'bullish_ratio': 0.5,
            'latest_tweet': None
        })
        
        # Add some random variation to make it look realistic
        score_variation = random.uniform(-0.1, 0.1)
        volume_variation = random.randint(-3, 3)
        
        # Apply variations to base values
        simulated_score = max(-1.0, min(1.0, default_data.get('score', 0) + score_variation))
        simulated_volume = max(1, default_data.get('volume', 10) + volume_variation)
        simulated_bullish = max(0.0, min(1.0, default_data.get('bullish_ratio', 0.5) + score_variation/5))
        
        # Create result
        result = {
            'score': simulated_score,
            'volume': simulated_volume,
            'bullish_ratio': simulated_bullish,
            'latest_tweet': "Simulated sentiment data - API rate limited",
            'timestamp': time.time(),
            'source': 'simulated'
        }
        
        self.logger.info(f"Generated simulated sentiment data for {symbol}: score={simulated_score:.2f}")
        return result