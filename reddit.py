"""
RedditSentimentAnalyzer - Crypto sentiment analysis from Reddit

This module provides Reddit sentiment analysis for cryptocurrency trading.
"""

import os
import json
import logging
import aiohttp
import asyncio
import time
import re
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import math
import pandas as pd
import numpy as np

class RedditSentimentAnalyzer:
    """Analyzes sentiment from Reddit posts and comments for crypto trading."""
    
    def __init__(self):
        """Initialize the Reddit sentiment analyzer."""
        self.logger = logging.getLogger("RedditSentimentAnalyzer")
        
        # API configurations
        self.reddit_client_id = os.environ.get("REDDIT_CLIENT_ID")
        self.reddit_client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
        self.reddit_user_agent = os.environ.get("REDDIT_USER_AGENT", "KryptosTradingBot/1.0")
        
        # Check API credentials
        self.api_available = bool(self.reddit_client_id) and bool(self.reddit_client_secret)
        if not self.api_available:
            self.logger.warning("Reddit API credentials not configured - using simulated data")
        
        # Token storage
        self.access_token = None
        self.token_expiry = 0
        
        # Setup caching with adaptive durations
        self.cache = {}
        self.disk_cache_path = 'cache/sentiment'
        os.makedirs(self.disk_cache_path, exist_ok=True)
        self.base_cache_duration = 3600  # 1 hour
        
        # Keywords dictionary for specific coins
        self.coin_keywords = {
            'XBTUSD': ['bitcoin', 'btc', 'xbt', 'satoshi', 'sats', '₿'],
            'ETHUSD': ['ethereum', 'eth', 'ether', 'vitalik', 'buterin', 'gwei'],
            'SOLUSD': ['solana', 'sol'],
            'AVAXUSD': ['avalanche', 'avax'],
            'XRPUSD': ['ripple', 'xrp'],
            'XDGUSD': ['dogecoin', 'doge']
        }
        
        # Subreddits to monitor
        self.crypto_subreddits = [
            'cryptocurrency', 'bitcoin', 'ethereum', 'cryptomarkets',
            'solana', 'avalanche', 'ripple', 'dogecoin',
            'cryptomoonshots', 'altcoin', 'defi', 'satoshistreetbets'
        ]
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds
        
        # Initialize NLP model for sentiment analysis
        self.nlp_model = self._initialize_nlp_model()
        
        # Create HTTP session for async requests
        self.session = None
        
        # Load cache from disk
        self._load_disk_cache()
        
        self.logger.info("RedditSentimentAnalyzer initialized")
    
    def _initialize_nlp_model(self):
        """Initialize VADER NLP model for sentiment analysis."""
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            return SentimentIntensityAnalyzer()
        except ImportError:
            self.logger.error("NLTK not installed for Reddit sentiment analysis")
            return None
    
    def _load_disk_cache(self):
        """Load cached sentiment data from disk."""
        try:
            cache_file = os.path.join(self.disk_cache_path, 'reddit_cache.json')
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    
                # Process cache data
                for symbol, data in cache_data.items():
                    # Convert timestamp strings back to float
                    if 'timestamp' in data:
                        data['timestamp'] = float(data['timestamp'])
                    self.cache[symbol] = data
                    
                self.logger.info(f"Loaded {len(cache_data)} items from Reddit cache")
        except Exception as e:
            self.logger.error(f"Error loading Reddit cache: {str(e)}")
    
    def _save_to_disk_cache(self):
        """Save current cache to disk."""
        try:
            cache_file = os.path.join(self.disk_cache_path, 'reddit_cache.json')
            with open(cache_file, 'w') as f:
                json.dump(self.cache, f)
                
            self.logger.info(f"Saved {len(self.cache)} items to Reddit cache")
        except Exception as e:
            self.logger.error(f"Error saving Reddit cache: {str(e)}")
    
    async def _get_access_token(self):
        """Get Reddit API access token."""
        if not self.api_available:
            return False
        
        # Check if token is still valid
        if self.access_token and time.time() < self.token_expiry:
            return True
        
        try:
            if self.session is None:
                self.session = aiohttp.ClientSession()
                
            auth = aiohttp.BasicAuth(
                login=self.reddit_client_id,
                password=self.reddit_client_secret
            )
            
            headers = {"User-Agent": self.reddit_user_agent}
            data = {"grant_type": "client_credentials"}
            
            async with self.session.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=auth,
                headers=headers,
                data=data
            ) as response:
                if response.status != 200:
                    self.logger.error(f"Failed to get Reddit access token: {response.status}")
                    return False
                
                token_data = await response.json()
                self.access_token = token_data.get("access_token")
                
                # Set expiry (slightly less than the actual expiry time)
                expires_in = token_data.get("expires_in", 3600)
                self.token_expiry = time.time() + (expires_in - 60)
                
                self.logger.info("Reddit API access token obtained")
                return True
                
        except Exception as e:
            self.logger.error(f"Error getting Reddit access token: {str(e)}")
            return False
    
    async def _close_session(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
    
    async def _respect_rate_limit(self):
        """Respect API rate limits by waiting if needed."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    async def _search_reddit(self, coin_symbol, subreddit="all", timeframe="day", limit=100):
        """Search Reddit for posts related to a specific coin."""
        if not await self._get_access_token():
            return None
        
        try:
            # Respect rate limits
            await self._respect_rate_limit()
            
            # Generate search query using coin keywords
            keywords = self.coin_keywords.get(coin_symbol, [coin_symbol.replace('USD', '')])
            search_term = ' OR '.join([f'"{keyword}"' for keyword in keywords])
            
            # Set up API request
            url = f"https://oauth.reddit.com/r/{subreddit}/search"
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "User-Agent": self.reddit_user_agent
            }
            
            params = {
                "q": search_term,
                "sort": "relevance",
                "t": timeframe,
                "limit": limit,
                "type": "link"
            }
            
            self.logger.info(f"Searching Reddit for {coin_symbol} in r/{subreddit}")
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 429:  # Too Many Requests
                    self.logger.warning("Reddit API rate limit reached, increasing backoff")
                    self.min_request_interval *= 1.5
                    return None
                
                if response.status != 200:
                    self.logger.error(f"Reddit API error: {response.status}")
                    return None
                
                data = await response.json()
                
                # Extract posts data
                posts = []
                if 'data' in data and 'children' in data['data']:
                    for child in data['data']['children']:
                        post_data = child['data']
                        posts.append({
                            'title': post_data.get('title', ''),
                            'selftext': post_data.get('selftext', ''),
                            'score': post_data.get('score', 0),
                            'num_comments': post_data.get('num_comments', 0),
                            'created_utc': post_data.get('created_utc', 0),
                            'subreddit': post_data.get('subreddit', ''),
                            'permalink': post_data.get('permalink', ''),
                            'id': post_data.get('id', '')
                        })
                
                self.logger.info(f"Found {len(posts)} Reddit posts for {coin_symbol}")
                return posts
                
        except Exception as e:
            self.logger.error(f"Error searching Reddit: {str(e)}")
            return None
    
    async def _get_comments(self, post_id, limit=50):
        """Get comments for a specific Reddit post."""
        if not await self._get_access_token():
            return None
        
        try:
            # Respect rate limits
            await self._respect_rate_limit()
            
            # Set up API request
            url = f"https://oauth.reddit.com/comments/{post_id}"
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "User-Agent": self.reddit_user_agent
            }
            
            params = {
                "limit": limit,
                "sort": "top"
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 429:  # Too Many Requests
                    self.logger.warning("Reddit API rate limit reached, increasing backoff")
                    self.min_request_interval *= 1.5
                    return None
                
                if response.status != 200:
                    self.logger.error(f"Reddit API error: {response.status}")
                    return None
                
                data = await response.json()
                
                # Extract comments
                comments = []
                if len(data) > 1 and 'data' in data[1] and 'children' in data[1]['data']:
                    for child in data[1]['data']['children']:
                        if child['kind'] == 't1' and 'data' in child:  # 't1' is a comment
                            comment_data = child['data']
                            comments.append({
                                'body': comment_data.get('body', ''),
                                'score': comment_data.get('score', 0),
                                'created_utc': comment_data.get('created_utc', 0)
                            })
                
                return comments
                
        except Exception as e:
            self.logger.error(f"Error getting comments: {str(e)}")
            return None
    
    def _analyze_text_sentiment(self, text):
        """Analyze sentiment in text using NLP."""
        if not text or not self.nlp_model:
            return 0
        
        # Clean text
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        
        # Get sentiment score from VADER
        try:
            sentiment = self.nlp_model.polarity_scores(text)
            compound_score = sentiment['compound']
            
            # Normalize to range similar to our target (-1 to 1)
            return compound_score
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {str(e)}")
            return 0
    
    def _analyze_crypto_specific_sentiment(self, text, coin_symbol):
        """Enhanced sentiment analysis with crypto-specific factors."""
        base_score = self._analyze_text_sentiment(text)
        
        # Look for price predictions
        price_pattern = r'\$(\d+[,\.]?\d*)[kK]?'
        price_matches = re.findall(price_pattern, text)
        
        # Bullish/bearish keywords
        bullish_terms = ['moon', 'bullish', 'buy', 'hodl', 'hold', 'up', 'gains', 'profit', 'long']
        bearish_terms = ['crash', 'dump', 'sell', 'short', 'bear', 'bearish', 'dip', 'down', 'loss']
        
        # Check for bullish/bearish terms
        text_lower = text.lower()
        bullish_count = sum(1 for term in bullish_terms if term in text_lower)
        bearish_count = sum(1 for term in bearish_terms if term in text_lower)
        
        # Adjust score based on these factors
        adjustment = 0
        
        # Price predictions
        for price_str in price_matches:
            try:
                price = float(price_str.replace(',', ''))
                # For Bitcoin, any high price prediction is usually bullish
                if coin_symbol == 'XBTUSD' and price > 50000:
                    adjustment += 0.1
                elif coin_symbol == 'ETHUSD' and price > 3000:
                    adjustment += 0.1
            except:
                pass
        
        # Bullish/bearish keyword sentiment
        if bullish_count > bearish_count:
            adjustment += 0.05 * min(3, bullish_count - bearish_count)
        elif bearish_count > bullish_count:
            adjustment -= 0.05 * min(3, bearish_count - bullish_count)
        
        # Cap the adjustment
        adjustment = max(-0.3, min(0.3, adjustment))
        
        # Combine with base score
        final_score = base_score + adjustment
        
        # Ensure score is within -1 to 1
        return max(-1, min(1, final_score))
    
    def _generate_weighted_sentiment_score(self, posts, comments_by_post=None):
        """Generate a weighted sentiment score from posts and comments."""
        if not posts:
            return 0, 0
        
        total_weight = 0
        total_weighted_score = 0
        total_engagement = 0
        
        # Process posts
        for post in posts:
            # Calculate engagement for this post
            engagement = post.get('score', 0) + post.get('num_comments', 0)
            total_engagement += engagement
            
            # Combined text content
            content = f"{post.get('title', '')} {post.get('selftext', '')}"
            
            # Get sentiment for post
            post_sentiment = self._analyze_crypto_specific_sentiment(content, 'BTC')  # Default to BTC
            
            # Weight based on engagement (score + comments)
            weight = 1 + (engagement / 10)  # Base weight plus engagement bonus
            
            # Add to totals
            total_weighted_score += post_sentiment * weight
            total_weight += weight
            
            # Process comments for this post if available
            post_id = post.get('id')
            if comments_by_post and post_id in comments_by_post:
                for comment in comments_by_post[post_id]:
                    comment_text = comment.get('body', '')
                    comment_score = comment.get('score', 0)
                    
                    # Get sentiment for comment
                    comment_sentiment = self._analyze_crypto_specific_sentiment(comment_text, 'BTC')
                    
                    # Weight based on score
                    comment_weight = 0.5 + (comment_score / 20)  # Lower base weight than posts
                    
                    # Add to totals
                    total_weighted_score += comment_sentiment * comment_weight
                    total_weight += comment_weight
        
        # Calculate final sentiment score
        if total_weight > 0:
            final_sentiment = total_weighted_score / total_weight
        else:
            final_sentiment = 0
        
        return final_sentiment, total_engagement
    
    def _generate_simulated_data(self, coin_symbol):
        """Generate simulated Reddit sentiment data when API is unavailable."""
        # Base sentiment values by coin
        base_sentiments = {
            'XBTUSD': 0.2,    # Slightly positive
            'ETHUSD': 0.15,   # Slightly positive
            'SOLUSD': 0.05,   # Neutral to slightly positive
            'AVAXUSD': 0.1,   # Slightly positive
            'XRPUSD': -0.05,  # Neutral to slightly negative
            'XDGUSD': 0.3     # More positive (active community)
        }
        
        # Get base sentiment
        base = base_sentiments.get(coin_symbol, 0)
        
        # Add some random variation (-0.2 to +0.2)
        variation = (random.random() * 0.4) - 0.2
        
        # Add time-based cyclicality (simplified)
        hour_of_day = datetime.now().hour
        day_cycle = math.sin(hour_of_day * math.pi / 12) * 0.1  # -0.1 to 0.1 throughout the day
        
        # Combine factors
        simulated_score = base + variation + day_cycle
        
        # Ensure within bounds
        simulated_score = max(-1, min(1, simulated_score))
        
        # Generate simulated volume based on coin popularity
        volumes = {
            'XBTUSD': random.randint(80, 150),
            'ETHUSD': random.randint(60, 120),
            'SOLUSD': random.randint(30, 70),
            'AVAXUSD': random.randint(20, 50),
            'XRPUSD': random.randint(20, 60),
            'XDGUSD': random.randint(40, 100)
        }
        
        volume = volumes.get(coin_symbol, random.randint(20, 50))
        
        # Generate sample post headlines for context
        headlines = [
            f"What's happening with {coin_symbol.replace('USD', '')} today?",
            f"My thoughts on {coin_symbol.replace('USD', '')} price action",
            f"Is {coin_symbol.replace('USD', '')} a good investment right now?",
            f"Technical analysis for {coin_symbol.replace('USD', '')} - bullish signals?",
            f"{coin_symbol.replace('USD', '')} shows interesting patterns today"
        ]
        
        sample_headline = random.choice(headlines)
        
        return {
            'score': simulated_score,
            'volume': volume,
            'sample_post': sample_headline,
            'timestamp': time.time(),
            'source': 'simulated'
        }
    
    async def get_reddit_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get Reddit sentiment analysis for a cryptocurrency symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'XBTUSD')
            
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
                
                # Return cached data if still valid (1 hour for Reddit data)
                if cache_age < self.base_cache_duration:
                    self.logger.info(f"Using cached Reddit data for {symbol}, age: {cache_age/60:.1f} minutes")
                    return cache_data
            
            # If API not available, use simulated data
            if not self.api_available:
                self.logger.info(f"Using simulated Reddit data for {symbol}")
                result = self._generate_simulated_data(symbol)
                self.cache[symbol] = result
                return result
            
            # Initialize session if needed
            if self.session is None:
                self.session = aiohttp.ClientSession()
            
            # Get Reddit posts from relevant subreddits
            all_posts = []
            
            # First, search coin-specific subreddit if available
            coin_name = symbol.replace('USD', '').lower()
            specific_subreddit = None
            
            # Map coins to specific subreddits
            subreddit_map = {
                'xbt': 'bitcoin',
                'btc': 'bitcoin',
                'eth': 'ethereum',
                'sol': 'solana',
                'avax': 'avalanche',
                'xrp': 'ripple',
                'xdg': 'dogecoin',
                'doge': 'dogecoin'
            }
            
            if coin_name in subreddit_map:
                specific_subreddit = subreddit_map[coin_name]
                coin_posts = await self._search_reddit(symbol, specific_subreddit, "day", 30)
                if coin_posts:
                    all_posts.extend(coin_posts)
            
            # Then search general crypto subreddits
            for subreddit in ['cryptocurrency', 'cryptomarkets', 'cryptomoonshots']:
                if subreddit != specific_subreddit:  # Skip if already searched
                    posts = await self._search_reddit(symbol, subreddit, "day", 20)
                    if posts:
                        all_posts.extend(posts)
            
            if not all_posts:
                self.logger.warning(f"No Reddit posts found for {symbol}, using simulated data")
                result = self._generate_simulated_data(symbol)
                self.cache[symbol] = result
                return result
            
            # Get comments for top posts by engagement
            comments_by_post = {}
            top_posts = sorted(all_posts, key=lambda x: x.get('score', 0) + x.get('num_comments', 0), reverse=True)[:5]
            
            for post in top_posts:
                post_id = post.get('id')
                if post_id:
                    comments = await self._get_comments(post_id, 20)
                    if comments:
                        comments_by_post[post_id] = comments
            
            # Calculate sentiment
            sentiment_score, volume = self._generate_weighted_sentiment_score(all_posts, comments_by_post)
            
            # Get sample headline from top post
            sample_post = top_posts[0]['title'] if top_posts else "No headline available"
            
            # Create result
            result = {
                'score': sentiment_score,
                'volume': volume,
                'sample_post': sample_post,
                'post_count': len(all_posts),
                'comment_count': sum(len(comments) for comments in comments_by_post.values()),
                'timestamp': current_time,
                'source': 'reddit_api'
            }
            
            # Cache the result
            self.cache[symbol] = result
            
            # Save cache to disk occasionally (10% chance)
            if random.random() < 0.1:
                self._save_to_disk_cache()
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error getting Reddit sentiment for {symbol}: {str(e)}")
            
            # Fallback to simulated data on error
            result = self._generate_simulated_data(symbol)
            return result
        
        finally:
            # Ensure session is eventually closed
            if self.session and random.random() < 0.1:  # 10% chance to close on each call
                await self._close_session()
    
    def get_subreddit_trending_coins(self) -> List[Dict[str, Any]]:
        """Get trending coins based on Reddit activity and sentiment."""
        try:
            # If using simulated data, return pre-defined trends
            if not self.api_available:
                return [
                    {'symbol': 'XBTUSD', 'sentiment': 0.3, 'mentions': 120, 'trend': 'rising'},
                    {'symbol': 'ETHUSD', 'sentiment': 0.2, 'mentions': 95, 'trend': 'stable'},
                    {'symbol': 'SOLUSD', 'sentiment': 0.5, 'mentions': 60, 'trend': 'hot'},
                    {'symbol': 'AVAXUSD', 'sentiment': 0.1, 'mentions': 30, 'trend': 'neutral'},
                    {'symbol': 'XRPUSD', 'sentiment': -0.1, 'mentions': 25, 'trend': 'declining'}
                ]
            
            # Collect data from cache for analysis
            trends = []
            for symbol, data in self.cache.items():
                if 'score' in data and 'volume' in data:
                    # Calculate trend based on volume and sentiment
                    trend = 'neutral'
                    if data['volume'] > 100 and data['score'] > 0.4:
                        trend = 'hot'
                    elif data['volume'] > 80 and data['score'] > 0.2:
                        trend = 'rising'
                    elif data['score'] < -0.2:
                        trend = 'declining'
                    elif data['volume'] > 50:
                        trend = 'stable'
                    
                    trends.append({
                        'symbol': symbol,
                        'sentiment': data['score'],
                        'mentions': data['volume'],
                        'trend': trend
                    })
            
            # Sort by mention volume
            trends.sort(key=lambda x: x['mentions'], reverse=True)
            
            return trends[:10]  # Return top 10
            
        except Exception as e:
            self.logger.error(f"Error getting trending coins: {str(e)}")
            return []

