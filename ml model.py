"""
KryptosTradingBot - ML Model Manager

This module handles the machine learning models for trading predictions.
"""

import os
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

class MLModelManager:
    """ML model management and prediction with ensemble methods."""
    
    def __init__(self):
        """Initialize the ML model manager."""
        self.logger = logging.getLogger("MLModelManager")
        
        # Model state
        self.scaler = StandardScaler()
        self.model = None
        self.feature_importance = {}
        self.is_trained = False
        self.min_samples = 100
        
        # Directory for model storage
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Standard features
        self._feature_names = [
            'returns', 'log_returns', 'rolling_std_20', 'volume_ma_ratio',
            'rsi', 'macd', 'bb_width', 'adx', 'sma_20', 'atr', 'cci',
            # Adding missing features that were in the error message
            'adx_neg', 'adx_pos', 'bb_lower', 'bb_position', 'bb_upper',
            'macd_signal', 'macd_diff', 'sma_50', 'sma_20_50_ratio',
            'volume_std', 'rsi_divergence', 'rolling_std_50'
        ]
        
        # Model performance metrics
        self.performance_history = {
            'train_accuracy': [],
            'val_accuracy': [],
            'live_accuracy': [],
            'feature_importance': {}
        }
    
    def load_model(self) -> bool:
        """Load ML model from disk."""
        try:
            model_path = os.path.join(self.models_dir, "ml_ensemble_model.joblib")
            scaler_path = os.path.join(self.models_dir, "feature_scaler.joblib")
            feature_list_path = os.path.join(self.models_dir, "feature_list.json")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                
                # Load feature list if available
                if os.path.exists(feature_list_path):
                    with open(feature_list_path, 'r') as f:
                        self._feature_names = json.load(f)
                    self.logger.info(f"Loaded feature list with {len(self._feature_names)} features")
                else:
                    self.logger.info(f"Using default feature list ({len(self._feature_names)})")
                
                # Load feature importance if available
                importance_path = os.path.join(self.models_dir, "feature_importance.json")
                if os.path.exists(importance_path):
                    with open(importance_path, 'r') as f:
                        self.feature_importance = json.load(f)
                
                # Load performance history if available
                history_path = os.path.join(self.models_dir, "performance_history.json")
                if os.path.exists(history_path):
                    with open(history_path, 'r') as f:
                        self.performance_history = json.load(f)
                
                self.is_trained = True
                self.logger.info("ML model loaded successfully")
                return True
            else:
                self.logger.warning("ML model files not found, need training")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading ML model: {str(e)}")
            return False
    
    def predict(self, features_df: pd.DataFrame) -> Dict:
        """Make predictions with scaled confidence scores.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained or self.model is None:
            return {'action': 'hold', 'confidence': 0.5}
                    
        try:
            # Get the last row for prediction (we'll create any missing columns later)
            features_row = features_df.iloc[[-1]]
            
            # Get the feature names the model was trained with
            required_features = self.get_trained_features()
            
            # Create a new DataFrame with all required features
            filtered_features = pd.DataFrame(index=features_row.index)
            
            # Add existing features from input data
            for feature in required_features:
                if feature in features_row.columns:
                    filtered_features[feature] = features_row[feature]
                else:
                    # If any required feature is missing, add a column of zeros
                    self.logger.warning(f"Added missing feature {feature} with zeros")
                    filtered_features[feature] = 0.0
            
            # Ensure feature order matches what the model expects
            filtered_features = filtered_features[required_features]
            
            # Scale the features
            try:
                features_scaled = self.scaler.transform(filtered_features)
            except Exception as scale_err:
                self.logger.error(f"Error scaling features: {str(scale_err)}")
                # Try direct prediction without scaling as fallback
                features_scaled = filtered_features.values
            
            # Get raw probabilities
            try:
                raw_probs = self.model.predict_proba(features_scaled)[-1]
                
                # Extract buy probability (class 1)
                if len(raw_probs) >= 2:
                    buy_prob = raw_probs[1]
                else:
                    # If only one class is returned, use direct prediction
                    pred = self.model.predict(features_scaled)[-1]
                    buy_prob = float(pred)
            except Exception as pred_err:
                self.logger.error(f"Error in model prediction: {str(pred_err)}")
                # Return neutral prediction on error
                return {'action': 'hold', 'confidence': 0.5}
            
            # Center the buy probability around 0.5
            # Transform from [0, 1] to [0.45, 0.55]
            base_confidence = 0.5 + (buy_prob - 0.5) * 0.2
            
            # Add technical confirmation adjustments
            latest = features_df.iloc[-1]
            adjustment = 0
            
            # RSI confirmation (small adjustment)
            if 'rsi' in latest:
                rsi = latest['rsi']
                if rsi < 30 and buy_prob > 0.5:  # Oversold + bullish signal
                    adjustment += 0.01
                elif rsi > 70 and buy_prob < 0.5:  # Overbought + bearish signal
                    adjustment -= 0.01
            
            # MACD confirmation (small adjustment)
            if 'macd' in latest and 'macd_signal' in latest:
                macd_diff = latest['macd'] - latest['macd_signal']
                if macd_diff > 0 and buy_prob > 0.5:  # Positive MACD + bullish
                    adjustment += 0.01
                elif macd_diff < 0 and buy_prob < 0.5:  # Negative MACD + bearish
                    adjustment -= 0.01
            
            # Volume confirmation (tiny adjustment)
            if 'volume_ma_ratio' in latest:
                vol_ratio = latest['volume_ma_ratio']
                if vol_ratio > 1.2:  # High volume
                    if buy_prob > 0.5:
                        adjustment += 0.005
                    else:
                        adjustment -= 0.005
            
            # Apply adjustment while maintaining bounds
            final_confidence = max(0.40, min(0.60, base_confidence + adjustment))
            
            # Determine action based on confidence
            if final_confidence > 0.53:
                action = 'buy'
            elif final_confidence < 0.47:  # More eager to take profits
                action = 'sell'
            else:
                action = 'hold'
            
            # Add feature influence for top influencing features
            feature_influences = {}
            if self.feature_importance:
                # Get the most important features
                top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                for feature, importance in top_features:
                    if feature in latest:
                        feature_influences[feature] = {
                            'value': float(latest[feature]),
                            'importance': float(importance)
                        }
            
            self.logger.info(f"ML Prediction: raw_prob={buy_prob:.3f}, "
                            f"adj={adjustment:.3f}, "
                            f"final={final_confidence:.3f}, "
                            f"action={action}")
                    
            return {
                'action': action,
                'confidence': final_confidence,
                'raw_probability': buy_prob,
                'adjustments': adjustment,
                'feature_influences': feature_influences
            }

        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            traceback.print_exc()
            return {'action': 'hold', 'confidence': 0.5}
    
    def prepare_features(self, df: pd.DataFrame, use_available_only: bool = False) -> pd.DataFrame:
        """Create comprehensive features with enhanced technical indicators."""
        try:
            # Verify required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.warning(f"Missing columns: {missing_columns}")
                
            # Create copy to avoid modifying original
            df_copy = df.copy()
            
            # Ensure numeric data
            for col in [c for c in required_columns if c in df_copy.columns]:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                
            # Remove any rows with NaN values in required columns
            for col in [c for c in required_columns if c in df_copy.columns]:
                df_copy = df_copy.dropna(subset=[col])
            
            # Create features dataframe with ALL required features
            features = pd.DataFrame(index=df_copy.index)
            
            # Add close price and returns
            if 'close' in df_copy.columns:
                features['close'] = df_copy['close']
                features['returns'] = df_copy['close'].pct_change()
                features['log_returns'] = np.log1p(features['returns'])
                features['rolling_std_20'] = features['returns'].rolling(window=20, min_periods=1).std()
                features['rolling_std_50'] = features['returns'].rolling(window=50, min_periods=1).std()
            else:
                features['close'] = 0
                features['returns'] = 0
                features['log_returns'] = 0
                features['rolling_std_20'] = 0
                features['rolling_std_50'] = 0
            
            # Volume features
            if 'volume' in df_copy.columns:
                features['volume'] = df_copy['volume']
                vol_ma = df_copy['volume'].rolling(window=20, min_periods=1).mean()
                features['volume_ma_ratio'] = df_copy['volume'] / vol_ma
                features['volume_std'] = df_copy['volume'].rolling(window=20, min_periods=1).std()
            else:
                features['volume'] = 0
                features['volume_ma_ratio'] = 1.0
                features['volume_std'] = 0.0
            
            # RSI - Relative Strength Index
            if 'close' in df_copy.columns:
                delta = df_copy['close'].diff()
                gain = (delta.where(delta > 0, 0)).fillna(0)
                loss = (-delta.where(delta < 0, 0)).fillna(0)
                avg_gain = gain.rolling(window=14, min_periods=1).mean()
                avg_loss = loss.rolling(window=14, min_periods=1).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
                features['rsi'] = 100 - (100 / (1 + rs))
                features['rsi'] = features['rsi'].fillna(50)
                features['rsi_divergence'] = features['rsi'].diff()
            else:
                features['rsi'] = 50
                features['rsi_divergence'] = 0
            
            # MACD - Moving Average Convergence Divergence
            if 'close' in df_copy.columns:
                ema12 = df_copy['close'].ewm(span=12, adjust=False).mean()
                ema26 = df_copy['close'].ewm(span=26, adjust=False).mean()
                features['macd'] = ema12 - ema26
                features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
                features['macd_diff'] = features['macd'] - features['macd_signal']
            else:
                features['macd'] = 0
                features['macd_signal'] = 0
                features['macd_diff'] = 0
            
            # Bollinger Bands
            if 'close' in df_copy.columns:
                features['sma_20'] = df_copy['close'].rolling(window=20, min_periods=1).mean()
                bb_std = df_copy['close'].rolling(window=20, min_periods=1).std()
                features['bb_upper'] = features['sma_20'] + (bb_std * 2)
                features['bb_lower'] = features['sma_20'] - (bb_std * 2)
                features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['sma_20']
                features['bb_position'] = (df_copy['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-10)
            else:
                features['sma_20'] = 0
                features['bb_width'] = 0.1
                features['bb_position'] = 0.5
                features['bb_upper'] = 0
                features['bb_lower'] = 0
            
            # Moving averages
            if 'close' in df_copy.columns:
                features['sma_50'] = df_copy['close'].rolling(window=50, min_periods=1).mean()
                features['sma_20_50_ratio'] = features['sma_20'] / features['sma_50']
            else:
                features['sma_50'] = 0
                features['sma_20_50_ratio'] = 1.0
            
            # ATR - Average True Range
            if all(col in df_copy.columns for col in ['high', 'low', 'close']):
                high_low = df_copy['high'] - df_copy['low']
                high_close = (df_copy['high'] - df_copy['close'].shift()).abs()
                low_close = (df_copy['low'] - df_copy['close'].shift()).abs()
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                features['atr'] = true_range.rolling(window=14, min_periods=1).mean()
            else:
                features['atr'] = 1.0
            
            # ADX - Average Directional Index
            if all(col in df_copy.columns for col in ['high', 'low', 'close']):
                # +DM and -DM
                up_move = df_copy['high'].diff()
                down_move = df_copy['low'].diff().multiply(-1)
                
                # Filter for actual moves
                plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
                minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
                
                # Smoothed +DM and -DM
                plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=1).mean().values
                minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=1).mean().values
                
                # True Range
                true_range_14 = pd.Series(true_range).rolling(window=14, min_periods=1).mean().values
                
                # +DI and -DI with protection against division by zero
                plus_di = np.where(true_range_14 != 0, (plus_dm_14 / true_range_14) * 100, 0)
                minus_di = np.where(true_range_14 != 0, (minus_dm_14 / true_range_14) * 100, 0)
                
                # DX
                dx_sum = plus_di + minus_di
                dx = np.zeros_like(dx_sum)
                nonzero_mask = dx_sum != 0
                if np.any(nonzero_mask):
                    dx[nonzero_mask] = (np.abs(plus_di[nonzero_mask] - minus_di[nonzero_mask]) / dx_sum[nonzero_mask]) * 100

                
                # ADX
                features['adx'] = pd.Series(dx).rolling(window=14, min_periods=1).mean()
                features['adx_pos'] = pd.Series(plus_di)
                features['adx_neg'] = pd.Series(minus_di)
            else:
                features['adx'] = 20
                features['adx_pos'] = 20
                features['adx_neg'] = 20
            
            # CCI - Commodity Channel Index
            if all(col in df_copy.columns for col in ['high', 'low', 'close']):
                typical_price = (df_copy['high'] + df_copy['low'] + df_copy['close']) / 3
                mean_dev = abs(typical_price - typical_price.rolling(window=20, min_periods=1).mean())
                mean_dev_avg = mean_dev.rolling(window=20, min_periods=1).mean()
                features['cci'] = (typical_price - typical_price.rolling(window=20, min_periods=1).mean()) / (0.015 * mean_dev_avg)
            else:
                features['cci'] = 0
            
            # Clean up any infinity or NaN values
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.ffill().bfill().fillna(0)
            
            # Log which features were successfully created
            created_features = features.columns.tolist()
            self.logger.info(f"Created {len(created_features)} features: {created_features}")
            
            # We need to always return ALL features, regardless of use_available_only
            # This is critical to fix the feature mismatch issue
            return features
                
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            traceback.print_exc()
            # Return empty DataFrame on error
            return pd.DataFrame()
    
    def create_labels(self, df: pd.DataFrame, forward_window: int = 12) -> np.ndarray:
        """Create labels for training.
        
        Args:
            df: DataFrame with price data
            forward_window: Window size for forward returns
            
        Returns:
            NumPy array of labels
        """
        try:
            if 'close' not in df.columns:
                raise ValueError("DataFrame must contain 'close' column")
                
            # Calculate forward returns at multiple timeframes
            short_forward = df['close'].shift(-forward_window//2) / df['close'] - 1
            medium_forward = df['close'].shift(-forward_window) / df['close'] - 1
            long_forward = df['close'].shift(-int(forward_window*1.5)) / df['close'] - 1
            
            # Calculate target thresholds - dynamic based on volatility
            volatility = df['close'].pct_change().rolling(forward_window).std()
            threshold = volatility * 1.5  # Significant move threshold
            
            # Initialize labels with neutral (0)
            labels = pd.Series(0, index=df.index)
            
            # Strong bullish signals
            bullish_condition = (
                (short_forward > threshold) |  # Short-term gain
                (medium_forward > threshold * 0.8) |  # Medium-term gain
                ((short_forward > 0) & (medium_forward > threshold * 0.7) & (long_forward > threshold * 0.6))  # Consistent gains
            )
            
            # Strong bearish signals
            bearish_condition = (
                (short_forward < -threshold) |  # Short-term loss
                (medium_forward < -threshold * 0.8) |  # Medium-term loss
                ((short_forward < 0) & (medium_forward < -threshold * 0.7) & (long_forward < -threshold * 0.6))  # Consistent losses
            )
            
            # Assign labels
            labels[bullish_condition] = 1  # Bullish
            labels[bearish_condition] = 0  # Bearish (using 0 instead of -1 for binary classification)
            
            # Remove future lookahead bias
            labels.iloc[-forward_window:] = np.nan
            
            # Log label distribution
            valid_labels = labels.dropna()
            if len(valid_labels) > 0:
                distribution = valid_labels.value_counts(normalize=True) * 100
                self.logger.info(f"Label distribution: {distribution.to_dict()}")
            
            return labels.values
            
        except Exception as e:
            self.logger.error(f"Error creating labels: {str(e)}")
            traceback.print_exc()
            return np.array([])
    
    def prepare_training_data(self, features: pd.DataFrame, labels: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare data for training by handling NaN values and balancing classes.
        
        Args:
            features: DataFrame with features
            labels: NumPy array of labels
            
        Returns:
            Tuple of (features_clean, labels_clean)
        """
        try:
            # Ensure features and labels have the same length
            if len(features) != len(labels):
                raise ValueError(f"Features length ({len(features)}) does not match labels length ({len(labels)})")
                
            # Remove NaN values from both features and labels
            valid_indices = ~np.isnan(labels)
            features_clean = features.loc[features.index[valid_indices]].copy()
            labels_clean = labels[valid_indices]
            
            # Check if we have enough data
            if len(features_clean) < self.min_samples:
                self.logger.warning(f"Not enough clean data for training: {len(features_clean)} samples (minimum {self.min_samples})")
                return None, None
            
            # For multi-class, convert to binary if needed (1 for buy, 0 for sell/hold)
            # Uncomment if binary classification is needed
            # labels_binary = (labels_clean > 0).astype(int)
            
            self.logger.info(f"Features shape after cleaning: {features_clean.shape}")
            self.logger.info(f"Labels shape after cleaning: {len(labels_clean)}")
            
            return features_clean, labels_clean
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            traceback.print_exc()
            return None, None

    def train_model(self, features_df: pd.DataFrame, labels: np.ndarray) -> bool:
        """Train ML model with ensemble methods and hyperparameter optimization.
        
        Args:
            features_df: DataFrame with features for training
            labels: NumPy array of binary labels (0/1)
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            # Print input shapes
            self.logger.info(f"Features shape: {features_df.shape}")
            self.logger.info(f"Labels shape: {labels.shape}")
            
            # Prepare training data
            features_clean, labels_clean = self.prepare_training_data(features_df, labels)
            
            if features_clean is None or labels_clean is None:
                self.logger.error("Failed to prepare training data")
                return False
            
            # Save the feature names used for training
            self._feature_names = features_clean.columns.tolist()
            
            # Split data with stratification to ensure balanced classes
            X_train, X_test, y_train, y_test = train_test_split(
                features_clean, labels_clean, test_size=0.2, random_state=42, stratify=labels_clean
            )
            
            # Scale features - important for model performance
            self.scaler = StandardScaler().fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Calculate class weights to handle imbalanced data
            unique_classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
            class_weight_dict = dict(zip(unique_classes, class_weights))
            
            self.logger.info(f"Class weights: {class_weight_dict}")
            
            # Build the ensemble model
            self.model = self.build_ensemble_model()
            
            # Print model summary
            self.logger.info(f"Training ensemble model with {len(X_train)} samples")
            
            # Train the model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate on test set
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Log evaluation metrics
            self.logger.info(f"Model Evaluation:")
            self.logger.info(f"- Accuracy: {accuracy:.4f}")
            self.logger.info(f"- Confusion Matrix:\n{cm}")
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Safely access report values with fallbacks for both string and numeric keys
            class1_key = '1' if '1' in report else 1
            if class1_key in report:
                self.logger.info(f"- Precision (Class 1): {report[class1_key]['precision']:.4f}")
                self.logger.info(f"- Recall (Class 1): {report[class1_key]['recall']:.4f}")
                self.logger.info(f"- F1-Score (Class 1): {report[class1_key]['f1-score']:.4f}")
            else:
                self.logger.info("- No Class 1 metrics available in classification report")
            
            # Get feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_importance = {}
                
                for i, feature in enumerate(features_clean.columns):
                    feature_importance[feature] = float(importances[i])
                
                # Sort by importance
                sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                
                self.logger.info("Feature Importance:")
                for feature, importance in sorted_importance[:10]:  # Top 10
                    self.logger.info(f"  {feature}: {importance:.4f}")
                
                self.feature_importance = feature_importance
                
                # Store in performance history
                self.performance_history['feature_importance'] = feature_importance
            elif hasattr(self.model, 'estimators_') and hasattr(self.model.estimators_[0], 'feature_importances_'):
                # For voting classifier, get importances from first estimator
                importances = self.model.estimators_[0].feature_importances_
                feature_importance = {}
                
                for i, feature in enumerate(features_clean.columns):
                    feature_importance[feature] = float(importances[i])
                
                # Sort by importance
                sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                
                self.logger.info("Feature Importance (from first estimator):")
                for feature, importance in sorted_importance[:10]:  # Top 10
                    self.logger.info(f"  {feature}: {importance:.4f}")
                
                self.feature_importance = feature_importance
                
                # Store in performance history
                self.performance_history['feature_importance'] = feature_importance
            
            # Update performance history
            self.performance_history['train_accuracy'].append({
                'timestamp': datetime.now().isoformat(),
                'accuracy': float(accuracy),
                'precision': float(report[class1_key]['precision']) if class1_key in report else 0.0,
                'recall': float(report[class1_key]['recall']) if class1_key in report else 0.0,
                'f1_score': float(report[class1_key]['f1-score']) if class1_key in report else 0.0,
                'sample_size': len(X_train)
            })
            
            # Set model as trained
            self.is_trained = True
            
            # Save the model
            self._save_model()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            traceback.print_exc()
            return False
            
    def get_trained_features(self):
        """Get the list of features that the model was trained with."""
        if self.model is None:
            # Return default feature list if model is not available
            return self._feature_names
        
        # Check if feature list is saved
        feature_list_path = os.path.join(self.models_dir, "feature_list.json")
        if os.path.exists(feature_list_path):
            try:
                with open(feature_list_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading feature list: {str(e)}")
        
        # Use stored feature names
        return self._feature_names

    def build_ensemble_model(self):
        """Build an ensemble model for more robust predictions."""
        try:
            # Base models
            rf = RandomForestClassifier(
                n_estimators=300, 
                max_depth=12,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=42
            )
            
            gb = GradientBoostingClassifier(
                n_estimators=200, 
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=42
            )
            
            # Neural network
            mlp = MLPClassifier(
                hidden_layer_sizes=(100,50), 
                max_iter=500,
                alpha=0.001,
                learning_rate_init=0.001,
                early_stopping=True,
                random_state=42
            )
            
            # Create voting ensemble
            ensemble = VotingClassifier(
                estimators=[
                    ('rf', rf), 
                    ('gb', gb), 
                    ('mlp', mlp)
                ], 
                voting='soft'
            )
            
            self.logger.info("Built ML ensemble model with Random Forest, Gradient Boost, and Neural Network")
            return ensemble
            
        except Exception as e:
            self.logger.error(f"Error building ensemble model: {str(e)}")
            # Fall back to RandomForest only
            return RandomForestClassifier(n_estimators=200, random_state=42)

    def _save_model(self):
        """Save the trained model and scaler to disk with feature list."""
        try:
            if not os.path.exists(self.models_dir):
                os.makedirs(self.models_dir)
                
            # Save model
            model_path = os.path.join(self.models_dir, "ml_ensemble_model.joblib")
            joblib.dump(self.model, model_path)
            
            # Save scaler
            scaler_path = os.path.join(self.models_dir, "feature_scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            
            # Save feature importance
            if self.feature_importance:
                with open(os.path.join(self.models_dir, "feature_importance.json"), 'w') as f:
                    json.dump(self.feature_importance, f)
                    
            # Save feature list for consistency between training and inference
            feature_list = self._feature_names
            with open(os.path.join(self.models_dir, "feature_list.json"), 'w') as f:
                json.dump(feature_list, f)
            
            # Save performance history
            with open(os.path.join(self.models_dir, "performance_history.json"), 'w') as f:
                # Convert numpy types to standard Python types for JSON serialization
                history_copy = self.performance_history.copy()
                if 'feature_importance' in history_copy:
                    history_copy['feature_importance'] = {k: float(v) for k, v in history_copy['feature_importance'].items()}
                json.dump(history_copy, f)
                
            self.logger.info(f"Model, scaler, feature list and metadata saved to {self.models_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False

    def evaluate_prediction_accuracy(self, predicted_actions, actual_results):
        """Track and evaluate prediction accuracy over time."""
        try:
            # Calculate accuracy metrics
            correct = 0
            for pred, actual in zip(predicted_actions, actual_results):
                if (pred == 'buy' and actual > 0) or (pred == 'sell' and actual < 0):
                    correct += 1
            
            accuracy = correct / len(predicted_actions) if predicted_actions else 0
            
            # Update live accuracy tracking
            self.performance_history['live_accuracy'].append({
                'timestamp': datetime.now().isoformat(),
                'accuracy': accuracy,
                'sample_size': len(predicted_actions)
            })
            
            # Calculate moving average accuracy
            recent_accuracies = [entry['accuracy'] for entry in self.performance_history['live_accuracy'][-10:]]
            avg_accuracy = sum(recent_accuracies) / len(recent_accuracies) if recent_accuracies else 0
            
            self.logger.info(f"Prediction accuracy: {accuracy:.4f}, Moving average: {avg_accuracy:.4f}")
            
            # Save updated performance metrics
            with open(os.path.join(self.models_dir, "performance_history.json"), 'w') as f:
                json.dump(self.performance_history, f)
                
            return accuracy, avg_accuracy
            
        except Exception as e:
            self.logger.error(f"Error evaluating prediction accuracy: {str(e)}")
            return 0, 0

    def feature_ablation_study(self, X, y, n_iterations=5):
        """Perform feature ablation study to identify truly important features."""
        try:
            self.logger.info("Starting feature ablation study...")
            
            # Prepare data
            X_clean, y_clean = self.prepare_training_data(X, y)
            if X_clean is None or y_clean is None:
                return {}
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=0.3, random_state=42
            )
            
            # Create base model for testing
            base_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            )
            
            # Baseline performance with all features
            base_model.fit(X_train, y_train)
            baseline_score = base_model.score(X_test, y_test)
            self.logger.info(f"Baseline accuracy with all features: {baseline_score:.4f}")
            
            # Feature importance dict
            feature_impact = {}
            
            # Test each feature by removing it
            for feature in X_clean.columns:
                scores = []
                
                for i in range(n_iterations):
                    # Create dataset without this feature
                    X_train_reduced = X_train.drop(columns=[feature])
                    X_test_reduced = X_test.drop(columns=[feature])
                    
                    # Train and evaluate
                    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42+i)
                    model.fit(X_train_reduced, y_train)
                    score = model.score(X_test_reduced, y_test)
                    scores.append(score)
                
                # Average score without this feature
                avg_score = sum(scores) / len(scores)
                
                # Impact = baseline_score - avg_score
                # Positive impact means removing the feature hurts performance
                impact = baseline_score - avg_score
                feature_impact[feature] = impact
                
                self.logger.info(f"Feature '{feature}' impact: {impact:.4f}")
            
            # Sort features by impact
            sorted_impact = sorted(feature_impact.items(), key=lambda x: x[1], reverse=True)
            self.logger.info(f"Top impactful features: {sorted_impact[:10]}")
            
            return feature_impact
            
        except Exception as e:
            self.logger.error(f"Error in feature ablation study: {str(e)}")
            return {}

    def get_feature_correlations(self, features_df: pd.DataFrame):
        """Calculate and visualize feature correlations."""
        try:
            # Calculate correlation matrix
            corr_matrix = features_df.corr()
            
            # Find highly correlated features
            high_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr = corr_matrix.iloc[i, j]
                    
                    if abs(corr) > 0.8:  # High correlation threshold
                        high_correlations.append((col1, col2, corr))
            
            # Sort by absolute correlation
            high_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # Log correlations
            if high_correlations:
                self.logger.info("Highly correlated features:")
                for col1, col2, corr in high_correlations[:10]:  # Top 10
                    self.logger.info(f"  {col1} - {col2}: {corr:.4f}")
            
            return high_correlations
                
        except Exception as e:
            self.logger.error(f"Error calculating feature correlations: {str(e)}")
            return []