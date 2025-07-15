"""
AITradingEnhancer - Advanced AI models for trading predictions

This module implements deep learning models for price movement prediction,
including LSTM and Transformer architectures.
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
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, GlobalAveragePooling1D, Input,
    MultiHeadAttention, LayerNormalization, Layer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.saving import register_keras_serializable
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

@register_keras_serializable(package="trading_models")
class TransformerBlock(Layer):
    """Transformer block with configurable dimensions for time series data."""
    def __init__(self, embed_dim=8, num_heads=4, ff_dim=32, dropout=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.att = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            value_dim=self.embed_dim // self.num_heads
        )
        
        self.ffn = Sequential([
            Dense(self.ff_dim, activation="relu"),
            Dense(self.embed_dim)
        ])
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(self.dropout_rate)
        self.dropout2 = Dropout(self.dropout_rate)

    def build(self, input_shape):
        """Build method to properly initialize the layer with input shape."""
        if input_shape is None:
            raise ValueError("Input shape cannot be None")
            
        # Mark as built
        self.built = True
        
        # Build layers individually with correct shapes
        self.layernorm1.build(input_shape)
        self.layernorm2.build(input_shape)
        self.dropout1.build(input_shape)
        self.dropout2.build(input_shape)
        
        # For MultiHeadAttention, we need to call it once to build it properly
        # Create dummy tensors with valid shape
        batch_size = 1
        seq_len = input_shape[1] if len(input_shape) > 1 else 1
        dummy_shape = (batch_size, seq_len, self.embed_dim)
        dummy_tensor = tf.random.normal(shape=dummy_shape)
        self.att(dummy_tensor, dummy_tensor)
        
        # Build the feed-forward network with the right shape
        ffn_input_shape = input_shape
        self.ffn.build(ffn_input_shape)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate
        })
        return config

class AITradingEnhancer:
    """Enhanced AI models for advanced trading decisions with LSTM and Transformer architectures."""
    def __init__(self):
        """Initialize the AI trading enhancer."""
        self.logger = logging.getLogger("AITradingEnhancer")
        
        # Model components
        self.lstm_model = None
        self.transformer_model = None
        self.hybrid_model = None  # Combining LSTM + Transformer
        self.feature_scaler = StandardScaler()
        
        # Model parameters
        self.sequence_length = 100
        self.n_features = 11  # Standard feature set size
        
        # Model directory
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
    
    def rebuild_models_for_tensorflow_compatibility(self):
        """Rebuild AI models for compatibility with current TensorFlow version."""
        try:
            self.logger.info("Rebuilding AI models for TensorFlow compatibility")
            
            # Build new LSTM model
            self.lstm_model = self.build_lstm_model()
            
            # Build new transformer model
            self.transformer_model = self.build_transformer_model()
            
            # Build new hybrid model
            self.hybrid_model = self.build_hybrid_model()
            
            # Save the rebuilt models
            self.save_models()
            
            self.logger.info("AI models rebuilt successfully for TensorFlow compatibility")
            return True
        except Exception as e:
            self.logger.error(f"Error rebuilding AI models: {str(e)}")
            traceback.print_exc()
            return False

    def load_models(self):
        """Load ML model from disk with improved error handling."""
        try:
            model_paths = {
                'lstm': os.path.join(self.models_dir, "lstm_model.keras"),
                'transformer': os.path.join(self.models_dir, "transformer_model.keras"),
                'hybrid': os.path.join(self.models_dir, "hybrid_model.keras"),
                'scaler': os.path.join(self.models_dir, "ai_feature_scaler.joblib"),
                'feature_count': os.path.join(self.models_dir, "ai_feature_count.json")
            }
            
            # Check if all files exist
            missing_files = [path for path, file_path in model_paths.items() if not os.path.exists(file_path)]
            
            if missing_files:
                self.logger.warning(f"Missing model files: {missing_files}")
                
                # Try to create directories if they don't exist
                os.makedirs(self.models_dir, exist_ok=True)
                
                # If essential files are missing, rebuild models
                if 'scaler' in missing_files or ('lstm' in missing_files and 'transformer' in missing_files):
                    self.logger.info("Essential model files missing, rebuilding models...")
                    return self.rebuild_models_for_current_features()
            
            # Try to load each model with proper error handling
            custom_objects = {'TransformerBlock': TransformerBlock}
            models_loaded = 0
            
            # Load feature count first
            if os.path.exists(model_paths['feature_count']):
                try:
                    with open(model_paths['feature_count'], 'r') as f:
                        saved_config = json.load(f)
                        saved_features = saved_config.get("n_features", 0)
                        
                        if saved_features != self.n_features:
                            self.logger.warning(f"Feature count mismatch: saved models use {saved_features} features, "
                                            f"but current setting is {self.n_features}")
                            
                            # Only rebuild if the mismatch is significant
                            if abs(saved_features - self.n_features) > 0:
                                self.logger.warning("Rebuilding models for current features")
                                return self.rebuild_models_for_current_features()
                            else:
                                # Use the saved feature count
                                self.n_features = saved_features
                                self.logger.info(f"Adjusted to saved feature count: {self.n_features}")
                except Exception as e:
                    self.logger.error(f"Error loading feature count: {str(e)}")
            
            # Load LSTM model
            if os.path.exists(model_paths['lstm']):
                try:
                    self.lstm_model = load_model(model_paths['lstm'], custom_objects=custom_objects)
                    
                    # Verify input shape
                    input_shape = self.lstm_model.input_shape
                    if input_shape and len(input_shape) >= 3 and input_shape[-1] != self.n_features:
                        self.logger.warning(f"LSTM model input shape mismatch: expects {input_shape[-1]} features, "
                                        f"but we have {self.n_features}")
                        
                        # Adjust feature count if model is valid
                        if input_shape[-1] > 0:
                            self.n_features = input_shape[-1]
                            self.logger.info(f"Adjusted to model's feature count: {self.n_features}")
                            models_loaded += 1
                    else:
                        self.logger.info("LSTM model loaded successfully")
                        models_loaded += 1
                except Exception as e:
                    self.logger.error(f"Failed to load LSTM model: {str(e)}")
                    self.lstm_model = None
            
            # Load Transformer model
            if os.path.exists(model_paths['transformer']):
                try:
                    self.transformer_model = load_model(model_paths['transformer'], custom_objects=custom_objects)
                    
                    # Verify input shape
                    input_shape = self.transformer_model.input_shape
                    if input_shape and len(input_shape) >= 3 and input_shape[-1] != self.n_features:
                        self.logger.warning(f"Transformer model input shape mismatch: expects {input_shape[-1]} features, "
                                        f"but we have {self.n_features}")
                        
                        # Don't count if shape mismatch
                        self.transformer_model = None
                    else:
                        self.logger.info("Transformer model loaded successfully")
                        models_loaded += 1
                except Exception as e:
                    self.logger.error(f"Failed to load Transformer model: {str(e)}")
                    self.transformer_model = None
            
            # Load Hybrid model
            if os.path.exists(model_paths['hybrid']):
                try:
                    self.hybrid_model = load_model(model_paths['hybrid'], custom_objects=custom_objects)
                    
                    # Verify input shape
                    input_shape = self.hybrid_model.input_shape
                    if input_shape and len(input_shape) >= 3 and input_shape[-1] != self.n_features:
                        self.logger.warning(f"Hybrid model input shape mismatch: expects {input_shape[-1]} features, "
                                        f"but we have {self.n_features}")
                        
                        # Don't count if shape mismatch
                        self.hybrid_model = None
                    else:
                        self.logger.info("Hybrid model loaded successfully")
                        models_loaded += 1
                except Exception as e:
                    self.logger.error(f"Failed to load Hybrid model: {str(e)}")
                    self.hybrid_model = None
            
            # Load feature scaler
            if os.path.exists(model_paths['scaler']):
                try:
                    self.feature_scaler = joblib.load(model_paths['scaler'])
                    self.logger.info("Feature scaler loaded successfully")
                except Exception as e:
                    self.logger.error(f"Failed to load feature scaler: {str(e)}")
                    self.feature_scaler = StandardScaler()
            else:
                # Create a new scaler
                self.feature_scaler = StandardScaler()
                self.logger.info("Created new feature scaler")
            
            # If no models loaded, rebuild them
            if models_loaded == 0:
                self.logger.warning("No valid models loaded, rebuilding...")
                return self.rebuild_models_for_current_features()
            
            # Save the feature count if it's been adjusted
            if 'feature_count' in missing_files:
                try:
                    with open(model_paths['feature_count'], 'w') as f:
                        json.dump({"n_features": self.n_features}, f)
                    self.logger.info(f"Saved feature count: {self.n_features}")
                except Exception as e:
                    self.logger.error(f"Error saving feature count: {str(e)}")
            
            return models_loaded > 0
        
        except Exception as e:
            self.logger.error(f"Error loading AI models: {str(e)}")
            traceback.print_exc()
            return False

    def rebuild_models_for_current_features(self):
        """Rebuild models to match current feature count."""
        try:
            self.logger.info(f"Rebuilding models for {self.n_features} features")
            
            # Make sure n_features is valid
            if self.n_features <= 0:
                self.n_features = 11  # Default to standard feature set
                self.logger.warning(f"Invalid feature count, reset to {self.n_features}")
            
            # Build new LSTM model
            self.lstm_model = self.build_lstm_model()
            
            # Build new transformer model
            self.transformer_model = self.build_transformer_model()
            
            # Build new hybrid model
            self.hybrid_model = self.build_hybrid_model()
            
            self.logger.info("Models rebuilt successfully")
            
            # Save the rebuilt models
            self.save_models()
            
            return True
        except Exception as e:
            self.logger.error(f"Error rebuilding models: {str(e)}")
            traceback.print_exc()
            return False

    def build_lstm_model(self, dropout_rate=0.2, units=128, learning_rate=0.001):
        """Build an LSTM model for time series prediction."""
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=(self.sequence_length, self.n_features),
                recurrent_dropout=0.0),
            Dropout(dropout_rate),
            LSTM(units//2, return_sequences=False),
            Dropout(dropout_rate),
            Dense(units//4, activation='relu'),
            Dropout(dropout_rate/2),
            Dense(1, activation='sigmoid')
        ])
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, 
                    loss='binary_crossentropy', 
                    metrics=['accuracy'])
        return model
    
    def build_transformer_model(self, dropout_rate=0.1, num_heads=3, ff_dim=44):
        """Build a Transformer model for time series prediction."""
        try:
            input_shape = (self.sequence_length, self.n_features)
            inputs = Input(shape=input_shape)
            
            # First transformer block with error handling
            try:
                x = TransformerBlock(
                    embed_dim=self.n_features,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout_rate
                )(inputs)
                
                # Second transformer block
                x = TransformerBlock(
                    embed_dim=self.n_features,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout_rate
                )(x)
            except Exception as e:
                # Fallback to simpler architecture if transformer blocks fail
                self.logger.warning(f"TransformerBlock error: {str(e)}. Using simpler architecture.")
                # Use 1D convolution as alternative
                x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(inputs)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Activation("relu")(x)
                x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="same")(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Activation("relu")(x)
            
            x = GlobalAveragePooling1D()(x)
            x = Dense(24, activation="relu")(x)
            x = Dropout(dropout_rate)(x)
            outputs = Dense(1, activation="sigmoid")(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            optimizer = Adam(learning_rate=0.0005)
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            return model
        except Exception as e:
            self.logger.error(f"Error building transformer model: {str(e)}")
            # Return a simple feed-forward network as ultimate fallback
            model = Sequential([
                Input(shape=(self.sequence_length, self.n_features)),
                GlobalAveragePooling1D(),
                Dense(32, activation="relu"),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1, activation="sigmoid")
            ])
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            return model
    
    def build_hybrid_model(self):
        """Build a hybrid model combining LSTM and Transformer."""
        input_shape = (self.sequence_length, self.n_features)
        inputs = Input(shape=input_shape)
        
        # LSTM branch
        lstm_output = LSTM(64, return_sequences=True)(inputs)
        lstm_output = LSTM(32, return_sequences=False)(lstm_output)
        lstm_output = Dense(16, activation='relu')(lstm_output)
        
        # Transformer branch
        transformer_block = TransformerBlock(
            embed_dim=self.n_features,
            num_heads=4,
            ff_dim=48,
            dropout=0.1
        )(inputs)
        transformer_output = GlobalAveragePooling1D()(transformer_block)
        transformer_output = Dense(16, activation='relu')(transformer_output)
        
        # Combine branches
        combined = tf.keras.layers.concatenate([lstm_output, transformer_output])
        combined = Dense(16, activation='relu')(combined)
        combined = Dropout(0.1)(combined)
        outputs = Dense(1, activation='sigmoid')(combined)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def save_models(self):
        """Save models to disk."""
        try:
            os.makedirs(self.models_dir, exist_ok=True)
            
            # Save LSTM model
            if self.lstm_model:
                self.lstm_model.save(os.path.join(self.models_dir, "lstm_model.keras"), save_format="keras")

            # Save transformer model
            if self.transformer_model:
                self.transformer_model.save(os.path.join(self.models_dir, "transformer_model.keras"), save_format="keras")

            # Save hybrid model
            if self.hybrid_model:
                self.hybrid_model.save(os.path.join(self.models_dir, "hybrid_model.keras"), save_format="keras")

            # Save feature scaler
            joblib.dump(self.feature_scaler, os.path.join(self.models_dir, "ai_feature_scaler.joblib"))
            
            # Save feature count
            with open(os.path.join(self.models_dir, "ai_feature_count.json"), 'w') as f:
                json.dump({"n_features": self.n_features}, f)
                
            self.logger.info(f"Models saved to {self.models_dir}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            return False
    
    def prepare_sequence_data(self, df: pd.DataFrame, sequence_length: int = 100, target_column: str = 'future_return'):
        """Prepare sequential data with standard features.
        
        Args:
            df: DataFrame with price data
            sequence_length: Length of sequences to create
            target_column: Column name for target variable
            
        Returns:
            Tuple of (X, y) with sequence data and labels
        """
        try:
            if df is None or df.empty:
                raise ValueError("DataFrame is empty or None")

            # Work on a copy to avoid modifying the original
            df = df.copy()

            # Define standard features needed for the model - EXACTLY 11 features
            # IMPORTANT: Use the SAME order every time
            all_features = [
                'returns', 'log_returns', 'rolling_std_20', 'volume_ma_ratio',
                'rsi', 'macd', 'bb_width', 'adx', 'sma_20', 'atr', 'cci'
            ]
            
            # Set the feature count
            self.n_features = len(all_features)
            
            # Log which features we're using
            self.logger.info(f"Using {len(all_features)} features: {all_features}")

            missing_features = [f for f in all_features if f not in df.columns]
            if missing_features:
                self.logger.warning(f"Missing features: {missing_features}")
                for f in missing_features:
                    if f == 'returns':
                        df.loc[:, 'returns'] = df['close'].pct_change().fillna(0)
                    elif f == 'log_returns':
                        df.loc[:, 'log_returns'] = np.log1p(df['returns'] if 'returns' in df else df['close'].pct_change()).fillna(0)
                    elif f == 'rolling_std_20':
                        df.loc[:, 'rolling_std_20'] = df['returns'].rolling(20, min_periods=1).std().fillna(0)
                    elif f == 'volume_ma_ratio':
                        df.loc[:, 'volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20, min_periods=1).mean().fillna(1) if 'volume' in df else 1.0
                    elif f == 'rsi':
                        if 'close' in df:
                            delta = df['close'].diff()
                            gain = (delta.where(delta > 0, 0)).fillna(0)
                            loss = (-delta.where(delta < 0, 0)).fillna(0)
                            avg_gain = gain.rolling(14, min_periods=1).mean()
                            avg_loss = loss.rolling(14, min_periods=1).mean()
                            rs = avg_gain / avg_loss.replace(0, np.nan)
                            df.loc[:, 'rsi'] = 100 - (100 / (1 + rs)).fillna(50)
                        else:
                            df.loc[:, 'rsi'] = 50
                    elif f == 'macd':
                        if 'close' in df:
                            ema12 = df['close'].ewm(span=12, adjust=False).mean()
                            ema26 = df['close'].ewm(span=26, adjust=False).mean()
                            df.loc[:, 'macd'] = ema12 - ema26
                        else:
                            df.loc[:, 'macd'] = 0
                    elif f == 'bb_width':
                        if 'close' in df:
                            sma20 = df['close'].rolling(20, min_periods=1).mean()
                            std20 = df['close'].rolling(20, min_periods=1).std()
                            df.loc[:, 'bb_width'] = (4 * std20 / sma20).fillna(0)
                        else:
                            df.loc[:, 'bb_width'] = 0.1
                    elif f == 'adx':
                        df.loc[:, 'adx'] = 20
                    elif f == 'sma_20':
                        df.loc[:, 'sma_20'] = df['close'].rolling(20, min_periods=1).mean() if 'close' in df else 0
                    elif f == 'atr':
                        df.loc[:, 'atr'] = 1.0
                    elif f == 'cci':
                        df.loc[:, 'cci'] = 0

            # Now we should have all 11 features
            features = all_features.copy()  # Use the same exact order every time
            
            # Ensure all features exist in the DataFrame
            for f in features:
                if f not in df.columns:
                    df.loc[:, f] = 0.0
                    
            # Label generation
            if target_column in df:
                df.loc[:, 'target'] = (df[target_column] > 0).astype(int)
            elif 'close' in df:
                df.loc[:, 'target'] = (df['close'].shift(-1) > df['close']).astype(int)
            else:
                self.logger.error("No 'close' column to generate target.")
                return np.zeros((1, sequence_length, 11)), np.array([0])

            # Convert all features to numeric and handle missing values
            for f in features:
                df.loc[:, f] = pd.to_numeric(df[f], errors='coerce')
            df.loc[:, features] = df[features].ffill().bfill().fillna(0)

            # Sequence creation
            X, y = [], []
            if len(df) < sequence_length:
                # If we don't have enough data, pad with zeros
                padding = pd.DataFrame(0, index=range(sequence_length - len(df)), columns=df.columns)
                df = pd.concat([padding, df], ignore_index=True)
                X.append(df[features].values)
                y.append(df['target'].iloc[-1])
            else:
                # Create sequences using sliding window
                for i in range(len(df) - sequence_length):
                    seq = df[features].iloc[i:i+sequence_length].values
                    label = df['target'].iloc[i+sequence_length-1]
                    if not np.isnan(seq).any():
                        X.append(seq)
                        y.append(label)

            if not X:
                self.logger.warning("No valid sequences, creating dummy.")
                return np.zeros((1, sequence_length, 11)), np.array([0])

            X = np.array(X)
            y = np.array(y)

            # Standardize features
            try:
                # Reshape to 2D for scaling
                reshaped = X.reshape(-1, X.shape[-1])
                # Fit and transform
                scaled = self.feature_scaler.fit_transform(reshaped)
                # Reshape back to 3D
                X = scaled.reshape(X.shape)
            except Exception as e:
                self.logger.error(f"Scaler failed: {e}")
                # Fallback to min-max scaling
                X_min, X_max = X.min(axis=(0, 1), keepdims=True), X.max(axis=(0, 1), keepdims=True)
                X = (X - X_min) / np.clip(X_max - X_min, 1e-6, None)

            self.logger.info(f"Created {len(X)} sequences with shape {X.shape}")
            return X, y

        except Exception as e:
            self.logger.error(f"Critical failure in sequence preparation: {str(e)}")
            traceback.print_exc()
            return np.zeros((1, sequence_length, 11)), np.array([0])
    
    def train_models(self, X, y, validation_split=0.2, epochs=50, batch_size=32):
        """Train all models with advanced training procedures and callbacks."""
        try:
            if len(X) < 100:
                self.logger.warning("Not enough data for training AI models")
                return False
                
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, shuffle=True, stratify=y, random_state=42
            )
            
            # Calculate class weights for imbalanced data
            unique_classes = np.unique(y_train)
            class_weights = compute_class_weight(
                class_weight='balanced', 
                classes=unique_classes, 
                y=y_train
            )
            class_weight_dict = dict(zip(unique_classes, class_weights))
            
            # Set up callbacks for better training
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.00001,
                    verbose=1
                )
            ]
            
            # Train LSTM model
            self.logger.info("Training LSTM model...")
            lstm_filepath = os.path.join(self.models_dir, "lstm_model.keras")
            lstm_checkpoint = ModelCheckpoint(
                lstm_filepath,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            
            self.lstm_model = self.build_lstm_model()
            lstm_history = self.lstm_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weight_dict,
                callbacks=callbacks + [lstm_checkpoint],
                verbose=1
            )
            
            # If checkpoint saved, load best model
            if os.path.exists(lstm_filepath):
                self.lstm_model = load_model(lstm_filepath, custom_objects={'TransformerBlock': TransformerBlock})
                
            # Train Transformer model
            self.logger.info("Training Transformer model...")
            transformer_filepath = os.path.join(self.models_dir, "transformer_model.keras")
            transformer_checkpoint = ModelCheckpoint(
                transformer_filepath, 
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )

            self.transformer_model = self.build_transformer_model()
            transformer_history = self.transformer_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weight_dict,
                callbacks=callbacks + [transformer_checkpoint],
                verbose=1
            )
            
            # If checkpoint saved, load best model
            if os.path.exists(transformer_filepath):
                self.transformer_model = load_model(transformer_filepath, custom_objects={'TransformerBlock': TransformerBlock})
                
            # Train hybrid model
            self.logger.info("Training hybrid model...")
            hybrid_filepath = os.path.join(self.models_dir, "hybrid_model.keras")
            hybrid_checkpoint = ModelCheckpoint(
                hybrid_filepath, 
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            
            self.hybrid_model = self.build_hybrid_model()
            hybrid_history = self.hybrid_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weight_dict,
                callbacks=callbacks + [hybrid_checkpoint],
                verbose=1
            )
            
            # If checkpoint saved, load best model
            if os.path.exists(hybrid_filepath):
                self.hybrid_model = load_model(hybrid_filepath, custom_objects={'TransformerBlock': TransformerBlock})
            
            # Evaluate all models on validation set
            self._evaluate_models(X_val, y_val)
            
            # Save feature scaler
            joblib.dump(self.feature_scaler, os.path.join(self.models_dir, "ai_feature_scaler.joblib"))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training AI models: {str(e)}")
            traceback.print_exc()
            return False
            
    def _evaluate_models(self, X_val, y_val):
        """Evaluate all trained models and log performance metrics."""
        try:
            models = {
                'LSTM': self.lstm_model,
                'Transformer': self.transformer_model,
                'Hybrid': self.hybrid_model
            }
            
            for name, model in models.items():
                if model is not None:
                    self.logger.info(f"Evaluating {name} model...")
                    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
                    
                    # Get predictions and calculate metrics
                    y_pred_proba = model.predict(X_val, verbose=0)
                    y_pred = (y_pred_proba > 0.5).astype(int)
                    
                    # Calculate confusion matrix
                    cm = confusion_matrix(y_val, y_pred)
                    
                    # Log evaluation metrics
                    self.logger.info(f"{name} Model Evaluation:")
                    self.logger.info(f"- Loss: {loss:.4f}")
                    self.logger.info(f"- Accuracy: {accuracy:.4f}")
                    self.logger.info(f"- Confusion Matrix:\n{cm}")
                    
                    # Classification report
                    report = classification_report(y_val, y_pred, output_dict=True)
                    self.logger.info(f"- Precision (Class 1): {report['1']['precision']:.4f}")
                    self.logger.info(f"- Recall (Class 1): {report['1']['recall']:.4f}")
                    self.logger.info(f"- F1-Score (Class 1): {report['1']['f1-score']:.4f}")
                    
        except Exception as e:
            self.logger.error(f"Error evaluating models: {str(e)}")
    
    def predict_next_movement(self, current_data: pd.DataFrame) -> dict:
        """Predict price movement with ensemble of models for higher accuracy."""
        try:
            # Check if models are loaded
            if not self.lstm_model and not self.transformer_model and not self.hybrid_model:
                return {'confidence': 0.5, 'direction': 'hold', 'models_used': 0}
            
            # First, determine whether we need exact feature count
            if self.lstm_model:
                # Check first layer input shape to determine how many features are expected
                input_shape = self.lstm_model.input_shape
                expected_features = input_shape[-1] if input_shape and len(input_shape) >= 3 else 11
                self.logger.info(f"Model expects {expected_features} features. Standard is 11.")
                
                # If model expects different feature count than we're using, we may need to rebuild
                if expected_features != self.n_features:
                    self.logger.warning("Feature count mismatch: Model expects different feature count. Consider retraining.")
            
            # Prepare sequence data with standard feature set
            X, _ = self.prepare_sequence_data(current_data, self.sequence_length)
            
            if len(X) == 0:
                self.logger.warning("No valid sequences for prediction")
                return {'confidence': 0.5, 'direction': 'hold', 'models_used': 0}
            
            # Initialize predictions
            predictions = {}
            weights = {'lstm': 0.3, 'transformer': 0.3, 'hybrid': 0.4}
            models_used = 0
            
            # Get predictions from each available model
            if self.lstm_model is not None:
                try:
                    lstm_pred = self.lstm_model.predict(X[-1:], verbose=0)[0][0]
                    predictions['lstm'] = lstm_pred
                    models_used += 1
                except Exception as e:
                    self.logger.error(f"Error using LSTM model: {str(e)}")
                    # Don't stop execution, continue with other models
                    
            if self.transformer_model is not None:
                try:
                    transformer_pred = self.transformer_model.predict(X[-1:], verbose=0)[0][0]
                    predictions['transformer'] = transformer_pred
                    models_used += 1
                except Exception as e:
                    self.logger.warning(f"Error using transformer model: {str(e)}")
                    
            if self.hybrid_model is not None:
                try:
                    hybrid_pred = self.hybrid_model.predict(X[-1:], verbose=0)[0][0]
                    predictions['hybrid'] = hybrid_pred
                    models_used += 1
                except Exception as e:
                    self.logger.warning(f"Error using hybrid model: {str(e)}")
            
            # Check if we have any predictions
            if models_used == 0:
                self.logger.warning("No models available for prediction")
                return {'confidence': 0.5, 'direction': 'hold', 'models_used': 0}
            
            # Combine predictions with weights or use available ones
            total_weight = 0
            combined_pred = 0
            
            for model_name, pred in predictions.items():
                model_weight = weights.get(model_name, 1.0 / models_used)
                combined_pred += pred * model_weight
                total_weight += model_weight
            
            # Normalize by actual weights used
            if total_weight > 0:
                combined_pred /= total_weight
            
            # Convert to confidence centered around 0.5 with a limited range
            raw_confidence = combined_pred
            centered_confidence = 0.5 + (combined_pred - 0.5) * 0.3
            
            # Ensure confidence stays in reasonable bounds
            confidence = max(0.35, min(0.65, centered_confidence))
            
            # Log individual model predictions
            prediction_str = ", ".join([f"{model}: {pred:.4f}" for model, pred in predictions.items()])
            self.logger.info(f"Model predictions: {prediction_str}")
            self.logger.info(f"Combined prediction: {raw_confidence:.4f}, Adjusted confidence: {confidence:.4f}")
            
            return {
                'confidence': confidence,
                'direction': 'buy' if raw_confidence > 0.5 else 'sell',
                'raw_confidence': raw_confidence,
                'models_used': models_used,
                'predictions': predictions
            }
            
        except Exception as e:
            self.logger.error(f"Error in AI prediction: {str(e)}")
            traceback.print_exc()
            return {'confidence': 0.5, 'direction': 'hold', 'models_used': 0}
