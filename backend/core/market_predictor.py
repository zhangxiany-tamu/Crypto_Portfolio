import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from typing import Dict, List, Tuple, Optional
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    XGBOOST_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')

class MarketPredictor:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.models = {}
        self.scalers = {}
        
    async def generate_predictions(self, symbols: List[str], start_date: str, 
                                 end_date: str, prediction_days: int = 30) -> Dict:
        """Generate market predictions for given symbols"""
        
        try:
            # Get price data
            price_data = await self.data_manager.get_price_data(symbols, start_date, end_date)
            
            predictions = {}
            for symbol in symbols:
                # Prepare features
                features_df = self._create_features(price_data[symbol])
                
                # Train model and make predictions
                model_results = self._train_and_predict(features_df, prediction_days)
                
                predictions[symbol] = {
                    'predicted_prices': model_results['predictions'].tolist(),
                    'confidence_intervals': model_results['confidence_intervals'],
                    'model_accuracy': model_results['accuracy'],
                    'trend_direction': model_results['trend'],
                    'volatility_forecast': model_results['volatility']
                }
            
            # Market sentiment analysis
            market_sentiment = self._analyze_market_sentiment(price_data)
            
            return {
                'predictions': predictions,
                'market_sentiment': market_sentiment,
                'prediction_period': prediction_days,
                'last_update': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            raise Exception(f"Prediction generation failed: {str(e)}")
    
    def _create_features(self, price_series: pd.Series) -> pd.DataFrame:
        """Create technical indicators and features for prediction"""
        
        df = pd.DataFrame({'price': price_series})
        
        # Basic price features
        df['returns'] = df['price'].pct_change()
        df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
        
        # Moving averages
        df['sma_5'] = df['price'].rolling(5).mean()
        df['sma_20'] = df['price'].rolling(20).mean()
        df['sma_50'] = df['price'].rolling(50).mean()
        df['ema_12'] = df['price'].ewm(span=12).mean()
        df['ema_26'] = df['price'].ewm(span=26).mean()
        
        # Volatility features
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        
        # Technical indicators
        # RSI
        df['rsi'] = self._calculate_rsi(df['price'])
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['price'].rolling(20).mean()
        bb_std = df['price'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Momentum features
        df['momentum_5'] = df['price'] / df['price'].shift(5) - 1
        df['momentum_20'] = df['price'] / df['price'].shift(20) - 1
        
        # Price position features
        df['price_position_5'] = (df['price'] - df['price'].rolling(5).min()) / (df['price'].rolling(5).max() - df['price'].rolling(5).min())
        df['price_position_20'] = (df['price'] - df['price'].rolling(20).min()) / (df['price'].rolling(20).max() - df['price'].rolling(20).min())
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'price_lag_{lag}'] = df['price'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        return df.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _train_and_predict(self, features_df: pd.DataFrame, prediction_days: int) -> Dict:
        """Train ML model and make predictions"""
        
        # Prepare target variable (next day price)
        target = features_df['price'].shift(-1).dropna()
        features = features_df[:-1].select_dtypes(include=[np.number])
        
        # Remove price from features to avoid data leakage
        feature_cols = [col for col in features.columns if 'price' not in col.lower() or 'lag' in col.lower()]
        X = features[feature_cols].fillna(0)
        y = target
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Configure custom time series cross-validation for daily data
        # Adjust validation window based on prediction horizon and available data
        # For daily data with typical 7-day predictions and 90-day training windows
        
        # Estimate if this is daily data based on typical training sizes
        is_likely_daily_data = len(X_train) < 500  # Less than ~1.5 years suggests daily data
        
        if is_likely_daily_data:
            # Daily data: validation window should match user's prediction period
            validation_window = prediction_days  # Use exactly what user wants to predict
            min_train_size = max(120, len(X_train) // 2)  # At least 120 days initial training
        else:
            # Weekly data: for weekly data, prediction_days likely represents days, convert to weeks
            validation_window = max(1, prediction_days // 7)  # Convert days to weeks
            min_train_size = max(60, len(X_train) // 4)  # At least 60 weeks
        
        # Create custom CV splits with fixed validation window sizes
        def create_custom_cv_splits(data_length, min_train_size, val_window):
            """Create custom CV splits with fixed validation window sizes"""
            splits = []
            
            # Start with minimum training size
            current_train_end = min_train_size - 1  # 0-indexed
            
            while current_train_end + val_window < data_length:
                # Training indices: from 0 to current_train_end
                train_indices = np.arange(0, current_train_end + 1)
                
                # Validation indices: next val_window weeks
                val_start = current_train_end + 1
                val_end = min(val_start + val_window - 1, data_length - 1)
                val_indices = np.arange(val_start, val_end + 1)
                
                splits.append((train_indices, val_indices))
                
                # Move to next fold (advance by validation window)
                current_train_end += val_window
                
                # Limit to maximum 12 folds for computational efficiency
                if len(splits) >= 12:
                    break
            
            return splits
        
        # Validate minimum data requirements and create appropriate CV splits
        required_min_data = min_train_size + (validation_window * 3)  # Need at least 3 validation periods
        
        if len(X_train) < required_min_data:
            # Insufficient data for proper CV - warn and use minimal validation
            print(f"Warning: Only {len(X_train)} days of training data available.")
            print(f"Using simplified validation with {min(3, max(2, len(X_train) // 30))} folds.")
            
            n_splits = min(3, max(2, len(X_train) // 30))
            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=None)
            cv_splits = list(tscv.split(X_train))
        else:
            # Sufficient data for custom CV
            cv_splits = create_custom_cv_splits(len(X_train), min_train_size, validation_window)
            n_splits = len(cv_splits)
            
            # Cap number of folds based on data type - fewer folds for longer minimum training
            max_folds = 6 if is_likely_daily_data else 10
            if n_splits > max_folds:
                cv_splits = cv_splits[:max_folds]
                n_splits = max_folds
        
        # Train ensemble of models with hyperparameter tuning
        models = {}
        predictions = {}
        accuracies = {}
        
        # Random Forest with optimized parameter grid for time series
        rf_param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 5],
            'max_features': ['sqrt']
        }
        
        
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            rf_param_grid,
            cv=cv_splits,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        rf_grid.fit(X_train_scaled, y_train)
        models['rf'] = rf_grid.best_estimator_
        
        # Linear Regression (no hyperparameters to tune)
        models['lr'] = LinearRegression()
        models['lr'].fit(X_train_scaled, y_train)
        
        # Lasso with EXTREME sparsity-encouraging alpha selection for crypto noise
        lasso_cv = LassoCV(
            alphas=np.logspace(-2, 2, 50),  # Reduced range: 0.01 to 100, fewer choices
            cv=cv_splits,
            random_state=42,
            max_iter=3000,  # Fewer iterations for faster execution
            tol=1e-4,  # Slightly relaxed tolerance
            selection='random'  # Random coordinate selection for better sparsity
        )
        lasso_cv.fit(X_train_scaled, y_train)
        models['lasso'] = lasso_cv
        
        # Neural Network with EXTREME sparsity-encouraging regularization for crypto noise
        # Split into two grids due to solver compatibility issues
        nn_param_grid_adam = {
            'hidden_layer_sizes': [(5,), (10,), (8, 5)],  # Smaller networks
            'alpha': [1.0, 10.0, 50.0],  # EXTREME L2 regularization
            'learning_rate_init': [0.001, 0.003],  # Lower rates
            'solver': ['adam'],
            'activation': ['relu'],
            'early_stopping': [True],
            'max_iter': [150]  # Fixed iterations
        }
        
        nn_param_grid_lbfgs = {
            'hidden_layer_sizes': [(5,), (10,)],  # Smaller networks
            'alpha': [1.0, 10.0],  # EXTREME L2 regularization
            'solver': ['lbfgs'],
            'activation': ['relu'],
            'early_stopping': [False],  # lbfgs doesn't support early_stopping
            'max_iter': [150]  # Fixed iterations
        }
        
        
        # Test both solver types and pick the best
        best_nn_score = float('-inf')
        best_nn_model = None
        
        for param_grid in [nn_param_grid_adam, nn_param_grid_lbfgs]:
            nn_grid = GridSearchCV(
                MLPRegressor(random_state=42, max_iter=1000, tol=1e-4),
                param_grid,
                cv=cv_splits,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            nn_grid.fit(X_train_scaled, y_train)
            
            if nn_grid.best_score_ > best_nn_score:
                best_nn_score = nn_grid.best_score_
                best_nn_model = nn_grid.best_estimator_
        
        models['nn'] = best_nn_model
        
        # XGBoost with EXTREME sparsity-encouraging regularization for crypto noise
        if XGBOOST_AVAILABLE:
            xgb_param_grid = {
                'n_estimators': [25, 50],  # Fewer trees
                'max_depth': [1, 2],  # Ultra-shallow trees for maximum sparsity
                'learning_rate': [0.01, 0.05],  # Lower rates
                'subsample': [0.5, 0.7],  # EXTREME subsampling for sparsity
                'colsample_bytree': [0.4, 0.6],  # EXTREME feature subsampling
                'reg_alpha': [5.0, 50.0],  # EXTREME L1 for sparsity
                'reg_lambda': [10.0, 100.0],  # EXTREME L2 regularization
                'min_child_weight': [10],  # Much higher minimum weights
                'gamma': [0.5]  # Minimum loss reduction for splits
            }
            
            
            xgb_grid = GridSearchCV(
                xgb.XGBRegressor(
                    random_state=42, 
                    verbosity=0,
                    objective='reg:squarederror'
                ),
                xgb_param_grid,
                cv=cv_splits,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            
            xgb_grid.fit(X_train_scaled, y_train)
            models['xgb'] = xgb_grid.best_estimator_
        else:
            # Fallback to Gradient Boosting with EXTREME sparsity-encouraging regularization for crypto noise
            gb_param_grid = {
                'n_estimators': [25, 50],  # Fewer trees
                'max_depth': [1, 2],  # Ultra-shallow trees for maximum sparsity
                'learning_rate': [0.01, 0.05],  # Lower learning rates
                'subsample': [0.5, 0.7],  # EXTREME subsampling for sparsity
                'min_samples_split': [20],  # Higher minimum samples for splits
                'min_samples_leaf': [10],  # Higher minimum samples in leaves
                'max_features': [0.4, 'sqrt']  # EXTREME feature subsampling
            }
            
            
            gb_grid = GridSearchCV(
                GradientBoostingRegressor(random_state=42),
                gb_param_grid,
                cv=cv_splits,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            
            gb_grid.fit(X_train_scaled, y_train)
            models['gb'] = gb_grid.best_estimator_
        
        # Make predictions and calculate accuracies
        for name, model in models.items():
            # All models are already fitted (RF via grid search, LR and Lasso directly)
            pred = model.predict(X_test_scaled)
            accuracy = r2_score(y_test, pred)
            
            predictions[name] = pred
            accuracies[name] = accuracy
        
        # Ensemble prediction (average)
        ensemble_pred = np.mean([predictions[name] for name in predictions], axis=0)
        ensemble_accuracy = r2_score(y_test, ensemble_pred)
        
        # Generate future predictions
        last_features = X.iloc[-1:].fillna(0)
        last_price = features_df['price'].iloc[-1]
        
        future_predictions = []
        predicted_prices = [last_price]  # Start with last known price
        
        for step in range(prediction_days):
            # Get current features for prediction
            current_features_raw = last_features.copy()
            
            # Update lag features with predicted prices
            if len(predicted_prices) >= 1:
                current_features_raw.loc[current_features_raw.index[0], 'returns_lag_1'] = (predicted_prices[-1] / predicted_prices[-2] - 1) if len(predicted_prices) > 1 else 0
            if len(predicted_prices) >= 2:
                current_features_raw.loc[current_features_raw.index[0], 'returns_lag_2'] = (predicted_prices[-2] / predicted_prices[-3] - 1) if len(predicted_prices) > 2 else 0
            if len(predicted_prices) >= 3:
                current_features_raw.loc[current_features_raw.index[0], 'returns_lag_3'] = (predicted_prices[-3] / predicted_prices[-4] - 1) if len(predicted_prices) > 3 else 0
            if len(predicted_prices) >= 5:
                current_features_raw.loc[current_features_raw.index[0], 'returns_lag_5'] = (predicted_prices[-5] / predicted_prices[-6] - 1) if len(predicted_prices) > 5 else 0
            if len(predicted_prices) >= 10:
                current_features_raw.loc[current_features_raw.index[0], 'returns_lag_10'] = (predicted_prices[-10] / predicted_prices[-11] - 1) if len(predicted_prices) > 10 else 0
                
            # Update momentum features
            current_features_raw.loc[current_features_raw.index[0], 'momentum_5'] = (predicted_prices[-1] / predicted_prices[-6] - 1) if len(predicted_prices) > 5 else 0
            current_features_raw.loc[current_features_raw.index[0], 'momentum_20'] = (predicted_prices[-1] / predicted_prices[-21] - 1) if len(predicted_prices) > 20 else 0
            
            # Update volatility features (simplified - using recent predictions)
            if len(predicted_prices) >= 6:
                recent_returns = [predicted_prices[i]/predicted_prices[i-1] - 1 for i in range(max(1, len(predicted_prices)-5), len(predicted_prices))]
                current_features_raw.loc[current_features_raw.index[0], 'volatility_5'] = np.std(recent_returns) if len(recent_returns) > 1 else 0
            
            # Scale features
            current_features_scaled = scaler.transform(current_features_raw.fillna(0))
            
            # Predict next price
            next_price_preds = [model.predict(current_features_scaled)[0] for model in models.values()]
            next_price = np.mean(next_price_preds)
            
            future_predictions.append(next_price)
            predicted_prices.append(next_price)
        
        # Calculate confidence intervals (simplified)
        price_std = features_df['price'].pct_change().std()
        confidence_intervals = [
            [pred * (1 - 1.96 * price_std), pred * (1 + 1.96 * price_std)]
            for pred in future_predictions
        ]
        
        # Trend analysis
        recent_prices = features_df['price'].tail(20)
        trend_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        trend_direction = 'bullish' if trend_slope > 0 else 'bearish'
        
        # Volatility forecast
        recent_volatility = features_df['returns'].tail(30).std() * np.sqrt(252)
        
        return {
            'predictions': np.array(future_predictions),
            'confidence_intervals': confidence_intervals,
            'accuracy': ensemble_accuracy,
            'trend': trend_direction,
            'volatility': recent_volatility
        }
    
    def _analyze_market_sentiment(self, price_data: pd.DataFrame) -> Dict:
        """Analyze overall market sentiment"""
        
        # Calculate market-wide metrics
        returns = price_data.pct_change().dropna()
        
        # Overall market trend
        avg_returns = returns.mean(axis=1)
        recent_trend = avg_returns.tail(30).mean()
        
        # Market correlation
        avg_correlation = returns.corr().values[np.triu_indices_from(returns.corr().values, k=1)].mean()
        
        # Volatility regime
        market_vol = avg_returns.std() * np.sqrt(252)
        vol_regime = 'high' if market_vol > 0.6 else 'medium' if market_vol > 0.3 else 'low'
        
        # Fear & Greed approximation based on volatility and returns
        fear_greed_score = min(100, max(0, 50 + (recent_trend * 1000) - (market_vol * 50)))
        
        if fear_greed_score > 75:
            sentiment = 'extreme_greed'
        elif fear_greed_score > 55:
            sentiment = 'greed'
        elif fear_greed_score > 45:
            sentiment = 'neutral'
        elif fear_greed_score > 25:
            sentiment = 'fear'
        else:
            sentiment = 'extreme_fear'
        
        return {
            'sentiment': sentiment,
            'fear_greed_score': fear_greed_score,
            'market_trend': 'bullish' if recent_trend > 0 else 'bearish',
            'volatility_regime': vol_regime,
            'correlation_level': 'high' if avg_correlation > 0.7 else 'medium' if avg_correlation > 0.4 else 'low'
        }