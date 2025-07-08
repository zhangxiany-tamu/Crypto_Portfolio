import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TechnicalAnalyzer:
    """
    Comprehensive technical analysis with bullish/bearish/neutral signals
    for short-term (1-7 days), medium-term (1-4 weeks), and long-term (1-3 months)
    """
    
    def __init__(self, price_data: pd.Series):
        """
        Initialize with price data
        
        Args:
            price_data: Pandas Series with datetime index and price values
        """
        self.price_data = price_data
        self.indicators = {}
        self.signals = {}
        
    def calculate_all_indicators(self) -> Dict:
        """Calculate all technical indicators"""
        
        # Moving Averages
        self.indicators['sma_7'] = self.price_data.rolling(7).mean()
        self.indicators['sma_21'] = self.price_data.rolling(21).mean()
        self.indicators['sma_50'] = self.price_data.rolling(50).mean()
        self.indicators['sma_200'] = self.price_data.rolling(200).mean()
        
        # Exponential Moving Averages
        self.indicators['ema_12'] = self.price_data.ewm(span=12).mean()
        self.indicators['ema_26'] = self.price_data.ewm(span=26).mean()
        self.indicators['ema_50'] = self.price_data.ewm(span=50).mean()
        
        # RSI (Relative Strength Index)
        self.indicators['rsi'] = self._calculate_rsi(period=14)
        
        # MACD (Moving Average Convergence Divergence)
        macd_data = self._calculate_macd()
        self.indicators.update(macd_data)
        
        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands()
        self.indicators.update(bb_data)
        
        # Stochastic Oscillator
        stoch_data = self._calculate_stochastic()
        self.indicators.update(stoch_data)
        
        # Williams %R
        self.indicators['williams_r'] = self._calculate_williams_r()
        
        # Commodity Channel Index (CCI)
        self.indicators['cci'] = self._calculate_cci()
        
        # Average True Range (ATR)
        self.indicators['atr'] = self._calculate_atr()
        
        # Volume indicators (if volume data available)
        if hasattr(self.price_data, 'volume'):
            self.indicators['volume_sma'] = self.price_data.volume.rolling(20).mean()
        
        # Momentum indicators
        self.indicators['momentum_10'] = self.price_data / self.price_data.shift(10) - 1
        self.indicators['momentum_20'] = self.price_data / self.price_data.shift(20) - 1
        
        # Price Rate of Change (ROC)
        self.indicators['roc_10'] = self._calculate_roc(period=10)
        self.indicators['roc_20'] = self._calculate_roc(period=20)
        
        return self.indicators
    
    def _calculate_rsi(self, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = self.price_data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD"""
        ema_fast = self.price_data.ewm(span=fast).mean()
        ema_slow = self.price_data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        
        return {
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram
        }
    
    def _calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2) -> Dict:
        """Calculate Bollinger Bands"""
        sma = self.price_data.rolling(period).mean()
        std = self.price_data.rolling(period).std()
        
        return {
            'bb_upper': sma + (std * std_dev),
            'bb_middle': sma,
            'bb_lower': sma - (std * std_dev),
            'bb_width': (sma + (std * std_dev)) - (sma - (std * std_dev)),
            'bb_position': (self.price_data - (sma - (std * std_dev))) / ((sma + (std * std_dev)) - (sma - (std * std_dev)))
        }
    
    def _calculate_stochastic(self, k_period: int = 14, d_period: int = 3) -> Dict:
        """Calculate Stochastic Oscillator"""
        low_min = self.price_data.rolling(k_period).min()
        high_max = self.price_data.rolling(k_period).max()
        
        k_percent = 100 * ((self.price_data - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(d_period).mean()
        
        return {
            'stoch_k': k_percent,
            'stoch_d': d_percent
        }
    
    def _calculate_williams_r(self, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        high_max = self.price_data.rolling(period).max()
        low_min = self.price_data.rolling(period).min()
        
        williams_r = -100 * ((high_max - self.price_data) / (high_max - low_min))
        return williams_r
    
    def _calculate_cci(self, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = self.price_data  # Simplified for price-only data
        sma_tp = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
    
    def _calculate_atr(self, period: int = 14) -> pd.Series:
        """Calculate Average True Range (simplified for price data)"""
        high_low = self.price_data.rolling(2).max() - self.price_data.rolling(2).min()
        atr = high_low.rolling(period).mean()
        return atr
    
    def _calculate_roc(self, period: int = 10) -> pd.Series:
        """Calculate Rate of Change"""
        roc = ((self.price_data - self.price_data.shift(period)) / self.price_data.shift(period)) * 100
        return roc
    
    def analyze_signals(self) -> Dict:
        """Analyze all signals for short, medium, and long term"""
        
        if not self.indicators:
            self.calculate_all_indicators()
        
        current_price = self.price_data.iloc[-1]
        
        # Get latest values
        latest_values = {}
        for key, indicator in self.indicators.items():
            if len(indicator.dropna()) > 0:
                latest_values[key] = indicator.iloc[-1]
        
        # Short-term signals (1-7 days)
        short_term = self._analyze_short_term_signals(current_price, latest_values)
        
        # Medium-term signals (1-4 weeks)
        medium_term = self._analyze_medium_term_signals(current_price, latest_values)
        
        # Long-term signals (1-3 months)
        long_term = self._analyze_long_term_signals(current_price, latest_values)
        
        self.signals = {
            'short_term': short_term,
            'medium_term': medium_term,
            'long_term': long_term,
            'overall': self._calculate_overall_signal(short_term, medium_term, long_term)
        }
        
        return self.signals
    
    def _analyze_short_term_signals(self, current_price: float, latest_values: Dict) -> Dict:
        """Analyze short-term signals (1-7 days)"""
        signals = []
        
        # RSI signals
        if 'rsi' in latest_values:
            rsi = latest_values['rsi']
            if rsi < 30:
                signals.append(('bullish', 'RSI oversold', 0.8))
            elif rsi > 70:
                signals.append(('bearish', 'RSI overbought', 0.8))
            elif 30 <= rsi <= 45:
                signals.append(('bullish', 'RSI bullish range', 0.4))
            elif 55 <= rsi <= 70:
                signals.append(('bearish', 'RSI bearish range', 0.4))
        
        # MACD signals
        if all(k in latest_values for k in ['macd_line', 'macd_signal', 'macd_histogram']):
            macd_line = latest_values['macd_line']
            macd_signal = latest_values['macd_signal']
            macd_hist = latest_values['macd_histogram']
            
            if macd_line > macd_signal and macd_hist > 0:
                signals.append(('bullish', 'MACD bullish crossover', 0.7))
            elif macd_line < macd_signal and macd_hist < 0:
                signals.append(('bearish', 'MACD bearish crossover', 0.7))
        
        # Stochastic signals
        if all(k in latest_values for k in ['stoch_k', 'stoch_d']):
            stoch_k = latest_values['stoch_k']
            stoch_d = latest_values['stoch_d']
            
            if stoch_k < 20 and stoch_d < 20:
                signals.append(('bullish', 'Stochastic oversold', 0.6))
            elif stoch_k > 80 and stoch_d > 80:
                signals.append(('bearish', 'Stochastic overbought', 0.6))
        
        # Williams %R signals
        if 'williams_r' in latest_values:
            williams_r = latest_values['williams_r']
            if williams_r < -80:
                signals.append(('bullish', 'Williams %R oversold', 0.5))
            elif williams_r > -20:
                signals.append(('bearish', 'Williams %R overbought', 0.5))
        
        # Short-term moving average signals
        if all(k in latest_values for k in ['sma_7', 'sma_21']):
            if current_price > latest_values['sma_7'] > latest_values['sma_21']:
                signals.append(('bullish', 'Price above short-term MAs', 0.6))
            elif current_price < latest_values['sma_7'] < latest_values['sma_21']:
                signals.append(('bearish', 'Price below short-term MAs', 0.6))
        
        # Fallback: if no signals were generated, create a basic price-based signal
        if not signals:
            if len(self.price_data) >= 2:
                recent_change = (self.price_data.iloc[-1] - self.price_data.iloc[-2]) / self.price_data.iloc[-2]
                if recent_change > 0.005:
                    signals.append(('bullish', 'Recent price increase', 0.3))
                elif recent_change < -0.005:
                    signals.append(('bearish', 'Recent price decrease', 0.3))
                else:
                    signals.append(('neutral', 'Price relatively stable', 0.2))
            else:
                signals.append(('neutral', 'Basic price analysis', 0.2))
        
        return self._consolidate_signals(signals, 'short_term')
    
    def _analyze_medium_term_signals(self, current_price: float, latest_values: Dict) -> Dict:
        """Analyze medium-term signals (1-4 weeks)"""
        signals = []
        
        # Medium-term moving averages
        if all(k in latest_values for k in ['sma_21', 'sma_50']):
            sma_21 = latest_values['sma_21']
            sma_50 = latest_values['sma_50']
            
            if current_price > sma_21 > sma_50:
                signals.append(('bullish', 'Bullish medium-term trend', 0.7))
            elif current_price < sma_21 < sma_50:
                signals.append(('bearish', 'Bearish medium-term trend', 0.7))
            else:
                # Add neutral signal for mixed conditions
                signals.append(('neutral', 'Mixed medium-term signals', 0.3))
        
        # Bollinger Bands signals
        if all(k in latest_values for k in ['bb_upper', 'bb_lower', 'bb_position']):
            bb_pos = latest_values['bb_position']
            
            if bb_pos < 0.2:
                signals.append(('bullish', 'Near Bollinger lower band', 0.6))
            elif bb_pos > 0.8:
                signals.append(('bearish', 'Near Bollinger upper band', 0.6))
            else:
                signals.append(('neutral', 'Within Bollinger bands', 0.3))
        
        # CCI signals
        if 'cci' in latest_values:
            cci = latest_values['cci']
            if cci < -100:
                signals.append(('bullish', 'CCI oversold', 0.5))
            elif cci > 100:
                signals.append(('bearish', 'CCI overbought', 0.5))
            else:
                signals.append(('neutral', 'CCI neutral range', 0.2))
        
        # Momentum signals
        if 'momentum_20' in latest_values:
            momentum = latest_values['momentum_20']
            if momentum > 0.05:
                signals.append(('bullish', 'Positive momentum', 0.5))
            elif momentum < -0.05:
                signals.append(('bearish', 'Negative momentum', 0.5))
            else:
                signals.append(('neutral', 'Neutral momentum', 0.2))
        
        # EMA crossover signals for medium term
        if all(k in latest_values for k in ['ema_12', 'ema_26']):
            ema_12 = latest_values['ema_12']
            ema_26 = latest_values['ema_26']
            
            if ema_12 > ema_26:
                signals.append(('bullish', 'EMA bullish crossover', 0.6))
            elif ema_12 < ema_26:
                signals.append(('bearish', 'EMA bearish crossover', 0.6))
        
        # Price position relative to moving averages
        if 'sma_21' in latest_values:
            sma_21 = latest_values['sma_21']
            price_deviation = (current_price - sma_21) / sma_21
            
            if price_deviation > 0.02:
                signals.append(('bullish', 'Price above SMA-21', 0.4))
            elif price_deviation < -0.02:
                signals.append(('bearish', 'Price below SMA-21', 0.4))
            else:
                signals.append(('neutral', 'Price near SMA-21', 0.2))
        
        # Fallback: if no signals were generated, create a basic price-based signal
        if not signals:
            if len(self.price_data) >= 2:
                recent_change = (self.price_data.iloc[-1] - self.price_data.iloc[-2]) / self.price_data.iloc[-2]
                if recent_change > 0.01:
                    signals.append(('bullish', 'Recent price increase', 0.3))
                elif recent_change < -0.01:
                    signals.append(('bearish', 'Recent price decrease', 0.3))
                else:
                    signals.append(('neutral', 'Price relatively stable', 0.2))
            else:
                signals.append(('neutral', 'Basic price analysis', 0.2))
        
        return self._consolidate_signals(signals, 'medium_term')
    
    def _analyze_long_term_signals(self, current_price: float, latest_values: Dict) -> Dict:
        """Analyze long-term signals (1-3 months)"""
        signals = []
        
        # Long-term moving averages
        if all(k in latest_values for k in ['sma_50', 'sma_200']):
            sma_50 = latest_values['sma_50']
            sma_200 = latest_values['sma_200']
            
            if current_price > sma_50 > sma_200:
                signals.append(('bullish', 'Golden cross pattern', 0.8))
            elif current_price < sma_50 < sma_200:
                signals.append(('bearish', 'Death cross pattern', 0.8))
        
        # Long-term trend analysis
        if 'sma_200' in latest_values:
            sma_200 = latest_values['sma_200']
            if current_price > sma_200 * 1.05:
                signals.append(('bullish', 'Strong uptrend', 0.7))
            elif current_price < sma_200 * 0.95:
                signals.append(('bearish', 'Strong downtrend', 0.7))
        
        # Rate of Change long-term
        if 'roc_20' in latest_values:
            roc = latest_values['roc_20']
            if roc > 10:
                signals.append(('bullish', 'Strong price momentum', 0.6))
            elif roc < -10:
                signals.append(('bearish', 'Weak price momentum', 0.6))
        
        # Fallback: if no signals were generated, create a basic price-based signal
        if not signals:
            if len(self.price_data) >= 2:
                recent_change = (self.price_data.iloc[-1] - self.price_data.iloc[-2]) / self.price_data.iloc[-2]
                if recent_change > 0.02:
                    signals.append(('bullish', 'Recent price increase', 0.3))
                elif recent_change < -0.02:
                    signals.append(('bearish', 'Recent price decrease', 0.3))
                else:
                    signals.append(('neutral', 'Price relatively stable', 0.2))
            else:
                signals.append(('neutral', 'Basic price analysis', 0.2))
        
        return self._consolidate_signals(signals, 'long_term')
    
    def _consolidate_signals(self, signals: List[Tuple], timeframe: str) -> Dict:
        """Consolidate signals into overall sentiment"""
        if not signals:
            return {
                'signal': 'neutral',
                'strength': 0.5,
                'confidence': 0.3,
                'reasons': [f'Limited technical indicators available for {timeframe} analysis'],
                'score': 0,
                'bullish_signals': 0,
                'bearish_signals': 0,
                'total_signals': 0
            }
        
        # Calculate weighted scores
        bullish_score = sum(weight for signal, reason, weight in signals if signal == 'bullish')
        bearish_score = sum(weight for signal, reason, weight in signals if signal == 'bearish')
        neutral_score = sum(weight for signal, reason, weight in signals if signal == 'neutral')
        
        total_signals = len(signals)
        total_weight = bullish_score + bearish_score
        
        # Determine overall signal
        if total_weight == 0:
            signal = 'neutral'
            strength = 0.5
        elif bullish_score > bearish_score:
            signal = 'bullish'
            strength = min(0.9, 0.5 + (bullish_score - bearish_score) / (2 * total_weight))
        elif bearish_score > bullish_score:
            signal = 'bearish'
            strength = min(0.9, 0.5 + (bearish_score - bullish_score) / (2 * total_weight))
        else:
            signal = 'neutral'
            strength = 0.5
        
        # Calculate confidence based on number of confirming signals
        confidence = min(0.9, 0.3 + (total_signals * 0.1))
        
        # Get reasons
        reasons = [reason for signal_type, reason, weight in signals]
        
        # Calculate final score (-1 to 1)
        score = (bullish_score - bearish_score) / max(total_weight, 1)
        
        return {
            'signal': signal,
            'strength': strength,
            'confidence': confidence,
            'reasons': reasons,
            'score': score,
            'bullish_signals': len([s for s in signals if s[0] == 'bullish']),
            'bearish_signals': len([s for s in signals if s[0] == 'bearish']),
            'total_signals': total_signals
        }
    
    def _calculate_overall_signal(self, short_term: Dict, medium_term: Dict, long_term: Dict) -> Dict:
        """Calculate overall signal combining all timeframes"""
        
        # Weight different timeframes
        weights = {'short_term': 0.3, 'medium_term': 0.4, 'long_term': 0.3}
        
        # Calculate weighted score
        weighted_score = (
            short_term['score'] * weights['short_term'] +
            medium_term['score'] * weights['medium_term'] +
            long_term['score'] * weights['long_term']
        )
        
        # Determine overall signal
        if weighted_score > 0.2:
            signal = 'bullish'
            strength = min(0.9, 0.5 + abs(weighted_score) / 2)
        elif weighted_score < -0.2:
            signal = 'bearish'
            strength = min(0.9, 0.5 + abs(weighted_score) / 2)
        else:
            signal = 'neutral'
            strength = 0.5
        
        # Calculate confidence
        avg_confidence = (short_term['confidence'] + medium_term['confidence'] + long_term['confidence']) / 3
        
        return {
            'signal': signal,
            'strength': strength,
            'confidence': avg_confidence,
            'weighted_score': weighted_score,
            'bullish_signals': short_term.get('bullish_signals', 0) + medium_term.get('bullish_signals', 0) + long_term.get('bullish_signals', 0),
            'bearish_signals': short_term.get('bearish_signals', 0) + medium_term.get('bearish_signals', 0) + long_term.get('bearish_signals', 0),
            'total_signals': short_term.get('total_signals', 0) + medium_term.get('total_signals', 0) + long_term.get('total_signals', 0),
            'timeframe_breakdown': {
                'short_term': short_term['signal'],
                'medium_term': medium_term['signal'],
                'long_term': long_term['signal']
            }
        }
    
    def get_signal_summary(self) -> Dict:
        """Get a comprehensive summary of all signals"""
        if not self.signals:
            self.analyze_signals()
        
        return {
            'current_price': self.price_data.iloc[-1],
            'signals': self.signals,
            'key_levels': self._get_key_levels(),
            'risk_metrics': self._calculate_risk_metrics()
        }
    
    def _get_key_levels(self) -> Dict:
        """Calculate key support and resistance levels"""
        if not self.indicators:
            return {}
        
        current_price = self.price_data.iloc[-1]
        
        levels = {}
        
        # Moving average levels
        for ma in ['sma_21', 'sma_50', 'sma_200']:
            if ma in self.indicators:
                levels[f'{ma}_level'] = self.indicators[ma].iloc[-1]
        
        # Bollinger Bands
        if 'bb_upper' in self.indicators:
            levels['resistance'] = self.indicators['bb_upper'].iloc[-1]
            levels['support'] = self.indicators['bb_lower'].iloc[-1]
        
        return levels
    
    def _calculate_risk_metrics(self) -> Dict:
        """Calculate risk metrics"""
        returns = self.price_data.pct_change().dropna()
        
        if len(returns) < 30:
            return {'insufficient_data': True}
        
        return {
            'volatility_30d': returns.tail(30).std() * np.sqrt(252),
            'volatility_7d': returns.tail(7).std() * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns)
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + self.price_data.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        return excess_returns / volatility if volatility > 0 else 0