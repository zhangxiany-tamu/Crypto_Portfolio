import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
import warnings
from scipy.signal import argrelextrema
from dataclasses import dataclass
warnings.filterwarnings('ignore')

@dataclass
class WavePoint:
    """Represents a significant price point in Elliott Wave analysis"""
    date: pd.Timestamp
    price: float
    point_type: str  # 'high' or 'low'
    wave_label: Optional[str] = None
    degree: Optional[str] = None
    
@dataclass
class FibonacciLevel:
    """Fibonacci retracement/extension level"""
    level: float
    price: float
    ratio: float
    level_type: str  # 'retracement' or 'extension'

@dataclass
class WavePattern:
    """Complete Elliott Wave pattern"""
    pattern_type: str  # 'impulse' or 'corrective'
    waves: List[WavePoint]
    start_point: WavePoint
    end_point: WavePoint
    fibonacci_levels: List[FibonacciLevel]
    confidence: float
    next_target: Optional[float] = None

class ElliottWaveAnalyzer:
    """
    Advanced Elliott Wave Analysis with pattern recognition and Fibonacci analysis
    
    Implements sophisticated algorithms for:
    - ZigZag indicator for swing detection
    - 5-wave impulse pattern identification
    - 3-wave corrective pattern recognition
    - Elliott Wave rule validation
    - Fibonacci retracement and extension calculations
    - Multi-degree wave analysis across timeframes
    """
    
    def __init__(self, price_data: pd.Series, min_swing_percentage: float = 2.0, degree: str = 'Intermediate'):
        """
        Initialize Elliott Wave Analyzer
        
        Args:
            price_data: Price series with datetime index
            min_swing_percentage: Minimum percentage move to constitute a swing
            degree: Elliott Wave degree for analysis scale
        """
        self.price_data = price_data.dropna()
        self.min_swing_percentage = min_swing_percentage
        self.degree = degree
        self.wave_points: List[WavePoint] = []
        self.patterns: List[WavePattern] = []
        self.fibonacci_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618]
        
        # Elliott Wave degree classifications with swing sensitivity
        self.wave_degrees = {
            'Primary': {'label': 'P', 'swing_pct': 8.0, 'timeframe': 'Long-term (6+ months)'},
            'Intermediate': {'label': 'I', 'swing_pct': 4.0, 'timeframe': 'Medium-term (1-6 months)'},
            'Minor': {'label': 'Min', 'swing_pct': 2.0, 'timeframe': 'Short-term (1-4 weeks)'},
            'Minute': {'label': 'Min', 'swing_pct': 1.0, 'timeframe': 'Very Short-term (1-7 days)'},
        }
    
    def detect_swing_points(self) -> List[WavePoint]:
        """
        Advanced swing point detection using adaptive ZigZag algorithm
        
        Returns:
            List of significant highs and lows that constitute wave points
        """
        if len(self.price_data) < 10:
            return []
        
        swing_points = []
        prices = self.price_data.values
        dates = self.price_data.index
        
        # Use the specified swing percentage directly (no adaptive override)
        adaptive_threshold = self.min_swing_percentage / 100
        
        # Find initial high/low
        if prices[0] < prices[1]:
            current_trend = 'up'
            last_extreme_idx = 0
            last_extreme_price = prices[0]
        else:
            current_trend = 'down'
            last_extreme_idx = 0
            last_extreme_price = prices[0]
        
        for i in range(1, len(prices)):
            current_price = prices[i]
            
            if current_trend == 'up':
                # Looking for higher highs or significant reversal
                if current_price > last_extreme_price:
                    last_extreme_idx = i
                    last_extreme_price = current_price
                elif (last_extreme_price - current_price) / last_extreme_price >= adaptive_threshold:
                    # Significant reversal - record the high
                    swing_points.append(WavePoint(
                        date=dates[last_extreme_idx],
                        price=last_extreme_price,
                        point_type='high'
                    ))
                    current_trend = 'down'
                    last_extreme_idx = i
                    last_extreme_price = current_price
            
            else:  # current_trend == 'down'
                # Looking for lower lows or significant reversal
                if current_price < last_extreme_price:
                    last_extreme_idx = i
                    last_extreme_price = current_price
                elif (current_price - last_extreme_price) / last_extreme_price >= adaptive_threshold:
                    # Significant reversal - record the low
                    swing_points.append(WavePoint(
                        date=dates[last_extreme_idx],
                        price=last_extreme_price,
                        point_type='low'
                    ))
                    current_trend = 'up'
                    last_extreme_idx = i
                    last_extreme_price = current_price
        
        # Add the final extreme point
        swing_points.append(WavePoint(
            date=dates[last_extreme_idx],
            price=last_extreme_price,
            point_type='high' if current_trend == 'up' else 'low'
        ))
        
        self.wave_points = swing_points
        return swing_points
    
    def validate_impulse_wave(self, points: List[WavePoint]) -> Tuple[bool, float]:
        """
        Validate if 5 points form a valid Elliott Wave impulse pattern
        
        Elliott Wave Rules:
        1. Wave 2 never retraces more than 100% of wave 1
        2. Wave 3 is never the shortest of waves 1, 3, and 5
        3. Wave 4 never overlaps with wave 1 price territory
        
        Args:
            points: List of 5 wave points (should alternate high-low or low-high)
            
        Returns:
            Tuple of (is_valid, confidence_score)
        """
        if len(points) != 5:
            return False, 0.0
        
        confidence = 1.0
        
        # Extract prices
        p0, p1, p2, p3, p4 = [point.price for point in points]
        
        # Determine if it's an upward or downward impulse
        if points[0].point_type == 'low' and points[4].point_type == 'high':
            # Upward impulse: Low-High-Low-High-Low expected
            wave1_length = p1 - p0
            wave2_retrace = p1 - p2
            wave3_length = p3 - p2
            wave4_retrace = p3 - p4
            wave5_length = p4 - p3 if len(points) > 4 else 0
            
            # Rule 1: Wave 2 retracement check
            if wave2_retrace / wave1_length > 1.0:
                return False, 0.0
            elif wave2_retrace / wave1_length > 0.786:
                confidence *= 0.7  # Deep retracement reduces confidence
            
            # Rule 2: Wave 3 should not be shortest
            wave_lengths = [wave1_length, wave3_length]
            if len(points) == 5:
                wave_lengths.append(wave5_length)
            
            if wave3_length <= min(wave_lengths):
                confidence *= 0.5  # Violates guideline
            
            # Rule 3: Wave 4 overlap check
            if p4 <= p1:  # Wave 4 low overlaps with Wave 1 high
                return False, 0.0
                
        elif points[0].point_type == 'high' and points[4].point_type == 'low':
            # Downward impulse: High-Low-High-Low-High expected
            wave1_length = p0 - p1
            wave2_retrace = p2 - p1
            wave3_length = p2 - p3
            wave4_retrace = p4 - p3
            
            # Rule 1: Wave 2 retracement check
            if wave2_retrace / wave1_length > 1.0:
                return False, 0.0
            elif wave2_retrace / wave1_length > 0.786:
                confidence *= 0.7
            
            # Rule 2: Wave 3 should not be shortest
            wave_lengths = [wave1_length, wave3_length]
            if wave3_length <= min(wave_lengths):
                confidence *= 0.5
            
            # Rule 3: Wave 4 overlap check
            if p4 >= p1:  # Wave 4 high overlaps with Wave 1 low
                return False, 0.0
        else:
            return False, 0.0
        
        return True, confidence
    
    def calculate_fibonacci_levels(self, start_point: WavePoint, end_point: WavePoint) -> List[FibonacciLevel]:
        """
        Calculate Fibonacci retracement and extension levels
        
        Args:
            start_point: Starting point of the move
            end_point: Ending point of the move
            
        Returns:
            List of Fibonacci levels with prices
        """
        levels = []
        price_diff = end_point.price - start_point.price
        
        # Retracement levels (from end_point back toward start_point)
        for ratio in self.fibonacci_ratios:
            if ratio <= 1.0:  # Retracements
                fib_price = end_point.price - (price_diff * ratio)
                levels.append(FibonacciLevel(
                    level=ratio,
                    price=fib_price,
                    ratio=ratio,
                    level_type='retracement'
                ))
        
        # Extension levels (beyond end_point)
        for ratio in self.fibonacci_ratios:
            if ratio > 1.0:  # Extensions
                fib_price = end_point.price + (price_diff * (ratio - 1))
                levels.append(FibonacciLevel(
                    level=ratio,
                    price=fib_price,
                    ratio=ratio,
                    level_type='extension'
                ))
        
        return levels
    
    def identify_wave_patterns(self) -> List[WavePattern]:
        """
        Identify Elliott Wave patterns in the detected swing points
        
        Returns:
            List of identified wave patterns with confidence scores
        """
        if not self.wave_points:
            self.detect_swing_points()
        
        patterns = []
        points = self.wave_points
        
        # Look for 5-wave impulse patterns
        for i in range(len(points) - 4):
            five_points = points[i:i+5]
            is_valid, confidence = self.validate_impulse_wave(five_points)
            
            if is_valid and confidence > 0.5:
                # Calculate Fibonacci levels for the entire pattern
                fib_levels = self.calculate_fibonacci_levels(five_points[0], five_points[4])
                
                # Determine next target based on pattern completion
                next_target = self._calculate_next_target(five_points)
                
                pattern = WavePattern(
                    pattern_type='impulse',
                    waves=five_points,
                    start_point=five_points[0],
                    end_point=five_points[4],
                    fibonacci_levels=fib_levels,
                    confidence=confidence,
                    next_target=next_target
                )
                patterns.append(pattern)
        
        # Look for 3-wave corrective patterns (simplified)
        for i in range(len(points) - 2):
            three_points = points[i:i+3]
            if self._is_valid_correction(three_points):
                fib_levels = self.calculate_fibonacci_levels(three_points[0], three_points[2])
                
                pattern = WavePattern(
                    pattern_type='corrective',
                    waves=three_points,
                    start_point=three_points[0],
                    end_point=three_points[2],
                    fibonacci_levels=fib_levels,
                    confidence=0.7,
                    next_target=None
                )
                patterns.append(pattern)
        
        self.patterns = patterns
        return patterns
    
    def _is_valid_correction(self, points: List[WavePoint]) -> bool:
        """Check if 3 points form a valid corrective pattern"""
        if len(points) != 3:
            return False
        
        p0, p1, p2 = [point.price for point in points]
        
        # Basic ABC correction validation
        if points[0].point_type != points[2].point_type:
            return False
        
        # Check for reasonable retracement (between 23.6% and 78.6%)
        if points[0].point_type == 'high':  # Downward correction
            total_move = p0 - p2
            retrace = p1 - p2
            retrace_ratio = retrace / total_move if total_move != 0 else 0
        else:  # Upward correction
            total_move = p2 - p0
            retrace = p0 - p1
            retrace_ratio = retrace / total_move if total_move != 0 else 0
        
        return 0.236 <= retrace_ratio <= 0.786
    
    def _calculate_next_target(self, wave_points: List[WavePoint]) -> Optional[float]:
        """Calculate potential next price target based on Elliott Wave theory"""
        if len(wave_points) < 5:
            return None
        
        # Simple projection: Wave 5 often equals Wave 1 in length
        wave1_length = abs(wave_points[1].price - wave_points[0].price)
        wave3_start = wave_points[2].price
        
        if wave_points[0].point_type == 'low':  # Upward impulse
            return wave3_start + wave1_length
        else:  # Downward impulse
            return wave3_start - wave1_length
    
    def get_current_wave_position(self) -> Dict:
        """
        Determine the current position within Elliott Wave patterns
        
        Returns:
            Dictionary with current wave analysis
        """
        if not self.patterns:
            self.identify_wave_patterns()
        
        current_price = self.price_data.iloc[-1]
        current_date = self.price_data.index[-1]
        
        # Find the most recent and relevant pattern
        active_pattern = None
        for pattern in reversed(self.patterns):
            if pattern.end_point.date <= current_date:
                active_pattern = pattern
                break
        
        if not active_pattern:
            return {
                'status': 'No clear Elliott Wave pattern identified',
                'confidence': 0.0,
                'current_wave': None,
                'next_target': None,
                'pattern_type': None
            }
        
        # Determine current wave position
        position_analysis = self._analyze_current_position(active_pattern, current_price)
        
        return {
            'status': 'Elliott Wave pattern identified',
            'confidence': active_pattern.confidence,
            'current_wave': position_analysis['current_wave'],
            'next_target': active_pattern.next_target,
            'pattern_type': active_pattern.pattern_type,
            'pattern_completion': position_analysis['completion_percentage'],
            'fibonacci_levels': active_pattern.fibonacci_levels,
            'key_levels': position_analysis['key_levels']
        }
    
    def _analyze_current_position(self, pattern: WavePattern, current_price: float) -> Dict:
        """Analyze current position within a wave pattern"""
        if pattern.pattern_type == 'impulse':
            # Determine which wave we're currently in
            wave_points = pattern.waves
            
            for i in range(len(wave_points) - 1):
                if wave_points[i].price <= current_price <= wave_points[i+1].price or \
                   wave_points[i].price >= current_price >= wave_points[i+1].price:
                    
                    wave_number = i + 1
                    completion = abs(current_price - wave_points[i].price) / \
                                abs(wave_points[i+1].price - wave_points[i].price)
                    
                    return {
                        'current_wave': f'Wave {wave_number}',
                        'completion_percentage': completion * 100,
                        'key_levels': {
                            'support': min(wave_points[i].price, wave_points[i+1].price),
                            'resistance': max(wave_points[i].price, wave_points[i+1].price)
                        }
                    }
        
        return {
            'current_wave': 'Pattern completed',
            'completion_percentage': 100.0,
            'key_levels': {
                'support': pattern.start_point.price,
                'resistance': pattern.end_point.price
            }
        }
    
    def get_clean_wave_analysis(self) -> Dict:
        """
        Get clean, focused Elliott Wave analysis for trading
        
        Returns:
            Dictionary with wave count, pattern type, and key levels
        """
        if not self.wave_points:
            self.detect_swing_points()
        if not self.patterns:
            self.identify_wave_patterns()
        
        # Get best pattern
        best_pattern = None
        if self.patterns:
            best_pattern = max(self.patterns, key=lambda p: p.confidence)
        
        # Analyze wave structure
        wave_structure = self._analyze_wave_structure()
        
        # Get key support/resistance levels
        key_levels = self._get_trading_levels()
        
        return {
            'wave_count': wave_structure['count'],
            'pattern_type': wave_structure['type'],
            'wave_direction': wave_structure['direction'],
            'current_position': wave_structure['current_position'],
            'pattern_confidence': best_pattern.confidence if best_pattern else 0.0,
            'key_levels': key_levels,
            'next_targets': self._get_next_targets(),
            'wave_labels': wave_structure['labels']
        }
    
    def _analyze_wave_structure(self) -> Dict:
        """Analyze the wave structure with adaptive timeframe-aware Elliott Wave analysis"""
        if len(self.wave_points) < 3:
            return {
                'count': 'Insufficient data',
                'type': 'Unknown',
                'direction': 'Unknown',
                'current_position': 'Unknown',
                'labels': []
            }
        
        # Adaptive analysis window based on swing percentage and data characteristics
        total_points = len(self.wave_points)
        data_length = len(self.price_data)
        
        # Dynamic window sizing based on timeframe characteristics
        if self.min_swing_percentage <= 1.0:  # Short term
            # For short term, focus on recent patterns but adapt to data availability
            base_window = 6
            analysis_window = min(base_window + (total_points // 5), total_points)
            wave_degree = "Minor"
            pattern_sensitivity = "high"
        elif self.min_swing_percentage <= 3.0:  # Medium term
            # Medium term needs broader context
            base_window = 8
            analysis_window = min(base_window + (total_points // 3), total_points)
            wave_degree = "Intermediate"
            pattern_sensitivity = "medium"
        else:  # Long term
            # Long term should consider most or all available patterns
            analysis_window = max(total_points // 2, min(10, total_points))
            wave_degree = "Primary"
            pattern_sensitivity = "low"
        
        # Ensure minimum window for pattern recognition
        analysis_window = max(3, min(analysis_window, total_points))
        points = self.wave_points[-analysis_window:]
        
        # Adaptive pattern recognition based on timeframe sensitivity
        confidence_threshold = {
            "high": 0.3,    # Lower threshold for short-term patterns
            "medium": 0.5,  # Standard threshold 
            "low": 0.7      # Higher threshold for long-term patterns
        }[pattern_sensitivity]
        
        # Adaptive pattern identification
        best_pattern = None
        best_confidence = 0
        
        # Try 5-wave impulse pattern with adaptive confidence
        if len(points) >= 5:
            for start_idx in range(max(0, len(points) - 8), len(points) - 4):
                test_points = points[start_idx:start_idx + 5]
                is_valid, confidence = self.validate_impulse_wave(test_points)
                
                if is_valid and confidence >= confidence_threshold and confidence > best_confidence:
                    direction = 'Up' if test_points[-1].price > test_points[0].price else 'Down'
                    
                    # Adaptive labeling based on position within larger structure
                    if len(points) > 5:
                        # Part of larger structure
                        labels = ['i', 'ii', 'iii', 'iv', 'v'] * 2  # Extended for complex patterns
                        labels = labels[-len(points):]
                    else:
                        labels = ['i', 'ii', 'iii', 'iv', 'v'][-len(points):]
                    
                    best_pattern = {
                        'count': '5-Wave',
                        'type': 'Impulse',
                        'direction': direction,
                        'current_position': f"Wave v ({wave_degree}, confidence: {confidence:.0%})",
                        'labels': labels,
                        'confidence': confidence
                    }
                    best_confidence = confidence
        
        # Try 3-wave corrective pattern with adaptive confidence
        if len(points) >= 3:
            for start_idx in range(max(0, len(points) - 6), len(points) - 2):
                test_points = points[start_idx:start_idx + 3]
                if self._is_three_wave_structure(test_points):
                    # Calculate basic confidence for corrective patterns
                    confidence = 0.6 if pattern_sensitivity == "high" else 0.7
                    
                    if confidence >= confidence_threshold and confidence > best_confidence:
                        direction = 'Down' if test_points[-1].price < test_points[0].price else 'Up'
                        
                        # Adaptive corrective labeling
                        if len(points) > 3:
                            # Multiple corrective cycles
                            labels = ['a', 'b', 'c'] * 3  # Extended for complex corrections
                            labels = labels[-len(points):]
                        else:
                            labels = ['a', 'b', 'c'][-len(points):]
                        
                        best_pattern = {
                            'count': '3-Wave',
                            'type': 'Corrective',
                            'direction': direction,
                            'current_position': f"Wave c ({wave_degree}, confidence: {confidence:.0%})",
                            'labels': labels,
                            'confidence': confidence
                        }
                        best_confidence = confidence
        
        # Return best pattern found
        if best_pattern:
            return best_pattern
        
        # Developing pattern analysis - use consistent notation
        position = f"{wave_degree} degree in progress ({len(points)} points)"
        
        # Use Roman numerals for developing impulse waves, letters for corrections
        if len(points) <= 5:
            labels = ['i', 'ii', 'iii', 'iv', 'v'][:len(points)]
        else:
            # Mixed pattern - alternate between impulse and corrective
            labels = ['i', 'ii', 'iii', 'iv', 'v', 'a', 'b', 'c'][:len(points)]
        
        return {
            'count': f'{len(points)}-Wave',
            'type': 'Developing',
            'direction': 'Up' if points[-1].price > points[0].price else 'Down',
            'current_position': position,
            'labels': labels
        }
    
    def _is_five_wave_structure(self, points: List[WavePoint]) -> bool:
        """Check if 5 points form a valid 5-wave structure"""
        if len(points) != 5:
            return False
        
        # Basic 5-wave validation
        prices = [p.price for p in points]
        
        # For upward impulse: 1<3<5 and 2>4 (in terms of peaks)
        if points[0].point_type == 'low':  # Starting from low
            wave1 = prices[1] - prices[0]  # Wave 1 up
            wave3 = prices[3] - prices[2]  # Wave 3 up  
            wave5 = prices[4] - prices[3]  # Wave 5 up
            
            # Wave 3 should not be shortest
            if wave3 >= wave1 and wave3 >= wave5:
                return True
        
        return False
    
    def _is_three_wave_structure(self, points: List[WavePoint]) -> bool:
        """Check if 3 points form a valid 3-wave correction"""
        if len(points) != 3:
            return False
        
        # Basic ABC correction validation
        # A down, B up, C down (or vice versa)
        return points[0].point_type != points[2].point_type
    
    def _get_trading_levels(self) -> Dict:
        """Get key support and resistance levels for trading - ENHANCED VERSION"""
        if len(self.wave_points) < 2:
            return {'support': [], 'resistance': []}
        
        current_price = self.price_data.iloc[-1]
        
        # Import the enhanced calculator with ZONES
        try:
            from support_resistance_zones import SupportResistanceZoneCalculator
            # Debug: Using zones module for proper zone support
            
            # Use enhanced S/R zone calculator
            zone_calculator = SupportResistanceZoneCalculator(self.price_data)
            
            # Get Fibonacci levels from patterns if they exist
            fibonacci_levels = []
            if self.patterns:
                for pattern in self.patterns:
                    if hasattr(pattern, 'fibonacci_levels'):
                        fibonacci_levels.extend(pattern.fibonacci_levels)
            
            # Get enhanced ZONES with minimum 1% distance
            enhanced_zones = zone_calculator.calculate_sr_zones(
                wave_points=self.wave_points if self.wave_points else [],
                fibonacci_levels=fibonacci_levels,
                min_zone_width_pct=0.3,
                max_zone_width_pct=2.0,
                min_distance_pct=1.0,  # Minimum 1% away from current price
                max_zones=5
            )
            
            # Convert to format with ZONE information for frontend
            support_levels = []
            for level in enhanced_zones['support_zones']:
                # Handle SupportResistanceZone objects
                if hasattr(level, 'zone_center'):  # Zone object format
                    support_levels.append({
                        'price': level.zone_center,  # Zone center (for compatibility)
                        'zone_low': level.zone_low,  # Zone bottom
                        'zone_high': level.zone_high,  # Zone top
                        'zone_width_pct': level.zone_width_pct,  # Zone width %
                        'type': 'support',
                        'strength': 'High' if level.strength > 70 else 'Medium' if level.strength > 40 else 'Low',
                        'enhanced_strength': level.strength,
                        'source': ', '.join(level.sources) if hasattr(level, 'sources') else 'unknown',
                        'distance_pct': level.distance_from_current_pct,
                        'touch_count': level.touch_count,
                        'is_zone': True  # This is definitely a zone
                    })
                else:  # Dictionary format
                    support_levels.append(level)
            
            resistance_levels = []
            for level in enhanced_zones['resistance_zones']:
                # Handle SupportResistanceZone objects
                if hasattr(level, 'zone_center'):  # Zone object format
                    resistance_levels.append({
                        'price': level.zone_center,  # Zone center (for compatibility)
                        'zone_low': level.zone_low,  # Zone bottom  
                        'zone_high': level.zone_high,  # Zone top
                        'zone_width_pct': level.zone_width_pct,  # Zone width %
                        'type': 'resistance',
                        'strength': 'High' if level.strength > 70 else 'Medium' if level.strength > 40 else 'Low',
                        'enhanced_strength': level.strength,
                        'source': ', '.join(level.sources) if hasattr(level, 'sources') else 'unknown',
                        'distance_pct': level.distance_from_current_pct,
                        'touch_count': level.touch_count,
                        'is_zone': True  # This is definitely a zone
                    })
                else:  # Dictionary format
                    resistance_levels.append(level)
            
            return {
                'support': support_levels[:3],  # Top 3 support levels
                'resistance': resistance_levels[:3]  # Top 3 resistance levels
            }
            
        except ImportError:
            # Fallback to old method if enhanced module not available
            # BUT with minimum distance filtering added
            support_levels = []
            resistance_levels = []
            
            # Get support/resistance from wave points
            for point in self.wave_points[-10:]:  # Check more points
                distance_pct = abs(point.price - current_price) / current_price * 100
                
                # Skip levels too close to current price (within 1%)
                if distance_pct < 1.0:
                    continue
                
                if point.price < current_price:
                    support_levels.append({
                        'price': point.price,
                        'type': point.point_type,
                        'strength': 'High' if point.point_type == 'low' else 'Medium',
                        'distance_pct': distance_pct
                    })
                else:
                    resistance_levels.append({
                        'price': point.price,
                        'type': point.point_type,
                        'strength': 'High' if point.point_type == 'high' else 'Medium',
                        'distance_pct': distance_pct
                    })
            
            # Sort by strength (High first) then by distance
            def sort_key(x):
                strength_score = 2 if x['strength'] == 'High' else 1
                return (-strength_score, x['distance_pct'])
            
            support_levels.sort(key=sort_key)
            resistance_levels.sort(key=sort_key)
            
            return {
                'support': support_levels[:3],  # Top 3 support levels
                'resistance': resistance_levels[:3]  # Top 3 resistance levels
            }
    
    def _get_next_targets(self) -> Dict:
        """Get next price targets based on Elliott Wave theory"""
        if len(self.wave_points) < 3:
            return {}
        
        targets = {}
        current_price = self.price_data.iloc[-1]
        
        # Simple target calculation based on recent waves
        recent_points = self.wave_points[-3:]
        
        if len(recent_points) >= 2:
            last_move = abs(recent_points[-1].price - recent_points[-2].price)
            
            if recent_points[-1].point_type == 'high':
                # Potential correction target
                targets['correction_target'] = recent_points[-1].price - (last_move * 0.618)
            else:
                # Potential extension target
                targets['extension_target'] = recent_points[-1].price + (last_move * 1.618)
        
        return targets
    
    def get_elliott_wave_summary(self) -> Dict:
        """
        Get comprehensive Elliott Wave analysis summary
        
        Returns:
            Complete analysis including patterns, current position, and forecasts
        """
        # Ensure analysis is complete
        if not self.wave_points:
            self.detect_swing_points()
        if not self.patterns:
            self.identify_wave_patterns()
        
        current_analysis = self.get_current_wave_position()
        
        # Statistical analysis
        pattern_stats = {
            'total_patterns': len(self.patterns),
            'impulse_patterns': len([p for p in self.patterns if p.pattern_type == 'impulse']),
            'corrective_patterns': len([p for p in self.patterns if p.pattern_type == 'corrective']),
            'avg_confidence': np.mean([p.confidence for p in self.patterns]) if self.patterns else 0.0
        }
        
        # Recent significant levels
        recent_levels = []
        if self.wave_points:
            for point in self.wave_points[-5:]:  # Last 5 significant points
                recent_levels.append({
                    'date': point.date,
                    'price': point.price,
                    'type': point.point_type,
                    'label': point.wave_label or f"{point.point_type.title()}"
                })
        
        return {
            'current_analysis': current_analysis,
            'pattern_statistics': pattern_stats,
            'recent_levels': recent_levels,
            'wave_points_detected': len(self.wave_points),
            'analysis_quality': self._assess_analysis_quality()
        }
    
    def _assess_analysis_quality(self) -> str:
        """Assess the quality of the Elliott Wave analysis"""
        if len(self.wave_points) < 5:
            return "Limited - Insufficient data points"
        elif len(self.patterns) == 0:
            return "Poor - No clear patterns identified"
        elif len(self.patterns) >= 1 and np.mean([p.confidence for p in self.patterns]) > 0.7:
            return "Good - Clear patterns with high confidence"
        elif len(self.patterns) >= 1:
            return "Fair - Patterns identified with moderate confidence"
        else:
            return "Limited - Analysis needs more data"
    
    @staticmethod
    def analyze_multiple_degrees(price_data: pd.Series) -> Dict:
        """
        Perform multi-timeframe Elliott Wave analysis across different degrees
        
        Args:
            price_data: Price series with datetime index
            
        Returns:
            Dictionary containing analysis for each degree and confluence zones
        """
        degrees = ['Primary', 'Intermediate', 'Minor', 'Minute']
        multi_analysis = {}
        all_fibonacci_levels = []
        
        for degree in degrees:
            try:
                # Get degree-specific swing percentage
                degree_config = {
                    'Primary': {'swing_pct': 8.0, 'timeframe': 'Long-term (6+ months)'},
                    'Intermediate': {'swing_pct': 4.0, 'timeframe': 'Medium-term (1-6 months)'},
                    'Minor': {'swing_pct': 2.0, 'timeframe': 'Short-term (1-4 weeks)'},
                    'Minute': {'swing_pct': 1.0, 'timeframe': 'Very Short-term (1-7 days)'},
                }
                
                swing_pct = degree_config[degree]['swing_pct']
                timeframe = degree_config[degree]['timeframe']
                
                # Create analyzer for this degree
                analyzer = ElliottWaveAnalyzer(price_data, swing_pct, degree)
                
                # Perform analysis
                summary = analyzer.get_elliott_wave_summary()
                
                # Store results
                multi_analysis[degree] = {
                    'analyzer': analyzer,
                    'summary': summary,
                    'timeframe': timeframe,
                    'swing_percentage': swing_pct,
                    'wave_count': len(analyzer.wave_points),
                    'pattern_count': len(analyzer.patterns)
                }
                
                # Collect Fibonacci levels for confluence analysis
                current_analysis = summary['current_analysis']
                if current_analysis.get('fibonacci_levels'):
                    for fib in current_analysis['fibonacci_levels']:
                        all_fibonacci_levels.append({
                            'degree': degree,
                            'price': fib.price,
                            'ratio': fib.ratio,
                            'level_type': fib.level_type,
                            'timeframe': timeframe
                        })
                        
            except Exception as e:
                multi_analysis[degree] = {
                    'error': str(e),
                    'timeframe': degree_config.get(degree, {}).get('timeframe', 'Unknown'),
                    'wave_count': 0,
                    'pattern_count': 0
                }
        
        # Find confluence zones
        confluence_zones = ElliottWaveAnalyzer._find_confluence_zones(all_fibonacci_levels, price_data.iloc[-1])
        
        return {
            'degree_analysis': multi_analysis,
            'confluence_zones': confluence_zones,
            'total_fibonacci_levels': len(all_fibonacci_levels),
            'analysis_timestamp': pd.Timestamp.now()
        }
    
    @staticmethod
    def _find_confluence_zones(fibonacci_levels: List[Dict], current_price: float) -> List[Dict]:
        """
        Find confluence zones where Fibonacci levels from different timeframes align
        
        Args:
            fibonacci_levels: List of Fibonacci level dictionaries
            current_price: Current price for relevance filtering
            
        Returns:
            List of confluence zones sorted by strength
        """
        confluence_zones = []
        
        # Group levels by price proximity (within 1% of each other)
        price_tolerance = 0.01  # 1%
        
        processed_indices = set()
        
        for i, level1 in enumerate(fibonacci_levels):
            if i in processed_indices:
                continue
                
            # Find all levels within tolerance of this level
            confluence_group = [level1]
            group_indices = {i}
            
            for j, level2 in enumerate(fibonacci_levels[i+1:], i+1):
                if j in processed_indices:
                    continue
                    
                price_diff = abs(level1['price'] - level2['price']) / level1['price']
                
                if price_diff <= price_tolerance:
                    confluence_group.append(level2)
                    group_indices.add(j)
            
            # Only consider confluences with 2+ levels from different timeframes
            if len(confluence_group) >= 2:
                degrees_in_group = set(level['degree'] for level in confluence_group)
                
                if len(degrees_in_group) >= 2:  # Must have levels from different timeframes
                    avg_price = np.mean([level['price'] for level in confluence_group])
                    
                    # Calculate relevance (closer to current price = more relevant)
                    distance_from_current = abs(avg_price - current_price) / current_price
                    relevance = max(0, 1 - distance_from_current * 2)  # 0-1 scale
                    
                    confluence_zones.append({
                        'price': avg_price,
                        'strength': len(confluence_group),
                        'degrees': list(degrees_in_group),
                        'levels': confluence_group,
                        'relevance': relevance,
                        'distance_from_current_pct': distance_from_current * 100
                    })
                    
                    processed_indices.update(group_indices)
        
        # Sort by strength and relevance
        confluence_zones.sort(key=lambda x: (x['strength'] * x['relevance']), reverse=True)
        
        return confluence_zones[:10]  # Return top 10 confluence zones