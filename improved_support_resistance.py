import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class SupportResistanceLevel:
    """Enhanced support/resistance level with detailed metrics"""
    price: float
    strength: float  # 0-100 score
    level_type: str  # 'support' or 'resistance'
    source: str  # 'wave_point', 'fibonacci', 'volume', 'psychological', 'moving_average'
    touch_count: int  # Number of times tested
    last_test_date: Optional[pd.Timestamp]
    distance_from_current_pct: float
    
class EnhancedSupportResistance:
    """Advanced support and resistance calculation with multiple methods"""
    
    def __init__(self, price_data: pd.Series, volume_data: Optional[pd.Series] = None):
        self.price_data = price_data
        self.volume_data = volume_data
        self.current_price = price_data.iloc[-1]
        
    def get_trading_levels(self, 
                          wave_points: List = None,
                          fibonacci_levels: List = None,
                          min_distance_pct: float = 1.0,
                          max_levels: int = 5) -> Dict:
        """
        Calculate support and resistance with multiple methods
        
        Args:
            wave_points: Elliott Wave swing points
            fibonacci_levels: Fibonacci retracement/extension levels
            min_distance_pct: Minimum distance from current price (default 1%)
            max_levels: Maximum number of levels to return per side
        """
        all_levels = []
        
        # 1. Wave-based S/R (if provided)
        if wave_points:
            wave_levels = self._get_wave_levels(wave_points)
            all_levels.extend(wave_levels)
        
        # 2. Fibonacci levels (if provided)
        if fibonacci_levels:
            fib_levels = self._get_fibonacci_levels(fibonacci_levels)
            all_levels.extend(fib_levels)
        
        # 3. Volume-based S/R (if volume data available)
        if self.volume_data is not None:
            volume_levels = self._get_volume_profile_levels()
            all_levels.extend(volume_levels)
        
        # 4. Psychological levels (round numbers)
        psych_levels = self._get_psychological_levels()
        all_levels.extend(psych_levels)
        
        # 5. Moving average S/R
        ma_levels = self._get_moving_average_levels()
        all_levels.extend(ma_levels)
        
        # 6. Historical pivot points
        pivot_levels = self._get_pivot_levels()
        all_levels.extend(pivot_levels)
        
        # Filter and score levels
        support_levels = []
        resistance_levels = []
        
        for level in all_levels:
            distance_pct = abs(level.price - self.current_price) / self.current_price * 100
            level.distance_from_current_pct = distance_pct
            
            # Skip levels too close to current price
            if distance_pct < min_distance_pct:
                continue
            
            # Skip levels too far (more than 20% away)
            if distance_pct > 20:
                continue
            
            if level.price < self.current_price:
                support_levels.append(level)
            else:
                resistance_levels.append(level)
        
        # Merge nearby levels and combine their strength
        support_levels = self._merge_nearby_levels(support_levels)
        resistance_levels = self._merge_nearby_levels(resistance_levels)
        
        # Sort by strength (strongest first) not proximity!
        support_levels.sort(key=lambda x: x.strength, reverse=True)
        resistance_levels.sort(key=lambda x: x.strength, reverse=True)
        
        # Then sort by distance for the top strongest levels
        top_support = sorted(support_levels[:max_levels*2], 
                           key=lambda x: x.distance_from_current_pct)[:max_levels]
        top_resistance = sorted(resistance_levels[:max_levels*2], 
                              key=lambda x: x.distance_from_current_pct)[:max_levels]
        
        return {
            'support': top_support,
            'resistance': top_resistance
        }
    
    def _get_wave_levels(self, wave_points: List) -> List[SupportResistanceLevel]:
        """Extract S/R from Elliott Wave points with touch count analysis"""
        levels = []
        price_counts = defaultdict(int)
        
        # Count how many times each price area is touched
        for point in wave_points:
            # Round to nearest 0.1% to group nearby prices
            rounded_price = round(point.price / self.current_price * 1000) / 1000 * self.current_price
            price_counts[rounded_price] += 1
        
        for point in wave_points[-10:]:  # Last 10 wave points
            touch_count = price_counts[round(point.price / self.current_price * 1000) / 1000 * self.current_price]
            
            # Calculate strength based on multiple factors
            strength = 30  # Base strength for wave point
            
            # Add strength for multiple touches
            strength += min(touch_count * 10, 30)
            
            # Add strength for recent points
            recency_factor = wave_points.index(point) / len(wave_points)
            strength += recency_factor * 20
            
            # Add strength for major highs/lows
            if point.point_type in ['high', 'low']:
                strength += 20
            
            level_type = 'support' if point.price < self.current_price else 'resistance'
            
            levels.append(SupportResistanceLevel(
                price=point.price,
                strength=min(strength, 100),
                level_type=level_type,
                source='wave_point',
                touch_count=touch_count,
                last_test_date=point.date if hasattr(point, 'date') else None,
                distance_from_current_pct=0  # Will be calculated later
            ))
        
        return levels
    
    def _get_fibonacci_levels(self, fibonacci_levels: List) -> List[SupportResistanceLevel]:
        """Convert Fibonacci levels to S/R with importance weighting"""
        levels = []
        
        # Key Fibonacci ratios have different importance
        key_ratios = {
            0.236: 40,
            0.382: 60,
            0.5: 70,
            0.618: 80,  # Golden ratio - most important
            0.786: 50,
            1.0: 60,
            1.272: 50,
            1.618: 75,  # Golden ratio extension
            2.618: 40
        }
        
        for fib in fibonacci_levels:
            strength = key_ratios.get(fib.ratio, 30)
            level_type = 'support' if fib.price < self.current_price else 'resistance'
            
            levels.append(SupportResistanceLevel(
                price=fib.price,
                strength=strength,
                level_type=level_type,
                source='fibonacci',
                touch_count=0,
                last_test_date=None,
                distance_from_current_pct=0
            ))
        
        return levels
    
    def _get_volume_profile_levels(self) -> List[SupportResistanceLevel]:
        """Find S/R levels based on volume profile (high volume nodes)"""
        levels = []
        
        if self.volume_data is None or len(self.volume_data) < 20:
            return levels
        
        # Create price bins and calculate volume at each price level
        price_bins = 50
        price_min = self.price_data.min()
        price_max = self.price_data.max()
        bins = np.linspace(price_min, price_max, price_bins)
        
        volume_profile = np.zeros(len(bins) - 1)
        
        for i in range(len(self.price_data)):
            price = self.price_data.iloc[i]
            volume = self.volume_data.iloc[i] if i < len(self.volume_data) else 0
            bin_idx = np.digitize(price, bins) - 1
            if 0 <= bin_idx < len(volume_profile):
                volume_profile[bin_idx] += volume
        
        # Find high volume nodes (top 20%)
        threshold = np.percentile(volume_profile, 80)
        
        for i, vol in enumerate(volume_profile):
            if vol > threshold:
                price = (bins[i] + bins[i + 1]) / 2
                strength = 40 + (vol / volume_profile.max()) * 40
                level_type = 'support' if price < self.current_price else 'resistance'
                
                levels.append(SupportResistanceLevel(
                    price=price,
                    strength=strength,
                    level_type=level_type,
                    source='volume',
                    touch_count=0,
                    last_test_date=None,
                    distance_from_current_pct=0
                ))
        
        return levels
    
    def _get_psychological_levels(self) -> List[SupportResistanceLevel]:
        """Find psychological S/R levels (round numbers)"""
        levels = []
        
        # Determine the order of magnitude for round numbers
        if self.current_price < 1:
            round_increments = [0.1, 0.25, 0.5]
        elif self.current_price < 10:
            round_increments = [1, 2.5, 5]
        elif self.current_price < 100:
            round_increments = [10, 25, 50]
        elif self.current_price < 1000:
            round_increments = [100, 250, 500]
        else:
            round_increments = [1000, 2500, 5000, 10000]
        
        for increment in round_increments:
            # Find nearest round numbers above and below current price
            below = (self.current_price // increment) * increment
            above = below + increment
            
            for price in [below, above]:
                if price <= 0:
                    continue
                
                # Stronger levels for rounder numbers
                if price % (increment * 10) == 0:
                    strength = 60
                elif price % (increment * 5) == 0:
                    strength = 50
                else:
                    strength = 40
                
                level_type = 'support' if price < self.current_price else 'resistance'
                
                levels.append(SupportResistanceLevel(
                    price=price,
                    strength=strength,
                    level_type=level_type,
                    source='psychological',
                    touch_count=0,
                    last_test_date=None,
                    distance_from_current_pct=0
                ))
        
        return levels
    
    def _get_moving_average_levels(self) -> List[SupportResistanceLevel]:
        """Calculate S/R from key moving averages"""
        levels = []
        
        # Key MA periods
        ma_configs = [
            (20, 40),   # 20-day MA
            (50, 50),   # 50-day MA
            (100, 55),  # 100-day MA
            (200, 70),  # 200-day MA (strongest)
        ]
        
        for period, strength_base in ma_configs:
            if len(self.price_data) >= period:
                ma_value = self.price_data.rolling(window=period).mean().iloc[-1]
                
                if pd.notna(ma_value):
                    # Check if price recently crossed this MA
                    recent_prices = self.price_data.iloc[-5:]
                    crosses = ((recent_prices.shift(1) < ma_value) & (recent_prices > ma_value)) | \
                             ((recent_prices.shift(1) > ma_value) & (recent_prices < ma_value))
                    
                    # Add strength if recently tested
                    strength = strength_base
                    if crosses.any():
                        strength += 20
                    
                    level_type = 'support' if ma_value < self.current_price else 'resistance'
                    
                    levels.append(SupportResistanceLevel(
                        price=ma_value,
                        strength=min(strength, 100),
                        level_type=level_type,
                        source='moving_average',
                        touch_count=crosses.sum(),
                        last_test_date=self.price_data.index[-1],
                        distance_from_current_pct=0
                    ))
        
        return levels
    
    def _get_pivot_levels(self) -> List[SupportResistanceLevel]:
        """Calculate pivot point S/R levels"""
        levels = []
        
        if len(self.price_data) < 2:
            return levels
        
        # Calculate daily pivot points
        high = self.price_data.rolling(window=1).max().iloc[-2]
        low = self.price_data.rolling(window=1).min().iloc[-2]
        close = self.price_data.iloc[-2]
        
        # Standard pivot calculation
        pivot = (high + low + close) / 3
        
        # Support and resistance levels
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        pivot_points = [
            (s3, 30, 'support'),
            (s2, 40, 'support'),
            (s1, 50, 'support'),
            (pivot, 60, 'support' if pivot < self.current_price else 'resistance'),
            (r1, 50, 'resistance'),
            (r2, 40, 'resistance'),
            (r3, 30, 'resistance'),
        ]
        
        for price, strength, level_type in pivot_points:
            if pd.notna(price) and price > 0:
                levels.append(SupportResistanceLevel(
                    price=price,
                    strength=strength,
                    level_type=level_type,
                    source='pivot',
                    touch_count=0,
                    last_test_date=None,
                    distance_from_current_pct=0
                ))
        
        return levels
    
    def _merge_nearby_levels(self, levels: List[SupportResistanceLevel], 
                            merge_threshold_pct: float = 0.5) -> List[SupportResistanceLevel]:
        """Merge levels that are very close together"""
        if not levels:
            return levels
        
        merged = []
        levels_sorted = sorted(levels, key=lambda x: x.price)
        
        current_group = [levels_sorted[0]]
        
        for level in levels_sorted[1:]:
            # Check if this level is close to the previous group
            price_diff_pct = abs(level.price - current_group[-1].price) / current_group[-1].price * 100
            
            if price_diff_pct < merge_threshold_pct:
                current_group.append(level)
            else:
                # Merge the current group and start a new one
                if current_group:
                    merged_level = self._merge_level_group(current_group)
                    merged.append(merged_level)
                current_group = [level]
        
        # Don't forget the last group
        if current_group:
            merged_level = self._merge_level_group(current_group)
            merged.append(merged_level)
        
        return merged
    
    def _merge_level_group(self, group: List[SupportResistanceLevel]) -> SupportResistanceLevel:
        """Merge a group of nearby levels into one stronger level"""
        if len(group) == 1:
            return group[0]
        
        # Weighted average price based on strength
        total_strength = sum(l.strength for l in group)
        weighted_price = sum(l.price * l.strength for l in group) / total_strength
        
        # Combined strength (with bonus for confluence)
        combined_strength = min(100, max(l.strength for l in group) + len(group) * 5)
        
        # Combine sources
        sources = list(set(l.source for l in group))
        source_str = ','.join(sources) if len(sources) > 1 else sources[0]
        
        # Total touch count
        total_touches = sum(l.touch_count for l in group)
        
        # Most recent test date
        test_dates = [l.last_test_date for l in group if l.last_test_date is not None]
        last_test = max(test_dates) if test_dates else None
        
        return SupportResistanceLevel(
            price=weighted_price,
            strength=combined_strength,
            level_type=group[0].level_type,
            source=source_str,
            touch_count=total_touches,
            last_test_date=last_test,
            distance_from_current_pct=group[0].distance_from_current_pct
        )


# Example usage with Elliott Wave Analyzer integration
def integrate_with_elliott_wave(analyzer, price_data, volume_data=None):
    """
    Integrate enhanced S/R with Elliott Wave Analyzer
    
    Args:
        analyzer: ElliottWaveAnalyzer instance
        price_data: Price series
        volume_data: Optional volume series
    """
    # Get wave points from analyzer
    wave_points = analyzer.wave_points if analyzer.wave_points else analyzer.detect_swing_points()
    
    # Get Fibonacci levels if patterns exist
    fibonacci_levels = []
    if analyzer.patterns:
        for pattern in analyzer.patterns:
            fibonacci_levels.extend(pattern.fibonacci_levels)
    
    # Create enhanced S/R calculator
    sr_calculator = EnhancedSupportResistance(price_data, volume_data)
    
    # Get enhanced trading levels
    trading_levels = sr_calculator.get_trading_levels(
        wave_points=wave_points,
        fibonacci_levels=fibonacci_levels,
        min_distance_pct=1.0,  # At least 1% away from current price
        max_levels=5
    )
    
    return trading_levels


# Patch for the existing ElliottWaveAnalyzer
def enhanced_get_trading_levels(self) -> Dict:
    """Enhanced version to replace the existing _get_trading_levels method"""
    
    # Use the enhanced calculator
    sr_calculator = EnhancedSupportResistance(self.price_data)
    
    # Get Fibonacci levels from patterns
    fibonacci_levels = []
    if self.patterns:
        for pattern in self.patterns:
            fibonacci_levels.extend(pattern.fibonacci_levels)
    
    return sr_calculator.get_trading_levels(
        wave_points=self.wave_points,
        fibonacci_levels=fibonacci_levels,
        min_distance_pct=1.0,
        max_levels=5
    )