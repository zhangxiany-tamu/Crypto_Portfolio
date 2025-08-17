"""
Support and Resistance ZONES Implementation
Based on established technical analysis principles that S/R are areas, not exact prices
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class SupportResistanceZone:
    """
    Represents a support/resistance ZONE rather than a single price
    This aligns with professional trading literature that emphasizes zones over lines
    """
    zone_center: float  # Central price of the zone
    zone_high: float    # Upper boundary of the zone
    zone_low: float     # Lower boundary of the zone
    zone_width_pct: float  # Width as percentage of price
    strength: float     # 0-100 strength score
    zone_type: str      # 'support' or 'resistance'
    sources: List[str]  # What identified this zone
    touch_count: int    # Number of times tested
    volume_concentration: float  # Volume traded in this zone (0-1)
    last_test_date: Optional[pd.Timestamp]
    distance_from_current_pct: float  # Distance of zone center from current price
    
    @property
    def zone_width(self) -> float:
        """Absolute width of the zone in price units"""
        return self.zone_high - self.zone_low
    
    def contains_price(self, price: float) -> bool:
        """Check if a price is within this zone"""
        return self.zone_low <= price <= self.zone_high
    
    def overlap_percentage(self, other: 'SupportResistanceZone') -> float:
        """Calculate how much two zones overlap (0-100%)"""
        overlap_low = max(self.zone_low, other.zone_low)
        overlap_high = min(self.zone_high, other.zone_high)
        
        if overlap_low >= overlap_high:
            return 0.0
        
        overlap_size = overlap_high - overlap_low
        self_size = self.zone_width
        
        return (overlap_size / self_size) * 100 if self_size > 0 else 0

class SupportResistanceZoneCalculator:
    """
    Advanced S/R Zone Calculator based on technical analysis best practices
    
    Key Concepts from Literature:
    1. S/R are zones, not exact prices (Murphy, Elder)
    2. Zone width varies with volatility (ATR-based sizing)
    3. High volume areas create stronger zones (Market Profile)
    4. Multiple timeframe confluence strengthens zones
    5. Zones can expand/contract based on market conditions
    """
    
    def __init__(self, 
                 price_data: pd.Series,
                 volume_data: Optional[pd.Series] = None,
                 high_data: Optional[pd.Series] = None,
                 low_data: Optional[pd.Series] = None):
        """
        Initialize zone calculator
        
        Args:
            price_data: Close prices
            volume_data: Volume data (optional)
            high_data: High prices for ATR calculation (optional)
            low_data: Low prices for ATR calculation (optional)
        """
        self.price_data = price_data
        self.volume_data = volume_data
        self.high_data = high_data
        self.low_data = low_data
        self.current_price = price_data.iloc[-1]
        
        # Calculate ATR for dynamic zone sizing
        self.atr = self._calculate_atr()
        
        # Calculate price volatility for zone width adjustment
        self.volatility_factor = self._calculate_volatility_factor()
    
    def _calculate_atr(self, period: int = 14) -> float:
        """
        Calculate Average True Range for dynamic zone sizing
        This is a standard method from Wilder (1978)
        """
        if self.high_data is None or self.low_data is None:
            # Estimate ATR from close prices if high/low not available
            returns = self.price_data.pct_change().dropna()
            return returns.std() * self.price_data.iloc[-1] * np.sqrt(period)
        
        # True Range calculation
        high_low = self.high_data - self.low_data
        high_close = abs(self.high_data - self.price_data.shift(1))
        low_close = abs(self.low_data - self.price_data.shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        return atr if pd.notna(atr) else self.price_data.iloc[-1] * 0.02
    
    def _calculate_volatility_factor(self) -> float:
        """
        Calculate volatility factor for zone width adjustment
        Higher volatility = wider zones (adaptive to market conditions)
        """
        if len(self.price_data) < 20:
            return 1.0
        
        # Calculate recent vs historical volatility
        recent_vol = self.price_data.iloc[-20:].pct_change().std()
        historical_vol = self.price_data.pct_change().std()
        
        if historical_vol > 0:
            return min(max(recent_vol / historical_vol, 0.5), 2.0)
        return 1.0
    
    def calculate_sr_zones(self,
                          wave_points: List = None,
                          fibonacci_levels: List = None,
                          min_zone_width_pct: float = 0.3,
                          max_zone_width_pct: float = 2.0,
                          min_distance_pct: float = 1.0,
                          max_zones: int = 5) -> Dict:
        """
        Calculate S/R zones using multiple methods
        
        Args:
            wave_points: Elliott Wave swing points
            fibonacci_levels: Fibonacci levels
            min_zone_width_pct: Minimum zone width as % of price
            max_zone_width_pct: Maximum zone width as % of price
            min_distance_pct: Minimum distance from current price
            max_zones: Maximum number of zones to return
            
        Returns:
            Dictionary with support and resistance zones
        """
        all_zones = []
        
        # 1. Volume-based zones (most reliable according to Market Profile theory)
        if self.volume_data is not None:
            volume_zones = self._create_volume_profile_zones()
            all_zones.extend(volume_zones)
        
        # 2. Wave structure zones
        if wave_points:
            wave_zones = self._create_wave_zones(wave_points)
            all_zones.extend(wave_zones)
        
        # 3. Fibonacci zones (use standard Fibonacci zone concepts)
        if fibonacci_levels:
            fib_zones = self._create_fibonacci_zones(fibonacci_levels)
            all_zones.extend(fib_zones)
        
        # 4. Price action zones (areas of consolidation)
        consolidation_zones = self._create_consolidation_zones()
        all_zones.extend(consolidation_zones)
        
        # 5. Dynamic zones based on ATR
        dynamic_zones = self._create_dynamic_zones()
        all_zones.extend(dynamic_zones)
        
        # Merge overlapping zones
        merged_zones = self._merge_overlapping_zones(all_zones)
        
        # Filter zones
        support_zones = []
        resistance_zones = []
        
        for zone in merged_zones:
            # Calculate distance from current price
            zone.distance_from_current_pct = abs(zone.zone_center - self.current_price) / self.current_price * 100
            
            # Skip zones too close to current price
            if zone.distance_from_current_pct < min_distance_pct:
                continue
            
            # Skip zones too far (more than 20%)
            if zone.distance_from_current_pct > 20:
                continue
            
            # Ensure zone width is within limits
            zone_width_pct = (zone.zone_width / zone.zone_center) * 100
            if zone_width_pct < min_zone_width_pct:
                # Expand zone to minimum width
                expansion = (min_zone_width_pct / 100 * zone.zone_center - zone.zone_width) / 2
                zone.zone_low -= expansion
                zone.zone_high += expansion
                zone.zone_width_pct = min_zone_width_pct
            elif zone_width_pct > max_zone_width_pct:
                # Contract zone to maximum width
                target_width = max_zone_width_pct / 100 * zone.zone_center
                contraction = (zone.zone_width - target_width) / 2
                zone.zone_low += contraction
                zone.zone_high -= contraction
                zone.zone_width_pct = max_zone_width_pct
            
            if zone.zone_center < self.current_price:
                zone.zone_type = 'support'
                support_zones.append(zone)
            else:
                zone.zone_type = 'resistance'
                resistance_zones.append(zone)
        
        # Sort by strength and select top zones
        support_zones.sort(key=lambda x: x.strength, reverse=True)
        resistance_zones.sort(key=lambda x: x.strength, reverse=True)
        
        return {
            'support_zones': support_zones[:max_zones],
            'resistance_zones': resistance_zones[:max_zones]
        }
    
    def _create_volume_profile_zones(self) -> List[SupportResistanceZone]:
        """
        Create zones based on Volume Profile (high volume nodes)
        This is based on Market Profile theory - high volume areas act as magnets
        """
        zones = []
        
        if self.volume_data is None or len(self.volume_data) < 20:
            return zones
        
        # Create volume profile with 30 bins
        price_bins = 30
        price_min = self.price_data.min()
        price_max = self.price_data.max()
        bins = np.linspace(price_min, price_max, price_bins)
        
        # Calculate volume at each price level
        volume_profile = np.zeros(len(bins) - 1)
        price_counts = np.zeros(len(bins) - 1)
        
        for i in range(len(self.price_data)):
            price = self.price_data.iloc[i]
            volume = self.volume_data.iloc[i] if i < len(self.volume_data) else 0
            bin_idx = np.digitize(price, bins) - 1
            if 0 <= bin_idx < len(volume_profile):
                volume_profile[bin_idx] += volume
                price_counts[bin_idx] += 1
        
        # Find high volume nodes (top 30%)
        threshold = np.percentile(volume_profile[volume_profile > 0], 70)
        
        for i, vol in enumerate(volume_profile):
            if vol > threshold:
                # Zone center is the middle of the bin
                zone_center = (bins[i] + bins[i + 1]) / 2
                
                # Zone width based on bin size and volume concentration
                base_width = bins[i + 1] - bins[i]
                volume_factor = vol / volume_profile.max()
                
                # Wider zones for high volume areas (they're stronger)
                zone_width = base_width * (1 + volume_factor)
                
                zone = SupportResistanceZone(
                    zone_center=zone_center,
                    zone_low=zone_center - zone_width / 2,
                    zone_high=zone_center + zone_width / 2,
                    zone_width_pct=(zone_width / zone_center) * 100,
                    strength=50 + (volume_factor * 40),  # 50-90 strength
                    zone_type='pending',  # Will be set later
                    sources=['volume_profile'],
                    touch_count=int(price_counts[i]),
                    volume_concentration=volume_factor,
                    last_test_date=None,
                    distance_from_current_pct=0
                )
                zones.append(zone)
        
        return zones
    
    def _create_wave_zones(self, wave_points: List) -> List[SupportResistanceZone]:
        """
        Create zones around Elliott Wave swing points
        Zone width based on local volatility and swing magnitude
        """
        zones = []
        
        for i, point in enumerate(wave_points[-10:]):  # Last 10 points
            # Calculate local volatility around this point
            point_idx = self.price_data.index.get_loc(point.date, method='nearest')
            local_window = 5
            
            start_idx = max(0, point_idx - local_window)
            end_idx = min(len(self.price_data), point_idx + local_window)
            
            local_prices = self.price_data.iloc[start_idx:end_idx]
            local_volatility = local_prices.pct_change().std()
            
            # Zone width based on ATR and local volatility
            zone_width = self.atr * self.volatility_factor * (1 + local_volatility * 10)
            zone_width_pct = (zone_width / point.price) * 100
            
            # Adjust zone width based on swing type
            if point.point_type == 'high':
                # Resistance zones slightly wider above
                zone_low = point.price - zone_width * 0.3
                zone_high = point.price + zone_width * 0.7
            elif point.point_type == 'low':
                # Support zones slightly wider below
                zone_low = point.price - zone_width * 0.7
                zone_high = point.price + zone_width * 0.3
            else:
                zone_low = point.price - zone_width / 2
                zone_high = point.price + zone_width / 2
            
            zone = SupportResistanceZone(
                zone_center=point.price,
                zone_low=zone_low,
                zone_high=zone_high,
                zone_width_pct=zone_width_pct,
                strength=60 if point.point_type in ['high', 'low'] else 40,
                zone_type='pending',
                sources=['elliott_wave'],
                touch_count=1,
                volume_concentration=0,
                last_test_date=point.date,
                distance_from_current_pct=0
            )
            zones.append(zone)
        
        return zones
    
    def _create_fibonacci_zones(self, fibonacci_levels: List) -> List[SupportResistanceZone]:
        """
        Create zones around Fibonacci levels
        Key Fibonacci zones get different widths based on importance
        """
        zones = []
        
        # Fibonacci importance and zone width multipliers
        fib_importance = {
            0.236: {'strength': 40, 'width_mult': 0.8},
            0.382: {'strength': 60, 'width_mult': 1.0},
            0.5: {'strength': 70, 'width_mult': 1.2},   # Important psychological level
            0.618: {'strength': 80, 'width_mult': 1.5},  # Golden ratio - widest zone
            0.786: {'strength': 50, 'width_mult': 0.9},
            1.0: {'strength': 60, 'width_mult': 1.1},
            1.272: {'strength': 50, 'width_mult': 0.9},
            1.618: {'strength': 75, 'width_mult': 1.3},  # Golden extension
            2.618: {'strength': 40, 'width_mult': 0.8}
        }
        
        for fib in fibonacci_levels:
            config = fib_importance.get(fib.ratio, {'strength': 30, 'width_mult': 0.7})
            
            # Base zone width from ATR
            base_width = self.atr * self.volatility_factor
            zone_width = base_width * config['width_mult']
            
            zone = SupportResistanceZone(
                zone_center=fib.price,
                zone_low=fib.price - zone_width / 2,
                zone_high=fib.price + zone_width / 2,
                zone_width_pct=(zone_width / fib.price) * 100,
                strength=config['strength'],
                zone_type='pending',
                sources=['fibonacci'],
                touch_count=0,
                volume_concentration=0,
                last_test_date=None,
                distance_from_current_pct=0
            )
            zones.append(zone)
        
        return zones
    
    def _create_consolidation_zones(self) -> List[SupportResistanceZone]:
        """
        Identify price consolidation areas (sideways movement)
        These often act as future S/R zones
        """
        zones = []
        
        if len(self.price_data) < 20:
            return zones
        
        # Look for periods of low volatility (consolidation)
        window = 10
        rolling_std = self.price_data.rolling(window=window).std()
        rolling_mean = self.price_data.rolling(window=window).mean()
        
        # Find consolidation periods (low volatility)
        median_std = rolling_std.median()
        consolidation_threshold = median_std * 0.7
        
        consolidation_periods = rolling_std < consolidation_threshold
        
        # Group consecutive consolidation periods
        i = 0
        while i < len(consolidation_periods) - window:
            if consolidation_periods.iloc[i]:
                # Start of consolidation
                start_idx = i
                while i < len(consolidation_periods) and consolidation_periods.iloc[i]:
                    i += 1
                end_idx = i
                
                # Get price range during consolidation
                consolidation_prices = self.price_data.iloc[start_idx:end_idx]
                zone_center = consolidation_prices.mean()
                zone_low = consolidation_prices.min()
                zone_high = consolidation_prices.max()
                
                # Only create zone if it's significant
                if (zone_high - zone_low) / zone_center > 0.005:  # At least 0.5% range
                    zone = SupportResistanceZone(
                        zone_center=zone_center,
                        zone_low=zone_low,
                        zone_high=zone_high,
                        zone_width_pct=((zone_high - zone_low) / zone_center) * 100,
                        strength=30 + min(30, (end_idx - start_idx)),  # Longer consolidation = stronger
                        zone_type='pending',
                        sources=['consolidation'],
                        touch_count=end_idx - start_idx,
                        volume_concentration=0,
                        last_test_date=self.price_data.index[end_idx - 1],
                        distance_from_current_pct=0
                    )
                    zones.append(zone)
            i += 1
        
        return zones
    
    def _create_dynamic_zones(self) -> List[SupportResistanceZone]:
        """
        Create dynamic S/R zones based on recent price action
        Uses concepts from floor trader pivots and Camarilla equations
        """
        zones = []
        
        if len(self.price_data) < 2:
            return zones
        
        # Daily pivot calculations (classic floor trader method)
        recent_high = self.high_data.iloc[-1] if self.high_data is not None else self.price_data.iloc[-1]
        recent_low = self.low_data.iloc[-1] if self.low_data is not None else self.price_data.iloc[-1]
        recent_close = self.price_data.iloc[-1]
        
        # Pivot point
        pivot = (recent_high + recent_low + recent_close) / 3
        
        # Range for zone width
        daily_range = recent_high - recent_low
        zone_width = daily_range * 0.25 * self.volatility_factor
        
        # Support and resistance levels with zones
        r1 = 2 * pivot - recent_low
        r2 = pivot + daily_range
        s1 = 2 * pivot - recent_high
        s2 = pivot - daily_range
        
        pivot_zones = [
            (s2, 30, 'S2'),
            (s1, 40, 'S1'),
            (pivot, 50, 'Pivot'),
            (r1, 40, 'R1'),
            (r2, 30, 'R2')
        ]
        
        for level, strength, label in pivot_zones:
            if pd.notna(level) and level > 0:
                zone = SupportResistanceZone(
                    zone_center=level,
                    zone_low=level - zone_width / 2,
                    zone_high=level + zone_width / 2,
                    zone_width_pct=(zone_width / level) * 100,
                    strength=strength,
                    zone_type='pending',
                    sources=[f'pivot_{label}'],
                    touch_count=0,
                    volume_concentration=0,
                    last_test_date=None,
                    distance_from_current_pct=0
                )
                zones.append(zone)
        
        return zones
    
    def _merge_overlapping_zones(self, zones: List[SupportResistanceZone], 
                                 overlap_threshold: float = 50) -> List[SupportResistanceZone]:
        """
        Merge zones that overlap significantly
        This creates stronger confluence zones
        """
        if not zones:
            return zones
        
        merged = []
        zones_sorted = sorted(zones, key=lambda x: x.zone_center)
        
        current_group = [zones_sorted[0]]
        
        for zone in zones_sorted[1:]:
            # Check overlap with last zone in current group
            overlap = current_group[-1].overlap_percentage(zone)
            
            if overlap >= overlap_threshold:
                current_group.append(zone)
            else:
                # Merge current group and start new one
                if current_group:
                    merged_zone = self._merge_zone_group(current_group)
                    merged.append(merged_zone)
                current_group = [zone]
        
        # Don't forget last group
        if current_group:
            merged_zone = self._merge_zone_group(current_group)
            merged.append(merged_zone)
        
        return merged
    
    def _merge_zone_group(self, group: List[SupportResistanceZone]) -> SupportResistanceZone:
        """
        Merge a group of overlapping zones into a single stronger zone
        """
        if len(group) == 1:
            return group[0]
        
        # Weight calculations by strength
        total_strength = sum(z.strength for z in group)
        
        # Weighted average center
        weighted_center = sum(z.zone_center * z.strength for z in group) / total_strength
        
        # Union of zone boundaries (expanded zone)
        zone_low = min(z.zone_low for z in group)
        zone_high = max(z.zone_high for z in group)
        
        # Combined strength with confluence bonus
        base_strength = max(z.strength for z in group)
        confluence_bonus = min(30, len(group) * 5)  # 5 points per overlapping zone
        combined_strength = min(100, base_strength + confluence_bonus)
        
        # Combine sources
        all_sources = []
        for z in group:
            all_sources.extend(z.sources)
        unique_sources = list(set(all_sources))
        
        # Total touches and volume
        total_touches = sum(z.touch_count for z in group)
        avg_volume = np.mean([z.volume_concentration for z in group])
        
        # Most recent test
        test_dates = [z.last_test_date for z in group if z.last_test_date is not None]
        last_test = max(test_dates) if test_dates else None
        
        return SupportResistanceZone(
            zone_center=weighted_center,
            zone_low=zone_low,
            zone_high=zone_high,
            zone_width_pct=((zone_high - zone_low) / weighted_center) * 100,
            strength=combined_strength,
            zone_type='pending',
            sources=unique_sources,
            touch_count=total_touches,
            volume_concentration=avg_volume,
            last_test_date=last_test,
            distance_from_current_pct=0
        )
    
    def visualize_zones(self, zones_dict: Dict) -> str:
        """
        Create a text visualization of S/R zones
        Useful for understanding zone relationships
        """
        all_zones = zones_dict.get('support_zones', []) + zones_dict.get('resistance_zones', [])
        
        if not all_zones:
            return "No zones identified"
        
        # Sort all zones by center price
        all_zones.sort(key=lambda x: x.zone_center, reverse=True)
        
        output = []
        output.append(f"Current Price: ${self.current_price:.2f}")
        output.append("=" * 60)
        
        for zone in all_zones:
            zone_type = "RES" if zone.zone_type == 'resistance' else "SUP"
            
            # Visual representation
            if zone.zone_type == 'resistance':
                visual = "▀" * int(zone.strength / 10)
            else:
                visual = "▄" * int(zone.strength / 10)
            
            output.append(f"{zone_type} Zone: ${zone.zone_low:.2f} - ${zone.zone_high:.2f}")
            output.append(f"  Center: ${zone.zone_center:.2f} ({zone.distance_from_current_pct:.1f}% away)")
            output.append(f"  Width: {zone.zone_width_pct:.2f}% | Strength: {zone.strength:.0f}/100")
            output.append(f"  Sources: {', '.join(zone.sources)}")
            output.append(f"  Visual: {visual}")
            output.append("")
        
        return "\n".join(output)


# Integration function for Elliott Wave Analyzer
def integrate_zones_with_elliott_wave(analyzer, price_data, volume_data=None, 
                                     high_data=None, low_data=None):
    """
    Integrate S/R zones with Elliott Wave Analyzer
    
    Returns zones instead of single price levels
    """
    # Get wave points from analyzer
    wave_points = analyzer.wave_points if analyzer.wave_points else analyzer.detect_swing_points()
    
    # Get Fibonacci levels if patterns exist
    fibonacci_levels = []
    if analyzer.patterns:
        for pattern in analyzer.patterns:
            fibonacci_levels.extend(pattern.fibonacci_levels)
    
    # Create zone calculator
    zone_calc = SupportResistanceZoneCalculator(price_data, volume_data, high_data, low_data)
    
    # Calculate zones
    zones = zone_calc.calculate_sr_zones(
        wave_points=wave_points,
        fibonacci_levels=fibonacci_levels,
        min_zone_width_pct=0.5,  # Minimum 0.5% zone width
        max_zone_width_pct=2.0,   # Maximum 2% zone width
        min_distance_pct=1.0,     # At least 1% from current price
        max_zones=5
    )
    
    return zones, zone_calc


# Example usage
if __name__ == "__main__":
    # Create sample data
    import datetime as dt
    
    dates = pd.date_range(end=dt.datetime.now(), periods=100, freq='D')
    np.random.seed(42)
    
    # Synthetic price data with clear levels
    prices = []
    volumes = []
    highs = []
    lows = []
    
    price = 100
    for i in range(100):
        # Create some structure
        if 95 < price < 97:
            price += np.random.uniform(0, 1)  # Support zone
        elif 103 < price < 105:
            price -= np.random.uniform(0, 1)  # Resistance zone
        else:
            price += np.random.uniform(-1, 1)
        
        daily_range = abs(np.random.normal(0, 1))
        high = price + daily_range
        low = price - daily_range
        
        prices.append(price)
        highs.append(high)
        lows.append(low)
        volumes.append(np.random.uniform(1000, 5000))
    
    price_series = pd.Series(prices, index=dates)
    volume_series = pd.Series(volumes, index=dates)
    high_series = pd.Series(highs, index=dates)
    low_series = pd.Series(lows, index=dates)
    
    # Create zone calculator
    zone_calc = SupportResistanceZoneCalculator(
        price_series, volume_series, high_series, low_series
    )
    
    # Calculate zones
    zones = zone_calc.calculate_sr_zones(
        min_zone_width_pct=0.5,
        max_zone_width_pct=2.0,
        min_distance_pct=1.0
    )
    
    # Visualize
    print(zone_calc.visualize_zones(zones))
    
    print("\n" + "=" * 60)
    print("ZONE ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\nSupport Zones: {len(zones['support_zones'])}")
    for i, zone in enumerate(zones['support_zones'], 1):
        print(f"{i}. ${zone.zone_low:.2f} - ${zone.zone_high:.2f}")
        print(f"   Width: {zone.zone_width_pct:.2f}% | Strength: {zone.strength:.0f}/100")
        print(f"   Distance: {zone.distance_from_current_pct:.1f}% | Sources: {', '.join(zone.sources)}")
    
    print(f"\nResistance Zones: {len(zones['resistance_zones'])}")
    for i, zone in enumerate(zones['resistance_zones'], 1):
        print(f"{i}. ${zone.zone_low:.2f} - ${zone.zone_high:.2f}")
        print(f"   Width: {zone.zone_width_pct:.2f}% | Strength: {zone.strength:.0f}/100")
        print(f"   Distance: {zone.distance_from_current_pct:.1f}% | Sources: {', '.join(zone.sources)}")