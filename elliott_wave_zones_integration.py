"""
Integration of S/R Zones with Elliott Wave Analyzer
Provides both single prices (for compatibility) and zones (for professional trading)
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

# Import our modules
from elliott_wave_analyzer import ElliottWaveAnalyzer
from support_resistance_zones import SupportResistanceZoneCalculator, SupportResistanceZone

class EnhancedElliottWaveAnalyzer(ElliottWaveAnalyzer):
    """
    Enhanced Elliott Wave Analyzer with S/R Zones support
    
    Extends the original analyzer to provide:
    1. Traditional single-price S/R (for compatibility)
    2. Professional S/R zones (for advanced trading)
    3. Volume-weighted analysis (when volume data available)
    4. Multi-source confluence zones
    """
    
    def __init__(self, 
                 price_data: pd.Series, 
                 volume_data: Optional[pd.Series] = None,
                 high_data: Optional[pd.Series] = None,
                 low_data: Optional[pd.Series] = None,
                 min_swing_percentage: float = 2.0, 
                 degree: str = 'Intermediate',
                 use_zones: bool = True):
        """
        Initialize Enhanced Elliott Wave Analyzer
        
        Args:
            price_data: Close price series
            volume_data: Volume series (optional, enables volume-based zones)
            high_data: High price series (optional, for better ATR calculation)
            low_data: Low price series (optional, for better ATR calculation)
            min_swing_percentage: Minimum swing for wave detection
            degree: Elliott Wave degree
            use_zones: Whether to calculate S/R zones (default True)
        """
        # Initialize parent Elliott Wave Analyzer
        super().__init__(price_data, min_swing_percentage, degree)
        
        # Store additional data
        self.volume_data = volume_data
        self.high_data = high_data
        self.low_data = low_data
        self.use_zones = use_zones
        
        # Initialize zone calculator if zones are enabled
        if self.use_zones:
            self.zone_calculator = SupportResistanceZoneCalculator(
                price_data, volume_data, high_data, low_data
            )
        else:
            self.zone_calculator = None
        
        # Cache for calculated zones
        self._calculated_zones = None
        self._zones_cache_timestamp = None
    
    def get_trading_levels(self, return_zones: bool = None) -> Dict:
        """
        Get trading levels as either single prices or zones
        
        Args:
            return_zones: If True, returns zones. If False, returns single prices.
                         If None, uses self.use_zones setting.
        
        Returns:
            Dictionary with either single price levels or zone objects
        """
        if return_zones is None:
            return_zones = self.use_zones
        
        if return_zones and self.zone_calculator is not None:
            return self._get_trading_zones()
        else:
            return self._get_trading_levels()  # Original single-price method
    
    def _get_trading_zones(self) -> Dict:
        """Get S/R zones for professional trading"""
        
        # Check cache first
        current_time = pd.Timestamp.now()
        if (self._calculated_zones is not None and 
            self._zones_cache_timestamp is not None and
            (current_time - self._zones_cache_timestamp).total_seconds() < 300):  # 5 min cache
            return self._calculated_zones
        
        # Ensure wave analysis is complete
        if not self.wave_points:
            self.detect_swing_points()
        if not self.patterns:
            self.identify_wave_patterns()
        
        # Get Fibonacci levels from patterns
        fibonacci_levels = []
        if self.patterns:
            for pattern in self.patterns:
                if hasattr(pattern, 'fibonacci_levels'):
                    fibonacci_levels.extend(pattern.fibonacci_levels)
        
        # Calculate zones
        zones = self.zone_calculator.calculate_sr_zones(
            wave_points=self.wave_points,
            fibonacci_levels=fibonacci_levels,
            min_zone_width_pct=0.3,    # Adaptive based on timeframe
            max_zone_width_pct=2.5,    # Allow wider zones for volatile markets
            min_distance_pct=1.0,      # Must be at least 1% from current price
            max_zones=5
        )
        
        # Cache results
        self._calculated_zones = zones
        self._zones_cache_timestamp = current_time
        
        return zones
    
    def get_enhanced_analysis(self) -> Dict:
        """
        Get comprehensive analysis including both traditional and zone-based insights
        
        Returns complete Elliott Wave analysis with zones for professional trading
        """
        # Get traditional Elliott Wave analysis
        traditional_analysis = self.get_elliott_wave_summary()
        
        # Get zone analysis if enabled
        zone_analysis = None
        if self.use_zones and self.zone_calculator:
            zones = self._get_trading_zones()
            zone_analysis = {
                'support_zones': zones.get('support_zones', []),
                'resistance_zones': zones.get('resistance_zones', []),
                'total_zones': len(zones.get('support_zones', [])) + len(zones.get('resistance_zones', [])),
                'strongest_support': zones.get('support_zones', [None])[0] if zones.get('support_zones') else None,
                'strongest_resistance': zones.get('resistance_zones', [None])[0] if zones.get('resistance_zones') else None,
                'zone_visualization': self.zone_calculator.visualize_zones(zones) if zones else None
            }
        
        # Enhanced wave position analysis
        enhanced_position = self._get_enhanced_wave_position()
        
        return {
            'traditional_analysis': traditional_analysis,
            'zone_analysis': zone_analysis,
            'enhanced_position': enhanced_position,
            'data_quality': self._assess_data_quality(),
            'trading_insights': self._generate_trading_insights()
        }
    
    def _get_enhanced_wave_position(self) -> Dict:
        """Enhanced wave position analysis with zone context"""
        
        base_position = self.get_current_wave_position()
        
        if not self.use_zones or not self.zone_calculator:
            return base_position
        
        # Add zone context
        current_price = self.price_data.iloc[-1]
        zones = self._get_trading_zones()
        
        # Check if current price is in any zone
        in_support_zone = None
        in_resistance_zone = None
        
        for zone in zones.get('support_zones', []):
            if zone.contains_price(current_price):
                in_support_zone = zone
                break
        
        for zone in zones.get('resistance_zones', []):
            if zone.contains_price(current_price):
                in_resistance_zone = zone
                break
        
        # Determine market context
        if in_support_zone:
            market_context = f"Inside support zone (${in_support_zone.zone_low:.2f} - ${in_support_zone.zone_high:.2f})"
            zone_strength = in_support_zone.strength
        elif in_resistance_zone:
            market_context = f"Inside resistance zone (${in_resistance_zone.zone_low:.2f} - ${in_resistance_zone.zone_high:.2f})"
            zone_strength = in_resistance_zone.strength
        else:
            market_context = "Between zones - no immediate S/R"
            zone_strength = 0
        
        base_position.update({
            'market_context': market_context,
            'current_zone_strength': zone_strength,
            'in_support_zone': in_support_zone is not None,
            'in_resistance_zone': in_resistance_zone is not None,
            'zone_trading_bias': self._determine_zone_bias(zones, current_price)
        })
        
        return base_position
    
    def _determine_zone_bias(self, zones: Dict, current_price: float) -> str:
        """Determine trading bias based on zone positioning"""
        
        support_zones = zones.get('support_zones', [])
        resistance_zones = zones.get('resistance_zones', [])
        
        if not support_zones and not resistance_zones:
            return "Neutral - No clear zones"
        
        # Distance to nearest zones
        nearest_support_distance = float('inf')
        nearest_resistance_distance = float('inf')
        
        if support_zones:
            nearest_support = support_zones[0]  # Strongest support
            nearest_support_distance = (current_price - nearest_support.zone_center) / current_price * 100
        
        if resistance_zones:
            nearest_resistance = resistance_zones[0]  # Strongest resistance
            nearest_resistance_distance = (nearest_resistance.zone_center - current_price) / current_price * 100
        
        # Determine bias
        if nearest_support_distance < 2 and nearest_support_distance > 0:
            return "Bullish - Near strong support"
        elif nearest_resistance_distance < 2 and nearest_resistance_distance > 0:
            return "Bearish - Near strong resistance"
        elif nearest_support_distance < nearest_resistance_distance:
            return "Cautiously Bullish - Support closer than resistance"
        elif nearest_resistance_distance < nearest_support_distance:
            return "Cautiously Bearish - Resistance closer than support"
        else:
            return "Neutral - Equidistant from zones"
    
    def _assess_data_quality(self) -> Dict:
        """Assess the quality of available data for analysis"""
        
        quality_score = 0
        max_score = 100
        quality_factors = []
        
        # Price data quality
        if len(self.price_data) >= 100:
            quality_score += 20
            quality_factors.append("Sufficient price history (100+ periods)")
        elif len(self.price_data) >= 50:
            quality_score += 10
            quality_factors.append("Moderate price history (50+ periods)")
        else:
            quality_factors.append("Limited price history (<50 periods)")
        
        # Volume data availability
        if self.volume_data is not None:
            quality_score += 25
            quality_factors.append("Volume data available (enables volume profile zones)")
        else:
            quality_factors.append("No volume data (volume-based analysis disabled)")
        
        # High/Low data for ATR
        if self.high_data is not None and self.low_data is not None:
            quality_score += 15
            quality_factors.append("OHLC data available (accurate ATR calculation)")
        else:
            quality_factors.append("Only close prices (estimated ATR)")
        
        # Wave pattern quality
        if len(self.wave_points) >= 5:
            quality_score += 20
            quality_factors.append("Multiple wave points detected")
        elif len(self.wave_points) >= 3:
            quality_score += 10
            quality_factors.append("Basic wave structure detected")
        else:
            quality_factors.append("Insufficient wave structure")
        
        # Pattern confidence
        if self.patterns:
            avg_confidence = np.mean([p.confidence for p in self.patterns])
            if avg_confidence > 0.7:
                quality_score += 20
                quality_factors.append("High confidence wave patterns")
            elif avg_confidence > 0.5:
                quality_score += 10
                quality_factors.append("Moderate confidence wave patterns")
            else:
                quality_factors.append("Low confidence wave patterns")
        else:
            quality_factors.append("No clear wave patterns identified")
        
        # Determine quality level
        if quality_score >= 80:
            quality_level = "Excellent"
        elif quality_score >= 60:
            quality_level = "Good"
        elif quality_score >= 40:
            quality_level = "Fair"
        else:
            quality_level = "Poor"
        
        return {
            'quality_score': quality_score,
            'quality_level': quality_level,
            'factors': quality_factors,
            'recommendations': self._get_quality_recommendations(quality_score)
        }
    
    def _get_quality_recommendations(self, quality_score: int) -> List[str]:
        """Get recommendations to improve analysis quality"""
        
        recommendations = []
        
        if quality_score < 40:
            recommendations.append("Consider using longer time period for more data")
            recommendations.append("Try different swing percentage for better wave detection")
        
        if self.volume_data is None:
            recommendations.append("Include volume data for volume-based S/R zones")
        
        if self.high_data is None or self.low_data is None:
            recommendations.append("Use OHLC data instead of close-only for better analysis")
        
        if len(self.patterns) == 0:
            recommendations.append("Adjust swing sensitivity or use different timeframe")
        
        if not recommendations:
            recommendations.append("Analysis quality is good - no specific improvements needed")
        
        return recommendations
    
    def _generate_trading_insights(self) -> Dict:
        """Generate actionable trading insights based on zones and waves"""
        
        insights = {
            'entry_opportunities': [],
            'risk_levels': [],
            'target_zones': [],
            'market_structure': 'Unknown'
        }
        
        if not self.use_zones or not self.zone_calculator:
            insights['note'] = "Zone analysis disabled - insights limited"
            return insights
        
        current_price = self.price_data.iloc[-1]
        zones = self._get_trading_zones()
        
        # Entry opportunities
        support_zones = zones.get('support_zones', [])
        resistance_zones = zones.get('resistance_zones', [])
        
        for zone in support_zones[:2]:  # Top 2 support zones
            if zone.distance_from_current_pct <= 5:  # Within 5%
                insights['entry_opportunities'].append({
                    'type': 'Long entry zone',
                    'zone_range': f"${zone.zone_low:.2f} - ${zone.zone_high:.2f}",
                    'strength': zone.strength,
                    'distance': f"{zone.distance_from_current_pct:.1f}%",
                    'confidence': 'High' if zone.strength > 70 else 'Medium'
                })
        
        for zone in resistance_zones[:2]:  # Top 2 resistance zones
            if zone.distance_from_current_pct <= 5:  # Within 5%
                insights['entry_opportunities'].append({
                    'type': 'Short entry zone',
                    'zone_range': f"${zone.zone_low:.2f} - ${zone.zone_high:.2f}",
                    'strength': zone.strength,
                    'distance': f"{zone.distance_from_current_pct:.1f}%",
                    'confidence': 'High' if zone.strength > 70 else 'Medium'
                })
        
        # Risk levels (stop loss zones)
        if support_zones:
            strongest_support = support_zones[0]
            insights['risk_levels'].append({
                'type': 'Long stop loss',
                'level': f"Below ${strongest_support.zone_low:.2f}",
                'reason': 'Break of strongest support zone'
            })
        
        if resistance_zones:
            strongest_resistance = resistance_zones[0]
            insights['risk_levels'].append({
                'type': 'Short stop loss',
                'level': f"Above ${strongest_resistance.zone_high:.2f}",
                'reason': 'Break of strongest resistance zone'
            })
        
        # Target zones (take profit areas)
        if support_zones and resistance_zones:
            insights['target_zones'].append({
                'type': 'Long target',
                'zone': f"${resistance_zones[0].zone_low:.2f} - ${resistance_zones[0].zone_high:.2f}",
                'reasoning': 'First resistance zone'
            })
            
            insights['target_zones'].append({
                'type': 'Short target',
                'zone': f"${support_zones[0].zone_low:.2f} - ${support_zones[0].zone_high:.2f}",
                'reasoning': 'First support zone'
            })
        
        # Market structure analysis
        if len(support_zones) > len(resistance_zones):
            insights['market_structure'] = 'Bullish structure (more support levels)'
        elif len(resistance_zones) > len(support_zones):
            insights['market_structure'] = 'Bearish structure (more resistance levels)'
        else:
            insights['market_structure'] = 'Balanced structure'
        
        return insights
    
    def export_analysis_summary(self) -> str:
        """Export complete analysis as formatted text"""
        
        analysis = self.get_enhanced_analysis()
        
        output = []
        output.append("=" * 80)
        output.append("ENHANCED ELLIOTT WAVE ANALYSIS WITH S/R ZONES")
        output.append("=" * 80)
        
        # Current market state
        current_price = self.price_data.iloc[-1]
        output.append(f"\nCurrent Price: ${current_price:.2f}")
        output.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Wave Analysis
        output.append("\n" + "=" * 50)
        output.append("ELLIOTT WAVE STRUCTURE")
        output.append("=" * 50)
        
        enhanced_pos = analysis['enhanced_position']
        output.append(f"Market Context: {enhanced_pos.get('market_context', 'Unknown')}")
        output.append(f"Zone Trading Bias: {enhanced_pos.get('zone_trading_bias', 'Unknown')}")
        
        # Zone Analysis
        if analysis['zone_analysis']:
            output.append("\n" + "=" * 50)
            output.append("SUPPORT & RESISTANCE ZONES")
            output.append("=" * 50)
            
            zone_viz = analysis['zone_analysis'].get('zone_visualization', '')
            if zone_viz:
                output.append(zone_viz)
        
        # Trading Insights
        insights = analysis['trading_insights']
        output.append("\n" + "=" * 50)
        output.append("TRADING INSIGHTS")
        output.append("=" * 50)
        
        output.append(f"\nMarket Structure: {insights['market_structure']}")
        
        if insights['entry_opportunities']:
            output.append("\nEntry Opportunities:")
            for opp in insights['entry_opportunities']:
                output.append(f"• {opp['type']}: {opp['zone_range']} (Confidence: {opp['confidence']})")
        
        if insights['risk_levels']:
            output.append("\nRisk Management:")
            for risk in insights['risk_levels']:
                output.append(f"• {risk['type']}: {risk['level']} - {risk['reason']}")
        
        # Data Quality
        quality = analysis['data_quality']
        output.append("\n" + "=" * 50)
        output.append("ANALYSIS QUALITY")
        output.append("=" * 50)
        output.append(f"\nQuality Score: {quality['quality_score']}/100 ({quality['quality_level']})")
        
        return "\n".join(output)


# Example usage and integration functions
def create_enhanced_analyzer(price_data: pd.Series,
                           volume_data: Optional[pd.Series] = None,
                           high_data: Optional[pd.Series] = None,
                           low_data: Optional[pd.Series] = None,
                           **kwargs) -> EnhancedElliottWaveAnalyzer:
    """
    Factory function to create enhanced analyzer with optimal settings
    """
    return EnhancedElliottWaveAnalyzer(
        price_data=price_data,
        volume_data=volume_data,
        high_data=high_data,
        low_data=low_data,
        use_zones=True,  # Enable zones by default
        **kwargs
    )


def upgrade_existing_analyzer(analyzer: ElliottWaveAnalyzer,
                            volume_data: Optional[pd.Series] = None,
                            high_data: Optional[pd.Series] = None,
                            low_data: Optional[pd.Series] = None) -> EnhancedElliottWaveAnalyzer:
    """
    Upgrade existing Elliott Wave Analyzer to enhanced version
    """
    return EnhancedElliottWaveAnalyzer(
        price_data=analyzer.price_data,
        volume_data=volume_data,
        high_data=high_data,
        low_data=low_data,
        min_swing_percentage=analyzer.min_swing_percentage,
        degree=analyzer.degree,
        use_zones=True
    )