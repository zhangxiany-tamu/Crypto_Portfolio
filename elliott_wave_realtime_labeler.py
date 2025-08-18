"""
Elliott Wave Real-Time Labeler and Predictor
Identifies current wave position and predicts next moves
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from elliott_wave_analyzer import ElliottWaveAnalyzer, WavePoint

class WaveScenario(Enum):
    """Possible wave scenarios"""
    IMPULSE_UP = "Bullish Impulse (i-ii-iii-iv-v up)"
    IMPULSE_DOWN = "Bearish Impulse (i-ii-iii-iv-v down)"
    CORRECTION_UP = "Corrective Rally (a-b-c up)"
    CORRECTION_DOWN = "Corrective Decline (a-b-c down)"
    UNCLEAR = "Unclear Pattern"

@dataclass
class WaveLabel:
    """Wave label with confidence"""
    point: WavePoint
    label: str  # e.g., "iii", "b", "iv?"
    confidence: float  # 0-1
    alternatives: List[str]  # Alternative labels
    
@dataclass
class WavePrediction:
    """Prediction for next wave move"""
    current_wave: str  # e.g., "Wave iii in progress"
    next_wave: str  # e.g., "Wave iv correction expected"
    target_range: Tuple[float, float]  # (min_target, max_target)
    direction: str  # "up" or "down"
    confidence: float
    timeframe_estimate: str  # e.g., "2-5 days"
    key_levels: Dict[str, float]  # Important levels to watch
    invalidation_level: float  # Level that invalidates this count

class ElliottWaveRealtimeLabeler(ElliottWaveAnalyzer):
    """
    Enhanced Elliott Wave Analyzer for real-time wave labeling and prediction
    
    Key Features:
    - Labels incomplete wave patterns
    - Identifies current wave in progress
    - Predicts next wave targets
    - Provides alternative counts
    - Configurable pattern recognition parameters
    """
    
    def __init__(self, 
                 price_data: pd.Series,
                 min_swing_percentage: float = 2.0,
                 degree: str = 'Intermediate',
                 wave_tolerance: float = 0.15,  # 15% tolerance for wave relationships
                 allow_extensions: bool = True,  # Allow wave extensions
                 max_wave_multiple: float = 3.0):  # Max ratio between waves
        """
        Initialize Real-time Wave Labeler
        
        Args:
            price_data: Price series
            min_swing_percentage: Minimum % move for swing detection
            degree: Wave degree (Primary, Intermediate, Minor, Minute)
            wave_tolerance: Tolerance for ideal wave relationships (0-1)
            allow_extensions: Allow extended waves (1.618x normal)
            max_wave_multiple: Maximum multiple between wave sizes
        """
        super().__init__(price_data, min_swing_percentage, degree)
        
        self.wave_tolerance = wave_tolerance
        self.allow_extensions = allow_extensions
        self.max_wave_multiple = max_wave_multiple
        
        # Wave relationship guidelines (from Elliott Wave theory)
        self.wave_relationships = {
            'wave2_retracement': [0.382, 0.5, 0.618, 0.786],  # Common wave 2 retracements
            'wave3_extension': [1.0, 1.618, 2.618],  # Wave 3 typically extends
            'wave4_retracement': [0.236, 0.382, 0.5],  # Wave 4 usually shallow
            'wave5_target': [0.618, 1.0, 1.618],  # Wave 5 relative to wave 1
            'waveb_retracement': [0.382, 0.5, 0.618, 0.786],  # B wave retracement
            'wavec_target': [1.0, 1.618]  # C wave typically equals or extends A
        }
        
        # Store labeled waves
        self.labeled_waves: List[WaveLabel] = []
        self.current_scenario: WaveScenario = WaveScenario.UNCLEAR
        self.alternative_scenarios: List[Dict] = []
    
    def label_current_waves(self) -> Dict:
        """
        Label all detected swing points with Elliott Wave notation
        Handles incomplete patterns and provides confidence scores
        
        Returns:
            Dictionary with labeled waves, current position, and predictions
        """
        if not self.wave_points:
            self.detect_swing_points()
        
        if len(self.wave_points) < 2:
            return {
                'status': 'Insufficient data',
                'labeled_waves': [],
                'current_position': 'Unknown',
                'prediction': None
            }
        
        # Try different pattern interpretations
        scenarios = []
        
        # Try impulse wave interpretation
        impulse_result = self._try_impulse_labeling()
        if impulse_result['confidence'] > 0:
            scenarios.append(impulse_result)
        
        # Try corrective wave interpretation
        corrective_result = self._try_corrective_labeling()
        if corrective_result['confidence'] > 0:
            scenarios.append(corrective_result)
        
        # Try complex/combination patterns
        complex_result = self._try_complex_labeling()
        if complex_result['confidence'] > 0:
            scenarios.append(complex_result)
        
        # Select best scenario
        if not scenarios:
            return {
                'status': 'No clear pattern',
                'labeled_waves': [],
                'current_position': 'Pattern unclear',
                'prediction': None
            }
        
        best_scenario = max(scenarios, key=lambda x: x['confidence'])
        self.labeled_waves = best_scenario['labels']
        self.current_scenario = best_scenario['scenario']
        self.alternative_scenarios = [s for s in scenarios if s != best_scenario]
        
        # Generate prediction based on best scenario
        prediction = self._generate_prediction(best_scenario)
        
        # Create comprehensive analysis
        return {
            'status': 'Pattern identified',
            'scenario': best_scenario['scenario'].value,
            'confidence': best_scenario['confidence'],
            'labeled_waves': self._format_wave_labels(best_scenario['labels']),
            'current_position': best_scenario['current_position'],
            'prediction': prediction,
            'alternative_counts': self._format_alternatives(),
            'key_observations': self._generate_observations(best_scenario)
        }
    
    def _try_impulse_labeling(self) -> Dict:
        """Try to label waves as an impulse pattern (incomplete or complete)"""
        points = self.wave_points[-9:]  # Look at last 9 points max (for extended waves)
        
        if len(points) < 2:
            return {'confidence': 0}
        
        # Determine direction
        is_upward = points[-1].price > points[0].price
        
        labels = []
        confidence = 0.5  # Base confidence
        
        # Try to fit impulse pattern
        if len(points) >= 2:
            # We have at least wave i
            labels.append(WaveLabel(
                point=points[0],
                label='(i)' if points[0].point_type == 'low' else 'i',
                confidence=0.8,
                alternatives=['1', 'a']
            ))
            
            labels.append(WaveLabel(
                point=points[1],
                label='(ii)' if points[1].point_type == 'high' else 'ii',
                confidence=0.7,
                alternatives=['2', 'b']
            ))
        
        if len(points) >= 3:
            # Check wave 2 retracement
            wave1_size = abs(points[1].price - points[0].price)
            wave2_size = abs(points[2].price - points[1].price)
            retracement = wave2_size / wave1_size if wave1_size > 0 else 0
            
            # Check if retracement is reasonable
            if 0.236 <= retracement <= 0.99:
                confidence += 0.1
                labels.append(WaveLabel(
                    point=points[2],
                    label='(iii)' if len(points) == 3 else 'iii',
                    confidence=0.7 if len(points) == 3 else 0.8,
                    alternatives=['3', 'c']
                ))
            else:
                confidence -= 0.2
        
        if len(points) >= 4:
            # Check wave 3 - should not be shortest
            wave3_size = abs(points[3].price - points[2].price)
            if wave3_size >= wave1_size * 0.618:  # Wave 3 reasonable
                confidence += 0.1
                labels.append(WaveLabel(
                    point=points[3],
                    label='(iv)' if len(points) == 4 else 'iv',
                    confidence=0.7,
                    alternatives=['4', 'a']
                ))
        
        if len(points) >= 5:
            # Check wave 4 - shouldn't overlap wave 1
            if is_upward:
                if points[4].price > points[1].price:  # No overlap
                    confidence += 0.15
                    labels.append(WaveLabel(
                        point=points[4],
                        label='(v)' if len(points) == 5 else 'v',
                        confidence=0.8,
                        alternatives=['5', 'c']
                    ))
                else:
                    confidence -= 0.3  # Overlap violation
            else:
                if points[4].price < points[1].price:  # No overlap
                    confidence += 0.15
                    labels.append(WaveLabel(
                        point=points[4],
                        label='(v)' if len(points) == 5 else 'v',
                        confidence=0.8,
                        alternatives=['5', 'c']
                    ))
        
        # Handle additional points as potential extensions or new cycle
        for i, point in enumerate(points[5:], 5):
            if i <= 7:  # Could be extended wave 5
                ext_num = i - 4
                labels.append(WaveLabel(
                    point=point,
                    label=f'v{ext_num}' if ext_num == 1 else f'v.{ext_num}',  # v1, v.2, v.3 (cleaner)
                    confidence=0.5,
                    alternatives=['a', 'i']
                ))
            else:
                labels.append(WaveLabel(
                    point=point,
                    label='?',
                    confidence=0.3,
                    alternatives=['new cycle']
                ))
        
        # Determine current position
        current_position = self._determine_current_position_impulse(labels, is_upward)
        
        return {
            'scenario': WaveScenario.IMPULSE_UP if is_upward else WaveScenario.IMPULSE_DOWN,
            'labels': labels,
            'confidence': min(1.0, max(0.0, confidence)),
            'current_position': current_position,
            'direction': 'up' if is_upward else 'down'
        }
    
    def _try_corrective_labeling(self) -> Dict:
        """Try to label waves as a corrective pattern (ABC)"""
        points = self.wave_points[-7:]  # Corrective patterns need fewer points
        
        if len(points) < 2:
            return {'confidence': 0}
        
        is_upward = points[-1].price > points[0].price
        labels = []
        confidence = 0.4  # Lower base confidence for corrections
        
        # Try ABC pattern
        if len(points) >= 2:
            labels.append(WaveLabel(
                point=points[0],
                label='a',
                confidence=0.7,
                alternatives=['w', '1']
            ))
            
            labels.append(WaveLabel(
                point=points[1],
                label='b',
                confidence=0.6,
                alternatives=['x', '2']
            ))
        
        if len(points) >= 3:
            # Check B retracement
            waveA_size = abs(points[1].price - points[0].price)
            waveB_size = abs(points[2].price - points[1].price)
            retracement = waveB_size / waveA_size if waveA_size > 0 else 0
            
            if 0.382 <= retracement <= 0.886:
                confidence += 0.2
                labels.append(WaveLabel(
                    point=points[2],
                    label='c' if len(points) == 3 else 'c',
                    confidence=0.7,
                    alternatives=['y', '3']
                ))
        
        # Handle additional points as complex correction
        for i, point in enumerate(points[3:], 3):
            if i == 3:
                labels.append(WaveLabel(
                    point=point,
                    label='x',  # Connector wave
                    confidence=0.5,
                    alternatives=['a']
                ))
            elif i == 4:
                labels.append(WaveLabel(
                    point=point,
                    label='a2',  # Second ABC
                    confidence=0.5,
                    alternatives=['w']
                ))
            else:
                labels.append(WaveLabel(
                    point=point,
                    label='?',
                    confidence=0.3,
                    alternatives=[]
                ))
        
        current_position = self._determine_current_position_corrective(labels)
        
        return {
            'scenario': WaveScenario.CORRECTION_UP if is_upward else WaveScenario.CORRECTION_DOWN,
            'labels': labels,
            'confidence': min(1.0, max(0.0, confidence)),
            'current_position': current_position,
            'direction': 'up' if is_upward else 'down'
        }
    
    def _try_complex_labeling(self) -> Dict:
        """Try to identify complex patterns (double three, triple three, etc.)"""
        # Simplified for now - could be expanded
        points = self.wave_points[-10:]
        
        if len(points) < 7:
            return {'confidence': 0}
        
        # Look for W-X-Y pattern (double three)
        confidence = 0.3
        labels = []
        
        for i, point in enumerate(points):
            if i % 3 == 0:
                labels.append(WaveLabel(
                    point=point,
                    label='w' if i == 0 else 'y' if i == 6 else 'x',
                    confidence=0.5,
                    alternatives=['a', str(i+1)]
                ))
            else:
                labels.append(WaveLabel(
                    point=point,
                    label='?',
                    confidence=0.3,
                    alternatives=[]
                ))
        
        return {
            'scenario': WaveScenario.UNCLEAR,
            'labels': labels,
            'confidence': confidence,
            'current_position': 'Complex correction in progress',
            'direction': 'sideways'
        }
    
    def _determine_current_position_impulse(self, labels: List[WaveLabel], is_upward: bool) -> str:
        """Determine current position in impulse wave"""
        if not labels:
            return "No waves identified"
        
        last_label = labels[-1].label
        current_price = self.price_data.iloc[-1]
        last_wave_price = labels[-1].point.price
        
        # Clean up label to get wave number
        wave_num = last_label.replace('(', '').replace(')', '').replace('?', '')
        
        direction = "up" if is_upward else "down"
        
        if 'v' in wave_num or '5' in wave_num:
            if (is_upward and current_price > last_wave_price) or \
               (not is_upward and current_price < last_wave_price):
                if 'v.' in wave_num or 'ext' in wave_num:
                    return f"Wave 5 extending {direction} (strong momentum)"
                else:
                    return f"Wave 5 extending {direction}"
            else:
                return f"Wave 5 complete, ABC correction expected"
        elif 'iv' in wave_num or '4' in wave_num:
            return f"Wave iv correction in progress"
        elif 'iii' in wave_num or '3' in wave_num:
            if (is_upward and current_price > last_wave_price) or \
               (not is_upward and current_price < last_wave_price):
                return f"Wave iii extending {direction} (strongest move)"
            else:
                return f"Wave iii possibly complete"
        elif 'ii' in wave_num or '2' in wave_num:
            return f"Wave ii correction in progress"
        elif 'i' in wave_num or '1' in wave_num:
            return f"Wave i in progress"
        else:
            return "Pattern developing"
    
    def _determine_current_position_corrective(self, labels: List[WaveLabel]) -> str:
        """Determine current position in corrective wave"""
        if not labels:
            return "No waves identified"
        
        last_label = labels[-1].label
        
        if 'c' in last_label:
            return "Wave C in progress (correction ending soon)"
        elif 'b' in last_label:
            return "Wave B retracement in progress"
        elif 'a' in last_label:
            return "Wave A of correction in progress"
        else:
            return "Complex correction developing"
    
    def _generate_prediction(self, scenario: Dict) -> WavePrediction:
        """Generate prediction for next wave move"""
        labels = scenario['labels']
        if not labels:
            return None
        
        current_price = self.price_data.iloc[-1]
        last_wave = labels[-1]
        
        # Calculate prediction based on wave position
        if 'impulse' in scenario['scenario'].value.lower():
            return self._predict_impulse_next_wave(labels, scenario['direction'] == 'up')
        else:
            return self._predict_corrective_next_wave(labels, scenario['direction'] == 'up')
    
    def _predict_impulse_next_wave(self, labels: List[WaveLabel], is_upward: bool) -> WavePrediction:
        """Predict next move in impulse pattern"""
        current_price = self.price_data.iloc[-1]
        
        if not labels:
            return None
        
        last_label = labels[-1].label.replace('(', '').replace(')', '').replace('?', '')
        
        # Get wave sizes for projections
        wave_sizes = []
        for i in range(len(labels) - 1):
            wave_sizes.append(abs(labels[i+1].point.price - labels[i].point.price))
        
        if 'v' in last_label or '5' in last_label:
            # Wave 5 complete or in progress - expect correction
            wave1_size = wave_sizes[0] if wave_sizes else current_price * 0.05
            
            # Typical correction targets
            min_correction = current_price - wave1_size * 0.382 if is_upward else current_price + wave1_size * 0.382
            max_correction = current_price - wave1_size * 0.618 if is_upward else current_price + wave1_size * 0.618
            
            return WavePrediction(
                current_wave="Wave v in progress" if len(labels) < 5 else "Wave v complete",
                next_wave="ABC correction expected",
                target_range=(min(min_correction, max_correction), max(min_correction, max_correction)),
                direction="down" if is_upward else "up",
                confidence=0.7,
                timeframe_estimate=f"{len(wave_sizes)*2}-{len(wave_sizes)*4} periods",
                key_levels={
                    '0.236 retracement': current_price - wave1_size * 0.236 if is_upward else current_price + wave1_size * 0.236,
                    '0.382 retracement': current_price - wave1_size * 0.382 if is_upward else current_price + wave1_size * 0.382,
                    '0.618 retracement': current_price - wave1_size * 0.618 if is_upward else current_price + wave1_size * 0.618
                },
                invalidation_level=labels[0].point.price  # Wave 1 low/high
            )
        
        elif 'iv' in last_label or '4' in last_label:
            # Wave 4 complete - expect wave 5
            if len(wave_sizes) >= 3:
                wave1_size = wave_sizes[0]
                wave3_size = wave_sizes[2]
                
                # Wave 5 typically equals wave 1 or 0.618 * wave 3
                min_target = current_price + wave1_size * 0.618 if is_upward else current_price - wave1_size * 0.618
                max_target = current_price + max(wave1_size, wave3_size * 0.618) if is_upward else current_price - max(wave1_size, wave3_size * 0.618)
            else:
                min_target = current_price * 1.02 if is_upward else current_price * 0.98
                max_target = current_price * 1.05 if is_upward else current_price * 0.95
            
            return WavePrediction(
                current_wave="Wave iv correction",
                next_wave="Wave v impulse expected",
                target_range=(min(min_target, max_target), max(min_target, max_target)),
                direction="up" if is_upward else "down",
                confidence=0.75,
                timeframe_estimate=f"{len(wave_sizes)}-{len(wave_sizes)*2} periods",
                key_levels={
                    'Wave 1 target': current_price + wave_sizes[0] if is_upward else current_price - wave_sizes[0],
                    '0.618 extension': current_price + wave_sizes[0] * 0.618 if is_upward else current_price - wave_sizes[0] * 0.618
                },
                invalidation_level=labels[1].point.price if len(labels) > 1 else current_price * 0.95
            )
        
        elif 'iii' in last_label or '3' in last_label:
            # Wave 3 in progress or complete - expect wave 4
            if len(wave_sizes) >= 2:
                wave3_size = wave_sizes[-1]
                
                # Wave 4 typically retraces 0.236-0.5 of wave 3
                min_retrace = current_price - wave3_size * 0.236 if is_upward else current_price + wave3_size * 0.236
                max_retrace = current_price - wave3_size * 0.5 if is_upward else current_price + wave3_size * 0.5
            else:
                min_retrace = current_price * 0.97 if is_upward else current_price * 1.03
                max_retrace = current_price * 0.95 if is_upward else current_price * 1.05
            
            return WavePrediction(
                current_wave="Wave iii impulse (strongest move)",
                next_wave="Wave iv correction expected",
                target_range=(min(min_retrace, max_retrace), max(min_retrace, max_retrace)),
                direction="down" if is_upward else "up",
                confidence=0.8,
                timeframe_estimate=f"{len(wave_sizes)//2}-{len(wave_sizes)} periods",
                key_levels={
                    '0.236 retracement': min_retrace,
                    '0.382 retracement': current_price - wave3_size * 0.382 if is_upward else current_price + wave3_size * 0.382,
                    '0.5 retracement': max_retrace
                },
                invalidation_level=labels[1].point.price if len(labels) > 1 else current_price
            )
        
        else:
            # Early stages
            return WavePrediction(
                current_wave="Early impulse development",
                next_wave="Pattern developing",
                target_range=(current_price * 0.95, current_price * 1.05),
                direction="unclear",
                confidence=0.4,
                timeframe_estimate="Uncertain",
                key_levels={},
                invalidation_level=labels[0].point.price if labels else current_price
            )
    
    def _predict_corrective_next_wave(self, labels: List[WaveLabel], is_upward: bool) -> WavePrediction:
        """Predict next move in corrective pattern"""
        current_price = self.price_data.iloc[-1]
        
        if not labels:
            return None
        
        last_label = labels[-1].label
        
        if 'c' in last_label:
            # Wave C ending - expect new impulse
            return WavePrediction(
                current_wave="Wave C of correction",
                next_wave="New impulse wave expected",
                target_range=(current_price * 0.95, current_price * 1.1),
                direction="up" if not is_upward else "down",  # Opposite of correction
                confidence=0.65,
                timeframe_estimate="Starting soon",
                key_levels={
                    'Correction end': current_price
                },
                invalidation_level=labels[0].point.price if labels else current_price
            )
        
        elif 'b' in last_label:
            # Wave B - expect wave C
            if len(labels) >= 2:
                waveA_size = abs(labels[1].point.price - labels[0].point.price)
                # Wave C often equals wave A
                target = current_price + waveA_size if is_upward else current_price - waveA_size
                
                return WavePrediction(
                    current_wave="Wave B retracement",
                    next_wave="Wave C expected (final leg)",
                    target_range=(target * 0.95, target * 1.05),
                    direction="up" if is_upward else "down",
                    confidence=0.7,
                    timeframe_estimate=f"{len(labels)*2} periods",
                    key_levels={
                        'Wave A equality': target,
                        '1.618 extension': current_price + waveA_size * 1.618 if is_upward else current_price - waveA_size * 1.618
                    },
                    invalidation_level=labels[0].point.price
                )
        
        return WavePrediction(
            current_wave="Correction developing",
            next_wave="Pattern unclear",
            target_range=(current_price * 0.95, current_price * 1.05),
            direction="unclear",
            confidence=0.3,
            timeframe_estimate="Uncertain",
            key_levels={},
            invalidation_level=current_price * 0.9
        )
    
    def _format_wave_labels(self, labels: List[WaveLabel]) -> List[Dict]:
        """Format wave labels for output"""
        formatted = []
        for label in labels:
            formatted.append({
                'date': label.point.date,
                'price': label.point.price,
                'type': label.point.point_type,
                'label': label.label,
                'confidence': f"{label.confidence:.0%}",
                'alternatives': label.alternatives
            })
        return formatted
    
    def _format_alternatives(self) -> List[Dict]:
        """Format alternative wave counts"""
        alternatives = []
        for scenario in self.alternative_scenarios[:2]:  # Top 2 alternatives
            alternatives.append({
                'scenario': scenario['scenario'].value,
                'confidence': f"{scenario['confidence']:.0%}",
                'position': scenario['current_position']
            })
        return alternatives
    
    def _generate_observations(self, scenario: Dict) -> List[str]:
        """Generate key observations about the wave structure"""
        observations = []
        labels = scenario['labels']
        
        if len(labels) >= 3:
            # Check for extensions
            wave_sizes = []
            for i in range(len(labels) - 1):
                wave_sizes.append(abs(labels[i+1].point.price - labels[i].point.price))
            
            if wave_sizes:
                largest_wave_idx = wave_sizes.index(max(wave_sizes))
                if largest_wave_idx == 2:
                    observations.append("Wave 3 appears extended (typical and bullish)")
                elif largest_wave_idx == 4:
                    observations.append("Wave 5 appears extended (potential exhaustion)")
        
        # Check current momentum
        if len(self.price_data) >= 20:
            recent_momentum = (self.price_data.iloc[-1] / self.price_data.iloc[-20] - 1) * 100
            if abs(recent_momentum) > 10:
                observations.append(f"Strong momentum: {recent_momentum:.1f}% over 20 periods")
            elif abs(recent_momentum) < 2:
                observations.append("Low momentum - potential consolidation")
        
        # Check for pattern clarity
        if scenario['confidence'] > 0.7:
            observations.append("High confidence pattern - clear wave structure")
        elif scenario['confidence'] < 0.5:
            observations.append("Low confidence - consider alternative counts")
        
        return observations
    
    def get_trading_signals(self) -> Dict:
        """
        Generate specific trading signals based on wave position
        
        Returns:
            Trading signals with entry, stop loss, and take profit levels
        """
        analysis = self.label_current_waves()
        
        if not analysis.get('prediction'):
            return {
                'signal': 'NEUTRAL',
                'reason': 'No clear wave pattern',
                'confidence': 0
            }
        
        prediction = analysis['prediction']
        current_price = self.price_data.iloc[-1]
        
        signals = {
            'signal': 'NEUTRAL',
            'reason': '',
            'entry_zone': None,
            'stop_loss': None,
            'take_profit': [],
            'risk_reward': None,
            'confidence': prediction.confidence
        }
        
        # Generate signals based on wave position
        if 'Wave iii' in prediction.current_wave and 'impulse' in analysis.get('scenario', '').lower():
            # Wave 3 is the strongest - ride the trend
            if prediction.direction == 'up':
                signals['signal'] = 'BUY'
                signals['reason'] = 'Wave iii impulse up - strongest move'
                signals['entry_zone'] = (current_price * 0.99, current_price * 1.01)
                signals['stop_loss'] = prediction.invalidation_level * 0.98
                signals['take_profit'] = [
                    current_price * 1.05,  # Conservative
                    current_price * 1.10,  # Target
                    current_price * 1.15   # Extended
                ]
            else:
                signals['signal'] = 'SELL'
                signals['reason'] = 'Wave iii impulse down - strongest move'
                signals['entry_zone'] = (current_price * 0.99, current_price * 1.01)
                signals['stop_loss'] = prediction.invalidation_level * 1.02
                signals['take_profit'] = [
                    current_price * 0.95,
                    current_price * 0.90,
                    current_price * 0.85
                ]
        
        elif 'Wave v' in prediction.current_wave and 'complete' in prediction.current_wave:
            # Wave 5 complete - counter-trend opportunity
            if prediction.direction == 'down':  # Expecting correction down
                signals['signal'] = 'SELL'
                signals['reason'] = 'Wave v complete - correction starting'
                signals['entry_zone'] = (current_price * 0.995, current_price * 1.005)
                signals['stop_loss'] = current_price * 1.02
                signals['take_profit'] = [
                    prediction.target_range[0],
                    prediction.target_range[1]
                ]
            else:
                signals['signal'] = 'BUY'
                signals['reason'] = 'Wave v complete - correction starting'
                signals['entry_zone'] = (current_price * 0.995, current_price * 1.005)
                signals['stop_loss'] = current_price * 0.98
                signals['take_profit'] = [
                    prediction.target_range[0],
                    prediction.target_range[1]
                ]
        
        elif 'Wave iv' in prediction.current_wave:
            # Wave 4 correction - prepare for wave 5
            signals['signal'] = 'WAIT'
            signals['reason'] = 'Wave iv correction - wait for completion'
            signals['entry_zone'] = prediction.target_range
        
        elif 'Wave C' in prediction.current_wave and 'ending' in prediction.next_wave:
            # Correction ending - major opportunity
            if 'down' in analysis.get('scenario', '').lower():
                signals['signal'] = 'BUY_PREPARE'
                signals['reason'] = 'Wave C ending - major bottom forming'
            else:
                signals['signal'] = 'SELL_PREPARE'
                signals['reason'] = 'Wave C ending - major top forming'
        
        # Calculate risk-reward if we have entry and targets
        if signals['stop_loss'] and signals['take_profit']:
            risk = abs(current_price - signals['stop_loss'])
            reward = abs(signals['take_profit'][0] - current_price)
            signals['risk_reward'] = reward / risk if risk > 0 else 0
        
        return signals
    
    def plot_wave_labels(self) -> Dict:
        """
        Generate data for plotting wave labels on a chart
        
        Returns:
            Dictionary with plotting data
        """
        analysis = self.label_current_waves()
        
        if not analysis.get('labeled_waves'):
            return {'error': 'No waves to plot'}
        
        plot_data = {
            'price_data': self.price_data.to_dict(),
            'wave_points': [],
            'annotations': [],
            'support_resistance': [],
            'prediction_zone': None
        }
        
        # Add wave points with labels
        for wave in analysis['labeled_waves']:
            plot_data['wave_points'].append({
                'x': wave['date'],
                'y': wave['price'],
                'label': wave['label'],
                'confidence': wave['confidence'],
                'type': wave['type']
            })
            
            # Add annotation
            plot_data['annotations'].append({
                'x': wave['date'],
                'y': wave['price'],
                'text': wave['label'],
                'showarrow': True,
                'arrowhead': 2,
                'arrowsize': 1,
                'arrowwidth': 2,
                'arrowcolor': 'blue' if wave['type'] == 'high' else 'red',
                'ax': 0,
                'ay': -30 if wave['type'] == 'high' else 30
            })
        
        # Add prediction zone if available
        if analysis.get('prediction'):
            pred = analysis['prediction']
            plot_data['prediction_zone'] = {
                'y0': pred.target_range[0],
                'y1': pred.target_range[1],
                'fillcolor': 'green' if pred.direction == 'up' else 'red',
                'opacity': 0.2,
                'label': pred.next_wave
            }
            
            # Add key levels
            for level_name, level_price in pred.key_levels.items():
                plot_data['support_resistance'].append({
                    'y': level_price,
                    'name': level_name,
                    'color': 'orange',
                    'dash': 'dash'
                })
        
        return plot_data


def demonstrate_labeling():
    """Demonstrate the real-time wave labeling with sample data"""
    
    # Create sample price data with clear wave pattern
    import datetime as dt
    
    dates = pd.date_range(end=dt.datetime.now(), periods=50, freq='D')
    
    # Create a partial impulse wave pattern (we're in wave 3)
    prices = []
    base_price = 100
    
    # Wave 1 up
    for i in range(10):
        prices.append(base_price + i * 1.5)
    
    # Wave 2 down (38.2% retracement)
    wave1_top = prices[-1]
    wave1_size = wave1_top - base_price
    for i in range(7):
        prices.append(wave1_top - (wave1_size * 0.382 * i / 6))
    
    # Wave 3 up (in progress - 1.618x of wave 1)
    wave2_bottom = prices[-1]
    for i in range(15):
        prices.append(wave2_bottom + (wave1_size * 1.618 * i / 20))  # Incomplete!
    
    # Add remaining dates with last price (current price)
    while len(prices) < 50:
        prices.append(prices[-1] + np.random.uniform(-0.5, 0.5))
    
    price_series = pd.Series(prices[:50], index=dates)
    
    # Create labeler
    labeler = ElliottWaveRealtimeLabeler(
        price_series,
        min_swing_percentage=2.0,
        wave_tolerance=0.15
    )
    
    # Get analysis
    print("=" * 80)
    print("ELLIOTT WAVE REAL-TIME ANALYSIS")
    print("=" * 80)
    
    analysis = labeler.label_current_waves()
    
    print(f"\nScenario: {analysis.get('scenario', 'Unknown')}")
    print(f"Confidence: {analysis.get('confidence', 0):.1%}")
    print(f"Current Position: {analysis.get('current_position', 'Unknown')}")
    
    print("\n" + "=" * 50)
    print("WAVE LABELS")
    print("=" * 50)
    
    for wave in analysis.get('labeled_waves', []):
        alt_text = f" (alt: {', '.join(wave['alternatives'])})" if wave['alternatives'] else ""
        print(f"{wave['date'].strftime('%Y-%m-%d')}: {wave['label']} @ ${wave['price']:.2f} "
              f"[{wave['confidence']}]{alt_text}")
    
    if analysis.get('prediction'):
        pred = analysis['prediction']
        print("\n" + "=" * 50)
        print("PREDICTION")
        print("=" * 50)
        print(f"Current Wave: {pred.current_wave}")
        print(f"Next Wave: {pred.next_wave}")
        print(f"Direction: {pred.direction}")
        print(f"Target Range: ${pred.target_range[0]:.2f} - ${pred.target_range[1]:.2f}")
        print(f"Confidence: {pred.confidence:.1%}")
        print(f"Timeframe: {pred.timeframe_estimate}")
        print(f"Invalidation: ${pred.invalidation_level:.2f}")
        
        if pred.key_levels:
            print("\nKey Levels:")
            for name, level in pred.key_levels.items():
                print(f"  {name}: ${level:.2f}")
    
    # Get trading signals
    print("\n" + "=" * 50)
    print("TRADING SIGNALS")
    print("=" * 50)
    
    signals = labeler.get_trading_signals()
    print(f"Signal: {signals['signal']}")
    print(f"Reason: {signals['reason']}")
    if signals.get('entry_zone'):
        print(f"Entry Zone: ${signals['entry_zone'][0]:.2f} - ${signals['entry_zone'][1]:.2f}")
    if signals.get('stop_loss'):
        print(f"Stop Loss: ${signals['stop_loss']:.2f}")
    if signals.get('take_profit'):
        print(f"Take Profit Targets: {', '.join([f'${tp:.2f}' for tp in signals['take_profit']])}")
    if signals.get('risk_reward'):
        print(f"Risk/Reward Ratio: 1:{signals['risk_reward']:.2f}")
    
    # Show alternative counts
    if analysis.get('alternative_counts'):
        print("\n" + "=" * 50)
        print("ALTERNATIVE COUNTS")
        print("=" * 50)
        for alt in analysis['alternative_counts']:
            print(f"• {alt['scenario']} ({alt['confidence']}) - {alt['position']}")
    
    # Show observations
    if analysis.get('key_observations'):
        print("\n" + "=" * 50)
        print("KEY OBSERVATIONS")
        print("=" * 50)
        for obs in analysis['key_observations']:
            print(f"• {obs}")


if __name__ == "__main__":
    demonstrate_labeling()