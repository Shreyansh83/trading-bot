import json
import os
from typing import Dict, Optional


class WeightManager:
    """
    Manages indicator weights for different symbols and strategies
    """
    
    def __init__(self, weights_file: str = "config/default_weights.json"):
        self.weights_file = weights_file
        self.default_weights = {
            'rsi': 0.15,
            'macd': 0.20,
            'bb': 0.15,
            'ema': 0.20,
            'sar': 0.15,
            'supertrend': 0.15,
            'buy_threshold': 0.3,
            'sell_threshold': -0.3
        }
        self.weights_cache = {}
        self._load_weights()
    
    def _load_weights(self):
        """Load weights from file or create default"""
        try:
            if os.path.exists(self.weights_file):
                with open(self.weights_file, 'r') as f:
                    self.weights_cache = json.load(f)
            else:
                self._save_default_weights()
        except Exception as e:
            print(f"Error loading weights: {e}, using defaults")
            self.weights_cache = {"default": self.default_weights}
    
    def _save_default_weights(self):
        """Save default weights to file"""
        os.makedirs(os.path.dirname(self.weights_file), exist_ok=True)
        self.weights_cache = {"default": self.default_weights}
        with open(self.weights_file, 'w') as f:
            json.dump(self.weights_cache, f, indent=2)
    
    def get_weights(self, symbol: str = "default") -> Dict[str, float]:
        """
        Get weights for a specific symbol
        
        Args:
            symbol: Trading symbol (e.g., "NSE:RELIANCE-EQ")
            
        Returns:
            Dictionary of weights
        """
        # Clean symbol for file storage (remove special characters)
        clean_symbol = symbol.replace(":", "_").replace("-", "_")
        
        if clean_symbol in self.weights_cache:
            return self.weights_cache[clean_symbol]
        elif "default" in self.weights_cache:
            return self.weights_cache["default"]
        else:
            return self.default_weights
    
    def set_weights(self, weights: Dict[str, float], symbol: str = "default"):
        """
        Set weights for a specific symbol
        
        Args:
            weights: Dictionary of weights
            symbol: Trading symbol
        """
        clean_symbol = symbol.replace(":", "_").replace("-", "_")
        self.weights_cache[clean_symbol] = weights
        self._save_weights()
    
    def _save_weights(self):
        """Save current weights to file"""
        try:
            with open(self.weights_file, 'w') as f:
                json.dump(self.weights_cache, f, indent=2)
        except Exception as e:
            print(f"Error saving weights: {e}")
    
    def normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize indicator weights so they sum to 1
        (excludes threshold values)
        
        Args:
            weights: Dictionary of weights
            
        Returns:
            Normalized weights dictionary
        """
        indicator_keys = ['rsi', 'macd', 'bb', 'ema', 'sar', 'supertrend']
        total_weight = sum(weights.get(key, 0) for key in indicator_keys)
        
        if total_weight > 0:
            normalized = weights.copy()
            for key in indicator_keys:
                if key in normalized:
                    normalized[key] = normalized[key] / total_weight
            return normalized
        else:
            return self.default_weights
    
    def update_weights_from_ml(self, ml_weights: Dict[str, float], symbol: str):
        """
        Update weights from ML model output
        
        Args:
            ml_weights: Weights from ML model
            symbol: Trading symbol
        """
        # Ensure ML weights are properly formatted and normalized
        current_weights = self.get_weights(symbol)
        current_weights.update(ml_weights)
        normalized_weights = self.normalize_weights(current_weights)
        self.set_weights(normalized_weights, symbol)