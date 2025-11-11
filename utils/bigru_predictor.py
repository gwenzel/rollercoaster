"""
BiGRU model integration for Streamlit app.
This module provides functions to load the trained BiGRU model and predict scores
from custom rollercoaster tracks generated in the app.
"""

import numpy as np
import pandas as pd
import torch
from bigru_score_predictor import CoasterScorePredictor
import os


class StreamlitBiGRUPredictor:
    """Wrapper for BiGRU predictor optimized for Streamlit usage."""
    
    def __init__(self, model_path="models/bigru_score_model.pth"):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to saved model checkpoint
        """
        self.model_path = model_path
        self.predictor = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                "Please train a model first using train_bigru_model.py or "
                "create a dummy model using create_dummy_model.py"
            )
        
        # Create predictor without data paths (we'll only use it for inference)
        self.predictor = CoasterScorePredictor(
            accel_data_dir="accel_data",  # Not used for inference
            ratings_data_path="ratings_data/dummy.csv"  # Not used for inference
        )
        
        # Load the trained model
        self.predictor.load_model(self.model_path)
    
    def predict_from_track(self, track_df: pd.DataFrame) -> float:
        """
        Predict score from track DataFrame.
        
        Args:
            track_df: DataFrame with columns ['x', 'y', 'acceleration']
            
        Returns:
            Predicted score (typically 3.0-5.0 range)
        """
        # Extract acceleration data
        if 'acceleration' not in track_df.columns:
            raise ValueError("Track DataFrame must have 'acceleration' column")
        
        acceleration = track_df['acceleration'].values
        
        # Reshape to (timesteps, features) - BiGRU expects 2D input
        acceleration_data = acceleration.reshape(-1, 1)
        
        # Predict using the model
        try:
            predicted_score = self.predictor.predict(acceleration_data)
            
            # Clip to reasonable range (models can sometimes predict outside training range)
            predicted_score = np.clip(predicted_score, 1.0, 5.0)
            
            return predicted_score
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Return a default value if prediction fails
            return 3.5
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if self.predictor is None or self.predictor.model is None:
            return {"error": "Model not loaded"}
        
        return {
            "sequence_length": self.predictor.seq_length,
            "input_size": self.predictor.input_size,
            "model_path": self.model_path,
            "device": str(self.predictor.device)
        }


# Singleton instance for Streamlit (cache the model loading)
_predictor_instance = None


def get_predictor(model_path="models/bigru_score_model.pth") -> StreamlitBiGRUPredictor:
    """
    Get or create the BiGRU predictor instance.
    Uses singleton pattern to avoid reloading model on every Streamlit rerun.
    
    Args:
        model_path: Path to saved model checkpoint
        
    Returns:
        StreamlitBiGRUPredictor instance
    """
    global _predictor_instance
    
    if _predictor_instance is None:
        _predictor_instance = StreamlitBiGRUPredictor(model_path=model_path)
    
    return _predictor_instance


def predict_score_bigru(track_df: pd.DataFrame, model_path="models/bigru_score_model.pth") -> float:
    """
    Convenience function to predict score from track DataFrame.
    
    Args:
        track_df: DataFrame with columns ['x', 'y', 'acceleration']
        model_path: Path to saved model checkpoint
        
    Returns:
        Predicted score
    """
    predictor = get_predictor(model_path)
    return predictor.predict_from_track(track_df)
