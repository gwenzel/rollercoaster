"""
BiGRU model integration for Streamlit app.
This module provides functions to load the trained BiGRU model and predict scores
from custom rollercoaster tracks generated in the app.

The model expects 3-axis accelerometer data (Lateral, Vertical, Longitudinal)
as recorded by wearables on rollercoaster riders.
"""

import numpy as np
import pandas as pd
import torch
from scripts.bigru_score_predictor import CoasterScorePredictor
from utils.accelerometer_transform import track_to_accelerometer_data
import os


class StreamlitBiGRUPredictor:
    """Wrapper for BiGRU predictor optimized for Streamlit usage."""
    
    def __init__(self, model_path="models/bigru_score_model.pth"):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to saved model checkpoint
        """
        # Always use the provided model path (models folder)
        self.model_path = model_path
        self.predictor = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model from notebook-generated format."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                "Please train a model first using the BiGRU notebook"
            )
        
        import torch
        import torch.nn as nn
        import pickle
        from sklearn.preprocessing import StandardScaler
        
        # Load checkpoint
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
        
        # Define the notebook's BiGRU model architecture (matched to checkpoint)
        class BiGRURegressor(nn.Module):
            def __init__(self, input_size_accel=3, input_size_airtime=4, hidden_size=128, num_layers=1, dropout=0.3):
                super(BiGRURegressor, self).__init__()
                self.gru = nn.GRU(
                    input_size=input_size_accel,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0,
                    bidirectional=True
                )
                self.airtime_head = nn.Sequential(
                    nn.Linear(input_size_airtime, 128),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                combined_size = hidden_size * 2 + 128  # 256 + 128 = 384
                self.head = nn.Sequential(
                    nn.Linear(combined_size, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(128, 1)
                )
            
            def forward(self, x_accel, x_airtime):
                _, h_n = self.gru(x_accel)
                h_forward = h_n[-2, :, :]
                h_backward = h_n[-1, :, :]
                gru_out = torch.cat((h_forward, h_backward), dim=1)
                airtime_out = self.airtime_head(x_airtime)
                combined = torch.cat((gru_out, airtime_out), dim=1)
                output = self.head(combined)
                return output
        
        # Create a minimal predictor object to hold the model
        class MinimalPredictor:
            def __init__(self):
                self.device = device
                self.model = None
                self.scaler_airtime = None
                self.scaler_rating = None
                self.input_size = 3
                self.seq_length = 100
        
        self.predictor = MinimalPredictor()
        
        # Try to load scalers from pickle files (notebook format)
        try:
            with open('models/scaler_airtime.pkl', 'rb') as f:
                self.predictor.scaler_airtime = pickle.load(f)
            with open('models/scaler_ratings.pkl', 'rb') as f:
                self.predictor.scaler_rating = pickle.load(f)
            print("✓ Loaded scalers from models folder (airtime & ratings)")
        except FileNotFoundError:
            # Fallback: try to get scalers from checkpoint
            self.predictor.scaler_airtime = checkpoint.get('scaler_airtime', None)
            self.predictor.scaler_rating = checkpoint.get('scaler_rating', None)
            
            # If still None, create dummy fitted scalers
            if self.predictor.scaler_airtime is None:
                self.predictor.scaler_airtime = StandardScaler()
                # Fit with dummy airtime feature ranges
                dummy_airtime = np.random.randn(1000, 4)
                self.predictor.scaler_airtime.fit(dummy_airtime)
                print("⚠ Created fitted scaler for airtime features (using dummy data)")
            
            if self.predictor.scaler_rating is None:
                self.predictor.scaler_rating = StandardScaler()
                # Fit with typical coaster rating range (3.0-5.0)
                dummy_scores = np.random.uniform(3.0, 5.0, size=(100, 1))
                self.predictor.scaler_rating.fit(dummy_scores)
                print("⚠ Created fitted scaler for ratings (using dummy data)")
        
        # Get model parameters
        self.predictor.input_size = checkpoint.get('input_size', 3)
        self.predictor.seq_length = checkpoint.get('seq_length', 100)
        
        # Create and load model with notebook architecture
        self.predictor.model = BiGRURegressor().to(device)
        # Accept either keyed or raw state_dict formats
        state_dict = checkpoint.get('model_state_dict', None)
        if state_dict is None:
            # If file is a raw state_dict, use it directly
            if all(isinstance(k, str) for k in checkpoint.keys()):
                state_dict = checkpoint
            else:
                raise KeyError(
                    "Checkpoint missing 'model_state_dict' and not a raw state_dict. "
                    "Please save with model_state_dict or provide a compatible file."
                )
        self.predictor.model.load_state_dict(state_dict)
        self.predictor.model.eval()
        
        print(f"✓ Model loaded from {self.model_path}")
        if 'val_loss' in checkpoint:
            print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
    
    def predict_from_track(self, track_df: pd.DataFrame) -> float:
        """
        Predict score from track DataFrame.
        
        Args:
            track_df: DataFrame with columns ['x', 'y'] from build_modular_track()
            
        Returns:
            Predicted score (typically 3.0-5.0 range)
        """
        # Convert track coordinates to accelerometer data (3-axis: Lateral, Vertical, Longitudinal)
        try:
            accel_df = track_to_accelerometer_data(track_df)
            
            # Extract the 3 acceleration axes
            acceleration_data = accel_df[['Lateral', 'Vertical', 'Longitudinal']].values
            
            # Shape: (timesteps, 3) - matches wearable accelerometer format
            print(f"Accelerometer data shape: {acceleration_data.shape}")
            print(f"Acceleration ranges - Lateral: [{acceleration_data[:, 0].min():.2f}, {acceleration_data[:, 0].max():.2f}]g, "
                  f"Vertical: [{acceleration_data[:, 1].min():.2f}, {acceleration_data[:, 1].max():.2f}]g, "
                  f"Longitudinal: [{acceleration_data[:, 2].min():.2f}, {acceleration_data[:, 2].max():.2f}]g")
            
        except Exception as e:
            print(f"Error converting track to accelerometer data: {e}")
            print("Falling back to simple approximation...")
            
            # Fallback: use track length and simple assumptions
            n = len(track_df)
            acceleration_data = np.zeros((n, 3))
            acceleration_data[:, 1] = 1.0  # 1g vertical (gravity)
        
        # Predict using the model
        try:
            import torch
            
            # Acceleration passes through (no accel scaler needed)
            accel_tensor = torch.FloatTensor(acceleration_data).unsqueeze(0).to(self.predictor.device)
            
            # Compute airtime features from Vertical g channel
            v = acceleration_data[:, 1].astype(float)
            airtime_mask = v < 0.2
            airtime_count = int(np.sum(airtime_mask))
            airtime_ratio = float(np.mean(airtime_mask)) if len(v) > 0 else 0.0
            min_vertical_g = float(np.min(v)) if len(v) > 0 else 1.0
            transitions = np.diff(airtime_mask.astype(int)) if len(v) > 1 else np.array([])
            segments = int(np.sum(transitions == 1)) if transitions.size > 0 else 0
            airtime_features = np.array([[airtime_count, airtime_ratio, min_vertical_g, segments]], dtype=float)
            # Normalize airtime features if scaler available
            if self.predictor.scaler_airtime is not None:
                airtime_features = self.predictor.scaler_airtime.transform(airtime_features)
            airtime_tensor = torch.FloatTensor(airtime_features).to(self.predictor.device)
            
            # Run prediction
            with torch.no_grad():
                prediction_normalized = self.predictor.model(accel_tensor, airtime_tensor)
            
            # Denormalize prediction if rating scaler available; else use raw
            pred_np = prediction_normalized.cpu().numpy().reshape(-1, 1)
            if self.predictor.scaler_score is not None:
                prediction = self.predictor.scaler_score.inverse_transform(pred_np)[0, 0]
            else:
                prediction = float(pred_np[0, 0])
            
            # Clip to reasonable range (models can sometimes predict outside training range)
            predicted_score = np.clip(prediction, 1.0, 5.0)
            print(f"Predicted (denorm, clipped): {predicted_score:.3f}")
            
            return float(predicted_score)
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
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
