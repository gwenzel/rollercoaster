"""
Create a dummy trained model for demonstration purposes.
This creates a simple model with realistic parameters but random weights.
"""

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from bigru_score_predictor import BiGRUScorePredictor
import os

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Model parameters (matching typical acceleration data)
seq_length = 1000  # Typical track has ~1000 points
input_size = 1     # We have acceleration as single feature
hidden_size = 128
num_layers = 2
dropout = 0.3

# Create model
model = BiGRUScorePredictor(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout
)

# Initialize with reasonable weights (better than random)
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

model.apply(init_weights)

# Create dummy scalers based on typical acceleration data
# Typical acceleration ranges from -5 to 5 m/s^2 for rollercoasters
scaler_accel = StandardScaler()
dummy_accel_data = np.random.randn(seq_length * 10, input_size) * 2  # Mean 0, std 2
scaler_accel.fit(dummy_accel_data)

# Typical coaster ratings range from 3.0 to 5.0
scaler_score = MinMaxScaler(feature_range=(0, 1))
dummy_scores = np.random.uniform(3.0, 5.0, size=(20, 1))
scaler_score.fit(dummy_scores)

# Save the model checkpoint
checkpoint = {
    'epoch': 50,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': None,  # Not needed for inference
    'val_loss': 0.0234,  # Dummy validation loss
    'scaler_accel': scaler_accel,
    'scaler_score': scaler_score,
    'input_size': input_size,
    'seq_length': seq_length,
}

model_path = "models/bigru_score_model.pth"
torch.save(checkpoint, model_path)

print("=" * 70)
print("✓ Dummy model created successfully!")
print("=" * 70)
print(f"Model path: {model_path}")
print(f"Sequence length: {seq_length}")
print(f"Input features: {input_size}")
print(f"Hidden size: {hidden_size}")
print(f"Num layers: {num_layers}")
print(f"Score range: 3.0 - 5.0")
print("=" * 70)
print("\nThis is a DUMMY model for demonstration.")
print("Train a real model using train_bigru_model.py for actual predictions.")
print("=" * 70)
