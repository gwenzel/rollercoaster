# BiGRU Score Prediction for Rollercoaster Designer

## Overview

The Streamlit rollercoaster designer now includes **AI-powered score prediction** using a Bidirectional GRU (BiGRU) neural network trained on real rollercoaster acceleration data and ratings.

## Features

- **Automatic Score Prediction**: As you design your rollercoaster, the BiGRU model automatically predicts a realistic score (1.0-5.0 scale)
- **Real-time Updates**: Score updates instantly when you modify track elements
- **Deep Learning**: Uses a 2-layer BiGRU with attention mechanism to analyze acceleration patterns
- **Dual Predictions**: Compare rule-based thrill score with AI-predicted rating

## Quick Start

### 1. Use the Dummy Model (for testing)

A pre-generated dummy model is already created in `models/bigru_score_model.pth`:

```bash
# Already done - just run Streamlit
streamlit run app.py
```

### 2. Train a Real Model (recommended)

To get accurate predictions based on actual rollercoaster data:

```bash
# Train the BiGRU model on your data
python train_bigru_model.py
```

This will:
- Load acceleration data from `accel_data/`
- Load ratings from `ratings_data/`
- Train a BiGRU model with progress bars
- Save the best model to `models/bigru_score_model.pth`

Training takes 5-10 minutes on CPU for ~20 coasters.

## How It Works

### Architecture

```
Input: Acceleration time series (1000 timesteps x 1 feature)
    ↓
Bidirectional GRU (128 hidden units, 2 layers)
    ↓
Attention Mechanism (focuses on important moments)
    ↓
Fully Connected Layers (256 → 64 → 32 → 1)
    ↓
Output: Predicted score (1.0 - 5.0)
```

### Prediction Pipeline

1. **Track Generation**: Your custom track is built from modular elements
2. **Acceleration Calculation**: Physics engine computes acceleration at each point
3. **Normalization**: Acceleration data is standardized using training statistics
4. **BiGRU Inference**: Neural network analyzes the acceleration pattern
5. **Score Output**: Predicted rating is denormalized to 1.0-5.0 scale

## Model Files

- `models/bigru_score_model.pth`: Trained model checkpoint
  - Contains: model weights, scalers, hyperparameters
  - Size: ~2-3 MB
  - Format: PyTorch `.pth` file

## File Structure

```
rollercoaster/
├── app.py                          # Main Streamlit app (with BiGRU integration)
├── bigru_score_predictor.py        # Core BiGRU model and training logic
├── train_bigru_model.py            # Training script
├── create_dummy_model.py           # Generate dummy model for testing
├── test_bigru_prediction.py        # Test BiGRU prediction
├── models/
│   └── bigru_score_model.pth      # Trained model (dummy or real)
└── utils/
    ├── bigru_predictor.py          # Streamlit integration wrapper
    ├── track.py                    # Track generation
    └── visualize.py                # 3D visualization
```

## Using in Streamlit

The BiGRU prediction is automatically integrated:

1. **Launch app**: `streamlit run app.py`
2. **Design track**: Add/remove elements, adjust parameters
3. **View predictions**: 
   - Left side: 3D track visualization
   - Right side: Rule-based thrill + BiGRU score

### Display

```
🎯 Predictions
┌─────────────────┬──────────────────┐
│ Rule-Based      │ 🧠 BiGRU Score   │
│ Thrill: 7.2/10  │ Score: 4.3/5.0   │
└─────────────────┴──────────────────┘
```

## API Usage

### Predict from DataFrame

```python
from utils.bigru_predictor import predict_score_bigru
import pandas as pd

# Assume track_df has 'acceleration' column
predicted_score = predict_score_bigru(track_df)
print(f"Predicted score: {predicted_score:.2f}/5.0")
```

### Load and Use Predictor Directly

```python
from utils.bigru_predictor import StreamlitBiGRUPredictor
import numpy as np

# Load model
predictor = StreamlitBiGRUPredictor("models/bigru_score_model.pth")

# Get model info
info = predictor.get_model_info()
print(f"Sequence length: {info['sequence_length']}")

# Predict from acceleration array
acceleration_data = np.random.randn(1000, 1)  # (timesteps, features)
score = predictor.predict(acceleration_data)
```

## Training Your Own Model

### Requirements

- At least 15-20 rollercoasters with acceleration data
- Rating data (1.0-5.0 scale)
- Name mapping between datasets (see `coaster_name_mapping.py`)

### Training Process

```bash
python train_bigru_model.py
```

**Progress bars show**:
- Data loading progress
- Epoch progress with live metrics
- Batch-level training/validation progress
- Time per epoch

**Output**:
- `models/bigru_score_model.pth`: Best model checkpoint
- `training_history.png`: Loss and MAE plots
- Console metrics: MSE, MAE, RMSE

### Hyperparameters

Default settings in `train_bigru_model.py`:

```python
hidden_size = 128          # GRU hidden units
num_layers = 2             # Stacked GRU layers
dropout = 0.3              # Dropout rate
learning_rate = 0.001      # Adam optimizer LR
batch_size = 4             # Small batch for small dataset
epochs = 100               # Max epochs (early stopping)
patience = 15              # Early stopping patience
```

## Troubleshooting

### Model Not Found Error

```
FileNotFoundError: Model file not found: models/bigru_score_model.pth
```

**Solution**: Create dummy model or train real model:
```bash
python create_dummy_model.py  # Quick dummy model
# OR
python train_bigru_model.py   # Train on real data
```

### PyTorch Version Issues

If you see `weights_only` errors, ensure you're using PyTorch 2.6+:

```bash
pip install --upgrade torch
```

### Prediction Errors

If predictions fail, the app will:
- Display "BiGRU Error" message
- Fall back to rule-based prediction
- Continue working normally

## Performance

- **Loading**: ~1-2 seconds (first time, then cached)
- **Prediction**: ~50-100ms per track
- **Memory**: ~100MB RAM for model
- **CPU**: Works fine on CPU (no GPU needed)

## Future Improvements

- [ ] Train on larger dataset (100+ coasters)
- [ ] Multi-output prediction (thrill, fear, smoothness)
- [ ] Confidence intervals for predictions
- [ ] Model ensemble for better accuracy
- [ ] Real-time retraining as users rate designs

## Credits

- **BiGRU Architecture**: Bidirectional Gated Recurrent Units with attention
- **Framework**: PyTorch 2.9.0
- **Preprocessing**: scikit-learn StandardScaler and MinMaxScaler
- **Visualization**: Plotly for 3D track rendering
- **Interface**: Streamlit for interactive design

---

**Note**: The dummy model generates random predictions. For accurate scores, train on real rollercoaster data using `train_bigru_model.py`.
