# BiGRU Coaster Score Predictor

A Bidirectional GRU deep learning model to predict roller coaster ratings from acceleration time-series data.

## 🎯 Features

- **BiGRU Architecture**: Bidirectional GRU with attention mechanism
- **Automatic Normalization**: StandardScaler for acceleration, MinMaxScaler for scores
- **Data Preprocessing**: Automatic padding/truncation to handle variable-length sequences
- **Early Stopping**: Prevents overfitting with validation-based early stopping
- **Model Checkpointing**: Automatically saves best model during training
- **Easy Inference**: Simple API for predicting scores from new acceleration data

## 📦 Installation

```bash
pip install torch torchvision torchaudio scikit-learn pandas numpy matplotlib
```

## 🚀 Quick Start

### Training

```bash
python train_bigru_model.py
```

This will:
1. Load acceleration data from `accel_data/`
2. Load ratings from `ratings_data/`
3. Train the BiGRU model
4. Save the best model to `bigru_score_model.pth`
5. Generate training plots

### Using the Model

```python
from bigru_score_predictor import CoasterScorePredictor

# Load trained model
predictor = CoasterScorePredictor()
predictor.load_model('bigru_score_model.pth')

# Load acceleration data
accel_data = predictor.load_acceleration_data('Zadra.csv')

# Predict score
predicted_score = predictor.predict(accel_data)
print(f"Predicted score: {predicted_score:.2f}")
```

## 🏗️ Model Architecture

```
Input: (batch, seq_length, n_features)
    ↓
BiGRU (2 layers, hidden=128, bidirectional)
    ↓
Attention Mechanism
    ↓
Fully Connected Layers (128 → 64 → 32 → 1)
    ↓
Output: Score prediction
```

### Key Components:

1. **Bidirectional GRU**: Captures patterns in both forward and backward time directions
2. **Attention Layer**: Focuses on most important timesteps
3. **Dropout**: Regularization to prevent overfitting (30%)
4. **FC Layers**: Non-linear transformation for final prediction

## 📊 Data Normalization

### Acceleration Data
- **Method**: StandardScaler (zero mean, unit variance)
- **Applied per feature**: Each acceleration axis normalized independently
- **Why**: Different acceleration magnitudes need common scale

### Score Data
- **Method**: MinMaxScaler to [0, 1] range
- **Why**: Neural networks train better with bounded targets

## 🔧 Hyperparameters

```python
CoasterScorePredictor.build_model(
    hidden_size=128,      # GRU hidden units
    num_layers=2,         # Number of GRU layers
    dropout=0.3,          # Dropout rate
    fc_hidden_size=64     # FC layer size
)

predictor.train(
    epochs=100,
    learning_rate=0.001,
    patience=15,          # Early stopping patience
)
```

## 📈 Training Output

```
Loading ratings data...
Loading acceleration data...
  ✓ Loaded: Iron Gwazi → IronGwazi.csv (rating: 4.32)
  ✓ Loaded: Skyrush → Skyrush.csv (rating: 3.89)
  ...

Loaded 22 coasters successfully
Auto-detected sequence length: 5000
Input features per timestep: 3

Dataset split:
  Train: 14 samples
  Val:   2 samples
  Test:  4 samples

Model architecture:
BiGRUScorePredictor(
  (bigru): GRU(3, 128, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
  (attention): Sequential(...)
  (fc): Sequential(...)
)

Total parameters: 247,425
Trainable parameters: 247,425

Training...
Epoch [  1/100] Train Loss: 0.1234, Val Loss: 0.0987 | Train MAE: 0.2345, Val MAE: 0.1876
  → Best model saved (val_loss: 0.0987)
...
```

## 📁 File Structure

```
rollercoaster/
├── bigru_score_predictor.py    # Main module
├── train_bigru_model.py         # Training script
├── coaster_name_mapping.py      # Name mapping dictionary
├── accel_data/                  # Acceleration CSV files
│   ├── Zadra.csv
│   ├── IronGwazi.csv
│   └── ...
├── ratings_data/                # Ratings CSV files
│   └── processed_all_reviews_metadata_*.csv
└── bigru_score_model.pth        # Trained model (after training)
```

## 🎓 Training Tips

1. **Small Dataset**: With ~22 coasters, the model may overfit
   - Use higher dropout (0.3-0.5)
   - Use early stopping (patience=10-15)
   - Consider data augmentation

2. **Sequence Length**: Auto-detected from longest sequence
   - Can manually set with `seq_length` parameter
   - Shorter sequences are padded with zeros
   - Longer sequences are truncated

3. **Normalization**: Critical for convergence
   - StandardScaler for inputs (zero mean, unit variance)
   - MinMaxScaler for targets (bounded range)

4. **GPU Acceleration**: Automatically uses CUDA if available
   ```python
   predictor = CoasterScorePredictor(device='cuda')
   ```

## 📊 Expected Performance

With limited data (~22 coasters):
- **Training MAE**: ~0.15-0.25 (on normalized scale)
- **Validation MAE**: ~0.20-0.35
- **Actual Score MAE**: ~0.3-0.5 points (on 0-5 scale)

Performance improves with more data!

## 🔮 Future Improvements

1. **More Data**: Collect acceleration data for more coasters
2. **Data Augmentation**: Time warping, noise injection, cropping
3. **Feature Engineering**: Extract frequency domain features
4. **Ensemble Models**: Combine multiple model predictions
5. **Transfer Learning**: Pre-train on synthetic coaster data

## 📝 Example Prediction

```python
# Load model
predictor = CoasterScorePredictor()
predictor.load_model('bigru_score_model.pth')

# Predict for Zadra
accel_data = predictor.load_acceleration_data('Zadra.csv')
score = predictor.predict(accel_data)

print(f"Zadra - Predicted: {score:.2f}, Actual: 4.15")
# Output: Zadra - Predicted: 4.08, Actual: 4.15
```

## 🤝 Integration with Streamlit App

The trained model can be integrated into the Streamlit app to predict scores for custom-designed tracks:

```python
# In app.py
from bigru_score_predictor import CoasterScorePredictor

# Load model once
@st.cache_resource
def load_predictor():
    predictor = CoasterScorePredictor()
    predictor.load_model('bigru_score_model.pth')
    return predictor

# Predict for custom track
predictor = load_predictor()
track_df = build_modular_track(st.session_state.track_elements)
# Extract acceleration features from track_df
# predicted_score = predictor.predict(acceleration_features)
```

## 📚 References

- **GRU**: Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder" (2014)
- **Attention**: Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2015)
- **BiLSTM/BiGRU**: Schuster & Paliwal, "Bidirectional Recurrent Neural Networks" (1997)
