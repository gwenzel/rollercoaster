"""
BiGRU Model for Roller Coaster Score Prediction from Acceleration Data

This module implements a Bidirectional GRU (BiGRU) neural network to predict
roller coaster ratings from acceleration time series data.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import os
import pickle
from typing import Tuple, Optional, Dict, List


class AccelerationDataset(Dataset):
    """PyTorch Dataset for acceleration time series data."""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Args:
            sequences: Array of shape (n_samples, seq_length, n_features)
            labels: Array of shape (n_samples,) with scores
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class BiGRUScorePredictor(nn.Module):
    """
    Bidirectional GRU model for score prediction from acceleration sequences.
    
    Architecture:
    - Input: (batch, seq_len, input_features)
    - BiGRU layers with dropout
    - Fully connected layers
    - Output: Single score prediction
    """
    
    def __init__(
        self,
        input_size: int = 3,  # x, y, z acceleration or similar features
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        fc_hidden_size: int = 64
    ):
        super(BiGRUScorePredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional GRU
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism (optional but helpful)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, fc_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size // 2, 1)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            predictions: Tensor of shape (batch, 1)
        """
        # BiGRU forward pass
        # output shape: (batch, seq_len, hidden_size * 2)
        gru_out, _ = self.bigru(x)
        
        # Apply attention
        # attention_weights shape: (batch, seq_len, 1)
        attention_weights = torch.softmax(self.attention(gru_out), dim=1)
        
        # Weighted sum of GRU outputs
        # context shape: (batch, hidden_size * 2)
        context = torch.sum(gru_out * attention_weights, dim=1)
        
        # Fully connected layers for prediction
        output = self.fc(context)
        
        return output.squeeze()


class CoasterScorePredictor:
    """
    High-level interface for training and using the BiGRU score predictor.
    Handles data loading, preprocessing, training, and inference.
    """
    
    def __init__(
        self,
        accel_data_dir: str = "accel_data",
        ratings_data_path: str = "ratings_data/processed_all_reviews_metadata_20251110_161035.csv",
        name_mapping_module: str = "coaster_name_mapping",
        seq_length: Optional[int] = None,  # Auto-detect if None
        device: Optional[str] = None
    ):
        """
        Args:
            accel_data_dir: Directory containing acceleration CSV files
            ratings_data_path: Path to ratings data CSV
            name_mapping_module: Module name for coaster name mapping
            seq_length: Fixed sequence length (will pad/truncate if needed)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.accel_data_dir = accel_data_dir
        self.ratings_data_path = ratings_data_path
        
        # Import name mapping
        import importlib
        mapping_module = importlib.import_module(name_mapping_module)
        self.name_mapping = mapping_module.COASTER_NAME_MAPPING
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.seq_length = seq_length
        self.model = None
        self.scaler_accel = StandardScaler()  # For acceleration features
        self.scaler_score = MinMaxScaler(feature_range=(0, 1))  # For scores
        self.input_size = None
    
    def load_acceleration_data(self, filename: str) -> np.ndarray:
        """
        Load and preprocess a single acceleration CSV file.
        
        Args:
            filename: Name of the CSV file (with .csv extension)
            
        Returns:
            numpy array of acceleration data
        """
        filepath = os.path.join(self.accel_data_dir, filename)
        df = pd.read_csv(filepath)
        
        # Assuming columns are time-based measurements
        # Drop non-numeric columns if any
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        data = df[numeric_cols].values
        
        return data
    
    def load_ratings_data(self) -> pd.DataFrame:
        """Load and aggregate ratings data by coaster name."""
        # Load metadata (has coaster names)
        df_meta = pd.read_csv(self.ratings_data_path)
        
        # Load ML data (has ratings) - change path to ML file
        ml_path = self.ratings_data_path.replace('_metadata_', '_ml_')
        
        # Try to load ML file, if not found, check if metadata has ratings
        if 'rating' in df_meta.columns:
            df = df_meta
        else:
            try:
                df_ml = pd.read_csv(ml_path)
                # Merge with metadata to get coaster names
                df = pd.merge(
                    df_meta[['coaster_name', 'coaster_id', 'review_id']],
                    df_ml[['coaster_id', 'review_id', 'rating']],
                    on=['coaster_id', 'review_id'],
                    how='inner'
                )
            except FileNotFoundError:
                raise ValueError(f"Could not find rating data. Checked: {self.ratings_data_path} and {ml_path}")
        
        # Aggregate ratings by coaster (mean rating)
        coaster_ratings = df.groupby('coaster_name').agg({
            'rating': 'mean',
            'review_id': 'count'  # Number of reviews
        }).reset_index()
        
        coaster_ratings.columns = ['coaster_name', 'avg_rating', 'num_reviews']
        
        return coaster_ratings
    
    def prepare_dataset(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load and prepare train/val/test datasets.
        
        Args:
            test_size: Fraction of data for testing
            val_size: Fraction of training data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            train_loader, val_loader, test_loader
        """
        print("Loading ratings data...")
        ratings_df = self.load_ratings_data()
        
        print("Loading acceleration data...")
        sequences = []
        scores = []
        coaster_names = []
        
        # Progress bar for loading coasters
        coaster_pbar = tqdm(self.name_mapping.items(), desc="Loading coasters", 
                           total=len(self.name_mapping))
        
        for ratings_name, accel_filename in coaster_pbar:
            # Check if coaster has ratings
            coaster_rating = ratings_df[ratings_df['coaster_name'] == ratings_name]
            
            coaster_pbar.set_postfix({'current': ratings_name[:20]})
            
            if coaster_rating.empty:
                tqdm.write(f"  ⚠ No ratings found for: {ratings_name}")
                continue
            
            # Load acceleration data
            try:
                accel_data = self.load_acceleration_data(f"{accel_filename}.csv")
                
                if len(accel_data) == 0:
                    tqdm.write(f"  ⚠ Empty acceleration data: {accel_filename}.csv")
                    continue
                
                sequences.append(accel_data)
                scores.append(coaster_rating['avg_rating'].values[0])
                coaster_names.append(ratings_name)
                tqdm.write(f"  ✓ Loaded: {ratings_name} → {accel_filename}.csv (rating: {coaster_rating['avg_rating'].values[0]:.2f})")
                
            except FileNotFoundError:
                tqdm.write(f"  ✗ File not found: {accel_filename}.csv")
                continue
            except Exception as e:
                tqdm.write(f"  ✗ Error loading {accel_filename}.csv: {e}")
                continue
        
        if len(sequences) == 0:
            raise ValueError("No valid coaster data loaded!")
        
        print(f"\nLoaded {len(sequences)} coasters successfully")
        
        # Determine sequence length and input features
        if self.seq_length is None:
            # Use maximum sequence length
            self.seq_length = max(len(seq) for seq in sequences)
            print(f"Auto-detected sequence length: {self.seq_length}")
        
        self.input_size = sequences[0].shape[1]
        print(f"Input features per timestep: {self.input_size}")
        
        # Normalize acceleration data and pad/truncate sequences
        print("\nNormalizing and padding sequences...")
        processed_sequences = []
        
        for seq in sequences:
            # Pad or truncate to fixed length
            if len(seq) < self.seq_length:
                # Pad with zeros
                padding = np.zeros((self.seq_length - len(seq), seq.shape[1]))
                seq = np.vstack([seq, padding])
            elif len(seq) > self.seq_length:
                # Truncate
                seq = seq[:self.seq_length]
            
            processed_sequences.append(seq)
        
        # Stack all sequences
        X = np.array(processed_sequences)  # Shape: (n_samples, seq_length, n_features)
        
        # Reshape for scaling: (n_samples * seq_length, n_features)
        n_samples, seq_len, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        
        # Fit and transform scaler
        X_normalized = self.scaler_accel.fit_transform(X_reshaped)
        X = X_normalized.reshape(n_samples, seq_len, n_features)
        
        # Normalize scores to [0, 1] range
        y = np.array(scores).reshape(-1, 1)
        y_normalized = self.scaler_score.fit_transform(y).flatten()
        
        print(f"Score range before normalization: [{y.min():.2f}, {y.max():.2f}]")
        print(f"Score range after normalization: [{y_normalized.min():.2f}, {y_normalized.max():.2f}]")
        
        # Split into train/val/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_normalized, test_size=test_size, random_state=random_state
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state
        )
        
        print(f"\nDataset split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val:   {len(X_val)} samples")
        print(f"  Test:  {len(X_test)} samples")
        
        # Create DataLoaders
        train_dataset = AccelerationDataset(X_train, y_train)
        val_dataset = AccelerationDataset(X_val, y_val)
        test_dataset = AccelerationDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def build_model(self, **model_kwargs):
        """
        Build the BiGRU model.
        
        Args:
            **model_kwargs: Arguments to pass to BiGRUScorePredictor
        """
        if self.input_size is None:
            raise ValueError("Must call prepare_dataset() before build_model()")
        
        self.model = BiGRUScorePredictor(
            input_size=self.input_size,
            **model_kwargs
        ).to(self.device)
        
        print(f"\nModel architecture:")
        print(self.model)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 0.001,
        patience: int = 15,
        save_path: str = "bigru_score_model.pth"
    ) -> Dict[str, List[float]]:
        """
        Train the model with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            learning_rate: Learning rate
            patience: Early stopping patience
            save_path: Path to save best model
            
        Returns:
            Dictionary with training history
        """
        if self.model is None:
            raise ValueError("Must call build_model() before train()")
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\nStarting training for {epochs} epochs...")
        print("=" * 70)
        
        # Progress bar for epochs
        epoch_pbar = tqdm(range(epochs), desc="Training Progress", position=0)
        
        for epoch in epoch_pbar:
            epoch_start_time = time.time()
            
            # Training
            self.model.train()
            train_losses = []
            train_maes = []
            
            # Progress bar for training batches
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                            leave=False, position=1)
            
            for sequences, labels in train_pbar:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                train_maes.append(torch.abs(outputs - labels).mean().item())
                
                # Update batch progress bar
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Validation
            self.model.eval()
            val_losses = []
            val_maes = []
            
            # Progress bar for validation batches
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", 
                          leave=False, position=1)
            
            with torch.no_grad():
                for sequences, labels in val_pbar:
                    sequences = sequences.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(sequences)
                    loss = criterion(outputs, labels)
                    
                    val_losses.append(loss.item())
                    val_maes.append(torch.abs(outputs - labels).mean().item())
                    
                    # Update batch progress bar
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Calculate epoch metrics
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            train_mae = np.mean(train_maes)
            val_mae = np.mean(val_maes)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_mae'].append(train_mae)
            history['val_mae'].append(val_mae)
            
            # Learning rate scheduler
            scheduler.step(val_loss)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Update epoch progress bar with metrics
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'val_mae': f'{val_mae:.4f}',
                'time': f'{epoch_time:.1f}s'
            })
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'scaler_accel': self.scaler_accel,
                    'scaler_score': self.scaler_score,
                    'input_size': self.input_size,
                    'seq_length': self.seq_length,
                }, save_path)
                tqdm.write(f"  ✓ Epoch {epoch+1}: Best model saved (val_loss: {val_loss:.4f}, val_mae: {val_mae:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    tqdm.write(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                    break
        
        epoch_pbar.close()
        
        print("\n" + "=" * 70)
        print("✓ Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"Final val MAE: {history['val_mae'][-1]:.4f}")
        print("=" * 70)
        
        return history
    
    def predict(self, acceleration_data: np.ndarray) -> float:
        """
        Predict score for a single coaster's acceleration data.
        
        Args:
            acceleration_data: Array of shape (seq_length, n_features)
            
        Returns:
            Predicted score (denormalized)
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        self.model.eval()
        
        # Preprocess
        data = acceleration_data.copy()
        
        # Pad or truncate
        if len(data) < self.seq_length:
            padding = np.zeros((self.seq_length - len(data), data.shape[1]))
            data = np.vstack([data, padding])
        elif len(data) > self.seq_length:
            data = data[:self.seq_length]
        
        # Normalize
        data_reshaped = data.reshape(-1, data.shape[1])
        data_normalized = self.scaler_accel.transform(data_reshaped)
        data = data_normalized.reshape(1, self.seq_length, -1)
        
        # Predict
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data).to(self.device)
            prediction = self.model(data_tensor).cpu().numpy()
        
        # Denormalize
        prediction = self.scaler_score.inverse_transform(prediction.reshape(-1, 1))[0, 0]
        
        return float(prediction)
    
    def load_model(self, model_path: str):
        """Load a saved model."""
        # PyTorch 2.6+ requires weights_only=False for loading scalers
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.input_size = checkpoint['input_size']
        self.seq_length = checkpoint['seq_length']
        self.scaler_accel = checkpoint['scaler_accel']
        self.scaler_score = checkpoint['scaler_score']
        
        # Rebuild model
        self.model = BiGRUScorePredictor(input_size=self.input_size).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"  Val loss at save: {checkpoint['val_loss']:.4f}")
        print(f"  Epoch: {checkpoint['epoch']}")


if __name__ == "__main__":
    # Example usage
    predictor = CoasterScorePredictor(
        accel_data_dir="accel_data",
        ratings_data_path="ratings_data/processed_all_reviews_metadata_20251110_161035.csv"
    )
    
    # Prepare data
    train_loader, val_loader, test_loader = predictor.prepare_dataset()
    
    # Build model
    predictor.build_model(
        hidden_size=128,
        num_layers=2,
        dropout=0.3
    )
    
    # Train
    history = predictor.train(
        train_loader,
        val_loader,
        epochs=30,
        learning_rate=0.001,
        patience=15
    )
    
    print("\nTraining complete!")
