"""
Training script for BiGRU score predictor.
Uses complete_coaster_mapping.csv with perfect matches and averaged ratings.
"""

import sys
import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.bigru_score_predictor import CoasterScorePredictor
import matplotlib.pyplot as plt
import torch

def plot_training_history(history, save_path='training_history.png'):
    """Plot training and validation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
    ax1.plot(history['val_loss'], label='Val Loss', marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE plot
    ax2.plot(history['train_mae'], label='Train MAE', marker='o', markersize=3)
    ax2.plot(history['val_mae'], label='Val MAE', marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Training and Validation MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.show()

def evaluate_model(predictor, test_loader):
    """Evaluate model on test set."""
    predictor.model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(predictor.device)
            outputs = predictor.model(sequences).cpu().numpy()
            
            # Denormalize
            predictions = predictor.scaler_score.inverse_transform(outputs.reshape(-1, 1)).flatten()
            true_scores = predictor.scaler_score.inverse_transform(labels.numpy().reshape(-1, 1)).flatten()
            
            all_predictions.extend(predictions)
            all_labels.extend(true_scores)
    
    # Calculate metrics
    import numpy as np
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    mse = np.mean((predictions - labels) ** 2)
    mae = np.mean(np.abs(predictions - labels))
    rmse = np.sqrt(mse)
    
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    print(f"MSE:  {mse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("="*60)
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(labels, predictions, alpha=0.6, s=100)
    plt.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Score', fontsize=12)
    plt.ylabel('Predicted Score', fontsize=12)
    plt.title('Predicted vs Actual Scores (Test Set)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    print("Predictions plot saved to predictions_vs_actual.png")
    plt.show()
    
    return {'mse': mse, 'mae': mae, 'rmse': rmse}

def load_training_data_from_mapping():
    """
    Load training data from complete_coaster_mapping.csv.
    Returns lists of accelerometer data and corresponding ratings.
    """
    from data_preparation import (
        prepare_training_data,
        load_last_accelerometer_file,
        calculate_airtime
    )
    
    # Prepare the mapping
    mapping_df = prepare_training_data(
        mapping_csv='../ratings_data/complete_coaster_mapping.csv',
        min_csv_count=1,
        min_ratings=10
    )
    
    # Load accelerometer data for each coaster
    accel_data_list = []
    ratings_list = []
    coaster_names = []
    airtime_stats_list = []
    
    print("\n" + "="*70)
    print("LOADING ACCELEROMETER DATA")
    print("="*70)
    
    for idx, row in mapping_df.iterrows():
        coaster_name = row['coaster_name']
        rating = row['avg_rating']
        full_path = row['full_path']
        
        # Load accelerometer data (last file)
        accel_df = load_last_accelerometer_file(f"../{full_path}")
        
        if accel_df is not None:
            # Calculate airtime statistics
            airtime_stats = calculate_airtime(accel_df)
            
            accel_data_list.append(accel_df)
            ratings_list.append(rating)
            coaster_names.append(coaster_name)
            airtime_stats_list.append(airtime_stats)
            
            if (idx + 1) % 50 == 0:
                print(f"  Loaded {idx + 1}/{len(mapping_df)} coasters...")
    
    print(f"\n✓ Successfully loaded {len(accel_data_list)}/{len(mapping_df)} coasters")
    
    # Show airtime statistics correlation
    if airtime_stats_list:
        airtimes = [stats['total_airtime'] for stats in airtime_stats_list]
        print(f"\n🎢 Airtime Statistics:")
        print(f"  Average total airtime: {np.mean(airtimes):.2f}s")
        print(f"  Max airtime: {np.max(airtimes):.2f}s")
        print(f"  Coasters with airtime (>0.5s): {sum(1 for a in airtimes if a > 0.5)}")
    
    return accel_data_list, ratings_list, coaster_names, airtime_stats_list


def main():
    print("="*70)
    print("BiGRU COASTER SCORE PREDICTOR - TRAINING")
    print("Using complete_coaster_mapping.csv")
    print("="*70)
    
    # Change to scripts directory to access data properly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Load data from complete mapping
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA FROM COMPLETE MAPPING")
    print("="*70)
    accel_data_list, ratings_list, coaster_names, airtime_stats = load_training_data_from_mapping()
    
    if len(accel_data_list) == 0:
        print("❌ No data loaded! Check paths and CSV files.")
        return
    
    # Initialize predictor (will create new data loading method)
    print("\n" + "="*70)
    print("STEP 2: INITIALIZING PREDICTOR")
    print("="*70)
    
    # TODO: Update CoasterScorePredictor to accept preloaded data
    # For now, use the existing structure but we'll need to refactor
    print("⚠️  Note: Model initialization needs refactoring to use new data format")
    print("   Current implementation uses old data loading method")
    
    # Prepare dataset
    print("\n" + "="*70)
    print("STEP 3: PREPARING PYTORCH DATALOADERS")
    print("="*70)
    
    # Convert to PyTorch format
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Prepare sequences and normalize
    sequences = []
    for accel_df in accel_data_list:
        # Extract 3-axis data (assuming columns: Lateral, Vertical, Longitudinal)
        if all(col in accel_df.columns for col in ['Lateral', 'Vertical', 'Longitudinal']):
            seq = accel_df[['Lateral', 'Vertical', 'Longitudinal']].values
        else:
            print(f"⚠️  Missing columns in accelerometer data, skipping...")
            continue
        sequences.append(seq)
    
    # Pad/truncate sequences to same length
    max_len = 1000  # Maximum sequence length
    X = []
    for seq in sequences:
        if len(seq) > max_len:
            X.append(seq[:max_len])
        else:
            pad_len = max_len - len(seq)
            X.append(np.vstack([seq, np.zeros((pad_len, 3))]))
    
    X = np.array(X, dtype=np.float32)
    y = np.array(ratings_list, dtype=np.float32)
    
    print(f"✓ Prepared {len(X)} sequences")
    print(f"  Sequence shape: {X.shape}")
    print(f"  Rating shape: {y.shape}")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42  # 0.125 x 0.8 = 0.1 of total
    )
    
    print(f"\n✓ Data split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # Normalize features
    scaler_X = StandardScaler()
    X_train_flat = X_train.reshape(-1, 3)
    scaler_X.fit(X_train_flat)
    
    X_train = scaler_X.transform(X_train.reshape(-1, 3)).reshape(X_train.shape)
    X_val = scaler_X.transform(X_val.reshape(-1, 3)).reshape(X_val.shape)
    X_test = scaler_X.transform(X_test.reshape(-1, 3)).reshape(X_test.shape)
    
    # Normalize targets
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test_scaled)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("✓ DataLoaders created")
    
    # Build and train model using CoasterScorePredictor
    print("\n⚠️  NOTE: Full integration with CoasterScorePredictor class needs refactoring")
    print("   For now, this script prepares data in the correct format.")
    print("   Next step: Update CoasterScorePredictor to accept preloaded DataLoaders")
    
    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETE")
    print("="*70)
    print(f"✓ {len(accel_data_list)} coasters loaded")
    print(f"✓ Perfect matches from complete_coaster_mapping.csv")
    print(f"✓ Duplicates averaged by coaster name")
    print(f"✓ Last CSV file used for each coaster")
    print(f"✓ Data split and normalized")
    print(f"✓ Ready for training!")
    
    # Show coaster name examples
    print("\n📋 Sample coasters in training set:")
    train_indices = list(range(len(X_train)))[:5]
    for i in train_indices:
        if i < len(coaster_names):
            print(f"  - {coaster_names[i]}: {ratings_list[i]:.2f}★")
    
    print("\n💡 Next steps:")
    print("  1. Update CoasterScorePredictor.build_model() to accept input_size=3")
    print("  2. Update CoasterScorePredictor.train() to accept DataLoaders")
    print("  3. Run training loop with prepared data")
    print("  4. Save trained model")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'coaster_names': coaster_names,
        'y_test': y_test  # Keep original scale for evaluation
    }

if __name__ == "__main__":
    main()
