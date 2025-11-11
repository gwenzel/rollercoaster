"""
Training script for BiGRU score predictor.
Run this to train the model on your acceleration and ratings data.
"""

import sys
sys.path.append('.')

from bigru_score_predictor import CoasterScorePredictor
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

def main():
    print("="*70)
    print("BiGRU COASTER SCORE PREDICTOR - TRAINING")
    print("="*70)
    
    # Initialize predictor
    predictor = CoasterScorePredictor(
        accel_data_dir="accel_data",
        ratings_data_path="ratings_data/processed_all_reviews_metadata_20251110_161035.csv"
    )
    
    # Prepare dataset
    print("\n" + "="*70)
    print("STEP 1: PREPARING DATASET")
    print("="*70)
    train_loader, val_loader, test_loader = predictor.prepare_dataset(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    # Build model
    print("\n" + "="*70)
    print("STEP 2: BUILDING MODEL")
    print("="*70)
    predictor.build_model(
        hidden_size=128,      # GRU hidden size
        num_layers=2,         # Number of GRU layers
        dropout=0.3,          # Dropout rate
        fc_hidden_size=64     # Fully connected layer size
    )
    
    # Train model
    print("\n" + "="*70)
    print("STEP 3: TRAINING MODEL")
    print("="*70)
    history = predictor.train(
        train_loader,
        val_loader,
        epochs=100,
        learning_rate=0.001,
        patience=15,
        save_path="bigru_score_model.pth"
    )
    
    # Plot training history
    print("\n" + "="*70)
    print("STEP 4: VISUALIZING TRAINING")
    print("="*70)
    plot_training_history(history)
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("STEP 5: EVALUATING ON TEST SET")
    print("="*70)
    test_metrics = evaluate_model(predictor, test_loader)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Model saved to: bigru_score_model.pth")
    print(f"Plots saved to: training_history.png, predictions_vs_actual.png")
    
    # Example prediction
    print("\n" + "="*70)
    print("EXAMPLE PREDICTION")
    print("="*70)
    try:
        # Load a sample coaster for prediction
        sample_data = predictor.load_acceleration_data("Zadra.csv")
        predicted_score = predictor.predict(sample_data)
        print(f"Zadra predicted score: {predicted_score:.2f}")
        
        # Get actual score
        ratings_df = predictor.load_ratings_data()
        actual_score = ratings_df[ratings_df['coaster_name'] == 'Zadra']['avg_rating'].values[0]
        print(f"Zadra actual score: {actual_score:.2f}")
        print(f"Error: {abs(predicted_score - actual_score):.2f}")
    except Exception as e:
        print(f"Could not run example prediction: {e}")

if __name__ == "__main__":
    main()
