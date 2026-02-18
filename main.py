# main.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import yaml
import argparse
import shutil
import datetime
import json

from data_utils import (
    load_training_data,
    create_train_val_test_splits,
    create_dataloaders
)
from model import KinaseSubstrateAttentionModel, count_parameters


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            kinase_emb = batch['kinase_embedding'].to(device)
            substrate_enc = batch['substrate_encoded'].to(device)
            labels = batch['label'].to(device)

            predictions, _ = model(kinase_emb, substrate_enc)
            predictions = predictions.squeeze()

            loss = criterion(predictions, labels)
            total_loss += loss.item()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    pred_binary = (all_predictions > 0.5).astype(int)

    metrics = {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy_score(all_labels, pred_binary),
        'precision': precision_score(all_labels, pred_binary, zero_division=0),
        'recall': recall_score(all_labels, pred_binary, zero_division=0),
        'f1': f1_score(all_labels, pred_binary, zero_division=0),
        'auc': roc_auc_score(all_labels, all_predictions)
    }

    return metrics, all_predictions, all_labels


# ============================================================================
# Training Function
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        kinase_emb = batch['kinase_embedding'].to(device)
        substrate_enc = batch['substrate_encoded'].to(device)
        labels = batch['label'].to(device)

        predictions, _ = model(kinase_emb, substrate_enc)
        predictions = predictions.squeeze()

        loss = criterion(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ============================================================================
# Main
# ============================================================================

def main():

    # ----------------------------------------------------------------------
    # Parse arguments
    # ----------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_dir", type=str, default=None)
    args = parser.parse_args()

    # ----------------------------------------------------------------------
    # Load YAML config
    # ----------------------------------------------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # ----------------------------------------------------------------------
    # Device
    # ----------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------------------------------------
    # Set random seeds
    # ----------------------------------------------------------------------
    torch.manual_seed(config["data"]["random_seed"])
    np.random.seed(config["data"]["random_seed"])

    # ----------------------------------------------------------------------
    # Create run directory
    # ----------------------------------------------------------------------
    if args.run_dir is not None:
        run_dir = args.run_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(
            config["experiment"]["save_dir"],
            f'{timestamp}_{config["experiment"]["name"]}'
        )

    os.makedirs(run_dir, exist_ok=True)

    out_files_dir = os.path.join(run_dir, "out_files")
    figures_dir = os.path.join(run_dir, "figures")
    os.makedirs(out_files_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Copy config for reproducibility
    shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))

    print("=" * 80)
    print("Kinase-Substrate Attention Model Training")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")

    # ======================================================================
    # Load Data
    # ======================================================================
    print("\n1. Loading data...")

    id_corrections = {
        'C0HM02': 'P24723',
    }

    df, kinase_embeddings = load_training_data(
        config["paths"]["train_pairs"],
        config["paths"]["kinase_embeddings"],
        id_remap=id_corrections
    )

    train_df, val_df, test_df = create_train_val_test_splits(
        df,
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
        random_seed=config["data"]["random_seed"]
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        kinase_embeddings,
        batch_size=config["training"]["batch_size"]
    )

    # ======================================================================
    # Initialize Model
    # ======================================================================
    print("\n2. Initializing model...")

    model = KinaseSubstrateAttentionModel(
        num_groups=config["model"]["num_groups"],
        substrate_embedding_dim=config["model"]["substrate_embedding_dim"],
        kinase_dim=config["model"]["kinase_dim"],
        attention_heads=config["model"]["attention_heads"],
        hidden_dim=config["model"]["hidden_dim"],
        dropout=config["model"]["dropout"]
    ).to(device)

    print(f"Total parameters: {count_parameters(model):,}")

    # ======================================================================
    # Training Setup
    # ======================================================================
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    patience_counter = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_auc': []
    }

    # ======================================================================
    # Training Loop
    # ======================================================================
    print("\n3. Training...")
    print("-" * 80)

    for epoch in range(config["training"]["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_metrics['loss'])

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_metrics['loss']:.4f}")
        print(f"Val Acc:    {val_metrics['accuracy']:.4f}")
        print(f"Val AUC:    {val_metrics['auc']:.4f}")
        print(f"Val F1:     {val_metrics['f1']:.4f}")

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': float(val_metrics['loss']),
                'val_metrics': {k: float(v) for k, v in val_metrics.items()},
            }, os.path.join(out_files_dir, 'best_model.pt'))

        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{config['training']['patience']}")

            if patience_counter >= config["training"]["patience"]:
                print("\nEarly stopping triggered!")
                break

    # ======================================================================
    # Test Evaluation
    # ======================================================================
    print("\n4. Final Evaluation on Test Set")
    print("=" * 80)

    checkpoint = torch.load(os.path.join(out_files_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print("\nTest Set Results:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    with open(os.path.join(out_files_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    np.savez(os.path.join(out_files_dir, 'test_predictions.npz'),
             predictions=test_preds,
             labels=test_labels)

    # ======================================================================
    # Plot Training History
    # ======================================================================
    print("\n5. Plotting training history...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history['val_accuracy'], label='Accuracy')
    axes[1].plot(history['val_auc'], label='AUC')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'training_history.png'), dpi=150)

    np.savez(os.path.join(out_files_dir, 'training_history.npz'), **history)

    print("\nTraining Complete!")
    print(f"Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
