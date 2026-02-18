# analyze_attention_reverse.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import yaml

from data_utils import load_training_data, create_train_val_test_splits, create_dataloaders
from model import KinaseSubstrateAttentionModel


def get_reverse_attention_weights(model, dataloader, device, num_examples=1000):
    """
    Extract attention by computing it in reverse direction.
    
    Instead of "substrate attends to kinase" (9→1, all weights are 1.0),
    we compute "kinase attends to substrate" (1→9, shows position importance).
    
    Returns:
        attention_weights: [num_examples, 9] - importance of each position
        labels: [num_examples] - true labels
        substrates: list of substrate sequences
        predictions: [num_examples] - model predictions
    """
    model.eval()
    
    all_attention = []
    all_labels = []
    all_substrates = []
    all_predictions = []
    
    examples_collected = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if examples_collected >= num_examples:
                break
                
            kinase_emb = batch['kinase_embedding'].to(device)
            substrate_enc = batch['substrate_encoded'].to(device)
            labels = batch['label']
            
            # Get embeddings from model
            substrate_emb = model.substrate_embedding(substrate_enc)  # [B, 9, 64]
            kinase_proj = model.kinase_projection(kinase_emb)  # [B, 64]
            kinase_proj = kinase_proj.unsqueeze(1)  # [B, 1, 64]
            
            # REVERSE attention: kinase (query) attends to substrate (key/value)
            _, attention_weights = model.cross_attention(
                query=kinase_proj,        # [B, 1, 64]
                key=substrate_emb,        # [B, 9, 64]
                value=substrate_emb,      # [B, 9, 64]
                need_weights=True
            )
            # attention_weights: [B, 1, 9]
            
            # Get predictions
            logits, _ = model(kinase_emb, substrate_enc)
            predictions = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # Process attention
            attention_weights = attention_weights.squeeze(1).cpu().numpy()  # [B, 9]
            
            batch_size = len(labels)
            all_attention.append(attention_weights)
            all_labels.extend(labels.numpy())
            all_substrates.extend(batch['substrate_seq'])
            all_predictions.extend(predictions if predictions.ndim > 0 else [predictions.item()])
            
            examples_collected += batch_size
    
    all_attention = np.vstack(all_attention)[:num_examples]
    all_labels = np.array(all_labels)[:num_examples]
    all_substrates = all_substrates[:num_examples]
    all_predictions = np.array(all_predictions)[:num_examples]
    
    print(f"\n✓ Extracted reverse attention weights")
    print(f"  Shape: {all_attention.shape}")
    print(f"  Example (first): {all_attention[0]}")
    print(f"  Sum (should be ~1.0): {all_attention[0].sum():.4f}")
    
    return all_attention, all_labels, all_substrates, all_predictions


def plot_position_attention(attention_weights, labels, save_path):
    """
    Plot average attention by position for positive vs negative examples.
    """
    # Separate by class
    pos_attention = attention_weights[labels == 1]
    neg_attention = attention_weights[labels == 0]
    
    # Average attention per position
    pos_avg = pos_attention.mean(axis=0)
    neg_avg = neg_attention.mean(axis=0)
    pos_std = pos_attention.std(axis=0)
    neg_std = neg_attention.std(axis=0)
    
    # Standard error
    pos_sem = pos_std / np.sqrt(len(pos_attention))
    neg_sem = neg_std / np.sqrt(len(neg_attention))
    
    # Position labels (CORRECTED - no P0!)
    positions = ['P-4', 'P-3', 'P-2', 'P-1', 'P+1', 'P+2', 'P+3', 'P+4', 'P+5']
    x = np.arange(len(positions))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ========================================================================
    # Plot 1: Bar plot comparison
    # ========================================================================
    width = 0.35
    axes[0, 0].bar(x - width/2, pos_avg, width, label='Phosphorylated', 
                   color='green', alpha=0.7, yerr=pos_sem, capsize=5)
    axes[0, 0].bar(x + width/2, neg_avg, width, label='Not Phosphorylated', 
                   color='red', alpha=0.7, yerr=neg_sem, capsize=5)
    # Note: P0 is between P-1 (index 3) and P+1 (index 4)
    axes[0, 0].axvline(x=3.5, color='black', linestyle='--', alpha=0.5, linewidth=2)
    axes[0, 0].set_xlabel('Substrate Position', fontsize=13)
    axes[0, 0].set_ylabel('Average Attention Weight', fontsize=13)
    axes[0, 0].set_title('Position Importance: Phosphorylated vs Not', 
                        fontsize=15, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(positions)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # Plot 2: Line plot
    # ========================================================================
    axes[0, 1].plot(positions, pos_avg, marker='o', linewidth=2, 
                   markersize=8, label='Phosphorylated', color='green')
    axes[0, 1].plot(positions, neg_avg, marker='s', linewidth=2, 
                   markersize=8, label='Not Phosphorylated', color='red')
    axes[0, 1].axvline(x=3.5, color='black', linestyle='--', alpha=0.5, linewidth=2)
    axes[0, 1].fill_between(x, pos_avg - pos_sem, pos_avg + pos_sem, 
                           alpha=0.2, color='green')
    axes[0, 1].fill_between(x, neg_avg - neg_sem, neg_avg + neg_sem, 
                           alpha=0.2, color='red')
    axes[0, 1].set_xlabel('Substrate Position', fontsize=13)
    axes[0, 1].set_ylabel('Average Attention Weight', fontsize=13)
    axes[0, 1].set_title('Attention Profile Across Positions', 
                        fontsize=15, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 3: Difference plot
    # ========================================================================
    difference = pos_avg - neg_avg
    colors = ['green' if d > 0 else 'red' for d in difference]
    axes[1, 0].bar(positions, difference, color=colors, alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 0].axvline(x=3.5, color='black', linestyle='--', alpha=0.5, linewidth=2)
    axes[1, 0].set_xlabel('Substrate Position', fontsize=13)
    axes[1, 0].set_ylabel('Attention Difference (Pos - Neg)', fontsize=13)
    axes[1, 0].set_title('Which Positions Distinguish Phosphorylation?', 
                        fontsize=15, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # Plot 4: Heatmap
    # ========================================================================
    n_samples_per_class = 30
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]
    
    sample_pos = np.random.choice(pos_indices, min(n_samples_per_class, len(pos_indices)), replace=False)
    sample_neg = np.random.choice(neg_indices, min(n_samples_per_class, len(neg_indices)), replace=False)
    sample_indices = np.concatenate([sample_pos, sample_neg])
    
    attention_sample = attention_weights[sample_indices]
    labels_sample = labels[sample_indices]
    
    # Sort by label
    sort_idx = np.argsort(labels_sample)
    attention_sample = attention_sample[sort_idx]
    
    im = axes[1, 1].imshow(attention_sample, aspect='auto', cmap='YlOrRd', 
                          vmin=0, vmax=attention_sample.max())
    axes[1, 1].set_xlabel('Substrate Position', fontsize=13)
    axes[1, 1].set_ylabel('Example (sorted: Neg → Pos)', fontsize=13)
    axes[1, 1].set_title('Attention Patterns (60 examples)', 
                        fontsize=15, fontweight='bold')
    axes[1, 1].set_xticks(range(9))
    axes[1, 1].set_xticklabels(positions)
    axes[1, 1].axhline(y=n_samples_per_class - 0.5, color='white', 
                      linestyle='--', linewidth=2)
    
    cbar = plt.colorbar(im, ax=axes[1, 1])
    cbar.set_label('Attention Weight', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.close()
    
    # ========================================================================
    # Print statistics
    # ========================================================================
    print("\n" + "="*70)
    print("REVERSE ATTENTION STATISTICS BY POSITION")
    print("="*70)
    print(f"{'Position':<10} {'Pos Mean':<12} {'Neg Mean':<12} {'Difference':<12} {'P-value'}")
    print("-"*70)
    
    from scipy import stats
    for i, pos in enumerate(positions):
        diff = pos_avg[i] - neg_avg[i]
        t_stat, p_val = stats.ttest_ind(pos_attention[:, i], neg_attention[:, i])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{pos:<10} {pos_avg[i]:<12.4f} {neg_avg[i]:<12.4f} {diff:<+12.4f} {p_val:.4f} {sig}")
    print("="*70)
    print("* p<0.05, ** p<0.01, *** p<0.001")
    print("\nNote: P0 (the actual phosphorylation site) is omitted from the 9-mer sequence")
    
    max_pos_idx = np.argmax(pos_avg)
    max_neg_idx = np.argmax(neg_avg)
    max_diff_idx = np.argmax(np.abs(difference))
    
    print(f"\n✓ Most attended position (Phosphorylated): {positions[max_pos_idx]} ({pos_avg[max_pos_idx]:.4f})")
    print(f"✓ Most attended position (Not phosphorylated): {positions[max_neg_idx]} ({neg_avg[max_neg_idx]:.4f})")
    print(f"✓ Biggest difference: {positions[max_diff_idx]} ({difference[max_diff_idx]:+.4f})")


def plot_high_confidence_attention(attention_weights, labels, substrates, predictions, save_path):
    """
    Show attention patterns for high-confidence predictions.
    """
    # Find high-confidence examples
    confident_pos_idx = np.where((predictions > 0.7) & (labels == 1))[0]
    confident_neg_idx = np.where((predictions < 0.3) & (labels == 0))[0]
    false_pos_idx = np.where((predictions > 0.7) & (labels == 0))[0]
    false_neg_idx = np.where((predictions < 0.3) & (labels == 1))[0]
    
    # Position labels (CORRECTED)
    positions = ['P-4', 'P-3', 'P-2', 'P-1', 'P+1', 'P+2', 'P+3', 'P+4', 'P+5']
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    
    def plot_example(ax, idx, color, title_prefix):
        """Plot a single attention example"""
        ax.bar(positions, attention_weights[idx], color=color, alpha=0.7)
        ax.set_ylim([0, attention_weights[idx].max() * 1.2])
        ax.axvline(x=3.5, color='black', linestyle='--', alpha=0.3, linewidth=1)
        ax.set_title(f"{title_prefix}\nPred: {predictions[idx]:.3f} | True: {int(labels[idx])}\n{substrates[idx]}", 
                    fontsize=9)
        ax.grid(True, alpha=0.2, axis='y')
    
    # Row 1: True Positives
    for i in range(3):
        if i < len(confident_pos_idx):
            plot_example(axes[0, i], confident_pos_idx[i], 'green', 'True Positive (Confident)')
        if i == 0:
            axes[0, i].set_ylabel('Attention', fontsize=11)
    
    # Row 2: True Negatives
    for i in range(3):
        if i < len(confident_neg_idx):
            plot_example(axes[1, i], confident_neg_idx[i], 'blue', 'True Negative (Confident)')
        if i == 0:
            axes[1, i].set_ylabel('Attention', fontsize=11)
    
    # Row 3: False Positives
    for i in range(3):
        if i < len(false_pos_idx):
            plot_example(axes[2, i], false_pos_idx[i], 'orange', 'False Positive (Error)')
        if i == 0:
            axes[2, i].set_ylabel('Attention', fontsize=11)
    
    # Row 4: False Negatives
    for i in range(3):
        if i < len(false_neg_idx):
            plot_example(axes[3, i], false_neg_idx[i], 'red', 'False Negative (Error)')
        axes[3, i].set_xlabel('Position', fontsize=10)
        if i == 0:
            axes[3, i].set_ylabel('Attention', fontsize=11)
    
    fig.suptitle('Reverse Attention Patterns: Confident Predictions & Errors', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze reverse attention weights')
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--num_examples", type=int, default=1000)
    args = parser.parse_args()
    
    run_dir = args.run_dir
    config_path = f"{run_dir}/config.yaml"
    model_path = f"{run_dir}/out_files/best_model.pt"
    figures_dir = f"{run_dir}/figures"
    
    print("="*80)
    print("REVERSE ATTENTION ANALYSIS")
    print("="*80)
    print(f"Run directory: {run_dir}")
    print("\nWhat is reverse attention?")
    print("  - Your model: Substrate (9) attends to Kinase (1)")
    print("  - For analysis: Kinase (1) attends to Substrate (9)")
    print("  - This shows which substrate positions are important!")
    print("  - Positions: P-4, P-3, P-2, P-1, [P0 omitted], P+1, P+2, P+3, P+4, P+5")
    print("="*80)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    print("\n1. Loading data...")
    id_corrections = {'C0HM02': 'P24723'}
    
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
    
    _, _, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        kinase_embeddings,
        batch_size=config["training"]["batch_size"]
    )
    
    # Load model
    print("\n2. Loading model...")
    model = KinaseSubstrateAttentionModel(
        num_groups=config["model"]["num_groups"],
        substrate_embedding_dim=config["model"]["substrate_embedding_dim"],
        kinase_dim=config["model"]["kinase_dim"],
        attention_heads=config["model"]["attention_heads"],
        hidden_dim=config["model"]["hidden_dim"],
        dropout=config["model"]["dropout"]
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    
    # Extract attention
    print(f"\n3. Computing reverse attention for {args.num_examples} examples...")
    attention_weights, labels, substrates, predictions = get_reverse_attention_weights(
        model, test_loader, device, num_examples=args.num_examples
    )
    
    print(f"✓ Extracted attention for {len(attention_weights)} examples")
    print(f"  - Positives: {labels.sum()}")
    print(f"  - Negatives: {(labels == 0).sum()}")
    
    # Generate plots
    print("\n4. Generating plots...")
    
    plot_position_attention(
        attention_weights, labels, 
        f"{figures_dir}/reverse_attention_by_position.png"
    )
    
    plot_high_confidence_attention(
        attention_weights, labels, substrates, predictions,
        f"{figures_dir}/reverse_attention_examples.png"
    )
    
    # Save data
    output_path = f"{run_dir}/out_files/reverse_attention_analysis.npz"
    np.savez(
        output_path,
        attention_weights=attention_weights,
        labels=labels,
        substrates=np.array(substrates),
        predictions=predictions
    )
    print(f"\n✓ Saved data to: {output_path}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()