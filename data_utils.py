# data_utils.py

import pandas as pd
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Biochemical property groups (5 groups + padding)
AA_GROUPS = {
    # Group 0: Hydrophobic
    'A': 0, 'V': 0, 'I': 0, 'L': 0, 'M': 0, 'F': 0, 'W': 0, 'P': 0,
    # Group 1: Polar uncharged
    'S': 1, 'T': 1, 'N': 1, 'Q': 1, 'C': 1, 'G': 1, 'Y': 1,
    # Group 2: Positive
    'K': 2, 'R': 2, 'H': 2,
    # Group 3: Negative
    'D': 3, 'E': 3,
    # Group 4: Padding (underscore)
    '_': 4
}

NUM_GROUPS = 5


def load_training_data(csv_path, embeddings_path, id_remap=None):
    """
    Load training data CSV and kinase embeddings.
    
    Args:
        csv_path: path to CSV file
        embeddings_path: path to pickle file with embeddings
        id_remap: optional dict mapping {old_id: new_id} for corrections
    
    Returns:
        df: pandas DataFrame with all training examples
        kinase_embeddings: dict mapping gene_symbol -> 1280-dim numpy array
    """
    print(f"Loading training data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Apply ID remapping if provided
    if id_remap is not None:
        print(f"\nApplying ID remapping for {len(id_remap)} kinase(s):")
        for old_id, new_id in id_remap.items():
            n_changed = (df['enzyme'] == old_id).sum()
            if n_changed > 0:
                print(f"  {old_id} -> {new_id} ({n_changed} examples)")
                df.loc[df['enzyme'] == old_id, 'enzyme'] = new_id
            else:
                print(f"  WARNING: {old_id} not found in data")
    
    print(f"\nLoading kinase embeddings from {embeddings_path}")
    with open(embeddings_path, 'rb') as f:
        kinase_embeddings = pickle.load(f)
    
    # Verify all kinases in dataset have embeddings
    unique_kinases = df['enzyme'].unique()
    missing = [k for k in unique_kinases if k not in kinase_embeddings]
    if missing:
        print(f"WARNING: {len(missing)} kinases missing embeddings: {missing[:5]}")
    
    print(f"Loaded {len(df)} examples ({df['label'].sum()} positive, {(df['label']==0).sum()} negative)")
    print(f"Loaded embeddings for {len(kinase_embeddings)} kinases")
    
    return df, kinase_embeddings


def encode_substrate_sequence(kmer):
    """
    Encode a 9-mer substrate sequence into biochemical property group indices.
    
    Args:
        kmer: string of length 9 (e.g., "SASPYPEHA")
    
    Returns:
        numpy array of shape (9,) with group indices 0-4
    """
    if len(kmer) != 9:
        raise ValueError(f"Expected 9-mer, got length {len(kmer)}: {kmer}")
    
    encoded = np.array([AA_GROUPS[aa] for aa in kmer], dtype=np.int64)
    return encoded


def create_train_val_test_splits(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Create stratified train/val/test splits preserving 1:2 pos:neg ratio.
    
    Args:
        df: DataFrame with training examples
        train_ratio: proportion for training (default 0.7)
        val_ratio: proportion for validation (default 0.15)
        test_ratio: proportion for testing (default 0.15)
        random_seed: for reproducibility
    
    Returns:
        train_df, val_df, test_df
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # First split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_ratio,
        stratify=df['label'],
        random_state=random_seed
    )
    
    # Second split: val vs test
    val_size_adjusted = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size_adjusted,
        stratify=temp_df['label'],
        random_state=random_seed
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} ({train_df['label'].sum()} pos, {(train_df['label']==0).sum()} neg)")
    print(f"  Val:   {len(val_df)} ({val_df['label'].sum()} pos, {(val_df['label']==0).sum()} neg)")
    print(f"  Test:  {len(test_df)} ({test_df['label'].sum()} pos, {(test_df['label']==0).sum()} neg)")
    
    return train_df, val_df, test_df


class KinaseSubstrateDataset(Dataset):
    """
    PyTorch Dataset for kinase-substrate interactions.
    """
    def __init__(self, df, kinase_embeddings):
        """
        Args:
            df: DataFrame with columns [enzyme_genesymbol, substrate_kmer, label]
            kinase_embeddings: dict mapping gene_symbol -> 1280-dim array
        """
        self.df = df.reset_index(drop=True)
        self.kinase_embeddings = kinase_embeddings
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get kinase embedding (1280-dim)
        kinase_id = row['enzyme']  # Using UniProt ID
        kinase_emb = self.kinase_embeddings[kinase_id]  # numpy array or tensor
        
        # Handle if it's already a tensor or numpy array
        if isinstance(kinase_emb, torch.Tensor):
            kinase_emb = kinase_emb.squeeze()  # Remove any extra dimensions
            kinase_tensor = kinase_emb.clone().detach().float()
        else:
            kinase_emb = np.array(kinase_emb).squeeze()  # Remove any extra dimensions
            kinase_tensor = torch.from_numpy(kinase_emb).float()
        
        # Encode substrate sequence (9 positions -> 9 group indices)
        substrate_seq = row['substrate_kmer']
        substrate_encoded = encode_substrate_sequence(substrate_seq)
        
        # Label
        label = row['label']
        
        return {
            'kinase_embedding': kinase_tensor,
            'substrate_encoded': torch.tensor(substrate_encoded, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float32),
            'kinase_name': kinase_id,  # for debugging/analysis
            'substrate_seq': substrate_seq  # for debugging/analysis
        }


def create_dataloaders(train_df, val_df, test_df, kinase_embeddings, batch_size=32):
    """
    Create PyTorch DataLoaders for train/val/test sets.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = KinaseSubstrateDataset(train_df, kinase_embeddings)
    val_dataset = KinaseSubstrateDataset(val_df, kinase_embeddings)
    test_dataset = KinaseSubstrateDataset(test_df, kinase_embeddings)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # adjust based on your system
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader