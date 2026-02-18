# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class KinaseSubstrateAttentionModel(nn.Module):
    """
    Cross-attention model for kinase-substrate interaction prediction.
    
    Architecture:
        1. Substrate embedding: Maps 5 biochemical groups to embedding space
        2. Cross-attention: Substrate positions attend to kinase embedding
        3. Pooling: Aggregate attended substrate representations
        4. MLP classifier: Predict phosphorylation probability
    """
    
    def __init__(
        self,
        num_groups=5,              # Number of biochemical property groups
        substrate_embedding_dim=64, # Dimension for substrate position embeddings
        kinase_dim=1280,           # ESM-2 embedding dimension
        attention_heads=4,         # Number of attention heads
        hidden_dim=256,            # Hidden layer dimension for MLP
        dropout=0.1                # Dropout rate
    ):
        super().__init__()
        
        self.num_groups = num_groups
        self.substrate_embedding_dim = substrate_embedding_dim
        self.kinase_dim = kinase_dim
        
        # 1. Substrate position embedding
        # Maps biochemical group indices (0-4) to embedding vectors
        self.substrate_embedding = nn.Embedding(
            num_embeddings=num_groups,
            embedding_dim=substrate_embedding_dim
        )
        
        # 2. Cross-attention layer
        # Substrate (query) attends to kinase (key/value)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=substrate_embedding_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Project kinase embedding to match substrate embedding dimension
        self.kinase_projection = nn.Linear(kinase_dim, substrate_embedding_dim)
        
        # 3. Pooling: aggregate the 9 attended substrate positions
        # Options: mean pooling, max pooling, or learnable pooling
        # We'll use mean pooling for simplicity
        
        # 4. MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(substrate_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output probability
        )
        
    def forward(self, kinase_embedding, substrate_encoded):
        """
        Forward pass.
        
        Args:
            kinase_embedding: [batch_size, 1280] - ESM-2 kinase embeddings
            substrate_encoded: [batch_size, 9] - Substrate sequences as group indices
        
        Returns:
            predictions: [batch_size, 1] - Phosphorylation probabilities
            attention_weights: [batch_size, 9, 1] - Attention weights (optional, for analysis)
        """
        batch_size = substrate_encoded.shape[0]
        
        # 1. Embed substrate sequence
        # [batch_size, 9] -> [batch_size, 9, substrate_embedding_dim]
        substrate_emb = self.substrate_embedding(substrate_encoded)
        
        # 2. Project kinase embedding to substrate embedding space
        # [batch_size, 1280] -> [batch_size, substrate_embedding_dim]
        kinase_proj = self.kinase_projection(kinase_embedding)
        
        # Reshape kinase for attention: [batch_size, 1, substrate_embedding_dim]
        # (treating kinase as a single "token" that substrate attends to)
        kinase_proj = kinase_proj.unsqueeze(1)
        
        # 3. Cross-attention: substrate (query) attends to kinase (key/value)
        # substrate_emb: [batch_size, 9, substrate_embedding_dim] (query)
        # kinase_proj: [batch_size, 1, substrate_embedding_dim] (key & value)
        attended_substrate, attention_weights = self.cross_attention(
            query=substrate_emb,      # [batch_size, 9, substrate_embedding_dim]
            key=kinase_proj,          # [batch_size, 1, substrate_embedding_dim]
            value=kinase_proj,        # [batch_size, 1, substrate_embedding_dim]
            need_weights=True
        )
        # attended_substrate: [batch_size, 9, substrate_embedding_dim]
        # attention_weights: [batch_size, 9, 1]
        
        # 4. Pool attended substrate representations
        # Mean pooling across the 9 positions
        pooled = attended_substrate.mean(dim=1)  # [batch_size, substrate_embedding_dim]
        
        # 5. Classify
        predictions = self.classifier(pooled)  # [batch_size, 1]
        
        return predictions, attention_weights
    
    def get_attention_weights(self, kinase_embedding, substrate_encoded):
        """
        Convenience method to get attention weights for interpretation.
        """
        with torch.no_grad():
            _, attention_weights = self.forward(kinase_embedding, substrate_encoded)
        return attention_weights


def count_parameters(model):
    """
    Count trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)