"""
Multi-modal classifier implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from .base import BaseMultiModalModel


class MultiModalClassifier(BaseMultiModalModel):
    """
    A multi-modal classifier that can handle text, image, and audio inputs.
    
    This model encodes each modality separately and then fuses them for classification.
    """
    
    def __init__(
        self,
        num_classes: int,
        text_dim: int = 768,
        image_dim: int = 2048,
        audio_dim: int = 512,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        fusion_method: str = "concat",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the multi-modal classifier.
        
        Args:
            num_classes: Number of output classes
            text_dim: Dimension of text embeddings
            image_dim: Dimension of image embeddings
            audio_dim: Dimension of audio embeddings
            hidden_dim: Dimension of hidden fusion layer
            dropout: Dropout probability
            fusion_method: Method for fusing modalities ('concat', 'attention', 'sum')
            config: Additional configuration
        """
        super().__init__(config)
        
        self.num_classes = num_classes
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.fusion_method = fusion_method
        
        # Modality encoders
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Fusion layer
        if fusion_method == "concat":
            fusion_input_dim = hidden_dim * 3  # Assuming all three modalities
        elif fusion_method == "attention":
            fusion_input_dim = hidden_dim
            self.attention = MultiModalAttention(hidden_dim, num_heads=8)
        elif fusion_method == "sum":
            fusion_input_dim = hidden_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
            
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def encode_modality(self, modality: str, data: torch.Tensor) -> torch.Tensor:
        """
        Encode a specific modality.
        
        Args:
            modality: Name of the modality ('text', 'image', 'audio')
            data: Input tensor for the specified modality
            
        Returns:
            Encoded representation tensor
        """
        if modality == "text":
            return self.text_encoder(data)
        elif modality == "image":
            return self.image_encoder(data)
        elif modality == "audio":
            return self.audio_encoder(data)
        else:
            raise ValueError(f"Unknown modality: {modality}")
    
    def fuse_modalities(self, encoded_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse encoded modalities using the specified fusion method.
        
        Args:
            encoded_inputs: Dictionary of encoded modality representations
            
        Returns:
            Fused representation
        """
        if self.fusion_method == "concat":
            # Concatenate all available modalities
            features = []
            for modality in ["text", "image", "audio"]:
                if modality in encoded_inputs:
                    features.append(encoded_inputs[modality])
                else:
                    # Add zero tensor if modality is missing
                    batch_size = list(encoded_inputs.values())[0].size(0)
                    device = list(encoded_inputs.values())[0].device
                    features.append(torch.zeros(batch_size, self.hidden_dim, device=device))
            
            return torch.cat(features, dim=1)
        
        elif self.fusion_method == "attention":
            # Use attention mechanism to fuse modalities
            modality_features = list(encoded_inputs.values())
            stacked_features = torch.stack(modality_features, dim=1)
            return self.attention(stacked_features)
        
        elif self.fusion_method == "sum":
            # Sum all modality representations
            fused = torch.zeros_like(list(encoded_inputs.values())[0])
            for features in encoded_inputs.values():
                fused += features
            return fused / len(encoded_inputs)  # Average instead of sum
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multi-modal classifier.
        
        Args:
            inputs: Dictionary containing input tensors for different modalities
                   Keys: 'text', 'image', 'audio' (any subset is allowed)
                   
        Returns:
            Dictionary containing 'logits' and 'predictions'
        """
        # Encode each available modality
        encoded_inputs = {}
        for modality, data in inputs.items():
            if modality in ["text", "image", "audio"]:
                encoded_inputs[modality] = self.encode_modality(modality, data)
        
        # Fuse modalities
        fused_features = self.fuse_modalities(encoded_inputs)
        
        # Classification
        logits = self.fusion(fused_features)
        predictions = torch.softmax(logits, dim=-1)
        
        return {
            "logits": logits,
            "predictions": predictions,
            "encoded_features": encoded_inputs,
            "fused_features": fused_features
        }
    
    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Make predictions on input data.
        
        Args:
            inputs: Dictionary containing input tensors for different modalities
            
        Returns:
            Predicted class indices
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs)
            return torch.argmax(outputs["predictions"], dim=-1)


class MultiModalAttention(nn.Module):
    """
    Multi-head attention mechanism for fusing multiple modalities.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head attention.
        
        Args:
            x: Input tensor of shape [batch_size, num_modalities, hidden_dim]
            
        Returns:
            Attended features of shape [batch_size, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute queries, keys, values
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection and mean pooling across modalities
        output = self.output(attended)
        return output.mean(dim=1)  # Average across modalities