"""
Base class for multi-modal models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union


class BaseMultiModalModel(nn.Module, ABC):
    """
    Abstract base class for all multi-modal models.
    
    This class defines the common interface that all multi-modal models
    should implement.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base multi-modal model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__()
        self.config = config or {}
        
    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            inputs: Dictionary containing input tensors for different modalities
                   Keys should specify the modality (e.g., 'text', 'image', 'audio')
                   
        Returns:
            Dictionary containing model outputs
        """
        pass
    
    @abstractmethod
    def encode_modality(self, modality: str, data: torch.Tensor) -> torch.Tensor:
        """
        Encode a specific modality into a common representation space.
        
        Args:
            modality: Name of the modality ('text', 'image', 'audio', etc.)
            data: Input tensor for the specified modality
            
        Returns:
            Encoded representation tensor
        """
        pass
    
    def get_modality_encoders(self) -> Dict[str, nn.Module]:
        """
        Get the encoder modules for each supported modality.
        
        Returns:
            Dictionary mapping modality names to their encoder modules
        """
        encoders = {}
        for name, module in self.named_modules():
            if 'encoder' in name.lower():
                modality = name.split('_')[0]  # Assume naming convention: modality_encoder
                encoders[modality] = module
        return encoders
    
    def freeze_modality_encoder(self, modality: str) -> None:
        """
        Freeze the parameters of a specific modality encoder.
        
        Args:
            modality: Name of the modality to freeze
        """
        encoders = self.get_modality_encoders()
        if modality in encoders:
            for param in encoders[modality].parameters():
                param.requires_grad = False
    
    def unfreeze_modality_encoder(self, modality: str) -> None:
        """
        Unfreeze the parameters of a specific modality encoder.
        
        Args:
            modality: Name of the modality to unfreeze
        """
        encoders = self.get_modality_encoders()
        if modality in encoders:
            for param in encoders[modality].parameters():
                param.requires_grad = True
    
    def get_parameter_count(self) -> Dict[str, int]:
        """
        Get the number of parameters in the model.
        
        Returns:
            Dictionary with total, trainable, and frozen parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params
        }
    
    def save_pretrained(self, save_path: str) -> None:
        """
        Save the model state and configuration.
        
        Args:
            save_path: Path to save the model
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_class': self.__class__.__name__
        }
        torch.save(save_dict, save_path)
    
    @classmethod
    def load_pretrained(cls, load_path: str) -> 'BaseMultiModalModel':
        """
        Load a pretrained model.
        
        Args:
            load_path: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(load_path, map_location='cpu')
        config = checkpoint.get('config', {})
        
        # Create model instance
        model = cls(config=config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model