#!/usr/bin/env python3
"""
Basic multi-modal classification example.

This script demonstrates how to use the MultiModalClassifier with synthetic data
representing text, image, and audio features.
"""

import torch
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.multimodal_classifier import MultiModalClassifier


def generate_synthetic_data(batch_size: int = 32, num_classes: int = 5):
    """
    Generate synthetic multi-modal data for demonstration.
    
    Args:
        batch_size: Number of samples in the batch
        num_classes: Number of classes for classification
        
    Returns:
        Dictionary containing synthetic inputs and labels
    """
    # Synthetic text features (e.g., from BERT)
    text_features = torch.randn(batch_size, 768)
    
    # Synthetic image features (e.g., from ResNet)
    image_features = torch.randn(batch_size, 2048)
    
    # Synthetic audio features (e.g., from audio encoder)
    audio_features = torch.randn(batch_size, 512)
    
    # Random labels
    labels = torch.randint(0, num_classes, (batch_size,))
    
    return {
        "inputs": {
            "text": text_features,
            "image": image_features,
            "audio": audio_features
        },
        "labels": labels
    }


def demonstrate_multimodal_classification():
    """
    Demonstrate basic multi-modal classification functionality.
    """
    print("🚀 Multi-Modal Classification Demo")
    print("=" * 50)
    
    # Model parameters
    num_classes = 5
    batch_size = 16
    
    # Create model
    print("📦 Creating MultiModalClassifier...")
    model = MultiModalClassifier(
        num_classes=num_classes,
        text_dim=768,
        image_dim=2048,
        audio_dim=512,
        hidden_dim=256,
        dropout=0.1,
        fusion_method="concat"
    )
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Generate synthetic data
    print("\n📊 Generating synthetic data...")
    data = generate_synthetic_data(batch_size=batch_size, num_classes=num_classes)
    
    print(f"✅ Generated data:")
    print(f"   - Text features: {data['inputs']['text'].shape}")
    print(f"   - Image features: {data['inputs']['image'].shape}")
    print(f"   - Audio features: {data['inputs']['audio'].shape}")
    print(f"   - Labels: {data['labels'].shape}")
    
    # Forward pass
    print("\n🔄 Running forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(data["inputs"])
    
    print(f"✅ Forward pass completed:")
    print(f"   - Logits shape: {outputs['logits'].shape}")
    print(f"   - Predictions shape: {outputs['predictions'].shape}")
    print(f"   - Fused features shape: {outputs['fused_features'].shape}")
    
    # Make predictions
    predictions = model.predict(data["inputs"])
    print(f"\n🎯 Predictions: {predictions}")
    print(f"   Ground truth: {data['labels']}")
    
    # Test with missing modalities
    print("\n🧪 Testing with missing modalities...")
    
    # Only text and image
    partial_inputs = {
        "text": data["inputs"]["text"],
        "image": data["inputs"]["image"]
    }
    
    partial_outputs = model(partial_inputs)
    print(f"✅ Text + Image only:")
    print(f"   - Predictions shape: {partial_outputs['predictions'].shape}")
    
    # Only text
    text_only_inputs = {"text": data["inputs"]["text"]}
    text_only_outputs = model(text_only_inputs)
    print(f"✅ Text only:")
    print(f"   - Predictions shape: {text_only_outputs['predictions'].shape}")
    
    # Test different fusion methods
    print("\n🔀 Testing different fusion methods...")
    
    fusion_methods = ["concat", "attention", "sum"]
    for fusion_method in fusion_methods:
        print(f"\n   Testing {fusion_method} fusion...")
        test_model = MultiModalClassifier(
            num_classes=num_classes,
            fusion_method=fusion_method,
            hidden_dim=256
        )
        
        test_model.eval()
        with torch.no_grad():
            test_outputs = test_model(data["inputs"])
        
        print(f"   ✅ {fusion_method}: Output shape {test_outputs['predictions'].shape}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("- Replace synthetic data with real multi-modal datasets")
    print("- Add training loop with proper loss function")
    print("- Implement evaluation metrics")
    print("- Experiment with different architectures")


def demonstrate_modality_encoding():
    """
    Demonstrate individual modality encoding.
    """
    print("\n🔍 Modality Encoding Demo")
    print("=" * 30)
    
    model = MultiModalClassifier(num_classes=5)
    
    # Test individual modality encoding
    modalities = ["text", "image", "audio"]
    dims = [768, 2048, 512]
    
    for modality, dim in zip(modalities, dims):
        test_input = torch.randn(8, dim)  # Batch of 8 samples
        encoded = model.encode_modality(modality, test_input)
        print(f"📋 {modality.capitalize()}: {test_input.shape} → {encoded.shape}")
    
    # Parameter count analysis
    param_counts = model.get_parameter_count()
    print(f"\n📊 Model Parameters:")
    print(f"   - Total: {param_counts['total']:,}")
    print(f"   - Trainable: {param_counts['trainable']:,}")
    print(f"   - Frozen: {param_counts['frozen']:,}")


if __name__ == "__main__":
    print("🎯 Multi-Modal Models (MMM) - Basic Example")
    print("HSE 2025 AI Course")
    print("=" * 60)
    
    try:
        demonstrate_multimodal_classification()
        demonstrate_modality_encoding()
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Thank you for using MMM! 🚀")