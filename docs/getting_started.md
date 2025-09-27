# Getting Started with Multi-Modal Models (MMM)

Welcome to the Multi-Modal Models framework for HSE 2025! This guide will help you get started with building and training multi-modal machine learning models.

## Quick Installation

```bash
# Clone the repository
git clone https://github.com/Skripkon/MMM.git
cd MMM

# Create a virtual environment (recommended)
python -m venv mmm_env
source mmm_env/bin/activate  # On Windows: mmm_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## First Steps

### 1. Run the Basic Example

```bash
python examples/basic_multimodal.py
```

This will demonstrate:
- Creating a multi-modal classifier
- Generating synthetic data
- Running forward passes
- Testing different fusion methods
- Handling missing modalities

### 2. Understanding the Architecture

The MMM framework is built around several key components:

#### Base Model (`src/models/base.py`)
- Abstract base class for all multi-modal models
- Provides common functionality like saving/loading, parameter counting
- Defines the interface that all models must implement

#### Multi-Modal Classifier (`src/models/multimodal_classifier.py`)
- Concrete implementation for classification tasks
- Supports text, image, and audio inputs
- Multiple fusion strategies (concatenation, attention, sum)
- Handles missing modalities gracefully

#### Dataset Handling (`src/data/multimodal_dataset.py`)
- PyTorch Dataset for multi-modal data
- Flexible data loading from various sources
- Built-in data validation and splitting

### 3. Basic Usage Example

```python
import torch
from mmm.models import MultiModalClassifier
from mmm.data import MultiModalDataset

# Create a model
model = MultiModalClassifier(
    num_classes=5,
    text_dim=768,      # BERT embeddings
    image_dim=2048,    # ResNet features
    audio_dim=512,     # Audio features
    fusion_method="attention"
)

# Prepare your data
inputs = {
    "text": torch.randn(32, 768),    # Batch of text features
    "image": torch.randn(32, 2048),  # Batch of image features
    "audio": torch.randn(32, 512)    # Batch of audio features
}

# Forward pass
outputs = model(inputs)
predictions = outputs["predictions"]  # Shape: [32, 5]
```

## Working with Real Data

### Text Data
The framework expects pre-processed text embeddings. You can use:
- BERT/RoBERTa embeddings (768-dim)
- Sentence-BERT embeddings
- Custom text encoders

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')

# Process text
text = "Your input text here"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
text_embeddings = bert_model(**inputs).last_hidden_state.mean(dim=1)
```

### Image Data
For images, extract features using pre-trained CNN models:

```python
import torchvision.transforms as transforms
from torchvision.models import resnet50

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Feature extraction
resnet = resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()  # Remove final layer
resnet.eval()

# Process image
image_features = resnet(preprocessed_image)  # Shape: [batch, 2048]
```

### Audio Data
Audio processing typically involves extracting spectral features:

```python
import librosa
import numpy as np

def extract_audio_features(audio_path, sr=16000, duration=10):
    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr, duration=duration)
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    # Or extract mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    
    return np.mean(mel_spec, axis=1)  # Average over time
```

## Configuration

The framework uses YAML configuration files for easy experimentation:

```yaml
# configs/my_experiment.yaml
model:
  name: "MultiModalClassifier"
  num_classes: 10
  fusion_method: "attention"
  hidden_dim: 512

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 50

data:
  train_data_path: "data/train.json"
  val_data_path: "data/val.json"
```

Load and use configurations:

```python
import yaml

with open('configs/my_experiment.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = MultiModalClassifier(**config['model'])
```

## Next Steps

1. **Explore Examples**: Check out more examples in the `examples/` directory
2. **Read Documentation**: Browse the `docs/` directory for detailed guides
3. **Try Different Fusion Methods**: Experiment with "concat", "attention", and "sum"
4. **Custom Models**: Extend the `BaseMultiModalModel` for your specific needs
5. **Real Datasets**: Apply the framework to your own multi-modal datasets

## Common Issues and Solutions

### Missing Dependencies
If you encounter import errors, ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### GPU Memory Issues
For large models, consider:
- Reducing batch size
- Using gradient checkpointing
- Mixed precision training

### Data Format Issues
Ensure your data follows the expected format:
- Text: Pre-computed embeddings as tensors
- Images: Preprocessed feature vectors
- Audio: Extracted feature representations

## Getting Help

- 📖 Check the documentation in `docs/`
- 🐛 Report issues on GitHub
- 💬 Ask questions in discussions
- 📧 Contact: ai-lab@hse.ru

Happy multi-modal learning! 🚀