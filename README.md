# Multi-Modal Models (MMM) | HSE 2025

A comprehensive framework for developing and deploying multi-modal machine learning models, created for the Higher School of Economics (HSE) 2025 curriculum.

## Overview

This project implements state-of-the-art multi-modal deep learning architectures that can process and understand multiple types of data simultaneously, including:

- **Text**: Natural language processing and understanding
- **Images**: Computer vision and image analysis
- **Audio**: Speech recognition and audio processing
- **Video**: Temporal visual understanding
- **Tabular Data**: Structured data analysis

## Key Features

- 🔄 **Cross-Modal Learning**: Models that learn representations across different modalities
- 🎯 **Task-Specific Architectures**: Specialized models for various multi-modal tasks
- 📊 **Evaluation Framework**: Comprehensive metrics for multi-modal model assessment
- 🛠️ **Easy Integration**: Simple APIs for incorporating multi-modal capabilities
- 📚 **Educational Resources**: Tutorials and examples for HSE students

## Supported Multi-Modal Tasks

1. **Vision-Language Understanding**
   - Image captioning
   - Visual question answering
   - Image-text retrieval

2. **Audio-Visual Processing**
   - Lip-reading
   - Audio-visual speech recognition
   - Video understanding with audio

3. **Text-Audio Integration**
   - Speech-to-text with context
   - Text-to-speech with emotion
   - Audio-text sentiment analysis

4. **Multi-Modal Classification**
   - Document analysis (text + images)
   - Social media content classification
   - Medical diagnosis (multiple data types)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Skripkon/MMM.git
cd MMM

# Install dependencies
pip install -r requirements.txt

# Run a simple multi-modal example
python examples/basic_multimodal.py
```

## Project Structure

```
MMM/
├── src/                    # Core implementation
│   ├── models/            # Multi-modal model architectures
│   ├── data/              # Data loading and preprocessing
│   ├── training/          # Training utilities
│   └── evaluation/        # Evaluation metrics and tools
├── examples/              # Example scripts and notebooks
├── datasets/              # Dataset utilities and loaders
├── docs/                  # Documentation
├── tests/                 # Unit tests
└── configs/               # Configuration files
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (optional, for GPU acceleration)

### Install from source
```bash
git clone https://github.com/Skripkon/MMM.git
cd MMM
pip install -e .
```

## Usage Examples

### Basic Multi-Modal Classification
```python
from mmm.models import MultiModalClassifier
from mmm.data import MultiModalDataset

# Load your multi-modal dataset
dataset = MultiModalDataset(
    text_data="path/to/text",
    image_data="path/to/images",
    labels="path/to/labels"
)

# Initialize model
model = MultiModalClassifier(
    text_dim=768,
    image_dim=2048,
    num_classes=10
)

# Train the model
model.fit(dataset)

# Make predictions
predictions = model.predict(new_data)
```

## Educational Context (HSE 2025)

This project is designed as a comprehensive learning resource for students in the HSE 2025 machine learning curriculum, covering:

- **Theoretical Foundations**: Understanding multi-modal learning principles
- **Practical Implementation**: Hands-on coding with real datasets
- **Research Applications**: Current trends in multi-modal AI
- **Industry Relevance**: Real-world applications and case studies

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{mmm2025,
  title={Multi-Modal Models Framework},
  author={HSE AI Lab},
  year={2025},
  url={https://github.com/Skripkon/MMM}
}
```

## Contact

For questions and support:
- 📧 Email: ai-lab@hse.ru
- 💬 Issues: [GitHub Issues](https://github.com/Skripkon/MMM/issues)
- 📖 Documentation: [Project Wiki](https://github.com/Skripkon/MMM/wiki)