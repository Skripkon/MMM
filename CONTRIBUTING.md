# Contributing to Multi-Modal Models (MMM)

Thank you for your interest in contributing to the MMM project! This document provides guidelines for contributing to the Multi-Modal Models framework for HSE 2025.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic knowledge of PyTorch and multi-modal machine learning

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/MMM.git
   cd MMM
   ```

2. **Create Development Environment**
   ```bash
   python -m venv dev_env
   source dev_env/bin/activate  # On Windows: dev_env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. **Install Development Tools**
   ```bash
   pip install pytest pytest-cov black flake8 mypy pre-commit
   pre-commit install
   ```

## Types of Contributions

We welcome various types of contributions:

### 🐛 Bug Reports
- Use GitHub Issues with the "bug" label
- Include Python version, OS, and error traceback
- Provide minimal code to reproduce the issue

### ✨ Feature Requests
- Use GitHub Issues with the "enhancement" label
- Describe the motivation and use case
- Provide example usage if possible

### 📝 Documentation
- Improve existing documentation
- Add new tutorials or examples
- Fix typos or clarify explanations

### 🔧 Code Contributions
- Bug fixes
- New features
- Performance improvements
- New model architectures

## Development Guidelines

### Code Style
We use Black for code formatting and follow PEP 8 guidelines:

```bash
# Format code
black src/ examples/ tests/

# Check style
flake8 src/ examples/ tests/

# Type checking
mypy src/
```

### Code Structure
- Follow the existing project structure
- Use type hints for all function signatures
- Write docstrings for all public methods
- Keep functions focused and well-named

### Example Code Style:
```python
def encode_modality(self, modality: str, data: torch.Tensor) -> torch.Tensor:
    """
    Encode a specific modality into a common representation space.
    
    Args:
        modality: Name of the modality ('text', 'image', 'audio', etc.)
        data: Input tensor for the specified modality
        
    Returns:
        Encoded representation tensor
        
    Raises:
        ValueError: If modality is not supported
    """
    if modality not in self.supported_modalities:
        raise ValueError(f"Unsupported modality: {modality}")
    
    return self.encoders[modality](data)
```

### Testing
- Write unit tests for all new functionality
- Ensure tests pass before submitting PR
- Aim for >80% code coverage

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Documentation
- Update docstrings for any changed functions
- Add examples for new features
- Update README.md if necessary

## Contribution Process

### 1. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

### 2. Make Changes
- Write code following our guidelines
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes
```bash
# Run all tests
pytest tests/

# Check code style
black --check src/ examples/ tests/
flake8 src/ examples/ tests/

# Run type checking
mypy src/
```

### 4. Commit Changes
Use clear, descriptive commit messages:
```bash
git commit -m "Add attention fusion mechanism for multi-modal classifier

- Implement MultiModalAttention class
- Add attention fusion option to MultiModalClassifier  
- Include unit tests and documentation
- Fixes #123"
```

### 5. Push and Create PR
```bash
git push origin feature/your-feature-name
```

Create a Pull Request on GitHub with:
- Clear title and description
- Link to related issues
- Screenshots/examples if applicable

## Pull Request Guidelines

### PR Template
When creating a PR, please include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Unit tests pass
- [ ] New tests added (if applicable)
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### Review Process
1. Automated checks (CI/CD) must pass
2. Code review by maintainers
3. Address feedback promptly
4. Squash commits if requested

## Specific Areas for Contribution

### High Priority
- 🔥 **New Model Architectures**: Vision-Language models, Audio-Visual models
- 🔥 **Training Utilities**: Advanced optimizers, learning rate schedulers
- 🔥 **Evaluation Metrics**: Cross-modal retrieval, attention visualization

### Medium Priority
- 📊 **Dataset Loaders**: Common multi-modal datasets
- 🎨 **Visualization Tools**: Attention maps, feature visualizations
- 📚 **Educational Content**: Tutorials, Jupyter notebooks

### Good First Issues
- 🐛 Bug fixes in existing code
- 📝 Documentation improvements
- ✅ Adding unit tests
- 🧹 Code cleanup and refactoring

## Educational Context (HSE 2025)

This project serves HSE students, so consider:
- **Clarity**: Code should be educational and well-documented
- **Examples**: Provide practical, real-world examples  
- **Difficulty Levels**: Mark contributions by difficulty (beginner/intermediate/advanced)
- **Learning Objectives**: Align with course learning goals

## Community Guidelines

### Be Respectful
- Use inclusive language
- Be constructive in feedback
- Help newcomers learn

### Be Patient
- Code review takes time
- Learning has a curve
- Everyone was a beginner once

### Be Collaborative
- Discuss ideas openly
- Share knowledge
- Credit contributors

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Acknowledged in academic publications (where applicable)

## Questions?

- 💬 **GitHub Discussions**: For general questions
- 🐛 **GitHub Issues**: For bug reports and feature requests
- 📧 **Email**: ai-lab@hse.ru for sensitive matters

## License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

---

Thank you for helping make MMM better for the HSE 2025 community! 🎓✨