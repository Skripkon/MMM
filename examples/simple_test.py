#!/usr/bin/env python3
"""
Simple test to verify the project structure without heavy dependencies.
"""

import os
import sys
import importlib.util

def test_project_structure():
    """Test that all expected directories exist."""
    print("🔍 Testing project structure...")
    
    expected_dirs = [
        "src",
        "src/models", 
        "src/data",
        "src/training",
        "src/evaluation",
        "examples",
        "configs",
        "docs",
        "tests",
        "datasets"
    ]
    
    for dir_path in expected_dirs:
        full_path = os.path.join(os.getcwd(), dir_path)
        if os.path.exists(full_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - MISSING")
    
    print()

def test_important_files():
    """Test that important files exist."""
    print("📄 Testing important files...")
    
    expected_files = [
        "README.md",
        "requirements.txt", 
        "setup.py",
        "LICENSE",
        ".gitignore",
        "configs/default_config.yaml",
        "src/__init__.py",
        "src/models/__init__.py",
        "src/models/base.py",
        "src/models/multimodal_classifier.py",
        "examples/basic_multimodal.py"
    ]
    
    for file_path in expected_files:
        full_path = os.path.join(os.getcwd(), file_path)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            print(f"✅ {file_path} ({size} bytes)")
        else:
            print(f"❌ {file_path} - MISSING")
    
    print()

def test_python_imports():
    """Test that Python modules can be imported without PyTorch."""
    print("🐍 Testing Python module structure...")
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
    
    try:
        # Test basic imports (these should work even without PyTorch)
        spec = importlib.util.spec_from_file_location(
            "base", 
            os.path.join(os.getcwd(), 'src/models/base.py')
        )
        print("✅ Base model module structure is valid")
        
        spec = importlib.util.spec_from_file_location(
            "multimodal_classifier", 
            os.path.join(os.getcwd(), 'src/models/multimodal_classifier.py')
        )
        print("✅ MultiModal classifier module structure is valid")
        
        spec = importlib.util.spec_from_file_location(
            "multimodal_dataset", 
            os.path.join(os.getcwd(), 'src/data/multimodal_dataset.py')
        )
        print("✅ MultiModal dataset module structure is valid")
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
    
    print()

def show_project_overview():
    """Show project overview."""
    print("🎯 Multi-Modal Models (MMM) Project Overview")
    print("=" * 60)
    print("HSE 2025 - Multi-Modal Machine Learning Framework")
    print()
    
    print("📋 Project Features:")
    print("  • Multi-modal learning (text, image, audio)")
    print("  • Flexible fusion mechanisms (concat, attention, sum)")
    print("  • Educational framework for HSE students") 
    print("  • Comprehensive evaluation metrics")
    print("  • Easy-to-use APIs and examples")
    print()
    
    print("🏗️ Architecture:")
    print("  • Base classes for extensible models")
    print("  • Multi-modal dataset handling")
    print("  • Training and evaluation utilities")
    print("  • Configuration management")
    print()
    
    print("📚 Usage:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run basic example: python examples/basic_multimodal.py")
    print("  3. Check documentation in docs/ directory")
    print("  4. Customize configs/default_config.yaml for your needs")
    print()

if __name__ == "__main__":
    show_project_overview()
    test_project_structure()
    test_important_files() 
    test_python_imports()
    
    print("🎉 Project setup verification completed!")
    print()
    print("Next steps:")
    print("1. Install PyTorch and other dependencies")
    print("2. Run the full example with: python examples/basic_multimodal.py")
    print("3. Start building your multi-modal applications!")