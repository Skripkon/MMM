# Multimodal Art Retrieval System

## Demo

<p align="center">
  <img src="demo_screenshots/demo1.png" alt="Demo 1" width="45%"/>
  <img src="demo_screenshots/demo2.jpeg" alt="Demo 2" width="45%"/>
</p>

⚠️ **HuggingFace Deployment Note**: Unable to deploy the demo on HuggingFace Spaces due to the 50GB storage constraint. The model files and FAISS indices exceed this limit.

<img src="demo_screenshots/cant_deploy.png" alt="Deployment Issue" width="300"/>

---

## Tasks Overview

### ✅ 1. Image→Image Retrieval (3 points) (CLIP)

### ✅ 2. Caption→Image Retrieval (3 points) (BLIP)

### ✅ 3. Omni→Image Retrieval (4 points) (CLIP + BLIP + LLaVa + deterministic filters)

### ✅ Bonus: Interactive Demo (3 points) (Gradio)
---

## Dataset
WikiArt dataset from HuggingFace (all 81k+ artworks)

## Implementation
See `solution.ipynb` for full implementation details and results visualization.

