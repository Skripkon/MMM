"""
Gradio App for Multimodal Image Retrieval
This app provides an interface to search for similar artwork using:
- Query Image (visual similarity)
- Query Text (caption similarity)
- Genre filter
- Tags filter
"""

import os
import pickle
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import faiss
import gradio as gr
from PIL import Image
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer


# ============================================================================
# Configuration
# ============================================================================
MODEL_NAME = "openai/clip-vit-base-patch32"
DATA_DIR = "data"

# ============================================================================
# Load Models and Data
# ============================================================================
print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained(MODEL_NAME)
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=True)
clip_tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

print("Loading dataset...")
dataset = load_dataset("huggan/wikiart", split="train")

# Genre mappings
genre_numeric_class2label = {
    i: label for i, label in enumerate(dataset.features["genre"].names)
}
genre_label2numeric = {
    label: i for i, label in genre_numeric_class2label.items()
}

print("Loading precomputed embeddings and indices...")
# Load image embeddings
with open(os.path.join(DATA_DIR, "image_embeddings.pkl"), "rb") as f:
    image_embeddings = pickle.load(f)

# Build FAISS index for images
dimension = image_embeddings.shape[1]
visual_index = faiss.IndexFlatIP(dimension)
visual_index.add(image_embeddings)

# Load caption embeddings
with open(os.path.join(DATA_DIR, "caption_embeddings.pkl"), "rb") as f:
    caption_embeddings = pickle.load(f)

# Build FAISS index for captions
caption_index = faiss.IndexFlatIP(caption_embeddings.shape[1])
caption_index.add(caption_embeddings.astype('float32'))

# Load genre to image IDs mapping
with open(os.path.join(DATA_DIR, "genre_to_image_ids.pkl"), "rb") as f:
    numeric_genre_to_image_ids = pickle.load(f)

genre_to_image_ids = {
    genre_numeric_class2label[genre_class]: ids 
    for genre_class, ids in numeric_genre_to_image_ids.items()
}

# Load tags
df_tags = pd.read_csv(os.path.join(DATA_DIR, "image_tags.csv"))

print("Setup complete!")


# ============================================================================
# Core Functions
# ============================================================================
def encode_images(images: list[Image.Image]) -> np.ndarray:
    """Encodes images using CLIP model"""
    inputs = clip_processor(images=images, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()


def encode_texts(texts: list[str]) -> np.ndarray:
    """Encode texts using CLIP text encoder"""
    inputs = clip_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()


def find_close_images_by_visual_index(query_image: Image.Image, n_similar: int = 100):
    """Find n_similar closest images to the query image using FAISS index"""
    query_embedding = encode_images([query_image])
    scores, similar_indices = visual_index.search(query_embedding, n_similar + 1)
    similar_indices = similar_indices[0][1:]  # Remove first element (the query itself)
    return similar_indices


def find_close_images_by_caption_index(query_caption: str, n_similar: int = 100):
    """Find n_similar closest images to the query caption using FAISS index"""
    query_caption_emb = encode_texts([query_caption]).astype('float32')
    scores, similar_indices = caption_index.search(query_caption_emb, n_similar)
    similar_indices = similar_indices[0]
    return similar_indices


def find_images_by_genre(genre: str) -> list[int]:
    """Find images by genre"""
    return genre_to_image_ids.get(genre, [])


def multimodal_retrieval(
    query_image: Image.Image = None, 
    query_text: str = None, 
    genre: str = None, 
    tags: list[str] = None, 
    n: int = 10
):
    """
    Multimodal retrieval function that combines multiple search modalities:
    1. Visual search by query image
    2. Text search by query caption
    3. Filter by genre
    4. Filter by tags
    
    Returns: List of image indices
    """
    found_images = set()

    # 1. Visual search
    if query_image is not None:
        similar_indices = find_close_images_by_visual_index(query_image, 100)
        found_images.update(similar_indices)

    # 2. Caption/text search
    if query_text is not None and query_text.strip() != "":
        similar_indices = find_close_images_by_caption_index(query_text, 100)
        found_images.update(similar_indices)

    # If neither image nor text provided, start with all images
    if not found_images:
        found_images = set(range(len(dataset)))

    # 3. Filter by genre
    if genre is not None and genre.strip() != "" and genre != "All":
        genre_ids = set(find_images_by_genre(genre))
        found_images = found_images & genre_ids

    # 4. Filter by tags
    if tags is not None and len(tags) > 0:
        filtered = set()
        for idx in found_images:
            tag_row = df_tags[df_tags["idx"] == idx]
            if not tag_row.empty:
                img_tags = tag_row["tags"].values[0].split()
                if any(tag.lower().strip() in img_tags for tag in tags if tag.strip()):
                    filtered.add(idx)
        found_images = filtered

    # Return random sample of results
    found_list = list(found_images)
    if len(found_list) > n:
        return random.sample(found_list, n)
    return found_list


# ============================================================================
# Gradio Interface
# ============================================================================
def gradio_multimodal_retrieval(
    query_image,
    query_text,
    genre,
    tags_text,
    num_results
):
    """
    Wrapper function for Gradio interface
    """
    # Parse tags from comma-separated string
    tags = [tag.strip() for tag in tags_text.split(",")] if tags_text else []
    tags = [tag for tag in tags if tag]  # Remove empty strings
    
    # Get results
    image_ids = multimodal_retrieval(
        query_image=query_image,
        query_text=query_text,
        genre=genre if genre != "All" else None,
        tags=tags if tags else None,
        n=num_results
    )
    
    # Convert to images for display
    result_images = [dataset[idx]["image"] for idx in image_ids]
    
    # Create info text
    info = f"Found {len(image_ids)} images"
    if not image_ids:
        info = "No images found matching your criteria. Try adjusting your search parameters."
    
    return result_images, info


# Build genre dropdown options
genre_options = ["All"] + sorted(list(genre_to_image_ids.keys()))

# Create Gradio interface
with gr.Blocks(title="Multimodal Art Search", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎨 Multimodal Art Retrieval System
    
    Search for similar artworks from the WikiArt dataset using multiple modalities:
    - **Image**: Upload an image to find visually similar artworks
    - **Text**: Enter a description to find matching artworks
    - **Genre**: Filter by art genre
    - **Tags**: Filter by comma-separated tags (e.g., "river, tree, mountain")
    
    You can use any combination of these search methods!
    
    This system uses:
    - **CLIP** (openai/clip-vit-base-patch32) for image and text encoding
    - **BLIP** for automatic image captioning
    - **LLaVA** for zero-shot tag generation
    - **FAISS** for fast similarity search
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            query_image = gr.Image(
                type="pil", 
                label="Query Image (optional)",
                height=300
            )
            query_text = gr.Textbox(
                label="Text Query (optional)",
                placeholder="e.g., 'a dark forest at night'",
                lines=2
            )
            genre = gr.Dropdown(
                choices=genre_options,
                label="Genre Filter (optional)",
                value="All"
            )
            tags_text = gr.Textbox(
                label="Tags (optional, comma-separated)",
                placeholder="e.g., river, tree, mountain",
                lines=1
            )
            num_results = gr.Slider(
                minimum=1,
                maximum=20,
                value=10,
                step=1,
                label="Number of Results"
            )
            search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            info_text = gr.Textbox(label="Search Info", interactive=False)
            result_gallery = gr.Gallery(
                label="Search Results",
                columns=5,
                rows=2,
                height="auto",
                object_fit="contain"
            )
    
    # Example searches
    gr.Markdown("### 💡 Example Searches")
    gr.Examples(
        examples=[
            [None, "a dark night", "landscape", "river, tree", 10],
            [None, "portrait of a woman", "portrait", "", 10],
            [None, "abstract geometric shapes", "abstract", "", 10],
            [None, "bright sunny day", "landscape", "sky, sun", 10],
        ],
        inputs=[query_image, query_text, genre, tags_text, num_results],
    )
    
    gr.Markdown("""
    ---
    ### About
    This is a homework project for the Multimodal Machine Learning course.
    Dataset: [WikiArt](https://huggingface.co/datasets/huggan/wikiart) (81,444 artworks)
    """)
    
    # Connect button to function
    search_btn.click(
        fn=gradio_multimodal_retrieval,
        inputs=[query_image, query_text, genre, tags_text, num_results],
        outputs=[result_gallery, info_text]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()

