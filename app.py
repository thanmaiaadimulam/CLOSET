#!/usr/bin/env python3
"""
Clothing Segmentation and Outfit Recommendation MVP
"""

import os
import gradio as gr
from pathlib import Path
import numpy as np
from PIL import Image
from datetime import datetime
import uuid

from app.models.segmentation import get_segmenter
from app.models.recommendation import OutfitRecommender
from app.utils.wardrobe import Wardrobe
from app.config import (
    UPLOAD_DIR, SEGMENTED_DIR, WARDROBE_DIR, CATEGORY_COLORS, CLOTHING_CATEGORIES
)


# Create necessary directories
for directory in [UPLOAD_DIR, SEGMENTED_DIR, WARDROBE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Initialize models and utilities
segmenter = None  # Lazy-load the segmentation model
recommender = OutfitRecommender()
wardrobe = Wardrobe()
output_gallery = None  # Will store the latest segmentation results


def load_segmentation_model():
    """Lazy-load the segmentation model."""
    global segmenter
    if segmenter is None:
        segmenter = get_segmenter("segformer")
    return segmenter


def process_image(image):
    """Process uploaded image through segmentation."""
    global output_gallery
    
    # Check if image was uploaded
    if image is None:
        output_gallery = None
        return None, "Please upload an image."
    
    # Save a copy of the image for visualization
    temp_image_path = None
    if isinstance(image, Image.Image):
        # Save a copy for visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"upload_{timestamp}_{unique_id}.jpg"
        save_path = os.path.join(UPLOAD_DIR, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)
        temp_image_path = save_path
    else:
        temp_image_path = image
    
    # Ensure segmentation model is loaded
    model = load_segmentation_model()
    
    # Process image
    try:
        # Pass the image directly to segment_image
        segmentation_path, extracted_items = model.segment_image(image, temp_image_path)
        
        # Format results
        if not extracted_items:
            result_images = [segmentation_path]
            output_gallery = result_images
            return result_images, "No clothing items detected in the image."
        
        result_images = [segmentation_path]
        for item_path, category in extracted_items:
            result_images.append(item_path)
        
        output_gallery = result_images
        message = f"Detected {len(extracted_items)} clothing items: " + ", ".join([cat for _, cat in extracted_items])
        
        return result_images, message
    except Exception as e:
        output_gallery = None
        return None, f"Error processing image: {str(e)}"


def add_to_wardrobe(item_index, category):
    """Add a clothing item to the wardrobe."""
    if item_index is None or item_index < 1:
        return "Please enter a valid item index (1 or higher)."
    
    # Convert to 0-based index
    selected_index = int(item_index) - 1
    
    if not category or category not in CLOTHING_CATEGORIES:
        return "Please select a valid category."
    
    # Get the latest segmentation results
    result_images = output_gallery
    
    if not result_images or selected_index >= len(result_images):
        return f"No valid item at index {item_index}. There are {len(result_images) if result_images else 0} items available."
    
    # Get the selected image path
    image_path = result_images[selected_index]
    
    if not image_path or not os.path.exists(image_path):
        return "Selected item not found."
        
    # Add to wardrobe
    item_id = wardrobe.add_item(image_path, category)
    
    return f"Added {category} to your wardrobe (ID: {item_id})."


def show_wardrobe():
    """Show all items in the wardrobe."""
    items = wardrobe.get_all_items()
    
    if not items:
        return [], "Your wardrobe is empty.", []
    
    image_paths = [item["image_path"] for item in items]
    categories = [item["category"] for item in items]
    
    message = f"You have {len(items)} items in your wardrobe."
    
    return image_paths, message, categories


def recommend_outfit(base_item_index=None):
    """Generate outfit recommendations."""
    items = wardrobe.get_all_items()
    
    if not items:
        return [], "Your wardrobe is empty. Add some items first."
    
    if len(items) < 2:
        return [], "You need at least 2 items in your wardrobe for outfit recommendations."
    
    # Choose a base item
    base_item_id = None
    if base_item_index is not None and 0 <= base_item_index < len(items):
        base_item_id = items[base_item_index]["id"]
    
    # Get recommendations
    outfits = recommender.recommend_outfit(base_item_id=base_item_id)
    
    if not outfits:
        return [], "Couldn't generate outfit recommendations. Try adding more diverse items to your wardrobe."
    
    # Format results
    outfit_images = []
    message = f"Generated {len(outfits)} outfit recommendations."
    
    for i, outfit in enumerate(outfits):
        outfit_images.extend([item["image_path"] for item in outfit])
        outfit_images.append(None)  # Separator between outfits
    
    return outfit_images[:-1], message  # Remove last separator


def create_gradio_interface():
    """Create a Gradio interface for the app."""
    global output_gallery
    
    with gr.Blocks(title="Clothing Segmentation and Outfit Recommendation") as demo:
        gr.Markdown("# Clothing Segmentation and Outfit Recommendation")
        gr.Markdown("""
        This application allows you to:
        1. Segment clothing items from images
        2. Add segmented items to your virtual wardrobe
        3. Get outfit recommendations based on your wardrobe
        """)
        
        with gr.Tab("Segment Clothing"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Upload Image", type="pil")
                    segment_btn = gr.Button("Segment Clothing")
                
                with gr.Column():
                    output_gallery = gr.Gallery(label="Segmentation Results", columns=2)
                    output_message = gr.Textbox(label="Results")
            
            segment_btn.click(
                process_image,
                inputs=[input_image],
                outputs=[output_gallery, output_message]
            )
            
            with gr.Row():
                wardrobe_category = gr.Dropdown(
                    choices=CLOTHING_CATEGORIES[1:],  # Skip background
                    label="Category"
                )
                item_index = gr.Number(value=1, label="Item Index (1 = segmentation, 2+ = items)", precision=0)
                add_btn = gr.Button("Add to Wardrobe")
            
            add_result = gr.Textbox(label="Add Result")
            
            add_btn.click(
                add_to_wardrobe,
                inputs=[item_index, wardrobe_category],
                outputs=[add_result]
            )
        
        with gr.Tab("My Wardrobe"):
            with gr.Row():
                refresh_btn = gr.Button("Refresh Wardrobe")
            
            with gr.Row():
                wardrobe_gallery = gr.Gallery(label="Your Wardrobe", columns=4)
                wardrobe_message = gr.Textbox(label="Wardrobe Status")
                wardrobe_categories = gr.State([])
            
            refresh_btn.click(
                show_wardrobe,
                inputs=[],
                outputs=[wardrobe_gallery, wardrobe_message, wardrobe_categories]
            )
        
        with gr.Tab("Outfit Recommendations"):
            with gr.Row():
                base_item_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    label="Base Item Index (leave at 0 for random selection)"
                )
                recommend_btn = gr.Button("Generate Outfit Recommendations")
            
            with gr.Row():
                outfit_gallery = gr.Gallery(label="Recommended Outfits", columns=4)
                outfit_message = gr.Textbox(label="Recommendation Results")
            
            recommend_btn.click(
                recommend_outfit,
                inputs=[base_item_slider],
                outputs=[outfit_gallery, outfit_message]
            )
        
        # Load wardrobe on startup
        demo.load(show_wardrobe, [], [wardrobe_gallery, wardrobe_message, wardrobe_categories])
        
    return demo


if __name__ == "__main__":
    # Create and launch the Gradio interface
    demo = create_gradio_interface()
    demo.launch(share=True) 