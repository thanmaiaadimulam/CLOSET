"""
Clothing segmentation module using pre-trained SegFormer model.
"""

import os
import torch
from transformers import SegformerForSemanticSegmentation
from huggingface_hub import hf_hub_download
from PIL import Image

from app.utils.image_processing import preprocess_image, postprocess_output, extract_clothing_items
from app.config import SEGFORMER_MODEL


class ClothingSegmenter:
    """Clothing segmentation model using SegFormer."""
    
    def __init__(self, model_name=SEGFORMER_MODEL):
        """
        Initialize the segmentation model.
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        print(f"Loading segmentation model {model_name} on {self.device}...")
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.")
    
    def segment_image(self, image_input, original_image_path=None, output_segmentation=True, extract_items=True):
        """
        Segment clothing items in an image.
        
        Args:
            image_input: Either a PIL Image object or a path to an image file
            original_image_path: Path to save the original image for visualization (can be None if image_input is a path)
            output_segmentation: Whether to output a segmentation visualization
            extract_items: Whether to extract individual clothing items
            
        Returns:
            Tuple of (segmentation_path, extracted_items)
        """
        # Determine the original image path for visualization
        if original_image_path is None:
            if not isinstance(image_input, Image.Image):
                original_image_path = image_input
            else:
                raise ValueError("When image_input is a PIL Image, original_image_path must be provided")
        
        # Preprocess image
        input_tensor = preprocess_image(image_input)
        input_tensor = input_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Process results
        segmentation_path = None
        extracted_items = None
        
        if output_segmentation:
            segmentation_path = postprocess_output(output, original_image_path)
            
        if extract_items:
            extracted_items = extract_clothing_items(output, original_image_path)
            
        return segmentation_path, extracted_items


class ClothingSegmenterU2NET:
    """Alternative clothing segmentation model using U2NET."""
    
    def __init__(self):
        """Initialize the U2NET segmentation model."""
        # This is a placeholder for the U2NET implementation
        # Would download the model from a repository or use a pre-trained model
        raise NotImplementedError("U2NET segmentation model is not implemented yet.")


def get_segmenter(model_type="segformer"):
    """
    Factory function to get the appropriate segmentation model.
    
    Args:
        model_type: Type of segmentation model to use
        
    Returns:
        Segmentation model
    """
    if model_type == "segformer":
        return ClothingSegmenter()
    elif model_type == "u2net":
        return ClothingSegmenterU2NET()
    else:
        raise ValueError(f"Unknown model type: {model_type}") 