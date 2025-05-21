"""
Image processing utilities for the clothing segmentation and outfit recommendation app.
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import uuid
from datetime import datetime

from app.config import CATEGORY_COLORS, IMAGE_SIZE, UPLOAD_DIR, SEGMENTED_DIR


def create_dirs():
    """Create necessary directories if they don't exist."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(SEGMENTED_DIR, exist_ok=True)


def preprocess_image(image_input, target_size=IMAGE_SIZE):
    """
    Preprocess image for model input.
    
    Args:
        image_input: Either a path to an image or a PIL Image object
        target_size: Size to resize image to
        
    Returns:
        Preprocessed tensor ready for model input
    """
    try:
        # Check if input is a PIL Image or a path
        if isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            # Load image from path
            image = Image.open(image_input).convert('RGB')
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Apply transforms
        tensor = transform(image)
        
        # Add batch dimension
        return tensor.unsqueeze(0)
    except Exception as e:
        print(f"Error preprocessing image {image_input}: {str(e)}")
        raise


def postprocess_output(output, original_image_path, category_idx=None):
    """
    Process model output to create a segmentation mask.
    
    Args:
        output: Model output tensor
        original_image_path: Path to original image
        category_idx: Index of category to segment (if None, all categories)
        
    Returns:
        Path to segmented image
    """
    # Get predictions
    logits = output.logits if hasattr(output, 'logits') else output
    predictions = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    
    # Get original image
    original_image = cv2.imread(original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_height, original_width = original_image.shape[:2]
    
    # Resize predictions to original image size
    predictions_resized = cv2.resize(
        predictions.astype(np.uint8), 
        (original_width, original_height), 
        interpolation=cv2.INTER_NEAREST
    )
    
    # Create segmentation overlay
    segmentation = np.zeros_like(original_image)
    
    # If specific category requested, only show that one
    if category_idx is not None:
        mask = predictions_resized == category_idx
        color = CATEGORY_COLORS.get(list(CATEGORY_COLORS.keys())[category_idx-1], (255, 255, 255))
        segmentation[mask] = color
    else:
        # Otherwise show all categories
        for category_id, color in enumerate(CATEGORY_COLORS.values(), start=1):
            mask = predictions_resized == category_id
            segmentation[mask] = color
    
    # Create alpha blend
    alpha = 0.5
    blended = cv2.addWeighted(original_image, 1-alpha, segmentation, alpha, 0)
    
    # Save result
    filename = f"segmented_{os.path.basename(original_image_path)}"
    output_path = os.path.join(SEGMENTED_DIR, filename)
    plt.figure(figsize=(10, 10))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    return output_path


def extract_clothing_items(output, original_image_path):
    """
    Extract individual clothing items from segmentation mask.
    
    Args:
        output: Model output tensor
        original_image_path: Path to original image
        
    Returns:
        List of paths to extracted clothing items
    """
    # Get predictions
    logits = output.logits if hasattr(output, 'logits') else output
    predictions = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    
    # Get original image
    original_image = cv2.imread(original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_height, original_width = original_image.shape[:2]
    
    # Resize predictions to original image size
    predictions_resized = cv2.resize(
        predictions.astype(np.uint8), 
        (original_width, original_height), 
        interpolation=cv2.INTER_NEAREST
    )
    
    extracted_paths = []
    
    # Skip background class (index 0)
    for category_id, category_name in enumerate(CATEGORY_COLORS.keys(), start=1):
        mask = predictions_resized == category_id
        
        # Skip if no pixels for this category
        if not np.any(mask):
            continue
            
        # Create masked image (clothing item on transparent background)
        rgba = np.zeros((original_height, original_width, 4), dtype=np.uint8)
        rgba[..., :3] = original_image
        rgba[..., 3] = 255  # Full opacity
        
        # Make background transparent
        rgba[~mask, 3] = 0
        
        # Create a unique filename
        item_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{category_name}_{timestamp}_{item_id}.png"
        output_path = os.path.join(SEGMENTED_DIR, filename)
        
        # Save the extracted item
        Image.fromarray(rgba).save(output_path)
        extracted_paths.append((output_path, category_name))
    
    return extracted_paths


def save_upload(uploaded_file):
    """
    Save uploaded file to the uploads directory.
    
    Args:
        uploaded_file: Uploaded file object
        
    Returns:
        Path to saved file
    """
    create_dirs()
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"upload_{timestamp}_{unique_id}.jpg"
    
    # Save the file
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file)
    
    return file_path 