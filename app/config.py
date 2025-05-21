"""
Configuration file for the clothing segmentation and outfit recommendation app.
"""

# Models
SEGFORMER_MODEL = "mattmdjaga/segformer_b2_clothes"  # Pre-trained SegFormer model from the PDF
U2NET_MODEL = "u2net_cloth_seg"  # Alternative model

# Processing
IMAGE_SIZE = 512  # Input image size for segmentation
CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for segmentation
MAX_ITEMS = 100  # Maximum number of clothing items to store

# Categories
CLOTHING_CATEGORIES = [
    "background",
    "top",
    "outer",
    "skirt",
    "dress",
    "pants",
    "leggings",
    "headwear",
    "eyeglass",
    "neckwear",
    "belt",
    "footwear",
    "bag",
    "hair",
    "face",
    "skin", 
    "ring",
    "wrist wearing",
]

# Colors for visualization
CATEGORY_COLORS = {
    "top": (255, 0, 0),       # Red
    "outer": (0, 255, 0),     # Green
    "skirt": (0, 0, 255),     # Blue
    "dress": (255, 255, 0),   # Yellow
    "pants": (255, 0, 255),   # Magenta
    "leggings": (0, 255, 255),# Cyan
    "headwear": (128, 0, 0),  # Dark Red
    "footwear": (0, 0, 128),  # Dark Blue
    "bag": (128, 128, 0),     # Olive
}

# Paths
TEMP_DIR = "app/static/temp"
UPLOAD_DIR = "app/static/uploads"
SEGMENTED_DIR = "app/static/segmented"
WARDROBE_DIR = "app/static/wardrobe" 