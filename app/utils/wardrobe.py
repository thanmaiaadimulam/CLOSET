"""
Wardrobe management utilities for the clothing segmentation and recommendation app.
"""

import os
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from app.config import WARDROBE_DIR


class Wardrobe:
    """Manages the user's clothing inventory."""
    
    def __init__(self):
        """Initialize the wardrobe."""
        os.makedirs(WARDROBE_DIR, exist_ok=True)
        self.metadata_file = os.path.join(WARDROBE_DIR, "metadata.json")
        self.items = self._load_metadata()
        
    def _load_metadata(self):
        """Load wardrobe metadata from file."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"items": []}
    
    def _save_metadata(self):
        """Save wardrobe metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.items, f, indent=2)
    
    def add_item(self, image_path, category, attributes=None):
        """
        Add a clothing item to the wardrobe.
        
        Args:
            image_path: Path to the image of the clothing item
            category: Category of the clothing item
            attributes: Additional attributes (color, style, etc.)
            
        Returns:
            Item ID
        """
        # Generate unique ID for the item
        item_id = str(uuid.uuid4())
        
        # Create destination path
        filename = Path(image_path).name
        dest_path = os.path.join(WARDROBE_DIR, filename)
        
        # Copy image to wardrobe directory
        shutil.copy(image_path, dest_path)
        
        # Create item metadata
        if attributes is None:
            attributes = {}
            
        item = {
            "id": item_id,
            "category": category,
            "image_path": dest_path,
            "attributes": attributes,
            "date_added": datetime.now().isoformat()
        }
        
        # Add to wardrobe
        self.items["items"].append(item)
        self._save_metadata()
        
        return item_id
    
    def remove_item(self, item_id):
        """
        Remove a clothing item from the wardrobe.
        
        Args:
            item_id: ID of the item to remove
            
        Returns:
            True if successful, False otherwise
        """
        # Find the item
        for i, item in enumerate(self.items["items"]):
            if item["id"] == item_id:
                # Delete the image file
                if os.path.exists(item["image_path"]):
                    os.remove(item["image_path"])
                
                # Remove from metadata
                self.items["items"].pop(i)
                self._save_metadata()
                return True
                
        return False
    
    def get_item(self, item_id):
        """
        Get a clothing item by ID.
        
        Args:
            item_id: ID of the item to get
            
        Returns:
            Item metadata or None if not found
        """
        for item in self.items["items"]:
            if item["id"] == item_id:
                return item
        return None
    
    def get_all_items(self):
        """
        Get all clothing items in the wardrobe.
        
        Returns:
            List of all items
        """
        return self.items["items"]
    
    def get_items_by_category(self, category):
        """
        Get clothing items by category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of items in the category
        """
        return [item for item in self.items["items"] if item["category"] == category] 