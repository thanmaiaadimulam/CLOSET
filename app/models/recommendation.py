"""
Outfit recommendation module using content-based filtering approach.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from app.utils.wardrobe import Wardrobe


class FeatureExtractor:
    """Extract visual features from clothing images."""
    
    def __init__(self):
        """Initialize the feature extractor model."""
        # Load pre-trained ResNet model
        self.model = models.resnet50(weights='IMAGENET1K_V2')
        
        # Remove the last layer (classification layer)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def extract_features(self, image_path):
        """
        Extract features from an image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Feature vector (numpy array)
        """
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(image_tensor)
        
        # Convert to numpy array and flatten
        features = features.squeeze().cpu().numpy()
        
        return features


class OutfitRecommender:
    """Recommend outfits based on content-based filtering."""
    
    def __init__(self):
        """Initialize the outfit recommender."""
        self.feature_extractor = FeatureExtractor()
        self.wardrobe = Wardrobe()
        self.features_cache = {}
    
    def _get_item_features(self, item):
        """
        Get features for a clothing item, using cache if available.
        
        Args:
            item: Wardrobe item metadata
            
        Returns:
            Feature vector
        """
        item_id = item["id"]
        
        # Check if features are already in cache
        if item_id in self.features_cache:
            return self.features_cache[item_id]
        
        # Extract features
        image_path = item["image_path"]
        features = self.feature_extractor.extract_features(image_path)
        
        # Cache features
        self.features_cache[item_id] = features
        
        return features
    
    def get_all_item_features(self):
        """
        Extract features for all items in the wardrobe.
        
        Returns:
            Dictionary mapping item IDs to feature vectors
        """
        items = self.wardrobe.get_all_items()
        
        for item in items:
            item_id = item["id"]
            if item_id not in self.features_cache:
                self._get_item_features(item)
        
        return self.features_cache
    
    def find_similar_items(self, item_id, top_n=5):
        """
        Find items that are visually similar to a given item.
        
        Args:
            item_id: ID of the reference item
            top_n: Number of similar items to return
            
        Returns:
            List of similar items with similarity scores
        """
        # Get the reference item and its features
        reference_item = self.wardrobe.get_item(item_id)
        if not reference_item:
            return []
        
        reference_features = self._get_item_features(reference_item)
        reference_category = reference_item["category"]
        
        # Get all items of the same category
        category_items = self.wardrobe.get_items_by_category(reference_category)
        
        # Calculate similarities
        similarities = []
        for item in category_items:
            if item["id"] == item_id:
                continue  # Skip the reference item
            
            item_features = self._get_item_features(item)
            similarity = cosine_similarity([reference_features], [item_features])[0][0]
            similarities.append((item, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N similar items
        return similarities[:top_n]
    
    def recommend_outfit(self, base_item_id=None, occasion=None, top_n=3):
        """
        Recommend a complete outfit based on a base item.
        
        Args:
            base_item_id: ID of the base item (optional)
            occasion: Occasion for the outfit (optional)
            top_n: Number of outfit recommendations to return
            
        Returns:
            List of recommended outfits, each containing a list of items
        """
        # Get all items in the wardrobe
        all_items = self.wardrobe.get_all_items()
        if not all_items:
            return []
        
        # If no base item specified, pick a random top or dress
        if not base_item_id:
            tops = self.wardrobe.get_items_by_category("top")
            dresses = self.wardrobe.get_items_by_category("dress")
            
            if tops:
                base_item = np.random.choice(tops)
                base_item_id = base_item["id"]
            elif dresses:
                base_item = np.random.choice(dresses)
                base_item_id = base_item["id"]
            else:
                # Pick any random item
                base_item = np.random.choice(all_items)
                base_item_id = base_item["id"]
        
        # Get the base item
        base_item = self.wardrobe.get_item(base_item_id)
        if not base_item:
            return []
        
        base_category = base_item["category"]
        base_features = self._get_item_features(base_item)
        
        # Generate outfits based on the base item
        if base_category == "dress":
            # For dresses, just add accessories, shoes, etc.
            outfit_categories = ["footwear", "bag", "headwear", "neckwear"]
        else:
            # For tops, add bottoms and accessories
            outfit_categories = ["pants", "skirt", "footwear", "bag", "headwear", "neckwear"]
        
        # Prepare potential outfit components
        outfit_components = {}
        for category in outfit_categories:
            items = self.wardrobe.get_items_by_category(category)
            if items:
                outfit_components[category] = []
                for item in items:
                    item_features = self._get_item_features(item)
                    similarity = cosine_similarity([base_features], [item_features])[0][0]
                    outfit_components[category].append((item, similarity))
                
                # Sort by compatibility
                outfit_components[category].sort(key=lambda x: x[1], reverse=True)
        
        # Generate outfits
        outfits = []
        for _ in range(top_n):
            outfit = [base_item]
            
            # Add one item from each category, with some randomness
            for category, items in outfit_components.items():
                if not items:
                    continue
                
                # Take from top 3 items with some randomness
                top_items = items[:3]
                weights = np.array([item[1] for item in top_items])
                weights = weights / np.sum(weights)  # Normalize
                
                try:
                    selected_idx = np.random.choice(len(top_items), p=weights)
                    outfit.append(top_items[selected_idx][0])
                except:
                    # Fallback if weights are invalid
                    if top_items:
                        outfit.append(top_items[0][0])
            
            outfits.append(outfit)
        
        return outfits 