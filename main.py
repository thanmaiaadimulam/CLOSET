import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
import sqlite3
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd

# Constants
IMAGE_SIZE = (224, 224)
FASHION_CATEGORIES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

COLOR_MAPPING = {
    "red": [(150, 0, 0), (255, 100, 100)],
    "blue": [(0, 0, 150), (100, 100, 255)],
    "green": [(0, 150, 0), (100, 255, 100)],
    "yellow": [(150, 150, 0), (255, 255, 100)],
    "black": [(0, 0, 0), (50, 50, 50)],
    "white": [(200, 200, 200), (255, 255, 255)],
    "gray": [(100, 100, 100), (150, 150, 150)]
}

# Initialize database
def init_db():
    conn = sqlite3.connect('fashion_closet.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS closet
    (id INTEGER PRIMARY KEY, 
     category TEXT, 
     color TEXT,
     feature_vector BLOB,
     image_path TEXT)
    ''')
    conn.commit()
    conn.close()

# Classification Pipeline
class ClassificationPipeline:
    def __init__(self):
        # Load base model - MobileNetV2 for feature extraction
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
        
        # For a real MVP, you would fine-tune this on a fashion dataset
        # Here we use a simpler approach just for demonstration
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')  # 10 fashion categories
        ])
        
        # In a real MVP, load pre-trained weights here
        # self.classifier.load_weights('fashion_classifier_weights.h5')
    
    def preprocess_image(self, img):
        """Resize and preprocess image for the model"""
        img = img.resize(IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)
    
    def extract_features(self, preprocessed_img):
        """Extract features using the CNN backbone"""
        features = self.feature_extractor.predict(preprocessed_img)
        return features
    
    def classify(self, features):
        """Classify the image based on extracted features"""
        # In a real MVP, this would use the trained classifier
        # Here we're just using a placeholder that returns a random category
        # In production, you'd use: predictions = self.classifier.predict(features)
        flattened_features = tf.keras.layers.GlobalAveragePooling2D()(features)
        # Simulating a prediction for MVP demonstration
        prediction = np.random.rand(10)
        prediction = prediction / np.sum(prediction)  # Normalize to get probabilities
        
        category_idx = np.argmax(prediction)
        confidence = prediction[category_idx]
        
        return FASHION_CATEGORIES[category_idx], confidence
    
    def detect_color(self, img):
        """Detect the dominant color in the image"""
        img_array = np.array(img)
        # Convert to RGB if image has an alpha channel
        if img_array.shape[-1] == 4:
            img_array = img_array[:,:,:3]
        
        # Calculate the average color
        avg_color = np.mean(img_array, axis=(0, 1))
        
        # Find the closest color in our mapping
        closest_color = "unknown"
        min_distance = float('inf')
        
        for color_name, (color_min, color_max) in COLOR_MAPPING.items():
            # Check if average color falls within the defined range
            if all(color_min[i] <= avg_color[i] <= color_max[i] for i in range(3)):
                # Calculate Euclidean distance to color midpoint
                color_mid = [(color_min[i] + color_max[i])/2 for i in range(3)]
                distance = sum((avg_color[i] - color_mid[i])**2 for i in range(3))
                
                if distance < min_distance:
                    min_distance = distance
                    closest_color = color_name
        
        return closest_color
    
    def process_image(self, img):
        """Process an image through the entire classification pipeline"""
        preprocessed_img = self.preprocess_image(img)
        features = self.extract_features(preprocessed_img)
        category, confidence = self.classify(features)
        color = self.detect_color(img)
        
        return {
            'category': category,
            'confidence': confidence,
            'color': color,
            'features': features
        }

# Recommendation Pipeline
class RecommendationEngine:
    def __init__(self, db_path='fashion_closet.db'):
        self.db_path = db_path
    
    def _get_closet_items(self, exclude_id=None):
        """Get all items from user's closet"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        if exclude_id:
            c.execute("SELECT id, category, color, feature_vector, image_path FROM closet WHERE id != ?", (exclude_id,))
        else:
            c.execute("SELECT id, category, color, feature_vector, image_path FROM closet")
            
        items = []
        for item in c.fetchall():
            item_id, category, color, feature_vector, image_path = item
            # Convert the BLOB back to numpy array
            feature_vector = np.frombuffer(feature_vector, dtype=np.float32).reshape(7, 7, 1280)
            items.append({
                'id': item_id,
                'category': category,
                'color': color,
                'features': feature_vector,
                'image_path': image_path
            })
        
        conn.close()
        return items
    
    def get_complementary_categories(self, category):
        """Determine which categories complement the given category"""
        complementary_mapping = {
            "T-shirt/top": ["Trouser", "Coat", "Sneaker"],
            "Trouser": ["T-shirt/top", "Shirt", "Sneaker", "Ankle boot"],
            "Pullover": ["Trouser", "Sneaker"],
            "Dress": ["Sandal", "Coat"],
            "Coat": ["T-shirt/top", "Dress", "Trouser"],
            "Sandal": ["Dress", "Trouser"],
            "Shirt": ["Trouser", "Coat"],
            "Sneaker": ["T-shirt/top", "Trouser", "Pullover"],
            "Bag": ["Dress", "Coat", "T-shirt/top"],
            "Ankle boot": ["Trouser", "Coat"]
        }
        
        return complementary_mapping.get(category, [])
    
    def calculate_similarity(self, features1, features2):
        """Calculate similarity between two feature vectors"""
        # Flatten features for cosine similarity
        f1 = features1.reshape(1, -1)
        f2 = features2.reshape(1, -1)
        
        return cosine_similarity(f1, f2)[0][0]
    
    def recommend_items(self, item_data, num_recommendations=2):
        """Recommend items based on the given item"""
        if not os.path.exists(self.db_path):
            return []
            
        # Get complementary categories
        complementary_categories = self.get_complementary_categories(item_data['category'])
        
        # Get all items from closet
        closet_items = self._get_closet_items()
        
        if not closet_items:
            return []
        
        # Filter items by complementary categories
        complementary_items = [item for item in closet_items 
                              if item['category'] in complementary_categories]
        
        if not complementary_items:
            # If no complementary items, return any items
            complementary_items = closet_items
        
        # Calculate similarity scores
        for item in complementary_items:
            item['similarity'] = self.calculate_similarity(
                item_data['features'], item['features'])
        
        # Sort by similarity
        complementary_items.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top N recommendations
        return complementary_items[:num_recommendations]

    def save_to_closet(self, item_data, image_path):
        """Save an item to the user's closet"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Convert feature vector to BLOB
        feature_blob = item_data['features'].tobytes()
        
        c.execute("""
        INSERT INTO closet (category, color, feature_vector, image_path)
        VALUES (?, ?, ?, ?)
        """, (item_data['category'], item_data['color'], feature_blob, image_path))
        
        conn.commit()
        item_id = c.lastrowid
        conn.close()
        
        return item_id

# Streamlit UI
def main():
    st.set_page_config(page_title="Fashion Classifier & Recommender", layout="wide")
    
    st.title("Fashion Item Classifier & Recommender")
    
    # Initialize database
    init_db()
    
    # Initialize pipelines
    classifier = ClassificationPipeline()
    recommender = RecommendationEngine()
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Classify & Recommend", "My Closet"])
    
    with tab1:
        st.header("Upload a Fashion Item")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.subheader("Uploaded Image")
                img = Image.open(uploaded_file)
                st.image(img, width=300)
                
                # Process the image
                with st.spinner("Classifying image..."):
                    item_data = classifier.process_image(img)
                
                # Display classification results
                st.subheader("Classification Results")
                st.write(f"**Category:** {item_data['category']}")
                st.write(f"**Color:** {item_data['color']}")
                st.write(f"**Confidence:** {item_data['confidence']:.2f}")
                
                # Save button
                if st.button("Save to My Closet"):
                    # Save the image file
                    save_dir = "saved_items"
                    os.makedirs(save_dir, exist_ok=True)
                    img_path = os.path.join(save_dir, uploaded_file.name)
                    img.save(img_path)
                    
                    # Save to database
                    item_id = recommender.save_to_closet(item_data, img_path)
                    st.success(f"Item saved to your closet (ID: {item_id})!")
            
            with col2:
                # Get recommendations
                with st.spinner("Finding recommendations..."):
                    recommendations = recommender.recommend_items(item_data)
                
                if recommendations:
                    st.subheader("Recommended Items")
                    for i, rec in enumerate(recommendations):
                        st.write(f"**Item {i+1}:** {rec['category']} ({rec['color']})")
                        if os.path.exists(rec['image_path']):
                            rec_img = Image.open(rec['image_path'])
                            st.image(rec_img, width=200)
                        st.write(f"Similarity Score: {rec['similarity']:.2f}")
                        st.divider()
                else:
                    st.info("No recommendations available. Add more items to your closet!")
    
    with tab2:
        st.header("My Fashion Closet")
        
        # Get all closet items
        closet_items = recommender._get_closet_items()
        
        if not closet_items:
            st.info("Your closet is empty. Upload and save items to build your collection!")
        else:
            # Display closet stats
            categories = [item['category'] for item in closet_items]
            category_counts = pd.Series(categories).value_counts()
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Closet Statistics")
                st.write(f"**Total Items:** {len(closet_items)}")
                st.write("**Categories:**")
                for category, count in category_counts.items():
                    st.write(f"- {category}: {count}")
            
            with col2:
                # Create a pie chart of categories
                fig, ax = plt.subplots()
                ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%')
                ax.set_title('Closet Composition')
                st.pyplot(fig)
            
            # Display all items
            st.subheader("All Items")
            
            # Create a grid layout
            cols = st.columns(3)
            
            for i, item in enumerate(closet_items):
                with cols[i % 3]:
                    st.write(f"**{item['category']}** ({item['color']})")
                    if os.path.exists(item['image_path']):
                        item_img = Image.open(item['image_path'])
                        st.image(item_img, width=150)
                    else:
                        st.error("Image not found")

if __name__ == "__main__":
    main()
