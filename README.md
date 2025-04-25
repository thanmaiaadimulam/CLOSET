# CLOSET a Fashion Recommendation System MVP

A simple MVP to get things started with fashion item classification and outfit recommendations.

## What it does

- Classifies uploaded fashion images into categories (shirts, pants, shoes, etc.)
- Detects dominant colors in fashion items
- Recommends complementary items from your virtual closet
- Maintains a personal fashion database to improve recommendations over time

## Quick Start

```bash
# Install dependencies
pip install streamlit tensorflow pillow numpy sklearn matplotlib pandas

# Run the application
streamlit run app.py
```

## Features

- **Image Classification**: Upload and classify fashion items
- **Recommendations**: Get outfit suggestions based on uploaded items
- **Virtual Closet**: Save items to build a personalized collection
- **Closet Analytics**: View statistics about your fashion collection

## Tech Stack

- Streamlit for UI
- TensorFlow/Keras for image processing
- SQLite for data storage
- MobileNetV2 for feature extraction

## Note

This is a minimal implementation to demonstrate the concept. The classification model is simulated and would be replaced with a properly trained model in production.
