# Clothing Segmentation and Outfit Recommendation App

This MVP application combines clothing segmentation and outfit recommendation using pre-trained deep learning models, as described in the research document.

## Features

1. **Clothing Segmentation**: Upload images and segment clothing items using the SegFormer model fine-tuned on clothing datasets.
2. **Wardrobe Management**: Add segmented clothing items to a virtual wardrobe.
3. **Outfit Recommendation**: Generate outfit recommendations based on clothing items in your wardrobe using a content-based approach.

## Example Clothing Segmentation
![segmented_upload_20250524_194552_b086f87e](https://github.com/user-attachments/assets/91af5f1c-8f65-4411-9c8c-9ae7fb428be0)



## Installation

1. Clone this repository:
```
git clone <repository-url>
cd <repository-directory>
```

2. Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

1. Run the application:
```
python app.py
```

2. Open the provided URL in your web browser (usually http://127.0.0.1:7860/).

3. Use the application:
   - **Segment Clothing**: Upload an image and click "Segment Clothing" to identify clothing items.
   - **Add to Wardrobe**: Select a category for a segmented item and add it to your wardrobe.
   - **View Wardrobe**: View all clothing items in your virtual wardrobe.
   - **Get Recommendations**: Generate outfit recommendations based on your wardrobe.

## Model Details

### Clothing Segmentation
- Uses the **SegFormer** transformer-based model (`mattmdjaga/segformer_b2_clothes`) fine-tuned on the ATR dataset.
- Can segment clothing into multiple categories including tops, bottoms, dresses, footwear, etc.

### Outfit Recommendation
- Uses a **content-based filtering** approach with visual features extracted using a pre-trained ResNet50 model.
- Calculates outfit compatibility based on feature similarity between clothing items.

## Dataset

This implementation doesn't require training, as it uses pre-trained models. However, the models were trained on datasets including:
- ATR (Atrous Textile Recognition)
- DeepFashion2
- Fashionpedia

## Directory Structure

```
app/
├── models/
│   ├── segmentation.py  # Clothing segmentation model
│   └── recommendation.py  # Outfit recommendation model
├── utils/
│   ├── image_processing.py  # Image processing utilities
│   └── wardrobe.py  # Wardrobe management utilities
├── static/
│   ├── uploads/  # Uploaded images
│   ├── segmented/  # Segmented clothing items
│   └── wardrobe/  # Wardrobe items
└── config.py  # Configuration settings

app.py  # Main application
requirements.txt  # Dependencies
```

## References

- SegFormer model: https://huggingface.co/mattmdjaga/segformer_b2_clothes
- Implementation based on research in clothing segmentation and outfit recommendation 
