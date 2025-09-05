import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from PIL import Image

img_size = 128
lesions = {
    "akiec": "Actinic Keratoses and Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "nv": "Melanocytic Nevi (common moles)",
    "mel": "Melanoma",
    "vasc": "Vascular Lesions"
}

def preprocess_img(img_path):
    # Load and resize image
    img = Image.open(img_path)
    img = img.resize((img_size, img_size))

    # Conver to numpy array and normalise
    img_array = np.array(img)
    img_array = img_array / 255.0

    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# Main execution
if __name__ == "__main__":
    img_path = "test_imgs/bcc2.jpg"

    # Load model
    print("Loading the trained model...")
    try:
        model = load_model("model/skin_lesion_classifier_model.keras")
        print("Model loaded successfully.")
    except Exception as e:
        print("Error: Couldn't load the model.")
        print(e)
        exit()

    # Get label mapping
    print("Loading label mapping...")
    try:
        md = pd.read_csv("data/raw/skin-cancer-mnist-ham10000/HAM10000_metadata.csv")
        le = LabelEncoder()
        le.fit(md["dx"])
        print("Label mapping loaded successfully.")
    except Exception as e:
        print("Error: Couldn't load label mapping")
        print(e)
        exit()

    # Make a prediction based on given image
    if not os.path.exists(img_path):
        print("Error: Image not found.")
    else:
        print("Preprocessing image...")
        pp_img = preprocess_img(img_path)

        print("Making a prediction...")
        predictions = model.predict(pp_img)

        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_label = le.inverse_transform([predicted_class_idx])[0]

        print(f"\nThe model predicts the skin lesion is: {lesions[predicted_label]}")
