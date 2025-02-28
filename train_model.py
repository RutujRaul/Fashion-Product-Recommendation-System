import os
import pickle
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

# Load MobileNetV2 model
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# Define dataset folder
DATASET_FOLDER = "static/images"
features = []
image_paths = []

for img_name in os.listdir(DATASET_FOLDER):
    img_path = os.path.join(DATASET_FOLDER, img_name)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    feature_vector = model.predict(img_array)[0]
    features.append(feature_vector)
    image_paths.append(img_path)

# Save extracted features
with open("static/features.pkl", "wb") as f:
    pickle.dump((np.array(features), image_paths), f)

print("Feature extraction completed. Saved to static/features.pkl")
