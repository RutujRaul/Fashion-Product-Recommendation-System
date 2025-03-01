import os
import pickle
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

with open("static/features.pkl", "rb") as f:
    features_dict = pickle.load(f)

if not features_dict:
    raise ValueError("Feature vectors are empty! Ensure features.pkl is correctly generated.")


image_filenames = list(features_dict.keys())
feature_vectors = np.array(list(features_dict.values()))


model = ResNet50(weights="imagenet", include_top=False, pooling="avg")


def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()


def find_similar_images(query_image_path):
   
    query_features = extract_features(query_image_path, model)

    if query_features is None or len(query_features) == 0:
        raise ValueError("Could not extract features from query image.")

    similarities = cosine_similarity([query_features], feature_vectors)[0]
    top_indices = np.argsort(similarities)[-4:][::-1]

    similar_images = []
    for idx in top_indices:
        image_filename = image_filenames[idx]
        image_path = os.path.join("static", "images", image_filename)  # Ensure correct path
        similar_images.append(image_path)

    return similar_images
