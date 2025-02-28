import os
import pickle
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load extracted features
FEATURES_PATH = "static/features.pkl"

if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError("features.pkl not found! Run train_model.py first.")

with open(FEATURES_PATH, "rb") as f:
    feature_vectors, image_paths = pickle.load(f)

# Load model
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return model.predict(img_array)[0]

def find_similar_images(query_img_path):
    query_features = extract_features(query_img_path).reshape(1, -1)
    similarities = cosine_similarity(query_features, feature_vectors)[0]
    indices = np.argsort(similarities)[::-1][:4]  # Get top 4 similar images
    return [image_paths[idx] for idx in indices]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded"

        file = request.files["file"]
        if file.filename == "":
            return "No file selected"

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        similar_images = find_similar_images(filepath)
        return render_template("index.html", uploaded_image=filepath, results=similar_images)

    return render_template("index.html", uploaded_image=None, results=None)

if __name__ == "__main__":
    app.run(debug=True)
