import os

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load the pretrained MobileNetV2 model trained on ImageNet
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Set up upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Define allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Local Calorie Database
food_calories = {
    "apple": 95,
    "pomegranate": 100,
    "banana": 105,
    "orange": 62,
    "carrot": 41,
    "chicken breast": 165,
    "rice": 206,
    "egg": 68,
    "pizza": 285,
    "burger": 354,
    "salad": 150,
    "potato": 77,
}


def get_calories(food_name):
    return food_calories.get(food_name.lower(), "Calories not available for this food")


# Route for the main page
@app.route("/")
def index():
    return render_template("index.html")


# Route for uploading the image
@app.route("/upload", methods=["POST"])
def upload_image():
    """Uploading image."""

    if "file" not in request.files:
        return "No file part"

    file = request.files["file"]

    # If no file is selected
    if file.filename == "":
        return "No selected file"

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Process image and predict food
        img = image.load_img(
            filepath, target_size=(224, 224)
        )  # Resize image to 224x224
        img_array = image.img_to_array(img)  # Convert image to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            img_array
        )  # Preprocess image

        # Make prediction
        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(
            predictions, top=1
        )[0]

        # Get predicted class (first result)
        predicted_class = decoded_predictions[0][1]  # The label of the class

        # Get the calories of the predicted class from the local database
        calories = get_calories(predicted_class)

        # Return results to the front-end
        return render_template(
            "index.html",
            image_url=f"/{filepath}",
            prediction=predicted_class,
            calories=calories,
        )


if __name__ == "__main__":
    app.run(debug=True)
