import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import io
# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# from PIL import Image

from flask import Flask, request, jsonify

arial_model = keras.models.load_model("optical-character-recognition/saved_models/Arial_Model/best_accuracy")
nik_model = keras.models.load_model("optical-character-recognition/saved_models/NIK_Model/best_accuracy")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        try:
            image_bytes = file.read()
            print(image_bytes)
            # pillow_img = Image.open(io.BytesIO(image_bytes)).convert('L')
            # tensor = transform_image(pillow_img)
            # prediction = predict(tensor)
            # data = {"prediction": int(prediction)}
            # return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"


if __name__ == "__main__":
    app.run(debug=True)