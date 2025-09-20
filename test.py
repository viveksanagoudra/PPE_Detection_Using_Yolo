from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import supervision as sv
from concurrent.futures import ThreadPoolExecutor
import os
import logging
from threading import Lock

# Initialize logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# ------------------------------
# Model setup
# ------------------------------
# Place your model weights in a "models" folder, not inside templates
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")

try:
    model = YOLO(MODEL_PATH)
    logging.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None  # Fail gracefully

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
executor = ThreadPoolExecutor(max_workers=2)
processing_lock = Lock()


# ------------------------------
# Helper function
# ------------------------------
def process_image(img_np):
    input_size = (640, 480)
    resized_img = cv2.resize(img_np, input_size)

    results = model(resized_img, conf=0.6, iou=0.5)[0]

    # Rescale detections back
    scale_x = img_np.shape[1] / input_size[0]
    scale_y = img_np.shape[0] / input_size[1]
    detections = sv.Detections.from_ultralytics(results)
    detections.xyxy[:, [0, 2]] *= scale_x
    detections.xyxy[:, [1, 3]] *= scale_y

    annotated_image = bounding_box_annotator.annotate(scene=img_np, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    _, buffer = cv2.imencode('.jpg', annotated_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buffer).decode('utf-8')


# ------------------------------
# Routes
# ------------------------------
@app.route("/")
def home():
    # Make sure "templates/test1.html" exists (renamed, no spaces)
    return render_template("test1.html")


@app.route("/index")
def index():
    # Make sure "templates/index.html" exists
    return render_template("index.html")


@app.route("/process_frame", methods=["POST"])
def process_frame():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        if processing_lock.locked():
            return jsonify({"image": None})

        data = request.get_json()
        image_data = data.get("image")
        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        img_str = base64.b64decode(image_data.split(",")[1])
        img_np = cv2.imdecode(np.frombuffer(img_str, np.uint8), cv2.IMREAD_COLOR)

        with processing_lock:
            annotated_image_str = process_image(img_np)

        return jsonify({"image": f"data:image/jpeg;base64,{annotated_image_str}"})
    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return jsonify({"error": str(e)}), 500


# ------------------------------
# Gunicorn entrypoint (no app.run here)
# ------------------------------
# Render/Heroku will use: gunicorn test:app --bind 0.0.0.0:$PORT
