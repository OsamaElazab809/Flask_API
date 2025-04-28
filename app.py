# Deploy on GitHup 
# git init
# git add .
# git commit -m "Initial commit"
# git config --global user.name "Osama Elazab"
# git config --global user.email "osamaelazab809@gmail.com"
# git branch -M main
# git remote add origin https://github.com/OsamaElazab809/flask.git // link of repositry
# git push -u origin main


from flask import Flask, request, jsonify
from ultralytics import YOLO
import easyocr
import base64
import cv2
import numpy as np


import torch
# Disable oneDNN/MKLDNN so conv2d uses the pure-PyTorch kernels
torch.backends.mkldnn.enabled = False

app = Flask(__name__)

# Load YOLO model
model = YOLO("best.pt")

# Initialize EasyOCR for Arabic
reader = easyocr.Reader(['ar'])  # Add 'en' if you want bilingual detection

def decode_base64_image(base64_string):
    try:
        img_data = base64.b64decode(base64_string)
        np_arr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        # print("Decoding error:", e)
        return None

def predict_license_plates(image):
    results = model(image)[0]
    plates = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        y1+=80 # 70 , 25 ,80 
        cropped = image[y1:y2, x1:x2]
        text = reader.readtext(cropped, detail=0)
        # text = reader.readtext(cropped)
        plates.extend(text)
    return plates

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image provided"}), 400
    
    image = decode_base64_image(data['image'])
    if image is None:
        return jsonify({"error": "Invalid base64 image"}), 400

    plates = predict_license_plates(image)
    # print(plates[0])
    print(plates)
    return jsonify({"license_plate": plates[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
