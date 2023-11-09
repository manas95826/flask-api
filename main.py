import cv2
from PIL import Image as PilImage
from PIL import ImageDraw
import numpy as np
import io
import base64
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_write_on_faces(image):
    texts_to_write = [
        "Happy face, this person has a retention rate of 69.",
        "Smiling face, spreading positivity!",
        "Serious face, focused and determined.",
        "Surprised face, something caught their attention!",
        "Confused face, deep in thought.",
        "Excited face, full of energy!",
        "Calm face, a picture of tranquility."
    ]

    # Convert the uploaded image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Convert the OpenCV image to a Pillow image
    pil_image = PilImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    for i, (x, y, w, h) in enumerate(faces):
        # Write text on the detected face
        if i < len(texts_to_write):
            text_to_write = texts_to_write[i]
            draw.text((x, y - 10), text_to_write, fill=(255, 0, 0, 0))

    # Convert the Pillow image back to OpenCV format
    image_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return image_with_text

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    uploaded_image = request.files['file']

    if uploaded_image.filename == '':
        return jsonify({"error": "No selected file"})

    if uploaded_image:
        # Read the uploaded image
        input_image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), -1)

        # Process the image
        result_image = detect_and_write_on_faces(input_image)

        # Convert the processed image to base64
        ret, buffer = cv2.imencode(".jpg", result_image)
        output_buffer = io.BytesIO(buffer)
        processed_image_base64 = base64.b64encode(output_buffer.getvalue()).decode()

        return jsonify({"processed_image": processed_image_base64})

if __name__ == '__main__':
    app.run()
