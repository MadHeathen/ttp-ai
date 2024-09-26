from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import io

# Initialize Flask app
app = Flask(__name__)

# Initialize the PaddleOCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Function to identify the document type
def identify_document_type(text):
    if "driving" in text.lower():
        return "Driving Licence"
    elif "permanent" in text.lower():
        return "PAN Card"
    elif "government" in text.lower():
        return "Aadhar Card"
    elif "republic" in text.lower():
        return "Passport"
    elif "election" in text.lower():
        return "Voter ID"
    elif "pay" in text.lower() and "ifs" in text.lower():
        return "Chequebook"
    elif "ifsc" in text.lower():
        return "Bank Passbook"
    else:
        return "Unknown Document"

# Define the OCR route for processing image uploads
@app.route('/process_document', methods=['POST'])
def process_document():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    
    # Read image and convert it to a format compatible with PaddleOCR
    image = Image.open(file.stream)
    image = np.array(image)

    # Perform OCR on the image
    result = ocr.ocr(image, cls=True)
    doc_text = ""

    # Extract the detected text
    for line in result:
        for word_info in line:
            detected_text = word_info[1][0]
            doc_text += detected_text

    # Identify document type
    document_type = identify_document_type(doc_text)

    # Return the detected text and document type as a JSON response
    return jsonify({"detected_text": doc_text, "document_type": document_type})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
