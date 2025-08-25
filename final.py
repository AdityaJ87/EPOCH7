import cv2
import pytesseract
import json
import re
import os
from datetime import datetime

# Configure Tesseract path (update as needed for your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path

# -----------------------------
# Step 1: Load the receipt image
# -----------------------------
def load_image(image_path):
    if not os.path.exists(image_path):
        raise ValueError(f"Image path does not exist: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image from path: {image_path}")
    return image

# -----------------------------
# Step 2: Preprocess the image for OCR
# -----------------------------
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Invert image if dark background with light text
    if gray.mean() < 127:
        gray = 255 - gray
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh

# -----------------------------
# Step 3: Extract text using OCR
# -----------------------------
def extract_text(image):
    custom_config = r'--oem 3 --psm 6'  # PSM 6 for general text block
    try:
        text = pytesseract.image_to_string(image, config=custom_config, lang='eng')
        if not text.strip():
            raise ValueError("OCR failed to extract any text")
        print("Extracted text:", text)  # Debug: Print extracted text
        return text
    except Exception as e:
        raise ValueError(f"OCR error: {str(e)}")

# -----------------------------
# Step 4: Convert OCR text to custom JSON
# -----------------------------
def ocr_text_to_custom_json(raw_text):
    """
    Converts given OCR text to the target JSON format.
    Keeps TransactionDT as a human-readable date string.
    Missing features are filled with default values.
    """
    # Patterns based on your text
    transaction_id_pattern = r'Transaction\s*ID\s*([A-Z0-9]+)'
    datetime_pattern = r'(\d{2}:\d{2}\s*[ap]m on \d{1,2} \w+ \d{4})'
    amount_pattern = r'%([\d,]+)'
    sender_pattern = r'Received from\s+(.*?)(?=%|\n)'

    transaction_id_match = re.search(transaction_id_pattern, raw_text, re.IGNORECASE)
    datetime_match = re.search(datetime_pattern, raw_text, re.IGNORECASE)
    amount_match = re.search(amount_pattern, raw_text)
    sender_match = re.search(sender_pattern, raw_text, re.IGNORECASE)

    # Keep the extracted datetime string as is (human-readable)
    transaction_dt = datetime_match.group(1) if datetime_match else ""

    # Build JSON dictionary matching your format
    data = {
        "TransactionID": transaction_id_match.group(1) if transaction_id_match else "",
        "TransactionDT": transaction_dt,  # human-readable datetime string
        "tabular_data": {
            "TransactionAmt": float(amount_match.group(1).replace(',', '')) if amount_match else 0.0,
            # Placeholder/fake values for the other features:
            "card1": 0,
            "card2": 0,
            "addr1": 0,
            "ProductCD_W": 0,
            "ProductCD_C": 0,
            "ProductCD_H": 0,
            "P_emaildomain_gmail.com": 0,
            "P_emaildomain_yahoo.com": 0,
            "R_emaildomain_gmail.com": 0,
            "R_emaildomain_yahoo.com": 0,
            "DeviceType_mobile": 0,
            "DeviceType_desktop": 0,
            "DeviceInfo_iPhone": 0,
            "DeviceInfo_Android": 0,
            "V1": 0.0,
            "V2": 0.0
        }
    }

    return data

# -----------------------------
# Main workflow
# -----------------------------
def main(image_path):
    try:
        # Validate image path
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise ValueError("Please provide a valid image file (.png, .jpg, .jpeg)")

        # Load and preprocess
        image = load_image(image_path)
        preprocessed = preprocess_image(image)

        # Extract text via OCR
        raw_text = extract_text(preprocessed)

        # Convert OCR text to JSON
        json_data = ocr_text_to_custom_json(raw_text)

        # Save JSON to file
        with open('receipt_data.json', 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)
        print("JSON data saved to receipt_data.json")
        print(json.dumps(json_data, indent=4))

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    image_path = input("Enter path to image: ")
    main(image_path)