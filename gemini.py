import cv2
import pytesseract
import json
import re
import os

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
# Step 4: Parse text to JSON
# -----------------------------
def parse_text_to_json(text):
    def extract_number(pattern, default=0):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1) or match.group(2) if match.groups() else None
            if value:
                return float(value.replace(",", ""))
        return default

    def extract_string(pattern, default="Unknown"):
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else default

    # Flexible patterns to handle various receipt formats
    data = {
        "TransactionID": extract_string(r'(?:Transaction\s*ID|TXN\s*ID)[:\s]*([\w\d-]+)', "T12345"),
        "TransactionDT": extract_number(r'(?:Transaction\s*Date|Date)[:\s]*(\d+)', 1056789),
        "tabular_data": {
            "TransactionAmt": extract_number(r'(?:\$|€|£|₹|Amount)[:\s]*([\d,]+\.?\d*)', 2500.00),
            "card1": extract_number(r'Card1[:\s]*(\d+)', 1500),
            "card2": extract_number(r'Card2[:\s]*(\d+)', 301),
            "addr1": extract_number(r'Addr1[:\s]*(\d+)', 325),
            "ProductCD_W": 1 if extract_string(r'Product\s*CD[:\s]*([A-Z])') == "W" else 0,
            "ProductCD_C": 1 if extract_string(r'Product\s*CD[:\s]*([A-Z])') == "C" else 0,
            "ProductCD_H": 1 if extract_string(r'Product\s*CD[:\s]*([A-Z])') == "H" else 0,
            "P_emaildomain_gmail.com": 1 if extract_string(r'P_emaildomain[:\s]*(\S+)') == "gmail.com" else 0,
            "P_emaildomain_yahoo.com": 1 if extract_string(r'P_emaildomain[:\s]*(\S+)') == "yahoo.com" else 0,
            "R_emaildomain_gmail.com": 1 if extract_string(r'R_emaildomain[:\s]*(\S+)') == "gmail.com" else 0,
            "R_emaildomain_yahoo.com": 1 if extract_string(r'R_emaildomain[:\s]*(\S+)') == "yahoo.com" else 0,
            "DeviceType_mobile": 1 if extract_string(r'DeviceType[:\s]*(\w+)') == "mobile" else 0,
            "DeviceType_desktop": 1 if extract_string(r'DeviceType[:\s]*(\w+)') == "desktop" else 0,
            "DeviceInfo_iPhone": 1 if extract_string(r'DeviceInfo[:\s]*(\w+)') == "iPhone" else 0,
            "DeviceInfo_Android": 1 if extract_string(r'DeviceInfo[:\s]*(\w+)') == "Android" else 0,
            "V1": 0.5,
            "V2": 0.3,
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
        text = extract_text(preprocessed)

        # Parse text to JSON
        json_data = parse_text_to_json(text)

        # Save JSON to file
        with open('receipt_data.json', 'w') as f:
            json.dump(json_data, f, indent=2)
        print("JSON data saved to receipt_data.json")

        return json_data

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    image_path = input("Enter path to receipt image: ")
    main(image_path)