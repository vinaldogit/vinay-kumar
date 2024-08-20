import easyocr
import os

class TextExtractionModel:
    def __init__(self, languages=['en']):
        # Initialize the EasyOCR reader with the specified languages
        self.reader = easyocr.Reader(languages)

    def extract_text(self, image_paths):
        extracted_data = {}

        for image_path in image_paths:
            if "text" in image_path:
                # Perform OCR on the image
                result = self.reader.readtext(image_path)
                extracted_text = " ".join([text[1] for text in result])  # Extract the recognized text
                
                # Store the text with the corresponding image ID
                image_id = os.path.splitext(os.path.basename(image_path))[0]
                extracted_data[image_id] = extracted_text

        return extracted_data

    def save_extracted_text(self, extracted_data, output_path='extracted_text.json'):
        with open(output_path, 'w') as f:
            num_recs = len(extracted_data)
            f.write(f"#numrecs={num_recs}\n")
            f.write("#ObjectId|ExtractedText\n")
            for key, value in extracted_data.items():
                f.write(f"{key}|{value}\n")
        print(f"Extracted text data saved to {output_path}")
