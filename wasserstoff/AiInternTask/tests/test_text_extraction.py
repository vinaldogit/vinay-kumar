from models import text_extraction_model as txtmod
import os

segmented_images_dir = "data/segmented_objects"
output_dir = "data/output"

def run_text_extraction_model_test():
    # Fetch the segmented images from the directory
    image_paths = [os.path.join(segmented_images_dir, fname) for fname in os.listdir(segmented_images_dir) if fname.endswith('.jpg')]

    if len(image_paths) == 0:
        return False

    # Run the text extraction model
    text_extractor = txtmod.TextExtractionModel(languages=['en'])
    extracted_data = text_extractor.extract_text(image_paths)
    text_extractor.save_extracted_text(extracted_data, output_path=f"{output_dir}/extracted_text.dat")
    return True

if __name__ == "__main__":
    print("TEST - Text Extraction Model: start")
    if run_text_extraction_model_test():
        print("TEST - Text Extraction Model: end\n")
    else:
        print("ERROR! No segmented files exists! Please run segmentation model first!")