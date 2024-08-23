from models import segmentation_model as segmod
import os

input_images_dir = "data/input_images"
segmented_images_dir = "data/segmented_objects"
output_dir = "data/output"

def run_segmentation_model(image_file_name):
    input_image_path = os.path.join(input_images_dir, image_file_name)
    
    # Check if exists in input images directory
    if not os.path.exists(input_image_path):
        False
    
    sModel = segmod.SegmentationModel()
    sModel.loadImage(input_image_path)
    sModel.execute()
    return True


def preprocess_uploaded_image(file_name, data):
    # Saving the uploaded image on disk
    filename = os.path.join(input_images_dir, file_name)
    
    if not os.path.exists(filename):
        with open(filename, 'wb') as f:
            f.write(data)
    
    # Make ouput directories
    os.makedirs(segmented_images_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":
    input_file_path = input("Enter input image name with complete path:")

    print("TEST - Segmentation Model: start")
    if run_segmentation_model(input_file_path):
        print("TEST - Segmentation Model: end\n")
    else:
        print("ERROR! No such file exists! Run test again with correct path")