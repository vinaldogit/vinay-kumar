from models import segmentation_model as segmod
import os

input_images_dir = "data/input_images"
segmented_images_dir = "data/segmented_objects"

def run_segmentation_model_test(image_file_name):
    input_image_path = os.path.join(input_images_dir, image_file_name)
    
    if not os.path.exists(input_image_path):
        return False
    
    sModel = segmod.SegmentationModel()
    sModel.loadImage(input_image_path)
    sModel.execute()
    return True

if __name__ == "__main__":
    input_file_path = input("Enter input image name with complete path:")

    print("TEST - Segmentation Model: start")
    if run_segmentation_model_test(input_file_path):
        print("TEST - Segmentation Model: end\n")
    else:
        print("ERROR! No such file exists! Run test again with correct path")