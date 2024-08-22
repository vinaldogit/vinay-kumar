from models import identification_model as idfmod
import os

segmented_images_dir = "data/segmented_objects"
output_dir = "data/output"

def run_identification_model_test():
    # Fetch the segmented images from the directory
    image_paths = [os.path.join(segmented_images_dir, fname) for fname in os.listdir(segmented_images_dir) if fname.endswith('.jpg')]

    if len(image_paths) == 0:
        return False

    # Run the identification model
    identifier = idfmod.IdentificationModel()
    descriptions = identifier.identify_objects(image_paths)
    identifier.save_descriptions(descriptions, output_path=f"{output_dir}/object_descriptions.dat")
    return True

if __name__ == "__main__":
    print("TEST - Identification Model: start")
    if run_identification_model_test():
        print("TEST - Identification Model: end\n")
    else:
        print("ERROR! No segmented files exists! Please run segmentation model first!")