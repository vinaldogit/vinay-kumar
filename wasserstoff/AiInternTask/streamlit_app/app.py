import sys
import os

# Correcting the path for the application
# Because of not able to find tests module error
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from PIL import Image
from tests_ import test_identification
from tests_ import test_segmentation
from tests_ import test_text_extraction
from tests_ import test_summarization
from utils import data_mapping
from utils import visualization

if __name__ == "__main__":
    # Title of the User Interface application
    st.title("AI Intern Project")

    # Upload an image
    orig_image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    orig_image_file_loaded = False
    orig_image_file_name = ""

    # Check if a file has been uploaded
    if orig_image_file is not None:
        # Open the image file
        image = Image.open(orig_image_file)
        orig_image_file_loaded = True
        orig_image_file_name = orig_image_file.name
        
        # Upload on disk (it will create one if not already present)
        test_segmentation.preprocess_uploaded_image(orig_image_file_name, orig_image_file.read())
        
        # Display the image
        st.write("Image file loaded successfully")
        st.image(image, caption="Upload Image", use_column_width=True)

    # Create a button to run segmentation model testing
    if st.button("Test Segmentation Model"):
        if orig_image_file_loaded == False:
            st.write("Input image is missing! Reselect the image and run test again")
        else:
            # Run segmentation test
            if test_segmentation.run_segmentation_model(orig_image_file_name):
                st.write("Segmentation Model Testing Complete! See results in data directory")
            else:
                st.write("ERROR! Image not present in the data/input_images directory")

    # Create a button to run identification model
    if st.button("Test Identification Model"):
        if test_identification.run_identification_model_test():
            st.write("Identification Model Testing Complete! See results in data directory")
        else:
            st.write("ERROR! Need to run segmentation model first!")

    # Create a button to run text extraction model
    if st.button("Test Text Extraction Model"):
        if test_text_extraction.run_text_extraction_model_test():
            st.write("Text Extraction Model Testing Complete! See results in data directory")
        else:
            st.write("ERROR! Need to run segmentation model first!")

    # Create a button to run summarization model
    if st.button("Test Summarization Model"):
        if test_summarization.run_summarization_model_test():
            st.write("Summarization Model Testing Complete! See results in data directory")
        else:
            st.write("ERROR! Need to run identification model first!")

    st.write("\n")
    st.write("CAUTION: Press button below when all model have run successfully at least once for correct outputs")

    # Initialize session state variables
    if 'image_files_dict' not in st.session_state:
        st.session_state.image_files_dict = {}
    
    if 'visualization_gen_flag' not in st.session_state:
        st.session_state.visualization_gen_flag = False
    
    if 'disp_image_path' not in st.session_state:
        st.session_state.disp_image_path = ""

    if st.button("Generate Visulizations"):
        # Create data mapping and run visualization
        dataMap = data_mapping.DataMapping()
        dataMap.map_data()

        mapping_file = dataMap.get_data_mapping_file_name()

        if orig_image_file_name == "":
            st.write("Select an image file first !!")
        else:
            input_image_path = "data/input_images/" + orig_image_file.name
        
            output_generator = visualization.Visualization(mapping_file, input_image_path)
            st.session_state.image_files_dict = output_generator.generate_final_output()
            st.write("Visualizations generated successfully! See data directory for the results")
            st.write("Press below buttons to show visualizations")
            st.session_state.visualization_gen_flag = True

    # Display buttons for different images
    if st.session_state.visualization_gen_flag and st.session_state.image_files_dict:
        if st.button("Display Annotated Image"):
            st.session_state.disp_image_path = st.session_state.image_files_dict.get("annotated_image", "")
        
        if st.button("Display Data Table Image"):
            st.session_state.disp_image_path = st.session_state.image_files_dict.get("data_table", "")
        
        if st.button("Display Master Annotated Image"):
            st.session_state.disp_image_path = st.session_state.image_files_dict.get("master_annotated_image", "")
        
        # Display the selected image
        if st.session_state.disp_image_path:
            st.write(f"Display: {st.session_state.disp_image_path}")
            disp_image = Image.open(st.session_state.disp_image_path)
            st.image(disp_image, caption="Displayed Image", use_column_width=True)

