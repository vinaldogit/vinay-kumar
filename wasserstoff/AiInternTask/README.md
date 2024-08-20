# AI Intern Project

This project is designed to demonstrate computer vision and natural language processing capabilities using Python. It provides a Streamlit UI for users to upload an image, performs segmentation on the image, identifies segmented objects, extracts text from the segments, summarizes the results, and create visualizations for the results.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Contributing](#contributing)

## Overview

The **AI Intern Project** is a comprehensive Python-based tool that allows users to interactively process images. It performs the following tasks:

1. **Image Segmentation:** Segments the input image into different regions.
2. **Object Identification:** Identifies the objects present in the segmented regions.
3. **Text Extraction:** Extracts any text found within the segmented regions.
4. **Summarization:** Summarizes the content of the image based on the identified objects and extracted text.
5. **Results Display:** Shows the segmented images, identified objects, extracted text, and the summary on the Streamlit UI.

## Features

- **Streamlit UI:** A simple and interactive user interface for uploading images and displaying results.
- **Image Segmentation:** Automatically segments uploaded images into different parts.
- **Object Identification:** Recognizes and identifies objects within each segmented part.
- **Text Extraction:** Extracts and reads any text present in the segmented images.
- **Summary Generation:** Summarizes the information extracted from the image, providing a concise overview.

## Installation

### Prerequisites

- Python 3.11.8+
- Pip / Pip3

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/vinaldogit/vinay-kumar/wasserstoff/AIInternTask.git
    cd AIInternTask
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit application:
    ```bash
    streamlit run streamlit_app/app.py
    ```

## Usage

1. Open the Streamlit app in your web browser.
2. Upload an image using the provided UI.
3. Use the buttons to apply segmentation, identification, text_extraction, summarization models on the input image.
4. Generate visualizations once all the models are applied on the image.
4. View the results directly in the UI with the buttons.

## Project Structure

AI Intern Project/
│
├── data/
│	├── input_images/			    # All the input images resides in this directory
│	├── output/					    # All the output tables and result images are saved in this directory
│	└── segmented_object/		    # All the segmented images resides here
├── models/
│	├── segmentation_model.py	    # Segmentation model implementation
│	├── identification_model.py	    # Identification model implementation
│	├── text_extraction_model.py	# Text extraction model implementation
│	└── summarization_model.py		# Summarization model implementation
├── streamlit_app/
│	└── app.py	                    # Stream lit UI application logic
├── tests/
│	├── test_identification.py	    # Identification model test
│	├── test_segmentation.py	    # Segmentation model test
│	├── text_extraction.py	        # Text Extraction model test
│	└── test_summarization.py		# Summarization model test
├── utils/
│	├── data_mapping.py 	        # Data mapping logic for the outputs generated
│	└── visualization.py		    # Visualization logic for the outputs generated
├── presentation.pptx               # Project presentation
├── requirements.txt                # Python dependencies
└── README.md                       # Markdown File

## Technologies

- **Python**: The primary programming language.
- **Streamlit**: Used for creating the web-based UI.
- **OpenCV**: For image processing and segmentation.
- **PyTorch**: For object identification and text extraction models.
- **Transformers**: For text summarization.
- **Pandas**: For data frame generation.
- **Matplotlib**: For image plotting or visualization.
- **Transformers**: For text summarization.

## Contributing

Contributions are welcome! Please fork this repository and create a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

