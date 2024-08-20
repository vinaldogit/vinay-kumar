# MODEL: segamentaion_model.py
# This mode is used to segment the portions in the input image

import cv2
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import matplotlib.pyplot as plt
from PIL import Image
import uuid
import sqlite3
import numpy as np
import os
import easyocr

class SegmentationModel:

    def __init__(self):
        print("Loading the sgmentation model (mask R-CNN model : pretrained model)")
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        # self.model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.eval()

        # Initialize EasyOCR for text detection
        self.ocr_reader = easyocr.Reader(['en'])

    def loadImage(self, imagePath):
        self.imagePath = imagePath
    
    def execute(self):
        self.imagePreprocessing()
        self.performInference()
        self.addToDB()
    
    def imagePreprocessing(self):
        self.origImage = cv2.imread(self.imagePath)
        self.image = cv2.cvtColor(self.origImage, cv2.COLOR_BGR2RGB)
        self.image = self.image.astype("float32") / 255
        self.image = torch.tensor(self.image).permute(2, 0, 1).unsqueeze(0)

    def performInference(self):
        with torch.no_grad():
            self.output = self.model(self.image)

    def addToDB(self):
        master_id = str(uuid.uuid4())   # Generate a master ID for the original image
        objects_metadata = self.save_segmented_objects(self.origImage, self.output, master_id)
        self.save_metadata_to_db(objects_metadata)
        print(f"Segmented objects saved successfully with master ID: {master_id}")

    def save_segmented_objects(self, orig_image, output, master_id, save_dir="data/segmented_objects", out_dir="data/output"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        objects_metadata = []
        image_np = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

        segmented_objects_dimension_list = {}       # To store the coordinated of segmented regions

        # Save segmented regions
        for i, box in enumerate(output[0]['boxes']):
            x1, y1, x2, y2 = map(int, box.tolist())
            segmented_object = orig_image[y1:y2, x1:x2]
            
            object_id = str(uuid.uuid4())
            object_filename = f"{object_id}.jpg"
            object_path = os.path.join(save_dir, object_filename)
            cv2.imwrite(object_path, segmented_object)

            segmented_objects_dimension_list[object_id] = [x1, y1, x2, y2]
            
            objects_metadata.append((object_id, master_id, object_path))

        # Save segmented regions that contains text using easyOCR
        ocr_results = self.ocr_reader.readtext(image_np)
        self.text_boxes = []
        for text in ocr_results:
            # Extracting bounding box coordinates
            box = text[0]  # This is a list of points defining the bounding box
            x_coords = [point[0] for point in box]
            y_coords = [point[1] for point in box]
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)

            # Convert coordinates to integers
            x1, x2 = int(round(x1)), int(round(x2))
            y1, y2 = int(round(y1)), int(round(y2))
            
            # Ensure coordinates are within image bounds
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, orig_image.shape[1]), min(y2, orig_image.shape[0])
            
            segmented_text = orig_image[y1:y2, x1:x2]
            
            text_id = str(uuid.uuid4())
            text_filename = f"text_{text_id}.jpg"
            text_path = os.path.join(save_dir, text_filename)
            cv2.imwrite(text_path, segmented_text)
            
            objects_metadata.append((text_id, master_id, text_path))
            self.text_boxes.append([x1, y1, x2, y2])
            text_id = "text_" + text_id
            segmented_objects_dimension_list[text_id] = [x1, y1, x2, y2]

        # Save the segmented object dimension list to a file
        out_path = os.path.join(out_dir, "segmented_regions_in_master_img.dat")
        with open(out_path, 'w') as f:
            num_recs = len(segmented_objects_dimension_list)
            f.write(f"#numrecs={num_recs}\n")
            f.write("#ObjectId|ObjectDimensionInMasterImage\n")
            for key, value in segmented_objects_dimension_list.items():
                x1, y1, x2, y2 = value
                f.write(f"{key}|{x1},{y1},{x2},{y2}\n")

        return objects_metadata
    
    def save_metadata_to_db(self, objects_metadata, save_dir="data/output", db_file='metadata.db'):
        db_path = os.path.join(save_dir, db_file)
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        # Create table if it doesn't exist
        c.execute('''CREATE TABLE IF NOT EXISTS objects_metadata
                    (object_id TEXT PRIMARY KEY, master_id TEXT, object_path TEXT)''')

        # Insert metadata into the table
        c.executemany('INSERT INTO objects_metadata (object_id, master_id, object_path) VALUES (?, ?, ?)', objects_metadata)

        conn.commit()
        conn.close()
