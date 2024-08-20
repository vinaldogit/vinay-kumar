import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from PIL import Image
import json
import os
import numpy as np

class Visualization:
    def __init__(self, mapping_file, original_image_path, output_dir='data/output', segmedted_rgn_file="segmented_regions_in_master_img.dat"):
        self.mapping_file = mapping_file
        self.original_image_path = original_image_path
        self.output_dir = output_dir
        self.segmented_rgn_file = os.path.join(output_dir, segmedted_rgn_file)

    # Load the JSON mapping file
    def load_mapping(self):
        with open(self.mapping_file, 'r') as f:
            return json.load(f)

    # Generate a table for the mapping
    def generate_table(self, mapping):
        rows = []
        for object_id, data in mapping.items():
            description = data.get("object_description", {}).get("object", None)
            extracted_text = data.get("text_descriptions", {})
            if isinstance(extracted_text, dict):
                extracted_text = extracted_text.get("object", "None")

            summary = data.get("summary", {}).get("object", None)

            row = {
                "Object ID": object_id,
                "Master ID": data["master_id"],
                "Description": description,
                "Extracted Text": extracted_text,
                "Summary": summary
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        return df
    
    # This will visualize and generate an annotated image with annotations
    def generate_annotation_image(self):
        seg_dmns = {}
        annotations_dict = {}

        # Read segmented regions coordinates
        with open(self.segmented_rgn_file) as f:
            lines = f.readlines()

            for line in lines[2:]:
                parts = line.strip().split('|')
                image_id = parts[0]
                x1, y1, x2, y2 = parts[1].strip().split(',')
                seg_dmns[image_id] = [x1, y1, x2, y2]
        
        # Read object descriptions
        object_descriptions_file = os.path.join(self.output_dir, "object_descriptions.dat")
        with open(object_descriptions_file) as f:
            lines = f.readlines()

            for line in lines[2:]:
                parts = line.strip().split('|')
                image_id = parts[0]
                annotation = parts[1]
                annotations_dict[image_id] = annotation
        
        # Plot the image with annotations
        orig_image = Image.open(self.original_image_path)
        orig_image = np.array(orig_image)
        plt.figure(figsize=(12, 8))
        plt.imshow(orig_image)
        plt.axis('off')

        ax = plt.gca()
        for key, value in seg_dmns.items():
            x1, y1, x2, y2 = value
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            width, height = x2 - x1, y2 - y1
            rect = plt.Rectangle((x1, y1), width, height, edgecolor='r', facecolor='none', linewidth=2)
            ax.add_patch(rect)

            annotation = annotations_dict.get(key, "Unknown")
            ax.text(x1, y1, annotation, color='blue')
        
        plt.legend(loc="upper left")
        annotated_image_path = f"{self.output_dir}/annotated_image.png"
        plt.savefig(annotated_image_path, bbox_inches='tight')
        plt.close()
        print(f"Annotated image saved to {annotated_image_path}")

    def save_table(self, df):
        fig, ax = plt.subplots(figsize=(10, len(df) * 0.8 + 1))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.5, 1.5)

        table_image_path = f"{self.output_dir}/data_table.png"
        plt.savefig(table_image_path, bbox_inches='tight')
        plt.close()
        print(f"Data table saved to {table_image_path}")

    # This will generate a final output image with annotations & attribute table
    def generate_final_output(self):
        mapping = self.load_mapping()

        # Generate the data table
        df = self.generate_table(mapping)
        self.save_table(df)

        # Annotate the original image
        self.generate_annotation_image()

        # Load the images using PIL
        image1 = Image.open(f"{self.output_dir}/annotated_image.png")
        image2 = Image.open(f"{self.output_dir}/data_table.png")

        # Create a figure with subplots: 2 rows and 1 column
        fig, axs = plt.subplots(2, 1, figsize=(6, 10))

        # Plot the first image on the first subplot
        axs[0].imshow(image1)
        axs[0].axis('off')

        # Plot the second image on the second subplot
        axs[1].imshow(image2)
        axs[1].axis('off')

        # Adjust spacing between plots to remove gaps
        plt.subplots_adjust(hspace=0)

        # Save the combined figure as a new image
        plt.savefig(f"{self.output_dir}/master_annotated_image.png", bbox_inches='tight', pad_inches=0, dpi=300)

        # Final combined output (just saving the paths for user use)
        final_output = {
            "annotated_image": f"{self.output_dir}/annotated_image.png",
            "data_table": f"{self.output_dir}/data_table.png",
            "master_annotated_image":f"{self.output_dir}/master_annotated_image.png"
        }

        print("Master image with annotations and attribute table is generated:-")
        print(final_output)
        return final_output
