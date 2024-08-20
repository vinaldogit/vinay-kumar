import json
import os
import sqlite3

class DataMapping:
    def __init__(self, output_dir="data/output"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.metadata = self.loadMetaData()
        self.descriptions = self.loadObjectDescriptionsData()
        self.extracted_texts = self.loadExtractedTextData()
        self.summaries = self.loadSummariesData()
        self.data_mapping_file_path = ""

    def loadMetaData(self, file="MetaData.db"):
        db_path = os.path.join(self.output_dir, file)
        metadata = {}

        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query the objects_metadata table
        cursor.execute("SELECT object_id, master_id, object_path FROM objects_metadata")
        rows = cursor.fetchall()

        # Convert the result to a dictionary
        for row in rows:
            object_id, master_id, object_path = row
            metadata[object_id] = {"master_id": master_id, "segmented_image_path": object_path}

        # Close the connection
        conn.close()

        return metadata

    def loadObjectDescriptionsData(self, file="object_descriptions.dat"):
        file_path = os.path.join(self.output_dir, file)
        return self.extractInfoFromDatFile(file_path)
    
    def loadExtractedTextData(self, file="extracted_text.dat"):
        file_path = os.path.join(self.output_dir, file)
        return self.extractInfoFromDatFile(file_path)
    
    def loadSummariesData(self, file="summaries.dat"):
        file_path = os.path.join(self.output_dir, file)
        return self.extractInfoFromDatFile(file_path)
    
    def extractInfoFromDatFile(self, file_path):
        info_dict = {}

        with open(file_path) as f:
            lines = f.readlines()

            for line in lines[2:]:
                parts = line.strip().split('|')
                image_id = parts[0]
                object_description = parts[1]
                info_dict[image_id] = {'object': object_description}
        
        return info_dict

    def map_data(self, output_file="data_mapping.json"):
        # Initialize the mapping dictionary
        mapping = {}

        # Iterate through all objects and map their data
        for image_id, meta in self.metadata.items():
            # Search for the object description
            description = self.descriptions.get(image_id, "None")
            text = self.extracted_texts.get(image_id, "None")
            summary = self.summaries.get(image_id, "None")

            # Check for text in case Description is None
            if description == "None":
                text_id = "text_" + image_id
                description = self.descriptions.get(str(text_id), "None")
                text = self.extracted_texts.get(text_id, "None")
                summary = self.summaries.get(str(text_id), "None")

            mapping[image_id] = {
                "master_id": meta["master_id"],
                "segmented_image_path": meta["segmented_image_path"],
                "object_description": description,
                "text_descriptions": text,
                "summary": summary
            }

        # Save the mapping to a JSON file
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w') as f:
            json.dump(mapping, f, indent=4)
        
        print(f"Data mapping saved to {output_path}")
        self.data_mapping_file_path = output_path

    def get_data_mapping_file_name(self):
        return self.data_mapping_file_path