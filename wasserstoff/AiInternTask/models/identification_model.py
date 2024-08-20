import torch
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel
import os
from PIL import Image

class IdentificationModel:
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def identify_objects(self, image_paths):
        descriptions = {}

        for image_path in image_paths:
            image = Image.open(image_path)
            image_id = os.path.splitext(os.path.basename(image_path))[0]

            if "text" in image_path:
                descriptions[image_id] = "It's plain text"
            else:
                inputs = self.processor(
                    text=["a photo of a cat", "a photo of a dog", "a photo of a person", "a photo of a car", "a photo of a tree"],
                    images=image,
                    return_tensors="pt",
                    padding=True)
            
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)  # Convert logits to probabilities

                # Map probabilities to corresponding labels
                labels = ["cat", "dog", "person", "car", "tree"]
                label_probs = {labels[i]: prob.item() for i, prob in enumerate(probs[0])}

                # Identify the most probable label
                identified_label = max(label_probs, key=label_probs.get)
                description = f"It's a {identified_label}"

                # Store the description
                descriptions[image_id] = description

        return descriptions

    def save_descriptions(self, descriptions, output_path='data/output/descriptions.dat'):
        with open(output_path, 'w') as f:
            num_recs = len(descriptions)
            f.write(f"#numrecs={num_recs}\n")
            f.write("#ObjectId|Description\n")
            for key, value in descriptions.items():
                f.write(f"{key}|{value}\n")
        print(f"Descriptions saved to {output_path}")
