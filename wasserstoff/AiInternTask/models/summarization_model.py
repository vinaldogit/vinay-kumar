from transformers import pipeline

class SummarizationModel:
    def __init__(self, inp_file="data/output/object_descriptions.dat", model_name="facebook/bart-large-cnn"):
        self.summarizer = pipeline("summarization", model=model_name)
        
        # Load descriptions
        self.descriptions = {}
        with open(inp_file, 'r') as f:
            lines = f.readlines()
            for line in lines[2:]:
                parts = line.strip().split('|')
                image_id = parts[0].replace('.jpg', '')
                object_description = parts[1]
                self.descriptions[image_id] = {'object': object_description}

    def summarize_text(self, text, max_length=10, min_length=5):
        # Generate a summary of the text
        summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']

    def generate_summaries(self, output_file='data/output/summaries.dat'):
        summaries = []

        for image_id, content in self.descriptions.items():
            object_description = content['object']
            
            full_text = f"Object Description: {object_description}"
            
            # Generate the summary
            summary = self.summarize_text(full_text)
            summaries.append((image_id, summary))

        # Save summaries to a file
        with open(output_file, 'w') as f:
            num_recs = len(summaries)
            f.write(f"#numrecs={num_recs}\n")
            f.write("#ObjectId|Summary\n")
            for image_id, summary in summaries:
                f.write(f"{image_id}|{summary}\n")

        print(f"Summaries saved to {output_file}")
