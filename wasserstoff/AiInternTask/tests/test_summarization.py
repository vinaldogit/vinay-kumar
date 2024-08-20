from models import summarization_model as summod
import os

segmented_images_dir = "data/segmented_objects"
output_dir = "data/output"

def run_summarization_model_test():
    inp_file = f"{output_dir}/object_descriptions.dat"

    # Check if the descriptions file is there or not
    if not os.path.exists(inp_file):
        return False
    
    # Run the Summarization model
    summurizer = summod.SummarizationModel(inp_file)
    summurizer.generate_summaries()
    return True

if __name__ == "__main__":
    print("TEST - Summarization Model: start")
    if run_summarization_model_test():
        print("TEST - Summarization Model: end\n")
    else:
        print("ERROR! No Objects Description Found! Please run identification model before running this model!")