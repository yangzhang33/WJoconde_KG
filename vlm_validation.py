"""
================================================================================
VLM Validation Script for Filtered "Depicts" Entities (Image-based)
================================================================================

Description:
    This script uses the VLM Visual Question Answering model to VALIDATE
    whether the filtered_depicts entities (from Word2Vec filtering) are actually
    depicted in heritage artwork IMAGES.

Task:
    Given an artwork's image and a list of filtered_depicts entities, the VLM VQA
    model determines whether each entity is visible in the image (Yes) or not (No).

Workflow:
    1. Load the combined output file (llm_bilp_combined_all.csv) containing:
       - Entity URIs and filtered_depicts
    2. Load the VLM VQA model
    3. For each artwork with non-empty filtered_depicts:
       - Load the corresponding image from the image folder
       - Resize to 224x224 for model input
       - For each entity, ask VLM: "Is there a '{entity}' in the image?"
       - Record Yes/No answers
    4. Save results to `filtered_depicts_vlm_validation.csv`

Input:
    - Combined predictions: llm_bilp_combined_all.csv (from llm_blip_combine.ipynb)
    - Images: images_all_en_about_images/

Output:
    - filtered_depicts_vlm_validation.csv: Contains entity, filtered_depicts, 
      vlm_response (Yes/No for each entity)

Model:
    - ""
================================================================================
"""

import os
import re
import pandas as pd
import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Allow large images
Image.MAX_IMAGE_PIXELS = None

# ============================================================================
# Configuration
# ============================================================================
INPUT_COMBINED_CSV = "llm_bilp_combined_all.csv"
IMAGE_FOLDER = "images_all_en_about_images"
OUTPUT_CSV = "filtered_depicts_vlm_validation.csv"

# ============================================================================
# Load Model
# ============================================================================
print("Loading BLIP VQA model...")
processor = BlipProcessor.from_pretrained("")
model = BlipForQuestionAnswering.from_pretrained("").to(
    "cuda" if torch.cuda.is_available() else "cpu"
)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# Load Data
# ============================================================================
print("Loading data files...")
df_combined = pd.read_csv(INPUT_COMBINED_CSV)


def convert_entity_to_image_filename(entity_uri):
    """
    Convert an entity URI to the corresponding image filename.
    
    Example:
        Input: "http://www.wikidata.org/entity/Q3399458"
        Output: "httpwwwwikidataorgentityQ3399458.jpg"
    """
    image_filename = re.sub(r'[^\w]', '', entity_uri) + '.jpg'
    return os.path.join(IMAGE_FOLDER, image_filename)


def process_list(column_value):
    """Convert comma-separated string to list."""
    if pd.isna(column_value) or str(column_value).strip() == "":
        return []
    return [x.strip() for x in str(column_value).split(",")]


def ask_vlm_about_entity(image, entity_label, processor, model, device):
    """
    Ask the VLM if a specific entity is present in the image.
    
    Returns:
        str: "yes" or "no"
    """
    question = f"Is there a '{entity_label}' in the image?"
    inputs = processor(image, question, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    answer = processor.decode(output[0], skip_special_tokens=True).lower()
    
    if "yes" in answer:
        return "yes"
    else:
        return "no"


# ============================================================================
# Main Processing
# ============================================================================
print("Processing filtered_depicts validation with VLM...")

results = []
missing_images = 0

# Filter rows that have non-empty filtered_depicts
df_to_process = df_combined[df_combined['filtered_depicts'].notna() & (df_combined['filtered_depicts'].str.strip() != '')]

print(f"Processing {len(df_to_process)} entities with filtered_depicts...")

for index, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="VLM Validation"):
    entity = row["from"]
    filtered_depicts_list = process_list(row["filtered_depicts"])
    
    if not filtered_depicts_list:
        continue
    
    # Get image path
    image_path = convert_entity_to_image_filename(entity)
    
    if not os.path.exists(image_path):
        print(f"\nImage not found: {image_path}")
        missing_images += 1
        # Store result with None responses
        results.append({
            "entity": entity,
            "filtered_depicts": row["filtered_depicts"],
            "vlm_response": json.dumps({"error": "image_not_found"})
        })
        continue
    
    try:
        # Load and resize image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Ask VLM about each filtered_depicts entity
        vlm_responses = {}
        for entity_label in filtered_depicts_list:
            answer = ask_vlm_about_entity(image, entity_label, processor, model, device)
            vlm_responses[entity_label] = answer
        
        # Store the results
        results.append({
            "entity": entity,
            "filtered_depicts": row["filtered_depicts"],
            "vlm_response": json.dumps(vlm_responses)
        })
        
    except Exception as e:
        print(f"\nError processing {entity}: {e}")
        results.append({
            "entity": entity,
            "filtered_depicts": row["filtered_depicts"],
            "vlm_response": json.dumps({"error": str(e)})
        })

# Save the results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nProcessing complete. Results saved to '{OUTPUT_CSV}'")
print(f"Total entities processed: {len(results)}")
print(f"Missing images: {missing_images}")
