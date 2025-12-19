"""
================================================================================
LLM Validation Script for Filtered "Depicts" Entities
================================================================================

Description:
    This script uses a Large Language Model to VALIDATE
    whether the filtered_depicts entities (from Word2Vec filtering) are actually
    depicted in heritage artworks based on their TEXT DESCRIPTIONS.

Task:
    Given an artwork's name, description, and a list of filtered_depicts entities,
    the LLM determines whether each entity is depicted (Yes) or not (No).

Workflow:
    1. Load the combined output file (llm_bilp_combined_all.csv) containing:
       - Entity URIs, descriptions, and filtered_depicts
    2. Load the LLaMA-3.1-8B-Instruct model
    3. For each artwork with non-empty filtered_depicts:
       - Create a chat prompt with the filtered entities list
       - Ask LLM: "Does this object depict [item]? Yes/No"
       - Parse JSON response with Yes/No for each entity
    4. Save results to `filtered_depicts_llm_validation.csv`

Input:
    - Combined predictions: llm_bilp_combined_all.csv (from llm_blip_combine.ipynb)
    - Entity descriptions: entity2text_long_en.csv
    - Entity names: entity2text.txt

Output:
    - filtered_depicts_llm_validation.csv: Contains entity, from_name, description,
      filtered_depicts, and llm_response (Yes/No for each entity)

Model:
    - ""
================================================================================
"""

from transformers import AutoTokenizer
import transformers
import torch
import os
import pandas as pd
import json
from tqdm import tqdm
import warnings
import huggingface_hub

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================
hf_api_key = ''
huggingface_hub.login(token=hf_api_key)

model_name = ""

# Input/Output files
INPUT_COMBINED_CSV = "llm_bilp_combined_all.csv"
ENTITY_DESC_CSV = "entity2text_long_en.csv"
ENTITY_NAME_TXT = "entity2text.txt"
OUTPUT_CSV = "filtered_depicts_llm_validation.csv"

# ============================================================================
# Load Model
# ============================================================================
print("Loading LLM model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

# ============================================================================
# Load Data
# ============================================================================
print("Loading data files...")
df_combined = pd.read_csv(INPUT_COMBINED_CSV)
df_entity_desc = pd.read_csv(ENTITY_DESC_CSV, delimiter=',')
df_entity_name = pd.read_csv(ENTITY_NAME_TXT, sep='\t', header=None, names=['entity', 'name'])


def get_entity_description(entity_uri, df_entity_desc):
    """Get the description text for a given entity URI."""
    matching_row = df_entity_desc[df_entity_desc['entity'] == entity_uri]
    if not matching_row.empty:
        return matching_row.iloc[0]['name']
    return None


def get_entity_name(entity_uri, df_entity_name):
    """Get the name for a given entity URI."""
    matching_row = df_entity_name[df_entity_name['entity'] == entity_uri]
    if not matching_row.empty:
        return matching_row.iloc[0]['name']
    return None


def heritage_object_depicts_chat_prompt(object_name, description, label_list):
    """
    Create a chat prompt to ask LLM if the heritage object depicts certain items.
    """
    system_prompt = "You are a chatbot tasked with determining whether a heritage object depicts certain items, responding only with 'Yes' or 'No' for each item, outputting your response in a valid JSON format."

    user_prompt = f"""
    **Heritage Object Name:** {object_name}
    **Object Description:** {description}
    **Label List:** [{label_list}]

    For each item in the label list, answer with either 'Yes' or 'No' depending on whether the object "{object_name}" depicts that item based on the description and your knowledge of the object. 
    Output **only once** in valid JSON format, with no extra explanation or repetition.

    Example output format:
    {{
        "item1": "Yes",
        "item2": "No",
        ...
    }}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return messages


def validate_json(content):
    """
    Validates JSON content.
    
    Returns:
        tuple: (is_valid, parsed_json, error_message)
    """
    try:
        parsed_json = json.loads(content)
        return True, parsed_json, None
    except json.JSONDecodeError as e:
        return False, None, f"JSON is invalid. Error: {str(e)}"


def extract_json_from_text(text):
    """
    Try to extract JSON from text that might contain extra content.
    """
    # Try to find JSON block in the text
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1:
        json_str = text[start_idx:end_idx+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    return None


def process_list(column_value):
    """Convert comma-separated string to list."""
    if pd.isna(column_value) or str(column_value).strip() == "":
        return []
    return [x.strip() for x in str(column_value).split(",")]


# ============================================================================
# Main Processing
# ============================================================================
print("Processing filtered_depicts validation with LLM...")

results = []
invalid_json_count = 0

# Filter rows that have non-empty filtered_depicts
df_to_process = df_combined[df_combined['filtered_depicts'].notna() & (df_combined['filtered_depicts'].str.strip() != '')]

print(f"Processing {len(df_to_process)} entities with filtered_depicts...")

for index, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="LLM Validation"):
    entity = row["from"]
    filtered_depicts_list = process_list(row["filtered_depicts"])
    
    if not filtered_depicts_list:
        continue
    
    # Get entity name and description
    object_name = get_entity_name(entity, df_entity_name)
    description = get_entity_description(entity, df_entity_desc)
    
    if object_name is None:
        object_name = entity.split('/')[-1]  # Use entity ID as fallback
    
    if description is None:
        description = "No description available."
    
    # Create the filtered_depicts string for the prompt
    filtered_depicts_str = ", ".join([f'"{item}"' for item in filtered_depicts_list])
    
    # Construct the prompt
    prompt = heritage_object_depicts_chat_prompt(object_name, description, filtered_depicts_str)
    
    # Generate response using the model
    try:
        sequences = pipeline(
            prompt,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=300,
        )
        
        # Extract the generated text
        content_json = sequences[0]['generated_text'][2]['content']
        is_valid, parsed_json, error_message = validate_json(content_json)
        
        if not is_valid:
            # Try to extract JSON from the text
            parsed_json = extract_json_from_text(content_json)
            if parsed_json is None:
                response_json = {"error": "Failed to parse JSON"}
                print(f"\nInvalid JSON for {object_name}: {error_message}")
                invalid_json_count += 1
            else:
                response_json = parsed_json
        else:
            response_json = parsed_json
            
    except Exception as e:
        response_json = {"error": str(e)}
        print(f"\nError processing {entity}: {e}")
        invalid_json_count += 1
    
    # Store the results
    results.append({
        "entity": entity,
        "from_name": object_name,
        "description": description,
        "filtered_depicts": row["filtered_depicts"],
        "llm_response": json.dumps(response_json)
    })

# Save the results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nProcessing complete. Results saved to '{OUTPUT_CSV}'")
print(f"Total entities processed: {len(results)}")
print(f"Number of invalid JSON responses: {invalid_json_count}")
