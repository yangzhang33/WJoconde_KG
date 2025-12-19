"""
================================================================================
Final Entity Selection Script
================================================================================

Description:
    This script combines the LLM and VLM validation results and selects the
    final entities that should be added to the knowledge graph. An entity is
    kept if at least ONE model (LLM or VLM) answers "Yes" for it.

Workflow:
    1. Load LLM validation results (filtered_depicts_llm_validation.csv)
    2. Load VLM validation results (filtered_depicts_vlm_validation.csv)
    3. For each entity in filtered_depicts:
       - Check if LLM said "Yes" OR VLM said "Yes"
       - Keep entities with at least one "Yes" answer
    4. Save final selected entities to `final_selected_entities.csv`

Input:
    - LLM validation: filtered_depicts_llm_validation.csv
    - VLM validation: filtered_depicts_vlm_validation.csv

Output:
    - final_selected_entities.csv: Contains entity, selected_depicts (entities
      that have at least one "Yes" from LLM or VLM), along with detailed
      LLM and VLM responses

================================================================================
"""

import pandas as pd
import json
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================
LLM_VALIDATION_CSV = "filtered_depicts_llm_validation.csv"
VLM_VALIDATION_CSV = "filtered_depicts_vlm_validation.csv"
OUTPUT_CSV = "final_selected_entities.csv"

# ============================================================================
# Load Data
# ============================================================================
print("Loading validation results...")
df_llm = pd.read_csv(LLM_VALIDATION_CSV)
df_vlm = pd.read_csv(VLM_VALIDATION_CSV)


def parse_response(response_str):
    """
    Parse the JSON response string into a dictionary.
    
    Returns:
        dict: Parsed response, or empty dict if parsing fails
    """
    if pd.isna(response_str):
        return {}
    try:
        return json.loads(response_str)
    except json.JSONDecodeError:
        return {}


def normalize_answer(answer):
    """
    Normalize answer to lowercase 'yes' or 'no'.
    """
    if isinstance(answer, str):
        answer_lower = answer.lower().strip()
        if "yes" in answer_lower:
            return "yes"
    return "no"


def process_list(column_value):
    """Convert comma-separated string to list."""
    if pd.isna(column_value) or str(column_value).strip() == "":
        return []
    return [x.strip() for x in str(column_value).split(",")]


# ============================================================================
# Merge and Process
# ============================================================================
print("Processing and merging validation results...")

# Merge LLM and VLM results on entity
df_merged = pd.merge(
    df_llm[['entity', 'filtered_depicts', 'llm_response']],
    df_vlm[['entity', 'vlm_response']],
    on='entity',
    how='outer'
)

# If there are entities only in one file, handle missing columns
df_merged['llm_response'] = df_merged['llm_response'].fillna('{}')
df_merged['vlm_response'] = df_merged['vlm_response'].fillna('{}')

# Also handle filtered_depicts from VLM if LLM is missing
if 'filtered_depicts_x' in df_merged.columns:
    df_merged['filtered_depicts'] = df_merged['filtered_depicts_x'].fillna(df_merged.get('filtered_depicts_y', ''))
    df_merged = df_merged.drop(columns=['filtered_depicts_x', 'filtered_depicts_y'], errors='ignore')

results = []

print(f"Processing {len(df_merged)} entities...")

for index, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Selecting entities"):
    entity = row['entity']
    filtered_depicts = row.get('filtered_depicts', '')
    filtered_depicts_list = process_list(filtered_depicts)
    
    if not filtered_depicts_list:
        continue
    
    # Parse responses
    llm_responses = parse_response(row['llm_response'])
    vlm_responses = parse_response(row['vlm_response'])
    
    # Check for errors in responses
    if "error" in llm_responses:
        llm_responses = {}
    if "error" in vlm_responses:
        vlm_responses = {}
    
    # Select entities that have at least one "Yes" from either model
    selected_entities = []
    entity_details = []
    
    for dep_entity in filtered_depicts_list:
        # Get responses for this entity (case-insensitive matching)
        llm_answer = "no"
        vlm_answer = "no"
        
        # Check LLM responses (try exact match and case-insensitive)
        for key, value in llm_responses.items():
            if key.lower().strip() == dep_entity.lower().strip():
                llm_answer = normalize_answer(value)
                break
        
        # Check VLM responses
        for key, value in vlm_responses.items():
            if key.lower().strip() == dep_entity.lower().strip():
                vlm_answer = normalize_answer(value)
                break
        
        # Keep if at least one says "yes"
        is_selected = (llm_answer == "yes" or vlm_answer == "yes")
        
        if is_selected:
            selected_entities.append(dep_entity)
        
        entity_details.append({
            "entity": dep_entity,
            "llm": llm_answer,
            "vlm": vlm_answer,
            "selected": is_selected
        })
    
    # Store results
    results.append({
        "entity": entity,
        "filtered_depicts": filtered_depicts,
        "selected_depicts": ", ".join(selected_entities) if selected_entities else "",
        "num_filtered": len(filtered_depicts_list),
        "num_selected": len(selected_entities),
        "entity_details": json.dumps(entity_details),
        "llm_response_raw": row['llm_response'],
        "vlm_response_raw": row['vlm_response']
    })

# Create results DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv(OUTPUT_CSV, index=False)

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*60)
print("FINAL ENTITY SELECTION SUMMARY")
print("="*60)

total_entities = len(results_df)
entities_with_selections = len(results_df[results_df['num_selected'] > 0])
total_filtered = results_df['num_filtered'].sum()
total_selected = results_df['num_selected'].sum()

print(f"Total artworks processed: {total_entities}")
print(f"Artworks with at least one selected entity: {entities_with_selections}")
print(f"Total filtered_depicts entities: {total_filtered}")
print(f"Total selected entities (at least one Yes): {total_selected}")
print(f"Selection rate: {total_selected/total_filtered*100:.2f}%" if total_filtered > 0 else "N/A")
print(f"\nResults saved to '{OUTPUT_CSV}'")

# Also save a simplified version with just the final selections
simplified_results = results_df[['entity', 'selected_depicts', 'num_selected']]
simplified_results = simplified_results[simplified_results['num_selected'] > 0]
simplified_output = "final_selected_entities_simple.csv"
simplified_results.to_csv(simplified_output, index=False)
print(f"Simplified results saved to '{simplified_output}'")

