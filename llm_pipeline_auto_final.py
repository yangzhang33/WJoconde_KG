"""
================================================================================
Large Language Model Pipeline for Heritage Object Relation Extraction
================================================================================

Description:
    This script uses a Large Language Model to automatically
    identify what entities/objects are depicted in heritage artworks based on their
    TEXT DESCRIPTIONS (not images). It processes textual metadata from the WJoconde
    dataset and extracts relations in JSON format.

Workflow:
    1. Load knowledge graph data (entities, relations) from CSV files
    2. Load the text generation model
    3. For each unique artwork entity in the knowledge graph:
       - Retrieve the entity's textual description
       - Generate a prompt 
       - Run inference to get a JSON list of depicted entities
       - Parse and validate the JSON output
    4. Retry failed entities (up to max_iterations) to handle parsing errors
    5. Save all valid JSON outputs to 'llm_valid_json_pairs.json'

Input:
    - Knowledge graph: 
    - Entity descriptions: 
    - Relation mappings: 
Output:
    - llm_valid_json_pairs.json: Dictionary mapping entity URIs to their 
      predicted relations in JSON format

Model:
    - (Text-only LLM)

================================================================================
"""

from transformers import AutoTokenizer
import transformers
import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

import os
import io
import IPython.display
from PIL import Image
import base64
import gensim.downloader as api
hf_api_key = '' 

import warnings
warnings.filterwarnings("ignore")

model = ""
# model = ""

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

import pandas as pd
df_kg = pd.read_csv('WJocondeMM_en.csv')
df_entities_about = pd.read_csv('entity2text_long_en.csv', delimiter=',')
df_relations = pd.read_csv('relation2text.txt', delimiter='\t', names=['relation', 'name'])


import pandas as pd

def get_triple(df, df_e, df_r, from_id, rel_id):
    # Filter df for rows with the given from_id and rel_id
    subset = df[(df['from'] == from_id) & (df['rel'] == rel_id)]
    
    # Get the list of to_ids from the filtered subset
    to_ids = subset['to'].tolist()
    
    # Retrieve the text for the provided from_id from df_e
    # (Assuming there is exactly one match)
    from_text_series = df_e.loc[df_e['entity'] == from_id, 'name']
    if from_text_series.empty:
        from_text = None
    else:
        from_text = from_text_series.iloc[0]
    
    # Retrieve the relation name for the provided rel_id from df_r
    rel_name_series = df_r.loc[df_r['relation'] == rel_id, 'name']
    if rel_name_series.empty:
        relation_name = None
    else:
        relation_name = rel_name_series.iloc[0]
    
    # Retrieve the text for each to_id from df_e
    # Note: This will return texts in the same order as they appear in df_e, 
    # not necessarily in the same order as in to_ids.
    to_texts = []
    for tid in to_ids:
        text_series = df_e.loc[df_e['entity'] == tid, 'name']
        if not text_series.empty:
            to_texts.append(text_series.iloc[0])
        else:
            to_texts.append(None)
    
    return [from_text, relation_name, to_texts]

# Example usage:
# Suppose your CSVs have been loaded as follows:
# df   = pd.read_csv("df.csv")
# df_e = pd.read_csv("df_e.csv")
# df_r = pd.read_csv("df_r.csv")
#
# Then call the function with desired from_id and rel_id:
result = get_triple(df_kg, df_entities_about, df_relations, from_id='http://www.wikidata.org/entity/Q3178376', rel_id='http://www.wikidata.org/prop/direct/P180')
print(len(result))  # Output: ['text_for_from_id', 'relation_name', ['text_for_to_id1', 'text_for_to_id2', ...]]
print(result[0])
print('-----------------')
print(result[1])
print('-----------------')
print(result[2])


# get list of from id
import pandas as pd

# Extract unique from_id values
unique_from_ids = df_kg["from"].unique()

# Optionally, convert the NumPy array to a Python list
unique_from_ids_list = unique_from_ids.tolist()

print(unique_from_ids_list)
print(len(unique_from_ids_list))



def create_llm_prompt(text_description, relation):
    """
    Creates a prompt that follows exactly the Definition, Role, Input,
    and Formatting shown in the image, while keeping the relation
    as a function input.
    """

    prompt = f"""
Definition
Knowledge graphs (KGs) represent a collection of interlinked descriptions of entities — objects, events, or concepts. KGs store data in the format <entity, relation, entity>. Knowledge graph completion involves adding missing entities and relations to enhance and enrich the existing graph, ensuring it provides a more comprehensive understanding of the data it represents.

Role
As a Large Language Model, your role is to assist in the completion of a knowledge graph by processing provided textual data, extracting <{relation}> properties, and then structuring this information in a predefined format.

Input
{text_description}

Formatting
Please format the extracted data in JSON, specifying '{relation}'. Each answer should contain only one or a few words from the given information. Here's how you should structure your response:
{{
  "{relation}": [
    "<entity name 1>",
    "<entity name 2>"
  ]
}}

Only output the JSON and nothing else.
"""
    return prompt



import json

def extract_json(llm_output):
    """
    Extracts and parses the JSON part from a string containing extra text.
    
    Parameters:
        llm_output (str): The output string from the LLM.
        
    Returns:
        dict or None: The parsed JSON object if extraction and parsing succeed, otherwise None.
    """
    # Find the first occurrence of "{" and the last occurrence of "}"
    start_index = llm_output.find('{')
    end_index = llm_output.rfind('}')
    
    # Ensure that both braces were found
    if start_index == -1 or end_index == -1:
        print("No JSON block found in the output.")
        return None
    
    # Extract the JSON substring
    json_str = llm_output[start_index:end_index+1]
    
    try:
        # Parse and return the JSON object
        json_obj = json.loads(json_str)
        return json_obj
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        return None
    
import json
from tqdm import tqdm

def process_unique_from_ids(unique_from_ids_list):
    """
    Processes a list of unique from_ids by generating an LLM prompt for each,
    extracting a JSON output from the generated text, and storing the valid JSON
    outputs as a pair (from_id, json_output). It also collects from_ids that did not
    produce valid JSON.
    
    Parameters:
        unique_from_ids_list (list): A list of unique from_id values.
        
    Returns:
        tuple: A tuple containing:
            - valid_json_pairs (dict): Dictionary mapping each from_id to its valid JSON output.
            - failed_from_ids (list): List of from_ids that did not produce valid JSON.
    """
    # Set the relation id you want to use.
    rel_id = 'http://www.wikidata.org/prop/direct/P180'
    
    # Dictionary to hold valid JSON outputs as {from_id: json_output}
    valid_json_pairs = {}
    
    # List to keep track of from_ids that do not produce valid JSON output
    failed_from_ids = []
    
    # Loop through each unique from_id using tqdm for a progress bar
    for from_id in tqdm(unique_from_ids_list, desc="Processing from_ids"):
        # Retrieve the triple (assumes get_triple returns [text, relation_name, list_of_texts])
        result = get_triple(df_kg, df_entities_about, df_relations, from_id, rel_id)
        
        # Create the input prompt for the LLM
        input_prompt = create_llm_prompt(result[0], result[1])
        
        # Generate output from the LLM.
        generation_output = pipeline(
            input_prompt,
            max_new_tokens=128,  # generates up to 128 new tokens beyond the prompt
            do_sample=True,      # use sampling
            temperature=0.7      # sampling temperature
        )
        
        # Extract the generated text (excluding the prompt)
        full_generated_text = generation_output[0]['generated_text']
        generated_text = full_generated_text[len(input_prompt):]
        
        # Extract the JSON part using the extract_json function
        json_output = extract_json(generated_text)
        
        # Check if a valid JSON object was extracted
        if json_output is None:
            failed_from_ids.append(from_id)
        else:
            valid_json_pairs[from_id] = json_output
            
    return valid_json_pairs, failed_from_ids


def process_until_all_success(unique_from_ids_list, max_iterations=20):
    """
    Repeatedly processes the provided from_ids until all produce valid JSON output,
    or until a maximum number of iterations is reached. In each iteration, new valid
    JSON pairs are added to the overall dictionary.

    Parameters:
        unique_from_ids_list (list): A list of unique from_id values to process.
        max_iterations (int): Maximum number of iterations to retry the failed from_ids.
    
    Returns:
        tuple: (all_valid_json_pairs, remaining_failed_ids)
            - all_valid_json_pairs (dict): Dictionary mapping each from_id to its valid JSON output.
            - remaining_failed_ids (list): from_ids that still did not produce valid JSON (if any).
    """
    all_valid_json_pairs = {}
    current_ids = unique_from_ids_list.copy()
    iteration = 0
    
    while current_ids and iteration < max_iterations:
        iteration += 1
        print(f"\nIteration {iteration}: Processing {len(current_ids)} from_ids.")
        valid_pairs, failed_ids = process_unique_from_ids(current_ids)
        
        # Add the new valid pairs to the overall dictionary.
        all_valid_json_pairs.update(valid_pairs)
        
        if not failed_ids:
            print("All from_ids processed successfully in this iteration.")
            break
        else:
            print(f"{len(failed_ids)} from_ids failed in iteration {iteration}. Retrying...")
        
        # Set current_ids to the list of failed IDs for the next iteration.
        current_ids = failed_ids

    if current_ids:
        print(f"\nProcessing completed after {iteration} iterations, but {len(current_ids)} from_ids still failed.")
    else:
        print(f"\nProcessing completed after {iteration} iterations with no remaining failures.")
    
    return all_valid_json_pairs, current_ids



# Example usage:
all_valid_json_pairs, remaining_failed_ids = process_until_all_success(unique_from_ids_list)

# Optionally, save the valid JSON pairs to a file.
output_filename = "llm_valid_json_pairs.json"
with open(output_filename, "w") as outfile:
    json.dump(all_valid_json_pairs, outfile, indent=4)
print(f"\nSaved valid JSON pairs to {output_filename}")

if remaining_failed_ids:
    print("\nThe following from_ids did not produce valid JSON output after maximum iterations:")
    print(remaining_failed_ids)
else:
    print("\nAll from_ids produced valid JSON output.")
