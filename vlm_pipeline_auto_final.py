"""
================================================================================
Vision-Language Model Pipeline for Heritage Object Relation Extraction
================================================================================

Description:
    This script uses a Vision-Language Model to automatically
    identify what entities/objects are depicted in heritage artwork images. It processes
    images from the WJoconde dataset and extracts relations in JSON format.

Workflow:
    1. Load knowledge graph data (entities, relations) from CSV files
    2. Load the vision-language model
    3. For each unique artwork entity in the knowledge graph:
       - Convert entity URI to corresponding image filename
       - Generate a VQA prompt
       - Run inference to get a JSON list of depicted entities
       - Parse and validate the JSON output
    4. Retry failed entities (up to max_iterations) to handle parsing errors
    5. Save all valid JSON outputs to 'blip_valid_json_pairs.json'

Input:
    - Knowledge graph: 
    - Entity descriptions: 
    - Relation mappings: 
    - Images: 
Output:
    - blip_valid_json_pairs.json: Dictionary mapping entity URIs to their 
      predicted relations in JSON format

Model:
    - (Vision-Language Model)

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

import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disables the limit


df_kg = pd.read_csv('WJocondeMM_en.csv')
df_entities_about = pd.read_csv('entity2text_long_en.csv', delimiter=',')
df_relations = pd.read_csv('relation2text.txt', delimiter='\t', names=['relation', 'name'])


from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "", torch_dtype=torch.float16, device_map="auto"
)

processor = AutoProcessor.from_pretrained("")



def create_vqa_question(relation):
    """
    Creates a VQA prompt that follows exactly the Definition / Role / Input / Formatting
    structure shown in the image, while keeping `relation` as an input.
    """

    question = f"""
Definition
Knowledge graphs (KGs) represent a collection of interlinked descriptions of entities — objects, events, or concepts. KGs store data in the format <entity, relation, entity>. Knowledge graph completion involves adding missing entities and relations to enhance and enrich the existing graph, ensuring it provides a more comprehensive understanding of the data it represents.

Role
As a Image Understanding Model, your role is to assist in the completion of a knowledge graph by processing provided image, extracting {relation} property, and then structuring this information in a predefined format.

Input
<image>

Formatting
Please format the extracted data in JSON, specifying '{relation}', the answer should contain only one or a few words from the given information. Here's how you should structure your response:
{{
  "{relation}": [
    "<entity name 1>",
    "<entity name 2>"
  ]
}}

Only output the JSON and nothing else.
"""
    return question



# get list of from id
import pandas as pd

# Extract unique from_id values
unique_from_ids = df_kg["from"].unique()

# Optionally, convert the NumPy array to a Python list
unique_from_ids_list = unique_from_ids.tolist()

# print(unique_from_ids_list)
# print(len(unique_from_ids_list))

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

import re

def convert_from_to_image_filename(from_url: str) -> str:
    """
    Converts a 'from' URL into the corresponding image filename.
    
    Example:
        Input: "http://www.wikidata.org/entity/Q3399458"
        Output: "httpwwwwikidataorgentityQ3399458.jpg"
    """
    image_filename = re.sub(r'[^\w]', '', from_url) + '.jpg'
    image_folder_path = "images_all_en_about_images"
    full_image_path = os.path.join(image_folder_path, image_filename)

    return full_image_path


from PIL import Image

def process_vision_info_with_resize(messages, max_size=(224, 224)):
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Resize images
    resized_images = [img.resize(max_size, Image.Resampling.LANCZOS) for img in image_inputs]
    
    return resized_images, video_inputs

def prepare_inference_inputs(converted_filename, question, processor, device="cuda"):
    """
    Prepares inputs for inference using a processor, given an image filename and a question.

    Parameters:
        converted_filename (str): The path to the converted image file.
        question (str): The question to be asked about the image.
        processor (object): The processor object to apply chat templates and process vision information.
        device (str): The device to move inputs to (default is "cuda").

    Returns:
        dict: A dictionary of inputs ready for model inference.
    """
    
    # Step 1: Create messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": converted_filename},
                {"type": "text", "text": question},
            ],
        }
    ]

    # Step 2: Apply chat template to get the text input
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Step 3: Process image and video inputs with resizing
    image_inputs, video_inputs = process_vision_info_with_resize(messages)

    # Step 4: Prepare inputs for inference
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Step 5: Move inputs to the specified device (default: CUDA)
    inputs = inputs.to(device)
    
    return inputs

import json

def parse_and_extract_json(llm_output):
    """
    Attempts to parse a JSON object from an LLM output string.
    If direct parsing fails, it tries to extract the JSON block from the string.

    Parameters:
        llm_output (str): The output string from the LLM.

    Returns:
        dict or None: The parsed JSON object if successful, otherwise None.
    """
    # First, try to clean markdown formatting and parse directly
    cleaned_output = llm_output.replace('```json\n', '').replace('\n```', '').strip()
    
    try:
        # Attempt direct parsing
        return json.loads(cleaned_output)
    except json.JSONDecodeError:
        print("Direct parsing failed. Attempting to extract JSON block...")
    
    # If direct parsing fails, attempt to extract the JSON block
    start_index = llm_output.find('{')
    end_index = llm_output.rfind('}')
    
    if start_index == -1 or end_index == -1:
        print("No JSON block found in the output.")
        return None
    
    json_str = llm_output[start_index:end_index+1]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print("Error decoding extracted JSON:", e)
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
        rel_name = get_triple(df_kg, df_entities_about, df_relations, from_id, rel_id)[1]

        converted_filename = convert_from_to_image_filename(from_id)
        question = create_vqa_question(rel_name)
        if os.path.isfile(converted_filename):

            inputs = prepare_inference_inputs(converted_filename, question, processor)
            
            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.7)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            json_output = parse_and_extract_json(output_text[0])

            # Check if a valid JSON object was extracted
            if json_output is None:
                failed_from_ids.append(from_id)
            else:
                valid_json_pairs[from_id] = json_output
        else:
            print(f"File not found: {converted_filename}")
            valid_json_pairs[from_id] = None
            
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



# process_unique_from_ids(unique_from_ids_list)

all_valid_json_pairs, remaining_failed_ids = process_until_all_success(unique_from_ids_list, max_iterations=5)

# Optionally, save the valid JSON pairs to a file.
output_filename = "blip_valid_json_pairs_test.json"
with open(output_filename, "w") as outfile:
    json.dump(all_valid_json_pairs, outfile, indent=4)
print(f"\nSaved valid JSON pairs to {output_filename}")

if remaining_failed_ids:
    print("\nThe following from_ids did not produce valid JSON output after maximum iterations:")
    print(remaining_failed_ids)
else:
    print("\nAll from_ids produced valid JSON output.")



# The following from_ids did not produce valid JSON output after maximum iterations:
# ['http://www.wikidata.org/entity/Q476458', 'http://www.wikidata.org/entity/Q17613629', 'http://www.wikidata.org/entity/Q2027662', 'http://www.wikidata.org/entity/Q16011124', 'http://www.wikidata.org/entity/Q3212142', 'http://www.wikidata.org/entity/Q18384768', 'http://www.wikidata.org/entity/Q5944788', 'http://www.wikidata.org/entity/Q531329', 'http://www.wikidata.org/entity/Q798034', 'http://www.wikidata.org/entity/Q97667173', 'http://www.wikidata.org/entity/Q3464047', 'http://www.wikidata.org/entity/Q1978815', 'http://www.wikidata.org/entity/Q738038', 'http://www.wikidata.org/entity/Q19947579', 'http://www.wikidata.org/entity/Q38484841', 'http://www.wikidata.org/entity/Q16020875', 'http://www.wikidata.org/entity/Q972295', 'http://www.wikidata.org/entity/Q3731232', 'http://www.wikidata.org/entity/Q3824633', 'http://www.wikidata.org/entity/Q2843410', 'http://www.wikidata.org/entity/Q4191448', 'http://www.wikidata.org/entity/Q17495466', 'http://www.wikidata.org/entity/Q7142987', 'http://www.wikidata.org/entity/Q654044', 'http://www.wikidata.org/entity/Q17354848', 'http://www.wikidata.org/entity/Q1133821', 'http://www.wikidata.org/entity/Q1212737', 'http://www.wikidata.org/entity/Q17583459', 'http://www.wikidata.org/entity/Q1799047', 'http://www.wikidata.org/entity/Q4790231', 'http://www.wikidata.org/entity/Q47128683', 'http://www.wikidata.org/entity/Q636001', 'http://www.wikidata.org/entity/Q17627500']