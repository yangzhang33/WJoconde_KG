# WJoconde Knowledge Graph Completion Pipeline

A multi-modal pipeline that extracts and validates "depicts" relations for heritage artworks using LLM (text) and VLM (image) models.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STEP 1: ENTITY EXTRACTION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  llm_pipeline_auto_final.py    →    Extract entities from TEXT descriptions │
│  vlm_pipeline_auto_final.py    →    Extract entities from IMAGES            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STEP 2: COMBINE & FILTER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  llm_blip_combine.ipynb        →    Merge LLM + VLM predictions             │
│                                →    Add ground truth labels                  │
│                                →    Filter with Word2Vec (remove redundant)  │
│                                →    Output: filtered_depicts                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STEP 3: VALIDATION                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  llm_validation.py             →    Ask LLM: "Is X depicted?" (Yes/No)      │
│  vlm_validation.py             →    Ask VLM: "Is X in image?" (Yes/No)      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                       STEP 4: FINAL SELECTION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  final_entity_selection.py     →    Keep entities if LLM OR VLM says "Yes"  │
│                                →    Output: final_selected_entities.csv     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `llm_pipeline_auto_final.py` | Extracts depicted entities from text using LLM |
| `vlm_pipeline_auto_final.py` | Extracts depicted entities from images using VLM |
| `llm_blip_combine.ipynb` | Combines results, adds ground truth, filters with Word2Vec |
| `llm_validation.py` | Validates filtered entities with LLM (text-based) |
| `vlm_validation.py` | Validates filtered entities with VLM (image-based) |
| `final_entity_selection.py` | Selects entities with at least one "Yes" vote |
| `evaluate.ipynb` | Evaluation metrics |

## Data Files

| File | Description |
|------|-------------|
| `WJocondeMM_en.csv` | Knowledge graph triples |
| `entity2text_long_en.csv` | Entity → description mapping |
| `entity2text.txt` | Entity → name mapping |
| `relation2text.txt` | Relation → name mapping |

## How to Run

```bash
# Step 1: Extract entities
python llm_pipeline_auto_final.py    # Output: llm_valid_json_pairs.json
python vlm_pipeline_auto_final.py    # Output: blip_valid_json_pairs.json

# Step 2: Combine and filter (run notebook cells)
# Output: llm_bilp_combined_all.csv (with filtered_depicts column)

# Step 3: Validate
python llm_validation.py             # Output: filtered_depicts_llm_validation.csv
python vlm_validation.py             # Output: filtered_depicts_vlm_validation.csv

# Step 4: Final selection
python final_entity_selection.py     # Output: final_selected_entities.csv
```

## Output

The final output `final_selected_entities.csv` contains:
- `entity`: Artwork URI
- `selected_depicts`: Entities confirmed by at least one model
- `num_selected`: Count of selected entities
- `entity_details`: Per-entity LLM/VLM responses

## Citation
