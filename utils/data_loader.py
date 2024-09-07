# utils/data_loader.py

import json
from typing import List, Dict

def load_dataset(file_path: str) -> List[Dict]:
    """
    Load dataset from a JSON file.
    
    Expected format:
    [
        {
            "question": "What is the capital of France?",
            "seed_entity": "France",
            "answer_entities": ["Paris"]
        },
        ...
    ]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Validate data format
    for item in data:
        assert "question" in item, "Each item must have a 'question' field"
        assert "seed_entity" in item, "Each item must have a 'seed_entity' field"
        assert "answer_entities" in item, "Each item must have an 'answer_entities' field"
    
    return data