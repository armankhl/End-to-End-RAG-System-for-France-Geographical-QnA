# utils.py
import json
import random
from typing import List, Dict

def load_and_sample_dataset(file_path: str, num_samples: int = -1) -> List[Dict]:
    """
    Loads a JSON dataset from a file and optionally samples a number of items.
    
    Args:
        file_path: Path to the JSON file.
        num_samples: Number of items to randomly sample. If -1, returns all items.
    
    Returns:
        A list of dictionaries from the dataset.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        raise
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON file.")
        raise
        
    if num_samples != -1 and num_samples < len(dataset):
        return random.sample(dataset, num_samples)
    
    return dataset