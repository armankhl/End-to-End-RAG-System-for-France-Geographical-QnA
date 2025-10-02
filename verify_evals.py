#!/usr/bin/env python3
"""
Simple script to verify that a JSON file has the same structure as sample_output.json using Pydantic
"""

import json
import sys
import argparse
from typing import Dict
from pydantic import BaseModel, Field, field_validator


class EvalOutput(BaseModel):
    """Model for evaluation output structure"""
    question: str = Field(..., description="The question string")
    answer: str = Field(..., description="The answer string")
    top_k: Dict[str, str] = Field(..., description="Dictionary with exactly 3 string key-value pairs")
    
    @field_validator('top_k')
    @classmethod
    def validate_top_k_has_three_items(cls, v):
        if len(v) != 3:
            raise ValueError(f"top_k must have exactly 3 items, got {len(v)}")
        return v


def validate_json_structure(filename):
    """
    Validate that a JSON file has the expected structure using Pydantic
    """
    try:
        # --- FIX Here: Explicitly specify UTF-8 encoding ---
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{filename}': {e}")
        return False
    
    # Check if data is a list
    if not isinstance(data, list):
        print(f"Error: Root element must be an array, got {type(data).__name__}")
        return False
    
    # Validate each item in the list
    for i, item in enumerate(data):
        try:
            EvalOutput(**item)
        except Exception as e:
            print(f"Error: Validation failed for item {i + 1} - {e}")
            return False
    
    print(f"Successfully validated {len(data)} items")
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify JSON file structure against sample_output.json using Pydantic")
    parser.add_argument("filename", help="Path to the JSON file to validate")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only output result, no details")
    
    args = parser.parse_args()
    
    is_valid = validate_json_structure(args.filename)
    
    if args.quiet:
        print("VALID" if is_valid else "INVALID")
    else:
        if is_valid:
            print(f"✓ '{args.filename}' has valid structure")
        else:
            print(f"✗ '{args.filename}' has invalid structure")
    
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
