#!/usr/bin/env python3
"""
Minimal GSM8K data preparation - no external dependencies except requests
"""

import json
import os
from pathlib import Path

def download_gsm8k_minimal():
    """Download GSM8K using only standard libraries"""
    
    print("Downloading GSM8K dataset (minimal version)...")
    
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    # Try using requests if available, otherwise urllib
    try:
        import requests
        
        # Direct URLs to GSM8K dataset
        train_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
        test_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        
        # Download train data
        print("Downloading training data...")
        response = requests.get(train_url)
        train_lines = response.text.strip().split('\n')
        
        train_data = []
        for line in train_lines:
            if line.strip():
                item = json.loads(line)
                train_data.append({
                    'question': item['question'],
                    'answer': item['answer'],
                    'final_answer': extract_final_answer(item['answer'])
                })
        
        # Download test data
        print("Downloading test data...")
        response = requests.get(test_url)
        test_lines = response.text.strip().split('\n')
        
        test_data = []
        for line in test_lines:
            if line.strip():
                item = json.loads(line)
                test_data.append({
                    'question': item['question'],
                    'answer': item['answer'],
                    'final_answer': extract_final_answer(item['answer'])
                })
                
    except ImportError:
        # Use urllib if requests is not available
        import urllib.request
        
        train_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
        test_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        
        # Download train data
        print("Downloading training data with urllib...")
        with urllib.request.urlopen(train_url) as response:
            train_lines = response.read().decode('utf-8').strip().split('\n')
        
        train_data = []
        for line in train_lines:
            if line.strip():
                item = json.loads(line)
                train_data.append({
                    'question': item['question'],
                    'answer': item['answer'],
                    'final_answer': extract_final_answer(item['answer'])
                })
        
        # Download test data  
        print("Downloading test data with urllib...")
        with urllib.request.urlopen(test_url) as response:
            test_lines = response.read().decode('utf-8').strip().split('\n')
        
        test_data = []
        for line in test_lines:
            if line.strip():
                item = json.loads(line)
                test_data.append({
                    'question': item['question'],
                    'answer': item['answer'],
                    'final_answer': extract_final_answer(item['answer'])
                })
    
    # Save processed data
    print("Saving processed data...")
    
    # Save 200 training examples for router development
    with open('data/gsm8k_train.json', 'w') as f:
        json.dump(train_data[:200], f, indent=2)
    
    # Save full test set
    with open('data/gsm8k_test.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\n✓ Successfully downloaded and processed GSM8K!")
    print(f"  Training examples saved: {min(200, len(train_data))}")
    print(f"  Test examples saved: {len(test_data)}")
    
    # Show samples
    if test_data:
        print(f"\nSample problem:")
        print(f"  Question: {test_data[0]['question'][:150]}...")
        print(f"  Answer: {test_data[0]['final_answer']}")
        
        print(f"\nAnother sample:")
        print(f"  Question: {test_data[1]['question'][:150]}...")
        print(f"  Answer: {test_data[1]['final_answer']}")
    
    return train_data[:200], test_data

def extract_final_answer(answer_text):
    """Extract numerical answer from GSM8K solution"""
    
    # GSM8K format: answer appears after ####
    if "####" in answer_text:
        answer = answer_text.split("####")[-1].strip()
        # Remove any non-numeric characters except decimal point
        import re
        # Extract number, handling commas and dollar signs
        match = re.search(r'[\$]?([\d,]+\.?\d*)', answer)
        if match:
            return match.group(1).replace(',', '')
        return answer
    
    # Fallback: find the last number
    import re
    numbers = re.findall(r'[\$]?([\d,]+\.?\d*)', answer_text)
    if numbers:
        return numbers[-1].replace(',', '')
    
    return ""

if __name__ == "__main__":
    train_data, test_data = download_gsm8k_minimal()
    
    # Verify the data
    print(f"\nData verification:")
    print(f"  First answer type: {type(test_data[0]['final_answer'])}")
    print(f"  Sample answers: {[d['final_answer'] for d in test_data[:5]]}")
