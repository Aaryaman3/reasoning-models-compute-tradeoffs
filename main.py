#!/usr/bin/env python3
"""
Robust baseline evaluation script for GCP VM.
Runs unattended with checkpointing, logging, and error handling.
"""

import os
import sys
import time
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
import signal

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from datasets import load_dataset

import tinker
from tinker import types

# Import config
import config

# ============================================================
# Setup Logging
# ============================================================

def setup_logging():
    """Setup comprehensive logging."""
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config.LOGS_DIR, f'run_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================
# Graceful Shutdown Handler
# ============================================================

class GracefulKiller:
    """Handle shutdown signals gracefully."""
    kill_now = False
    
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    
    def exit_gracefully(self, signum, frame):
        logger.warning("Received shutdown signal. Saving progress...")
        self.kill_now = True

killer = GracefulKiller()

# ============================================================
# Helper Functions
# ============================================================

def extract_numerical_answer(text):
    """Extract numerical answer from text."""
    if '####' in text:
        answer = text.split('####')[-1].strip()
        answer = answer.replace(',', '').strip()
        try:
            return float(answer)
        except:
            pass
    
    # Boxed format
    boxed_pattern = r'\$?\\boxed\{([-+]?\d+\.?\d*)\}\$?'
    match = re.search(boxed_pattern, text)
    if match:
        try:
            return float(match.group(1))
        except:
            pass
    
    # Final answer format
    final_patterns = [
        r'(?:final answer|answer) is:?\s*\$?\s*([-+]?\d+\.?\d*)',
        r'\\boxed\{([-+]?\d+\.?\d*)\}',
    ]
    
    for pattern in final_patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except:
                continue
    
    # Last number
    clean_text = re.split(r'<\|eot_id\|>|I\'d be happy|How can I assist', text)[0]
    numbers = re.findall(r'([-+]?\d+\.?\d*)', clean_text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except:
            pass
    
    return None

def extract_mcq_answer(text):
    """Extract multiple choice answer (A, B, C, D) - handles Llama tokens."""
    # Remove special tokens
    text = text.replace('<|eot_id|>', '').replace('<|end_of_text|>', '')
    text = text.replace('<｜end▁of▁sentence｜>', '').strip()
    text_upper = text.upper().strip()
    
    # If response is just a single letter - perfect!
    if text_upper in ['A', 'B', 'C', 'D']:
        return text_upper
    
    # Letter at the very start
    if len(text_upper) > 0 and text_upper[0] in ['A', 'B', 'C', 'D']:
        return text_upper[0]
    
    # Pattern 1: "THE ANSWER IS B" or "ANSWER IS B"
    pattern1 = r'(?:THE\s+)?ANSWER\s+IS\s+([A-D])'
    match = re.search(pattern1, text_upper)
    if match:
        return match.group(1)
    
    # Pattern 2: "ANSWER: B"
    pattern2 = r'ANSWER:\s*([A-D])'
    match = re.search(pattern2, text_upper)
    if match:
        return match.group(1)
    
    # Pattern 3: "(C)" format
    pattern3 = r'\(([A-D])\)'
    match = re.search(pattern3, text_upper)
    if match:
        return match.group(1)
    
    # Pattern 4: any A/B/C/D in the text
    pattern4 = r'([A-D])'
    match = re.search(pattern4, text_upper)
    if match:
        return match.group(1)
    
    return None


def evaluate_answer(model_output, ground_truth, question_type='numerical'):
    """Evaluate answer based on type."""
    if question_type == 'numerical':
        model_ans = extract_numerical_answer(model_output)
        gt_ans = extract_numerical_answer(ground_truth)
        
        if model_ans is None or gt_ans is None:
            return False
        
        return abs(model_ans - gt_ans) < 0.01
    
    elif question_type == 'mcq':
        model_ans = extract_mcq_answer(model_output)
        
        if isinstance(ground_truth, int):
            gt_ans = chr(65 + ground_truth)
        else:
            gt_ans = str(ground_truth).upper().strip()
        
        return model_ans == gt_ans if model_ans else False
    
    return False

# ============================================================
# Tinker Sampling with Retry Logic
# ============================================================

def sample_from_tinker_robust(client, tokenizer, problem_text, max_tokens=400, max_retries=3):
    """Sample with retry logic and error handling."""
    
    for attempt in range(max_retries):
        try:
            prompt_tokens = tokenizer.encode(problem_text)
            prompt = types.ModelInput.from_ints(prompt_tokens)
            
            sampling_params = types.SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0
            )
            
            future = client.sample(
                prompt=prompt,
                num_samples=1,
                sampling_params=sampling_params
            )
            
            result = future.result()
            
            # Try different response structures
            try:
                output_tokens = result.sequences[0].tokens
            except (AttributeError, IndexError):
                try:
                    output_tokens = result.samples[0].tokens
                except:
                    output_tokens = result.tokens
            
            output_text = tokenizer.decode(output_tokens)
            token_count = len(output_tokens)
            
            return output_text, token_count
            
        except Exception as e:
            logger.warning(f"Sampling attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(config.RETRY_DELAY)
            else:
                logger.error(f"All retry attempts failed: {e}")
                return None, 0
    
    return None, 0

# ============================================================
# Dataset Loaders
# ============================================================

def load_gsm8k():
    """Load GSM8K dataset."""
    logger.info("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")
    return dataset['test']


def load_sat_mixed():
    """Load MMLU mixed subjects as SAT replacement."""
    logger.info("Loading MMLU mixed subjects...")
    
    try:
        from datasets import load_dataset
        
        mixed_data = []
        
        # Math subjects
        math_subjects = ['high_school_mathematics', 'elementary_mathematics', 'abstract_algebra']
        
        # Humanities subjects  
        humanities_subjects = ['high_school_us_history', 'philosophy', 'moral_scenarios']
        
        for subject in math_subjects:
            dataset = load_dataset("cais/mmlu", subject, split='test')
            for ex in dataset:
                mixed_data.append({
                    'question': ex['question'],
                    'choices': ex['choices'],
                    'answer': ex['answer'],  # 0, 1, 2, 3
                    'type': 'math',
                    'passage': ''
                })
        
        for subject in humanities_subjects:
            dataset = load_dataset("cais/mmlu", subject, split='test')
            for ex in dataset:
                mixed_data.append({
                    'question': ex['question'],
                    'choices': ex['choices'],
                    'answer': ex['answer'],
                    'type': 'humanities',
                    'passage': ''
                })
        
        logger.info(f"Loaded {len(mixed_data)} MMLU problems")
        logger.info(f"  - Math: {sum(1 for x in mixed_data if x['type']=='math')}")
        logger.info(f"  - Humanities: {sum(1 for x in mixed_data if x['type']=='humanities')}")
        
        return mixed_data
        
    except Exception as e:
        logger.error(f"Failed to load MMLU data: {e}")
        return []

# ============================================================
# Prompt Formatters
# ============================================================

def format_gsm8k_prompt(question):
    """Format GSM8K prompt."""
    return f"Solve this math problem step by step:\n\n{question}\n\nSolution:"

def format_sat_prompt(problem):
    """Format SAT/MMLU MCQ prompt - uses Llama chat format."""
    
    # Llama-3 chat format with system message
    prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    prompt += "You are a test-taking assistant. When given a multiple choice question, you respond with ONLY a single letter: A, B, C, or D. You do not explain, calculate, or write anything else. Just the letter."
    prompt += "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    
    # Add passage if exists
    if problem.get('passage') and problem['passage'].strip():
        prompt += f"Context: {problem['passage']}\n\n"
    
    # Add question
    prompt += f"Question: {problem['question']}\n\n"
    
    # Add choices
    if problem.get('choices'):
        for i, choice in enumerate(problem['choices']):
            letter = chr(65 + i)  # A, B, C, D
            prompt += f"{letter}) {choice}\n"
    
    prompt += "\nAnswer (one letter only):"
    prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    return prompt

# ============================================================
# Checkpoint Management
# ============================================================

class CheckpointManager:
    """Manage checkpoints for resumable execution."""
    
    def __init__(self, dataset_name, model_name):
        self.dataset_name = dataset_name
        self.model_name = model_name.replace('/', '_')
        self.checkpoint_dir = Path(config.CHECKPOINT_DIR) / dataset_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_file = self.checkpoint_dir / f"{self.model_name}_checkpoint.json"
        self.results_file = self.checkpoint_dir / f"{self.model_name}_results.csv"
    
    def load_checkpoint(self):
        """Load existing checkpoint if available."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"Loaded checkpoint: {checkpoint['completed_samples']} samples completed")
            return checkpoint
        return None
    
    def save_checkpoint(self, results, completed_samples, total_samples):
        """Save checkpoint."""
        checkpoint = {
            'completed_samples': completed_samples,
            'total_samples': total_samples,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)
        
        # Save results DataFrame
        df = pd.DataFrame(results)
        df.to_csv(self.results_file, index=False)
        
        logger.info(f"Checkpoint saved: {completed_samples}/{total_samples} samples")
    
    def load_results(self):
        """Load existing results."""
        if self.results_file.exists():
            return pd.read_csv(self.results_file).to_dict('records')
        return []
    
    def finalize(self, results, stats):
        """Save final results."""
        # Save results
        df = pd.DataFrame(results)
        final_file = Path(config.RESULTS_DIR) / f"{self.dataset_name}_{self.model_name}_final.csv"
        df.to_csv(final_file, index=False)
        
        # Save stats
        stats_file = Path(config.RESULTS_DIR) / f"{self.dataset_name}_{self.model_name}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Remove checkpoint
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        
        logger.info(f"Final results saved to {final_file}")

# ============================================================
# Main Evaluation Function
# ============================================================

def evaluate_dataset(dataset_name, dataset, client, tokenizer, model_name, 
                    max_samples, max_tokens, question_type='numerical'):
    """
    Evaluate model on dataset with checkpointing and error handling.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting: {dataset_name} - {model_name}")
    logger.info(f"{'='*60}")
    
    # Setup checkpoint manager
    checkpoint_mgr = CheckpointManager(dataset_name, model_name)
    
    # Load checkpoint if exists
    checkpoint = checkpoint_mgr.load_checkpoint()
    if checkpoint:
        results = checkpoint_mgr.load_results()
        start_idx = checkpoint['completed_samples']
        logger.info(f"Resuming from sample {start_idx}")
    else:
        results = []
        start_idx = 0
    
    # Prepare dataset subset
    if isinstance(dataset, list):
        subset = dataset[start_idx:max_samples]
    else:
        subset = list(dataset.select(range(start_idx, min(max_samples, len(dataset)))))
    
    # Statistics tracking
    correct = sum(1 for r in results if r['correct'])
    total_tokens = sum(r['tokens'] for r in results)
    
    # Domain-specific tracking for SAT
    domain_stats = {}
    
    # Evaluation loop
    pbar = tqdm(enumerate(subset, start=start_idx), 
                total=max_samples, 
                initial=start_idx,
                desc=f"{model_name}")
    
    for i, example in pbar:
        # Check for shutdown signal
        if killer.kill_now:
            logger.warning("Shutdown signal received. Saving progress...")
            checkpoint_mgr.save_checkpoint(results, len(results), max_samples)
            sys.exit(0)
        
        # Format prompt based on dataset
        if dataset_name == 'gsm8k':
            question = example['question']
            ground_truth = example['answer']
            prompt = format_gsm8k_prompt(question)
            q_type = 'numerical'
        elif dataset_name == 'sat':
            question = example['question']
            ground_truth = example['answer']
            prompt = format_sat_prompt(example)
            q_type = 'mcq'
            domain = example.get('type', 'unknown')
        
        # Sample from model
        output, tokens = sample_from_tinker_robust(
            client, tokenizer, prompt, max_tokens, config.MAX_RETRIES
        )
        
        if output is None:
            logger.warning(f"Sample {i} failed after retries. Skipping.")
            continue
        
        # Evaluate
        is_correct = evaluate_answer(output, ground_truth, q_type)
        correct += int(is_correct)
        total_tokens += tokens
        
        # Track domain stats for SAT
        if dataset_name == 'sat':
            if domain not in domain_stats:
                domain_stats[domain] = {'correct': 0, 'total': 0}
            domain_stats[domain]['total'] += 1
            domain_stats[domain]['correct'] += int(is_correct)
        
        # Save result
        result = {
            'question': question,
            'output': output,
            'correct': is_correct,
            'tokens': tokens
        }
        
        if dataset_name == 'sat':
            result['type'] = domain
        
        results.append(result)
        
        # Update progress bar
        accuracy = correct / len(results)
        avg_tokens = total_tokens / len(results)
        pbar.set_postfix({
            'acc': f'{accuracy:.1%}',
            'tokens': f'{avg_tokens:.0f}'
        })
        
        # Checkpoint
        if (len(results)) % config.CHECKPOINT_INTERVAL == 0:
            checkpoint_mgr.save_checkpoint(results, len(results), max_samples)
    
    # Calculate final statistics
    accuracy = correct / len(results) if results else 0
    avg_tokens = total_tokens / len(results) if results else 0
    
    stats = {
        'dataset': dataset_name,
        'model': model_name,
        'total_samples': len(results),
        'correct': correct,
        'accuracy': accuracy,
        'avg_tokens': avg_tokens,
        'timestamp': datetime.now().isoformat()
    }
    
    # Add domain stats for SAT
    if dataset_name == 'sat':
        for domain, domain_data in domain_stats.items():
            stats[f'{domain}_accuracy'] = domain_data['correct'] / domain_data['total']
            stats[f'{domain}_correct'] = domain_data['correct']
            stats[f'{domain}_total'] = domain_data['total']
    
    # Log results
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS: {dataset_name} - {model_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy: {accuracy:.2%} ({correct}/{len(results)})")
    logger.info(f"Avg Tokens: {avg_tokens:.1f}")
    
    if dataset_name == 'sat':
        for domain in domain_stats:
            domain_acc = domain_stats[domain]['correct'] / domain_stats[domain]['total']
            logger.info(f"{domain.capitalize()} Accuracy: {domain_acc:.2%}")
    
    logger.info(f"{'='*60}\n")
    
    # Finalize
    checkpoint_mgr.finalize(results, stats)
    
    return results, stats

# ============================================================
# Tinker Client Setup
# ============================================================

def setup_tinker():
    """Initialize Tinker clients."""
    logger.info("Initializing Tinker clients...")
    
    os.environ['TINKER_API_KEY'] = config.TINKER_API_KEY
    
    try:
        from tinker_cookbook import tokenizer_utils
        
        service_client = tinker.ServiceClient()
        
        # Small model
        small_client = service_client.create_sampling_client(base_model=config.SMALL_MODEL)
        small_tokenizer = tokenizer_utils.get_tokenizer(config.SMALL_MODEL)
        
        # Large model
        large_client = service_client.create_sampling_client(base_model=config.LARGE_MODEL)
        large_tokenizer = tokenizer_utils.get_tokenizer(config.LARGE_MODEL)
        
        logger.info(f"✓ Small model: {config.SMALL_MODEL}")
        logger.info(f"✓ Large model: {config.LARGE_MODEL}")
        
        return (small_client, small_tokenizer, large_client, large_tokenizer)
        
    except Exception as e:
        logger.error(f"Failed to initialize Tinker: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

# ============================================================
# Main Execution
# ============================================================

def main():
    """Main execution function."""
    start_time = time.time()
    
    logger.info("="*60)
    logger.info("BASELINE EVALUATION - PRODUCTION RUN")
    logger.info("="*60)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directories
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Setup Tinker
    small_client, small_tokenizer, large_client, large_tokenizer = setup_tinker()
    
    # Track all results
    all_stats = []
    
    # ============================================================
    # GSM8K Evaluation
    # ============================================================
    
    if config.DATASETS['gsm8k']['enabled']:
        try:
            logger.info("\n" + "="*60)
            logger.info("DATASET: GSM8K")
            logger.info("="*60)
            
            gsm8k_data = load_gsm8k()
            
            # Small model
            logger.info("\n>>> Evaluating Small Model on GSM8K")
            _, stats = evaluate_dataset(
                'gsm8k',
                gsm8k_data,
                small_client,
                small_tokenizer,
                config.SMALL_MODEL,
                config.DATASETS['gsm8k']['num_samples'],
                config.DATASETS['gsm8k']['max_tokens'],
                'numerical'
            )
            all_stats.append(stats)
            
            # Large model
            logger.info("\n>>> Evaluating Large Model on GSM8K")
            _, stats = evaluate_dataset(
                'gsm8k',
                gsm8k_data,
                large_client,
                large_tokenizer,
                config.LARGE_MODEL,
                config.DATASETS['gsm8k']['num_samples'],
                config.DATASETS['gsm8k']['max_tokens'],
                'numerical'
            )
            all_stats.append(stats)
            
        except Exception as e:
            logger.error(f"GSM8K evaluation failed: {e}")
            logger.error(traceback.format_exc())
    
    # ============================================================
    # SAT Evaluation
    # ============================================================
    if config.DATASETS['sat']['enabled']:
        try:
            logger.info("\n" + "="*60)
            logger.info("DATASET: MMLU (Mixed Math + Humanities)")  # Updated
            logger.info("="*60)
            
            sat_data = load_sat_mixed()
            
            if sat_data:
                # Small model
                logger.info("\n>>> Evaluating Small Model on SAT")
                _, stats = evaluate_dataset(
                    'sat',
                    sat_data,
                    small_client,
                    small_tokenizer,
                    config.SMALL_MODEL,
                    config.DATASETS['sat']['num_samples'],
                    config.DATASETS['sat']['max_tokens'],
                    'mcq'
                )
                all_stats.append(stats)
                
                # Large model
                logger.info("\n>>> Evaluating Large Model on SAT")
                _, stats = evaluate_dataset(
                    'sat',
                    sat_data,
                    large_client,
                    large_tokenizer,
                    config.LARGE_MODEL,
                    config.DATASETS['sat']['num_samples'],
                    config.DATASETS['sat']['max_tokens'],
                    'mcq'
                )
                all_stats.append(stats)
            
        except Exception as e:
            logger.error(f"SAT evaluation failed: {e}")
            logger.error(traceback.format_exc())
    
    # ============================================================
    # Summary
    # ============================================================
    
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total duration: {duration/3600:.2f} hours")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save summary
    summary_file = Path(config.RESULTS_DIR) / 'summary_all.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'stats': all_stats,
            'duration_hours': duration/3600,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    logger.info(f"\nSummary saved to: {summary_file}")
    logger.info("All results saved to: " + config.RESULTS_DIR)
    
    # Print summary table
    logger.info("\n" + "="*60)
    logger.info("SUMMARY TABLE")
    logger.info("="*60)
    
    for stat in all_stats:
        logger.info(f"\n{stat['dataset'].upper()} - {stat['model'].split('/')[-1]}")
        logger.info(f"  Accuracy: {stat['accuracy']:.2%}")
        logger.info(f"  Avg Tokens: {stat['avg_tokens']:.1f}")
        
        if stat['dataset'] == 'sat':
            if 'math_accuracy' in stat:
                logger.info(f"  Math Accuracy: {stat['math_accuracy']:.2%}")
            if 'english_accuracy' in stat:
                logger.info(f"  English Accuracy: {stat['english_accuracy']:.2%}")
    
    logger.info("\n" + "="*60)
    logger.info("Done! 🎉")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user. Progress saved.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\nFatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)