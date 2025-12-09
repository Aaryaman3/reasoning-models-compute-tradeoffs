#!/usr/bin/env python3
"""
Generate router training data from baseline results.
Creates labels: 0 (route to small) or 1 (route to large)
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# Cost Analysis Configuration
# ============================================================

# Pricing per 1K tokens (adjust based on actual Tinker pricing)
PRICING = {
    'small': 0.0001,  # Llama-3.1-8B per 1K tokens
    'large': 0.001,   # DeepSeek-V3.1 per 1K tokens (10x)
}

# Parameter counts for compute estimation
PARAMETERS = {
    'small': 8e9,     # 8 billion
    'large': 37e9,    # 37 billion active (MoE)
}

def calculate_cost(model_type, total_tokens):
    """Calculate monetary cost."""
    price_per_1k = PRICING[model_type]
    return (total_tokens / 1000) * price_per_1k

def calculate_compute_units(model_type, total_tokens):
    """Calculate compute units (FLOPs proxy)."""
    params = PARAMETERS[model_type]
    return params * total_tokens / 1e12  # Return in TFLOPs

# ============================================================
# Generate Training Data
# ============================================================

def generate_from_existing_results(results_dir='./results', dataset='gsm8k'):
    """
    Generate router training data from existing baseline results.
    Fast method - uses data we already have.
    """
    logger.info("="*60)
    logger.info(f"Generating Router Training Data from {dataset.upper()}")
    logger.info("="*60)
    
    # Load baseline results
    small_file = Path(results_dir) / f'{dataset}_meta-llama_Llama-3.1-8B-Instruct_final.csv'
    large_file = Path(results_dir) / f'{dataset}_deepseek-ai_DeepSeek-V3.1_final.csv'
    
    if not small_file.exists() or not large_file.exists():
        logger.error(f"Missing baseline results files for {dataset}!")
        logger.error(f"  Expected: {small_file}")
        logger.error(f"  Expected: {large_file}")
        return None
    
    logger.info(f"Loading baseline results from {dataset}...")
    small_df = pd.read_csv(small_file)
    large_df = pd.read_csv(large_file)
    
    # Verify lengths match
    if len(small_df) != len(large_df):
        logger.error(f"Result lengths don't match! Small: {len(small_df)}, Large: {len(large_df)}")
        return None
    
    logger.info(f"✓ Loaded {len(small_df)} samples")
    
    # Create training data
    training_data = pd.DataFrame({
        'question': small_df['question'],
        'small_correct': small_df['correct'],
        'large_correct': large_df['correct'],
        'small_tokens': small_df['tokens'],
        'large_tokens': large_df['tokens'],
    })
    
    # Generate labels
    def assign_label(row):
        """
        Label = 0: Use small model (small is sufficient)
        Label = 1: Use large model (small fails, large succeeds)
        
        Logic:
        - If small correct → small sufficient (label 0)
        - If small wrong but large correct → need large (label 1)
        - If both wrong → use cheap small (label 0)
        """
        if row['small_correct']:
            return 0  # Small is sufficient
        elif row['large_correct']:
            return 1  # Need large
        else:
            return 0  # Both fail, use cheap
    
    training_data['label'] = training_data.apply(assign_label, axis=1)
    
    # Calculate problem categories
    both_correct = ((training_data['small_correct']) & (training_data['large_correct'])).sum()
    only_large = ((~training_data['small_correct']) & (training_data['large_correct'])).sum()
    only_small = ((training_data['small_correct']) & (~training_data['large_correct'])).sum()
    both_wrong = ((~training_data['small_correct']) & (~training_data['large_correct'])).sum()
    
    # Cost analysis
    total_small_tokens = training_data['small_tokens'].sum()
    total_large_tokens = training_data['large_tokens'].sum()
    
    small_cost = calculate_cost('small', total_small_tokens)
    large_cost = calculate_cost('large', total_large_tokens)
    
    small_compute = calculate_compute_units('small', total_small_tokens)
    large_compute = calculate_compute_units('large', total_large_tokens)
    
    # Statistics
    total = len(training_data)
    label_counts = training_data['label'].value_counts()
    
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING DATA STATISTICS")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples: {total}")
    logger.info(f"\nLabel Distribution:")
    logger.info(f"  Route to Small (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/total*100:.1f}%)")
    logger.info(f"  Route to Large (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/total*100:.1f}%)")
    
    logger.info(f"\nProblem Breakdown:")
    logger.info(f"  Both correct: {both_correct} ({both_correct/total*100:.1f}%)")
    logger.info(f"  Only large correct: {only_large} ({only_large/total*100:.1f}%) ← Router target!")
    logger.info(f"  Only small correct: {only_small} ({only_small/total*100:.1f}%)")
    logger.info(f"  Both wrong: {both_wrong} ({both_wrong/total*100:.1f}%)")
    
    logger.info(f"\n{'='*60}")
    logger.info("COST ANALYSIS")
    logger.info(f"{'='*60}")
    logger.info(f"\nToken Usage:")
    logger.info(f"  Small model: {total_small_tokens:,} tokens")
    logger.info(f"  Large model: {total_large_tokens:,} tokens")
    logger.info(f"  Ratio: {total_large_tokens/total_small_tokens:.2f}x")
    
    logger.info(f"\nEstimated Costs (hypothetical pricing):")
    logger.info(f"  Small model: ${small_cost:.2f}")
    logger.info(f"  Large model: ${large_cost:.2f}")
    logger.info(f"  Cost ratio: {large_cost/small_cost:.1f}x more expensive")
    
    logger.info(f"\nComputational Cost:")
    logger.info(f"  Small model: {small_compute:.1f} TFLOPs")
    logger.info(f"  Large model: {large_compute:.1f} TFLOPs")
    logger.info(f"  Compute ratio: {large_compute/small_compute:.1f}x")
    
    # Accuracy metrics
    small_acc = training_data['small_correct'].mean()
    large_acc = training_data['large_correct'].mean()
    
    logger.info(f"\nAccuracy:")
    logger.info(f"  Small model: {small_acc*100:.1f}%")
    logger.info(f"  Large model: {large_acc*100:.1f}%")
    logger.info(f"  Gap: {(large_acc-small_acc)*100:.1f} percentage points")
    
    logger.info(f"\nCost per Correct Answer:")
    logger.info(f"  Small: ${small_cost/(small_acc*total):.4f}")
    logger.info(f"  Large: ${large_cost/(large_acc*total):.4f}")
    
    # Router value proposition
    logger.info(f"\n{'='*60}")
    logger.info("ROUTER VALUE PROPOSITION")
    logger.info(f"{'='*60}")
    
    # Optimal routing (only use large when small fails AND large succeeds)
    optimal_large_usage = only_large / total
    optimal_cost = small_cost * (1 - optimal_large_usage) + large_cost * optimal_large_usage
    optimal_accuracy = (both_correct + only_large) / total
    
    logger.info(f"\nOptimal Router Strategy:")
    logger.info(f"  Use large for: {only_large} problems ({optimal_large_usage*100:.1f}%)")
    logger.info(f"  Accuracy: {optimal_accuracy*100:.1f}%")
    logger.info(f"  Cost: ${optimal_cost:.2f}")
    logger.info(f"  Savings vs Always-Large: {(1 - optimal_cost/large_cost)*100:.0f}%")
    logger.info(f"  Accuracy loss vs Always-Large: {(large_acc - optimal_accuracy)*100:.1f}%")
    
    # Check if distribution is reasonable
    if label_counts.get(1, 0) / total < 0.10:
        logger.warning("\n⚠️  WARNING: Very few 'route to large' labels (<10%)")
        logger.warning("    Router might struggle to learn when to use large model")
        logger.warning("    Small model performance is very good!")
    elif label_counts.get(1, 0) / total > 0.50:
        logger.warning("\n⚠️  WARNING: Many 'route to large' labels (>50%)")
        logger.warning("    Small model struggles significantly")
        logger.warning("    Router will mostly route to large")
    else:
        logger.info("\n✓ Label distribution looks good for training")
    
    # Save training data
    output_file = f'router_training_data_{dataset}.csv'
    training_data.to_csv(output_file, index=False)
    
    logger.info(f"\n✓ Saved to: {output_file}")
    logger.info(f"{'='*60}\n")
    
    # Save summary stats
    stats = {
        'dataset': dataset,
        'total_samples': total,
        'label_distribution': {
            'small': int(label_counts.get(0, 0)),
            'large': int(label_counts.get(1, 0))
        },
        'problem_breakdown': {
            'both_correct': int(both_correct),
            'only_large_correct': int(only_large),
            'only_small_correct': int(only_small),
            'both_wrong': int(both_wrong)
        },
        'percentages': {
            'route_to_small': float(label_counts.get(0, 0) / total * 100),
            'route_to_large': float(label_counts.get(1, 0) / total * 100)
        },
        'costs': {
            'small_total_tokens': int(total_small_tokens),
            'large_total_tokens': int(total_large_tokens),
            'small_cost': float(small_cost),
            'large_cost': float(large_cost),
            'optimal_router_cost': float(optimal_cost),
            'savings_vs_always_large': float((1 - optimal_cost/large_cost)*100)
        },
        'accuracy': {
            'small': float(small_acc),
            'large': float(large_acc),
            'optimal_router': float(optimal_accuracy)
        }
    }
    
    stats_file = f'router_training_stats_{dataset}.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"✓ Saved statistics to: {stats_file}")
    
    return training_data


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    logger.info("\n" + "="*60)
    logger.info("ROUTER TRAINING DATA GENERATION")
    logger.info("="*60)
    
    # Generate from GSM8K
    logger.info("\nGenerating training data from GSM8K baseline results...")
    gsm8k_data = generate_from_existing_results(dataset='gsm8k')
    
    if gsm8k_data is not None:
        logger.info("\n✓ GSM8K router training data ready!")
    else:
        logger.error("\n❌ Failed to generate GSM8K training data")
    
    # Optionally generate from MMLU (if you want)
    # logger.info("\nGenerating training data from MMLU baseline results...")
    # mmlu_data = generate_from_existing_results(dataset='sat')
    
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    if gsm8k_data is not None:
        logger.info("\n✓ Router training data generation complete!")
        logger.info("\nGenerated files:")
        logger.info("  - router_training_data_gsm8k.csv")
        logger.info("  - router_training_stats_gsm8k.json")
        logger.info("\nNext steps:")
        logger.info("  1. Review the training data distribution")
        logger.info("  2. Train router: python train_router.py")
        logger.info("  3. Evaluate router performance")
    else:
        logger.error("\n❌ Training data generation failed")
        logger.error("Check that baseline result files exist in ./results/")
    
    logger.info("\n" + "="*60)