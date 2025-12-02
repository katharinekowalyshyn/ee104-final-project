"""
Create custom training/test split from full Birkbeck dataset.

Usage:
    python create_custom_split.py --train_size 15000 --test_size 3000
"""

import sys
import os
import argparse
import random

sys.path.insert(0, os.path.dirname(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location("norvig_spell", "norvig-spell.py")
norvig_spell = importlib.util.module_from_spec(spec)
spec.loader.exec_module(norvig_spell)

BirkbeckTestset = norvig_spell.BirkbeckTestset

def create_custom_split(train_file='datasets/birbeck_train.txt', 
                        test_file='datasets/birbeck_test.txt',
                        train_size=15000,
                        test_size=3000,
                        output_train='datasets/birbeck_custom_train.txt',
                        output_test='datasets/birbeck_custom_test.txt',
                        seed=42):
    """
    Create custom train/test split from full dataset.
    
    Strategy:
    1. Load all pairs from both files
    2. Shuffle randomly
    3. Take first train_size for training, next test_size for test
    """
    print("="*60)
    print("Creating Custom Train/Test Split")
    print("="*60)
    
    # Load all pairs
    print(f"\nStep 1: Loading full dataset...")
    train_pairs = BirkbeckTestset(open(train_file))
    test_pairs = BirkbeckTestset(open(test_file))
    all_pairs = train_pairs + test_pairs
    
    print(f"  Training file: {len(train_pairs):,} pairs")
    print(f"  Test file: {len(test_pairs):,} pairs")
    print(f"  Total: {len(all_pairs):,} pairs")
    
    # Check if we have enough data
    if len(all_pairs) < train_size + test_size:
        print(f"\n⚠ WARNING: Not enough data!")
        print(f"  Need: {train_size + test_size:,} pairs")
        print(f"  Have: {len(all_pairs):,} pairs")
        print(f"  Using all available data...")
        train_size = min(train_size, len(all_pairs) - test_size)
        test_size = min(test_size, len(all_pairs) - train_size)
    
    # Shuffle
    print(f"\nStep 2: Shuffling data (seed={seed})...")
    random.seed(seed)
    shuffled = all_pairs.copy()
    random.shuffle(shuffled)
    print(f"  ✓ Shuffled {len(shuffled):,} pairs")
    
    # Split
    print(f"\nStep 3: Splitting data...")
    train_split = shuffled[:train_size]
    test_split = shuffled[train_size:train_size + test_size]
    
    print(f"  Training set: {len(train_split):,} pairs")
    print(f"  Test set: {len(test_split):,} pairs")
    print(f"  Remaining: {len(shuffled) - train_size - test_size:,} pairs (not used)")
    
    # Group by correct word (Birkbeck format)
    def group_by_word(pairs):
        """Group pairs by correct word (Birkbeck format: $word followed by misspellings)."""
        word_groups = {}
        for correct, wrong in pairs:
            if correct not in word_groups:
                word_groups[correct] = []
            word_groups[correct].append(wrong)
        return word_groups
    
    print(f"\nStep 4: Formatting in Birkbeck format...")
    train_groups = group_by_word(train_split)
    test_groups = group_by_word(test_split)
    
    print(f"  Training: {len(train_groups):,} word groups")
    print(f"  Test: {len(test_groups):,} word groups")
    
    # Write training set
    print(f"\nStep 5: Writing training set to {output_train}...")
    with open(output_train, 'w') as f:
        for correct in sorted(train_groups.keys()):
            f.write(f'${correct}\n')
            for wrong in train_groups[correct]:
                f.write(f'{wrong}\n')
    print(f"  ✓ Saved {len(train_split):,} pairs ({len(train_groups):,} word groups)")
    
    # Write test set
    print(f"\nStep 6: Writing test set to {output_test}...")
    with open(output_test, 'w') as f:
        for correct in sorted(test_groups.keys()):
            f.write(f'${correct}\n')
            for wrong in test_groups[correct]:
                f.write(f'{wrong}\n')
    print(f"  ✓ Saved {len(test_split):,} pairs ({len(test_groups):,} word groups)")
    
    # Verify
    print(f"\nStep 7: Verifying files...")
    verify_train = BirkbeckTestset(open(output_train))
    verify_test = BirkbeckTestset(open(output_test))
    
    print(f"  Training file: {len(verify_train):,} pairs ✓")
    print(f"  Test file: {len(verify_test):,} pairs ✓")
    
    if len(verify_train) == train_size and len(verify_test) == test_size:
        print(f"\n✓ SUCCESS: Files created correctly!")
    else:
        print(f"\n⚠ WARNING: Size mismatch!")
        print(f"  Expected: {train_size:,} train, {test_size:,} test")
        print(f"  Got: {len(verify_train):,} train, {len(verify_test):,} test")
    
    print("\n" + "="*60)
    print("Split Complete!")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  Training: {output_train} ({len(verify_train):,} pairs)")
    print(f"  Test: {output_test} ({len(verify_test):,} pairs)")

def main():
    parser = argparse.ArgumentParser(description='Create custom train/test split')
    parser.add_argument('--train_size', type=int, default=15000,
                       help='Number of training pairs (default: 15000)')
    parser.add_argument('--test_size', type=int, default=3000,
                       help='Number of test pairs (default: 3000)')
    parser.add_argument('--train_file', type=str, default='datasets/birbeck_train.txt',
                       help='Input training file (default: datasets/birbeck_train.txt)')
    parser.add_argument('--test_file', type=str, default='datasets/birbeck_test.txt',
                       help='Input test file (default: datasets/birbeck_test.txt)')
    parser.add_argument('--output_train', type=str, default='datasets/birbeck_custom_train.txt',
                       help='Output training file (default: datasets/birbeck_custom_train.txt)')
    parser.add_argument('--output_test', type=str, default='datasets/birbeck_custom_test.txt',
                       help='Output test file (default: datasets/birbeck_custom_test.txt)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    create_custom_split(
        train_file=args.train_file,
        test_file=args.test_file,
        train_size=args.train_size,
        test_size=args.test_size,
        output_train=args.output_train,
        output_test=args.output_test,
        seed=args.seed
    )

if __name__ == "__main__":
    main()

