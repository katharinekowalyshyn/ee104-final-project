"""
Bayesian Spell Checker - Kernighan-Church-Gale (1990) Method

Implements the noisy channel model from:
"A Spelling Correction Program Based on a Noisy Channel Model"
Kernighan, Church, and Gale (1990)

Key features:
- Pr(correct|typo) ∝ Pr(correct) * Pr(typo|correct)
- Pr(correct): Language model from word frequencies
- Pr(typo|correct): Error model using confusion matrices trained from actual typos
- Uses empirical confusion matrices (not QWERTY proximity)

Usage:
    python spell-check.py --train datasets/birbeck_large_train.txt --test datasets/birbeck_large_test.txt
"""

import math
import sys
import os
import json
import argparse
from collections import defaultdict, Counter
from typing import List, Tuple, Set, Optional
import time

# Import functions from norvig-spell.py
sys.path.insert(0, os.path.dirname(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location("norvig_spell", "norvig-spell.py")
norvig_spell = importlib.util.module_from_spec(spec)
spec.loader.exec_module(norvig_spell)

# Import Norvig's functions
edits1 = norvig_spell.edits1
edits2 = norvig_spell.edits2
BirkbeckTestset = norvig_spell.BirkbeckTestset

# Get base WORDS dictionary from big.txt
BASE_WORDS = norvig_spell.WORDS

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ============================================================================
# CONSTANTS
# ============================================================================

MAX_EDIT_DISTANCE = 2
NGRAM_SIMILARITY_THRESHOLD = 0.3
LENGTH_TOLERANCE = 3

MIN_PROB = 1e-10

# ============================================================================
# VOCABULARY BUILDING - THE KEY FIX
# ============================================================================

def build_vocabulary(base_words: Counter, 
                     train_tests: List[Tuple[str, str]],
                     include_train_correct_words: bool = True) -> Tuple[Counter, Set[str]]:
    """
    Build vocabulary following Norvig's approach.
    
    Vocabulary = big.txt (general English) + optionally training correct words (domain-specific)
    
    IMPORTANT: Does NOT include:
    - Test set words (data leakage!)
    - Misspellings from training set (they're typos, not valid words)
    
    Args:
        base_words: Words from big.txt (general English corpus, like Norvig)
        train_tests: Training typo pairs (for extracting domain-specific correct words)
        include_train_correct_words: If True, add correct words from training set
                                     (helps with domain-specific vocabulary)
    
    Returns:
        - Word frequency dictionary (for language model)
        - Set of known words (for candidate generation)
    """
    print("\n" + "="*60)
    print("Building Vocabulary (Norvig-style)")
    print("="*60)
    
    # Start with base WORDS from big.txt (like Norvig)
    vocab_words = Counter(base_words)
    print(f"\n1. Base vocabulary from big.txt: {len(base_words):,} words")
    print(f"   Total tokens: {sum(base_words.values()):,}")
    
    if include_train_correct_words:
        # Add correct words from training set (domain-specific vocabulary)
        # But NOT misspellings - those are typos, not valid words!
        train_correct_words = Counter()
        for correct, wrong in train_tests:
            correct_lower = correct.lower()
            if correct_lower.isalpha():
                train_correct_words[correct_lower] += 1
        
        # Find new words (not in big.txt)
        new_words = set(train_correct_words.keys()) - set(base_words.keys())
        print(f"\n2. Training set correct words: {len(train_correct_words):,} unique words")
        print(f"   New words (not in big.txt): {len(new_words):,} words")
        
        if new_words:
            print(f"\n   Sample new domain-specific words (first 20):")
            for i, word in enumerate(sorted(new_words)[:20], 1):
                count = train_correct_words[word]
                print(f"   {i:2d}. {word:20s} (appears {count:3d} times in training)")
        
        # Merge: Use big.txt frequencies for words in both
        # Add new words with minimal frequency (for domain-specific vocabulary)
        for word, count in train_correct_words.items():
            if word in vocab_words:
                # Word exists in big.txt - keep big.txt frequency (it's more reliable)
                pass
            else:
                # New domain-specific word - add with minimal frequency
                vocab_words[word] = max(count, 1)
        
        print(f"\n3. Final vocabulary: {len(vocab_words):,} words")
        print(f"   → Added {len(new_words):,} domain-specific words from training set")
    else:
        print(f"\n2. Using ONLY big.txt vocabulary (no training words added)")
        print(f"   Final vocabulary: {len(vocab_words):,} words")
    
    # Create set for fast membership testing
    vocab_set = set(vocab_words.keys())
    
    print(f"\n✓ Vocabulary ready for candidate generation")
    print(f"  (Test set words will NOT be included - proper evaluation!)")
    
    return vocab_words, vocab_set

# ============================================================================
# HELPER FUNCTIONS (same as before)
# ============================================================================

def character_ngram_similarity(s1: str, s2: str, n: int = 2) -> float:
    """Character n-gram similarity."""
    def get_ngrams(s):
        s = s.lower()
        return set(s[i:i+n] for i in range(len(s) - n + 1))
    
    ngrams1, ngrams2 = get_ngrams(s1), get_ngrams(s2)
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    return intersection / union if union > 0 else 0.0

# ============================================================================
# MODELS
# ============================================================================

class LanguageModel:
    """Language model using COMBINED vocabulary."""
    
    def __init__(self, word_counts: Counter):
        self.WORDS = word_counts
        self.total_words = sum(word_counts.values())
        print(f"\nLanguage Model initialized:")
        print(f"  Vocabulary: {len(self.WORDS):,} words")
        print(f"  Total tokens: {self.total_words:,}")
    
    def probability(self, word: str) -> float:
        """
        P(word) with Laplace smoothing (KCG method).
        KCG uses: (freq(c) + 0.5) / N
        """
        word_lower = word.lower()
        # KCG smoothing: (freq + 0.5) / (total + 0.5 * vocab_size)
        freq = self.WORDS.get(word_lower, 0)
        return (freq + 0.5) / (self.total_words + 0.5 * len(self.WORDS))
    
    def log_probability(self, word: str) -> float:
        """Log probability."""
        return math.log(self.probability(word) + MIN_PROB)

class ErrorModel:
    """
    Error model using confusion matrices (Kernighan-Church-Gale method).
    Trained from actual typo data, not QWERTY proximity.
    """
    
    def __init__(self, train_tests: Optional[List[Tuple[str, str]]] = None):
        # Confusion matrices: substitution, insertion, deletion, reversal
        self.substitution_counts = defaultdict(lambda: defaultdict(int))  # sub[X][Y] = count of X->Y
        self.insertion_counts = defaultdict(int)  # ins[X] = count of inserting X
        self.deletion_counts = defaultdict(int)  # del[X] = count of deleting X
        self.reversal_counts = defaultdict(lambda: defaultdict(int))  # rev[XY] = count of reversing XY
        
        # Total counts for normalization
        self.total_substitutions = 0
        self.total_insertions = 0
        self.total_deletions = 0
        self.total_reversals = 0
        
        if train_tests:
            self.train_from_corpus(train_tests)
        else:
            # Initialize with uniform priors if no training data
            self._initialize_uniform()
    
    def train_from_corpus(self, train_tests: List[Tuple[str, str]]):
        """Train confusion matrices from training corpus (KCG method)."""
        print("  Training error model from corpus...")
        
        for correct, typo in train_tests:
            self._extract_errors(correct.lower(), typo.lower())
        
        # Convert counts to probabilities with smoothing
        self._normalize_probabilities()
        
        print(f"    Trained on {len(train_tests):,} pairs")
        print(f"    Substitutions: {self.total_substitutions:,}, "
              f"Insertions: {self.total_insertions:,}, "
              f"Deletions: {self.total_deletions:,}, "
              f"Reversals: {self.total_reversals:,}")
    
    def _extract_errors(self, correct: str, typo: str):
        """Extract edit operations using dynamic programming alignment."""
        # Use DP to find optimal alignment
        m, n = len(typo), len(correct)
        dp = [[(0, None)] * (n + 1) for _ in range(m + 1)]
        
        # Initialize
        for i in range(m + 1):
            dp[i][0] = (i, 'ins')
        for j in range(n + 1):
            dp[0][j] = (j, 'del')
        dp[0][0] = (0, None)
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if typo[i-1] == correct[j-1]:
                    dp[i][j] = (dp[i-1][j-1][0], 'match')
                else:
                    # Substitution
                    sub_cost = dp[i-1][j-1][0] + 1
                    # Insertion
                    ins_cost = dp[i-1][j][0] + 1
                    # Deletion
                    del_cost = dp[i][j-1][0] + 1
                    
                    if sub_cost <= ins_cost and sub_cost <= del_cost:
                        dp[i][j] = (sub_cost, 'sub')
                    elif ins_cost <= del_cost:
                        dp[i][j] = (ins_cost, 'ins')
                    else:
                        dp[i][j] = (del_cost, 'del')
        
        # Backtrack to extract operations
        i, j = m, n
        operations = []
        while i > 0 or j > 0:
            op = dp[i][j][1]
            if op == 'match':
                i -= 1
                j -= 1
            elif op == 'sub':
                self.substitution_counts[correct[j-1]][typo[i-1]] += 1
                self.total_substitutions += 1
                operations.append(('sub', correct[j-1], typo[i-1]))
                i -= 1
                j -= 1
            elif op == 'ins':
                self.insertion_counts[typo[i-1]] += 1
                self.total_insertions += 1
                operations.append(('ins', typo[i-1]))
                i -= 1
            elif op == 'del':
                self.deletion_counts[correct[j-1]] += 1
                self.total_deletions += 1
                operations.append(('del', correct[j-1]))
                j -= 1
        
        # Check for reversals (adjacent transpositions) - check before other edits
        # This helps catch transpositions that might be split into substitutions
        if len(typo) == len(correct):
            for i in range(len(typo) - 1):
                if (typo[i] == correct[i+1] and typo[i+1] == correct[i] and 
                    typo[i] != correct[i]):
                    # Only count if not already accounted for by substitutions
                    if not any(op[0] == 'sub' and (op[1] == correct[i] or op[2] == correct[i]) 
                              for op in operations):
                        self.reversal_counts[correct[i]][correct[i+1]] += 1
                        self.total_reversals += 1
    
    def _normalize_probabilities(self):
        """Convert counts to probabilities with Laplace smoothing."""
        # Smoothing constant (like KCG's 0.5)
        alpha = 0.5
        
        # Normalize substitution probabilities
        chars = 'abcdefghijklmnopqrstuvwxyz'
        for c1 in chars:
            total = sum(self.substitution_counts[c1].values())
            if total > 0:
                for c2 in chars:
                    count = self.substitution_counts[c1][c2]
                    self.substitution_counts[c1][c2] = (count + alpha) / (total + alpha * 26)
        
        # Normalize insertion probabilities
        if self.total_insertions > 0:
            for c in chars:
                count = self.insertion_counts[c]
                self.insertion_counts[c] = (count + alpha) / (self.total_insertions + alpha * 26)
        
        # Normalize deletion probabilities
        if self.total_deletions > 0:
        for c in chars:
                count = self.deletion_counts[c]
                self.deletion_counts[c] = (count + alpha) / (self.total_deletions + alpha * 26)
        
        # Normalize reversal probabilities
        for c1 in chars:
            total = sum(self.reversal_counts[c1].values())
            if total > 0:
                for c2 in chars:
                    count = self.reversal_counts[c1][c2]
                    self.reversal_counts[c1][c2] = (count + alpha) / (total + alpha * 26)
    
    def _initialize_uniform(self):
        """Initialize with uniform probabilities if no training data."""
        chars = 'abcdefghijklmnopqrstuvwxyz'
        uniform_prob = 1.0 / 26
        
        for c1 in chars:
            for c2 in chars:
                if c1 != c2:
                    self.substitution_counts[c1][c2] = uniform_prob
            self.insertion_counts[c1] = uniform_prob
            self.deletion_counts[c1] = uniform_prob
    
    def substitution_prob(self, correct_char: str, typo_char: str) -> float:
        """P(typo_char | correct_char) for substitution."""
        return self.substitution_counts.get(correct_char.lower(), {}).get(typo_char.lower(), 0.0001)
    
    def insertion_prob(self, char: str) -> float:
        """P(insert char) for insertion."""
        return self.insertion_counts.get(char.lower(), 0.0001)
    
    def deletion_prob(self, char: str) -> float:
        """P(delete char) for deletion."""
        return self.deletion_counts.get(char.lower(), 0.0001)
    
    def reversal_prob(self, char1: str, char2: str) -> float:
        """P(reverse char1 and char2) for reversal."""
        return self.reversal_counts.get(char1.lower(), {}).get(char2.lower(), 0.0001)

# ============================================================================
# ERROR PROBABILITY
# ============================================================================

def calculate_error_probability(typo: str, correct: str, error_model: ErrorModel) -> float:
    """
    Calculate P(typo | correct) using confusion matrices (KCG method).
    Improved: Better handling of multi-edit errors and learned operation weights.
    """
    typo_lower, correct_lower = typo.lower(), correct.lower()
    
    if typo_lower == correct_lower:
        return 1.0
    
    m, n = len(typo_lower), len(correct_lower)
    
    # Learn operation weights from error model statistics
    # Higher weight for more common operations
    total_ops = (error_model.total_substitutions + error_model.total_insertions + 
                 error_model.total_deletions + error_model.total_reversals)
    
    if total_ops > 0:
        match_weight = 0.95  # Matches are very likely
        sub_weight = error_model.total_substitutions / total_ops * 0.5
        ins_weight = error_model.total_insertions / total_ops * 0.5
        del_weight = error_model.total_deletions / total_ops * 0.5
        rev_weight = error_model.total_reversals / total_ops * 0.3
    else:
        # Default weights if no training data
        match_weight = 0.95
        sub_weight = 0.3
        ins_weight = 0.3
        del_weight = 0.3
        rev_weight = 0.2
    
    # DP table: dp[i][j] = probability of generating typo[:i] from correct[:j]
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 1.0
    
    # Initialize: all deletions
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] * error_model.deletion_prob(correct_lower[j-1]) * del_weight
    
    # Initialize: all insertions
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] * error_model.insertion_prob(typo_lower[i-1]) * ins_weight
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Match (no error)
            if typo_lower[i-1] == correct_lower[j-1]:
                match_prob = dp[i-1][j-1] * match_weight
            else:
                match_prob = 0.0
            
            # Substitution
            sub_prob = dp[i-1][j-1] * error_model.substitution_prob(
                correct_lower[j-1], typo_lower[i-1]) * sub_weight
            
            # Deletion
            del_prob = dp[i][j-1] * error_model.deletion_prob(correct_lower[j-1]) * del_weight
            
            # Insertion
            ins_prob = dp[i-1][j] * error_model.insertion_prob(typo_lower[i-1]) * ins_weight
            
            # Reversal (adjacent transposition)
            rev_prob = 0.0
            if i > 1 and j > 1 and (typo_lower[i-2] == correct_lower[j-1] and 
                                   typo_lower[i-1] == correct_lower[j-2]):
                rev_prob = dp[i-2][j-2] * error_model.reversal_prob(
                    correct_lower[j-2], correct_lower[j-1]) * rev_weight
            
            # Sum all possibilities (not max - this is probability, not Viterbi)
            dp[i][j] = match_prob + sub_prob + del_prob + ins_prob + rev_prob
            dp[i][j] = max(dp[i][j], MIN_PROB)  # Avoid zero
    
    # Normalize by edit distance to prevent very small probabilities for long words
    # This helps with multi-edit errors (64% of cases)
    base_prob = dp[m][n]
    edit_dist = abs(m - n) + sum(1 for i in range(min(m, n)) if typo_lower[i] != correct_lower[i])
    
    # Apply length penalty more gently for multi-edit errors
    if edit_dist > 2:
        # Multi-edit errors are common (64%), so don't penalize too heavily
        length_penalty = 1.0 / (1.0 + 0.05 * edit_dist)
        base_prob *= length_penalty
    
    return max(base_prob, MIN_PROB)

# ============================================================================
# CANDIDATE GENERATION - Now uses combined vocabulary!
# ============================================================================

def known(words: Set[str], vocab_set: Set[str]) -> Set[str]:
    """Filter to known words using our combined vocabulary."""
    return set(w for w in words if w in vocab_set)

def generate_candidates(typo: str, vocab_set: Set[str], max_distance: int = MAX_EDIT_DISTANCE) -> List[str]:
    """
    Generate candidates using COMBINED vocabulary.
    Improved: Better handling of multi-edit errors and common patterns.
    """
    typo_lower = typo.lower()
    candidates_set = set()
    
    # Check if already correct
    if typo_lower in vocab_set:
        candidates_set.add(typo_lower)
    
    # Try edits1
    edits1_candidates = known(edits1(typo_lower), vocab_set)
    if edits1_candidates:
        candidates_set.update(edits1_candidates)
    
    # Always try edits2 (64% of errors are multi-edit!)
    if max_distance >= 2:
        edits2_candidates = known(edits2(typo_lower), vocab_set)
        if edits2_candidates:
            candidates_set.update(edits2_candidates)
    
    # For 3+ edit distance errors, use n-gram similarity as additional candidates
    # This helps with the 64% of multi-edit errors
    if len(candidates_set) < 10:
        ngram_candidates = fallback_ngram_candidates(typo_lower, vocab_set, max_candidates=30)
        candidates_set.update(ngram_candidates)
    
    # Special handling for common patterns
    # Double letter errors (add/remove double letters)
    double_letter_candidates = generate_double_letter_candidates(typo_lower, vocab_set)
    candidates_set.update(double_letter_candidates)
    
    # Vowel error candidates (common vowel substitutions)
    vowel_candidates = generate_vowel_candidates(typo_lower, vocab_set)
    candidates_set.update(vowel_candidates)
    
    return list(candidates_set)

def generate_double_letter_candidates(typo: str, vocab_set: Set[str], max_candidates: int = 10) -> Set[str]:
    """Generate candidates by adding/removing double letters."""
    candidates = set()
    
    # Remove double letters
    for i in range(len(typo) - 1):
        if typo[i] == typo[i+1]:
            candidate = typo[:i] + typo[i+1:]
            if candidate in vocab_set:
                candidates.add(candidate)
    
    # Add double letters (at positions where single letter exists)
    for i in range(len(typo)):
        candidate = typo[:i] + typo[i] + typo[i:]
        if candidate in vocab_set:
            candidates.add(candidate)
    
    return candidates

def generate_vowel_candidates(typo: str, vocab_set: Set[str], max_candidates: int = 15) -> Set[str]:
    """Generate candidates by substituting common vowel confusions."""
    vowels = 'aeiou'
    common_confusions = {
        'i': 'e', 'e': 'i',
        'a': 'e', 'e': 'a',
        'u': 'e', 'e': 'u',
        'u': 'o', 'o': 'u'
    }
    
    candidates = set()
    
    # Try single vowel substitutions
    for i, char in enumerate(typo):
        if char in vowels and char in common_confusions:
            candidate = typo[:i] + common_confusions[char] + typo[i+1:]
            if candidate in vocab_set:
                candidates.add(candidate)
    
    return candidates

def fallback_ngram_candidates(typo: str, vocab_set: Set[str], max_candidates: int = 20) -> Set[str]:
    """Fallback using n-gram similarity on combined vocabulary."""
    typo_lower = typo.lower()
    min_len = max(1, len(typo_lower) - LENGTH_TOLERANCE)
    max_len = len(typo_lower) + LENGTH_TOLERANCE
    
    best_matches = []
    
    # Check words with similar length
    # Limit search to reasonable subset for performance
    checked = 0
    max_checks = 10000  # Performance limit
    
    for word in vocab_set:
        if checked >= max_checks:
            break
        if min_len <= len(word) <= max_len:
            checked += 1
            sim = character_ngram_similarity(typo_lower, word, n=2)
            if sim > NGRAM_SIMILARITY_THRESHOLD:
                best_matches.append((word, sim))
    
    best_matches.sort(key=lambda x: x[1], reverse=True)
    return set(word for word, _ in best_matches[:max_candidates])

# ============================================================================
# SPELL CHECKING
# ============================================================================

def spell_check(typo: str, language_model: LanguageModel, error_model: ErrorModel, 
                vocab_set: Set[str], top_k: int = 3) -> List[Tuple[str, float]]:
    """Spell check using Bayes' theorem."""
    
    candidates = generate_candidates(typo, vocab_set, max_distance=MAX_EDIT_DISTANCE)
    if not candidates:
        return []
    
    scores = []
    for candidate in candidates:
        # KCG method: Pr(correct|typo) ∝ Pr(correct) * Pr(typo|correct)
        error_prob = calculate_error_probability(typo, candidate, error_model)
        lang_prob = language_model.probability(candidate)
        
        # Score = Pr(correct) * Pr(typo|correct)
        score = lang_prob * error_prob
        
        scores.append((candidate, score))
    
    # Sort by score (descending) - KCG method uses raw probabilities
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Normalize probabilities (as KCG does)
    total_score = sum(score for _, score in scores)
    if total_score > 0:
        normalized = [(word, score / total_score) for word, score in scores]
        return normalized[:top_k]
    else:
    return scores[:top_k]

# ============================================================================
# EVALUATION
# ============================================================================

def collect_detailed_stats(tests, language_model, error_model, vocab_set, verbose=True):
    """Collect detailed statistics including top-1, top-3, and top-5 accuracy."""
    stats = {
        'total': len(tests),
        'top1_correct': 0,
        'top3_correct': 0,
        'top5_correct': 0,
        'unknown_correct': 0,
        'valid_misspellings': 0,
        'no_candidates': 0,
        'word_lengths': defaultdict(lambda: {'top1_correct': 0, 'top3_correct': 0, 'top5_correct': 0, 'total': 0}),
        'error_types': defaultdict(int),
    }
    
    if verbose:
        print(f"  Processing {len(tests):,} test cases...")
        start_time = time.time()
    
    for idx, (right, wrong) in enumerate(tests):
        if verbose and (idx + 1) % max(1, len(tests) // 20) == 0:
            elapsed = time.time() - start_time
            progress = (idx + 1) / len(tests)
            eta = (elapsed / progress - elapsed) if progress > 0 else 0
            print(f"    Progress: {idx+1:,}/{len(tests):,} ({progress*100:.1f}%) - ETA: {eta:.0f}s", end='\r')
        
        length = len(wrong)
        stats['word_lengths'][length]['total'] += 1
        
        if wrong.lower() in vocab_set:
            stats['valid_misspellings'] += 1
        
        if right.lower() not in vocab_set:
            stats['unknown_correct'] += 1
        
        suggestions = spell_check(wrong, language_model, error_model, vocab_set, top_k=5)
        if suggestions:
            if suggestions[0][0].lower() == right.lower():
                stats['top1_correct'] += 1
                stats['word_lengths'][length]['top1_correct'] += 1
            
            top3_words = [w.lower() for w, _ in suggestions[:3]]
            if right.lower() in top3_words:
                stats['top3_correct'] += 1
                stats['word_lengths'][length]['top3_correct'] += 1
            
            top5_words = [w.lower() for w, _ in suggestions[:5]]
            if right.lower() in top5_words:
                stats['top5_correct'] += 1
                stats['word_lengths'][length]['top5_correct'] += 1
        else:
            stats['no_candidates'] += 1
            stats['error_types']['no_candidates'] += 1
        
        if suggestions and suggestions[0][0].lower() != right.lower():
            if right.lower() not in vocab_set:
                stats['error_types']['unknown_word'] += 1
            elif wrong.lower() in vocab_set:
                stats['error_types']['valid_misspelling'] += 1
            else:
                stats['error_types']['wrong_candidate'] += 1
    
    if verbose:
        print()
    
    return stats

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(train_stats, test_stats, output_dir='figures'):
    """Create visualization figures."""
    if not HAS_MATPLOTLIB:
        print("  Skipping visualizations (matplotlib not available)")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Accuracy comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    datasets = ['Training', 'Test']
    train_top1 = train_stats['top1_correct'] / train_stats['total']
    test_top1 = test_stats['top1_correct'] / test_stats['total']
    train_top3 = train_stats['top3_correct'] / train_stats['total']
    test_top3 = test_stats['top3_correct'] / test_stats['total']
    train_top5 = train_stats['top5_correct'] / train_stats['total']
    test_top5 = test_stats['top5_correct'] / test_stats['total']
    
    x = np.arange(len(datasets))
    width = 0.25
    
    axes[0].bar(x - width, [train_top1, test_top1], width, label='Top-1', color='#2ecc71', alpha=0.7)
    axes[0].bar(x, [train_top3, test_top3], width, label='Top-3', color='#3498db', alpha=0.7)
    axes[0].bar(x + width, [train_top5, test_top5], width, label='Top-5', color='#9b59b6', alpha=0.7)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Top-1, Top-3, and Top-5 Accuracy')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    for i, (t1, t3, t5) in enumerate([(train_top1, train_top3, train_top5), (test_top1, test_top3, test_top5)]):
        axes[0].text(i - width, t1 + 0.02, f'{t1:.1%}', ha='center', va='bottom', fontsize=9)
        axes[0].text(i, t3 + 0.02, f'{t3:.1%}', ha='center', va='bottom', fontsize=9)
        axes[0].text(i + width, t5 + 0.02, f'{t5:.1%}', ha='center', va='bottom', fontsize=9)
    
    # Error types
    error_types = ['Unknown\nWord', 'Valid\nMisspelling', 'Wrong\nCandidate', 'No\nCandidates']
    train_errs = [train_stats['error_types'][et] for et in 
                  ['unknown_word', 'valid_misspelling', 'wrong_candidate', 'no_candidates']]
    test_errs = [test_stats['error_types'][et] for et in 
                 ['unknown_word', 'valid_misspelling', 'wrong_candidate', 'no_candidates']]
    
    x = np.arange(len(error_types))
    axes[1].bar(x - width/2, train_errs, width, label='Training', color='#3498db', alpha=0.7)
    axes[1].bar(x + width/2, test_errs, width, label='Test', color='#9b59b6', alpha=0.7)
    axes[1].set_ylabel('Count')
    axes[1].set_title('Error Type Distribution')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(error_types)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fixed_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved {output_dir}/fixed_accuracy_comparison.png")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Bayesian Spell Checker (KCG Method)')
    parser.add_argument('--train', type=str, default='datasets/birbeck_large_train.txt',
                       help='Training dataset (default: datasets/birbeck_large_train.txt - 28,000 pairs)')
    parser.add_argument('--test', type=str, default='datasets/birbeck_large_test.txt',
                       help='Test dataset (default: datasets/birbeck_large_test.txt - 6,000 pairs)')
    parser.add_argument('--output', type=str, default='fixed_spell_checker_results.json')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Bayesian Spell Checker (Norvig-style Dataset Usage)")
    print("Language Model: big.txt + training correct words")
    print("Error Model: Trained on training set only")
    print("Evaluation: Test set (NOT in vocabulary!)")
    print("="*60)
    
    # Load datasets
    print(f"\nStep 1: Loading datasets...")
    print(f"  Loading training set: {args.train}")
    train_tests = BirkbeckTestset(open(args.train))
    print(f"  ✓ Training: {len(train_tests):,} pairs")
    print(f"  Loading test set: {args.test}")
    test_tests = BirkbeckTestset(open(args.test))
    print(f"  ✓ Test: {len(test_tests):,} pairs")
    
    if 'birbeck_large_train.txt' in args.train and 'birbeck_large_test.txt' in args.test:
        print(f"\n  ✓ Using LARGE split (28,000 train / 6,000 test)")
    elif 'birbeck_train.txt' in args.train and 'birbeck_test.txt' in args.test:
        print(f"\n  ✓ Using FULL dataset")
        print(f"    Training: {len(train_tests):,} pairs")
        print(f"    Test: {len(test_tests):,} pairs")
    
    # Build vocabulary following Norvig's approach
    # Language model: big.txt (general English) + optionally training correct words
    # Does NOT include test set words (data leakage!)
    vocab_words, vocab_set = build_vocabulary(BASE_WORDS, train_tests, include_train_correct_words=True)
    
    # Initialize models
    print(f"\nStep 2: Initializing models...")
    # Language model uses big.txt frequencies (like Norvig) + optional training words
    language_model = LanguageModel(vocab_words)
    
    # Train error model from corpus (KCG method)
    error_model = ErrorModel(train_tests=train_tests)
    
    # Evaluate
    print(f"\nStep 3: Evaluating on training set...")
    train_stats = collect_detailed_stats(train_tests, language_model, error_model, vocab_set)
    
    print(f"\nStep 4: Evaluating on test set...")
    test_stats = collect_detailed_stats(test_tests, language_model, error_model, vocab_set)
    
    # Print results
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    
    print(f"\nTraining Set:")
    print(f"  Total: {train_stats['total']:,}")
    print(f"  Top-1 accuracy: {train_stats['top1_correct']/train_stats['total']:.1%} ({train_stats['top1_correct']:,}/{train_stats['total']:,})")
    print(f"  Top-3 accuracy: {train_stats['top3_correct']/train_stats['total']:.1%} ({train_stats['top3_correct']:,}/{train_stats['total']:,})")
    print(f"  Top-5 accuracy: {train_stats['top5_correct']/train_stats['total']:.1%} ({train_stats['top5_correct']:,}/{train_stats['total']:,})")
    print(f"  Unknown words: {train_stats['unknown_correct']:,} ({train_stats['unknown_correct']/train_stats['total']:.1%})")
    
    print(f"\nTest Set:")
    print(f"  Total: {test_stats['total']:,}")
    print(f"  Top-1 accuracy: {test_stats['top1_correct']/test_stats['total']:.1%} ({test_stats['top1_correct']:,}/{test_stats['total']:,})")
    print(f"  Top-3 accuracy: {test_stats['top3_correct']/test_stats['total']:.1%} ({test_stats['top3_correct']:,}/{test_stats['total']:,})")
    print(f"  Top-5 accuracy: {test_stats['top5_correct']/test_stats['total']:.1%} ({test_stats['top5_correct']:,}/{test_stats['total']:,})")
    print(f"  Unknown words: {test_stats['unknown_correct']:,} ({test_stats['unknown_correct']/test_stats['total']:.1%})")
    
    unknown_pct = test_stats['unknown_correct'] / test_stats['total'] * 100
    if unknown_pct < 1:
        print(f"\n✓ EXCELLENT: Only {unknown_pct:.1f}% unknown words (was ~20%!)")
    elif unknown_pct < 5:
        print(f"\n✓ SUCCESS: Only {unknown_pct:.1f}% unknown words (was ~20%!)")
    elif unknown_pct < 10:
        print(f"\n✓ GOOD: Reduced to {unknown_pct:.1f}% unknown words (was ~20%)")
    else:
        print(f"\n⚠ WARNING: Still {unknown_pct:.1f}% unknown words")
        print(f"   This may indicate test set has words not in training corpus.")
        print(f"   Consider: expanding base vocabulary or handling OOV words better.")
    
    # Save results
    print(f"\nStep 5: Saving results...")
    results = {
        'vocabulary_info': {
            'base_words_count': len(BASE_WORDS),
            'vocab_words_count': len(vocab_words),
            'new_words_added': len(vocab_words) - len(BASE_WORDS),
            'note': 'Vocabulary includes big.txt + training correct words (not test set!)'
        },
        'training': {
            'total': train_stats['total'],
            'top1_correct': train_stats['top1_correct'],
            'top3_correct': train_stats['top3_correct'],
            'top5_correct': train_stats['top5_correct'],
            'top1_accuracy': train_stats['top1_correct'] / train_stats['total'],
            'top3_accuracy': train_stats['top3_correct'] / train_stats['total'],
            'top5_accuracy': train_stats['top5_correct'] / train_stats['total'],
            'unknown_correct': train_stats['unknown_correct'],
            'valid_misspellings': train_stats['valid_misspellings'],
            'error_types': dict(train_stats['error_types'])
        },
        'test': {
            'total': test_stats['total'],
            'top1_correct': test_stats['top1_correct'],
            'top3_correct': test_stats['top3_correct'],
            'top5_correct': test_stats['top5_correct'],
            'top1_accuracy': test_stats['top1_correct'] / test_stats['total'],
            'top3_accuracy': test_stats['top3_correct'] / test_stats['total'],
            'top5_accuracy': test_stats['top5_correct'] / test_stats['total'],
            'unknown_correct': test_stats['unknown_correct'],
            'valid_misspellings': test_stats['valid_misspellings'],
            'error_types': dict(test_stats['error_types'])
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved results to {args.output}")
    
    # Create visualizations
    print(f"\nStep 6: Creating visualizations...")
    create_visualizations(train_stats, test_stats)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
