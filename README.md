# EE104 Final Project: Natural Language Processing Applications

This project implements three core NLP applications: a Bayesian spell checker using the Kernighan-Church-Gale (1990) method, a Hidden Markov Model (HMM) Part-of-Speech (POS) tagger, and a Grammarly-style writing analysis tool.

## Project Overview

This repository contains implementations of fundamental NLP algorithms that demonstrate probabilistic modeling, sequence labeling, and text analysis techniques. Each component addresses a different aspect of language processing:

1. **Spell Checker**: Corrects spelling errors using a noisy channel model
2. **POS Tagger**: Tags words with their grammatical parts of speech using HMMs
3. **Grammarly Analysis**: Analyzes text for writing quality, style, and grammatical issues

---

## 1. Bayesian Spell Checker

### Overview

The spell checker implements the **Kernighan-Church-Gale (1990)** noisy channel model for spelling correction. It uses Bayes' theorem to find the most likely correct word given a misspelling.

### How It Works

The spell checker follows the probabilistic framework:

**Pr(correct | typo) ∝ Pr(correct) × Pr(typo | correct)**

#### Components:

1. **Language Model (Pr(correct))**
   - Estimates word probabilities from a large English corpus (`big.txt`)
   - Uses Laplace smoothing: `(freq + 0.5) / (total + 0.5 × vocab_size)`
   - Combines general English vocabulary with domain-specific words from training data

2. **Error Model (Pr(typo | correct))**
   - Trains empirical confusion matrices from actual typo data (Birkbeck corpus)
   - Models four types of edit operations:
     - **Substitutions**: Character replacements (e.g., `e` → `i`)
     - **Insertions**: Extra characters added
     - **Deletions**: Characters removed
     - **Reversals**: Adjacent character transpositions
   - Uses dynamic programming to align correct and misspelled words
   - Learns operation probabilities from training data (not keyboard proximity)

3. **Candidate Generation**
   - Generates correction candidates using edit distance (up to 2 edits)
   - Includes special handlers for common patterns:
     - Double letter errors (add/remove duplicates)
     - Vowel substitution errors (common confusions like `i`↔`e`)
   - Falls back to n-gram similarity for multi-edit errors

4. **Scoring & Ranking**
   - Scores each candidate: `score = P(correct) × P(typo | correct)`
   - Returns top-K candidates ranked by probability
   - Normalizes probabilities for interpretability

### Performance

- **Top-1 Accuracy**: ~38.6% on test set (6,000 pairs)
- **Top-3 Accuracy**: ~50.5%
- **Top-5 Accuracy**: ~54.3%
- Trained on 28,000 typo pairs from Birkbeck corpus

### Usage

```bash
cd spell-checker
python spell-check.py --train datasets/birbeck_large_train.txt --test datasets/birbeck_large_test.txt
```

### Key Features

- **No Data Leakage**: Test set words excluded from vocabulary
- **Empirical Error Model**: Learns from actual typos, not assumptions
- **Handles Multi-Edit Errors**: 64% of errors require 3+ edits
- **Comprehensive Evaluation**: Tracks Top-1, Top-3, Top-5 accuracy

---

## 2. HMM Part-of-Speech Tagger

### Overview
This module implements a probabilistic **Part-of-Speech (POS) tagger** using a **Hidden Markov Model (HMM)** trained on the Universal Dependencies English Web Treebank (UD-EWT).  
Given a sentence, the model assigns a grammatical tag (e.g., NOUN, VERB, ADP) to each word by estimating:

- Transition probabilities: P(tagᵢ | tagᵢ₋₁)  
- Emission probabilities: P(wordᵢ | tagᵢ)

A strong **Most-Frequent-Tag baseline** is also included for comparison.

---

### How It Works

#### 🧠 Components
- **Training**
  - Reads UD-EWT `.conllu` files  
  - Counts tag transitions and tag-word emissions  
  - Applies add-α Laplace smoothing  
  - Handles rare words using an `<UNK>` token  
  - Stores probabilities in log-space for numerical stability  

- **Decoding (Viterbi Algorithm)**
  - Computes the most likely sequence of POS tags  
  - Dynamic programming recurrence over 17 UPOS tags  
  - Uses backpointers to reconstruct the optimal tag path  

- **Features**
  - Rare/unknown word handling  
  - Transition matrix visualization  
  - Confusion matrix and per-tag accuracy  
  - Emission distribution plots for ambiguous words  
  - Average log-likelihood calculation  

---

### 📊 Performance Summary
- **Baseline accuracy:** 86.20%  
- **HMM tagger accuracy:** 89.32%  
- Performs extremely well on unambiguous categories (DET, PRON, PUNCT)  
- Major confusions: PROPN vs NOUN, VERB vs AUX  
- Handles ambiguous words like *can* and *like* in a probabilistic way  

---

### 📁 Files
Located in: `pos_tagger/`

- `HMM_POS_tagger.py` — full implementation  
- `README_POS.md` — detailed documentation for this module  
- `figures/` — transition matrix, confusion matrix, emission plots, HMM diagram  

---

### ▶ Usage

```bash
cd pos_tagger
python HMM_POS_tagger.py
```
---

## 3. Grammarly-Style Writing Analysis

### Overview

The Grammarly analysis compares Grammarly's product and component 

### Usage

```bash
python grammarly_analysis.py [input_file]
```

---

## Project Structure

```
ee104-final-project/
├── spell-checker/
│   ├── spell-check.py          # Main KCG spell checker
│   ├── norvig-spell.py          # Base implementation (Norvig)
│   ├── create_custom_split.py   # Dataset splitting utility
│   ├── datasets/                 # Training/test datasets
│   │   ├── big.txt              # Base vocabulary corpus
│   │   ├── birbeck_large_train.txt
│   │   ├── birbeck_large_test.txt
│   │   ├── birbeck_train.txt
│   │   └── birbeck_test.txt
│   ├── figures/                 # Generated visualizations
│   └── spell_checker_results.json
├── pos-tagger/
│   ├── HMM_POS_tagger.py # Hidden Markov Model POS tagger
│   ├── README_POS.md # Documentation for this module
│   ├── data/
│   │   ├── en_ewt-ud-train.conllu
│   │   ├── en_ewt-ud-dev.conllu
│   │   └── en_ewt-ud-test.conllu
│   └── figures/
│       ├── transition_matrix.png
│       ├── confusion_matrix.png
│       ├── emission_can.png
│       └── emission_like.png
├── grammarly-analysis/
│   └── grammarly_analysis.py    # Writing analysis tool
└── README.md
```

---

## Dependencies

- Python 3.7+
- `numpy` (for numerical operations)
- `matplotlib` (for visualizations, optional)
- Standard library: `collections`, `argparse`, `json`, `re`

Install dependencies:
```bash
pip install numpy matplotlib
```

---

## Key Algorithms & Techniques

### Spell Checker
- **Noisy Channel Model**: Bayes' theorem for error correction
- **Dynamic Programming**: Optimal alignment for error extraction
- **Confusion Matrices**: Empirical error probability learning
- **Edit Distance**: Levenshtein distance for candidate generation

### POS Tagger

### Grammarly Analysis

---

## Results & Evaluation

### Spell Checker Results
- Achieves **38.6% Top-1 accuracy** on 6,000 test pairs
- **54.3% Top-5 accuracy** (correct word in top 5 suggestions)
- Only **3.8% unknown words** (down from ~20% with vocabulary improvements)
- Handles multi-edit errors effectively (64% of test cases)

### Evaluation Metrics
- **Accuracy**: Top-1, Top-3, Top-5 correction rates
- **Error Analysis**: Unknown words, valid misspellings, wrong candidates
- **Performance**: Processing time and throughput

---

## References

- Kernighan, M. D., Church, K. W., & Gale, W. A. (1990). "A Spelling Correction Program Based on a Noisy Channel Model." *Proceedings of COLING-90*.
- Norvig, P. (2007). "How to Write a Spelling Corrector." http://norvig.com/spell-correct.html
- Jurafsky, D., & Martin, J. H. (2020). *Speech and Language Processing* (3rd ed.). Chapters on HMMs and POS tagging.

---

## License

See [LICENSE](LICENSE) file for details.
