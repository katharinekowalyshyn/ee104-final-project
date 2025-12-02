# Part-of-Speech Tagging with Hidden Markov Models (HMM)
Author: Kendrick Koumba  
Course: EE 104 — Probabilistic Systems  
Component: POS Tagging Section  
Dataset: Universal Dependencies (UD) English Web Treebank (EWT)

---

## Overview

This module implements a Part-of-Speech (POS) tagger using a Hidden Markov Model (HMM) trained on the UD English Web Treebank (EWT).  
The goal is to assign a syntactic category (e.g., NOUN, VERB, ADP) to each word in a sentence by modeling:

- A Markov chain over POS tags (hidden states)
- Emissions of words from tags (observations)
- Transition probabilities P(tᵢ | tᵢ₋₁)
- Emission probabilities P(wᵢ | tᵢ)

This implementation includes:

- A Most-Frequent-Tag baseline
- An HMM with Laplace smoothing
- Viterbi decoding for MAP inference
- Handling of unknown/rare words
- Evaluation metrics (accuracy, per-tag accuracy, log-likelihood)
- Visualizations, including:
  - HMM structure diagram
  - Transition matrix heatmap
  - Confusion matrix
  - Emission distributions (“can”, “like”)

This module corresponds to the 10 pages POS Tagging section of the project *Do Probabilities Suffice?*.

---

## Repository Structure

```
/pos_tagger/
│
├── HMM_POS_tagger.py          # Main HMM implementation
├── README.md                  # This file
│
├── data/
│   ├── en_ewt-ud-train.conllu
│   ├── en_ewt-ud-dev.conllu
│   └── en_ewt-ud-test.conllu
│
└── figures/
    ├── pos_hmm_structure.png
    ├── transition_matrix.png
    ├── confusion_matrix.png
    ├── emission_can.png
    ├── emission_like.png
```

---

## Running the Code

### 1. Install dependencies

```bash
pip install numpy matplotlib
```

### 2. Download the UD dataset

Download UD-EWT from:  
https://universaldependencies.org/treebanks/en_ewt/

Place the `.conllu` files inside the `/data` directory.

### 3. Run the tagger

```bash
python HMM_POS_tagger.py
```

You should see output including:

```
Baseline test accuracy: 0.8620
HMM dev accuracy: 0.8906
HMM test accuracy: 0.8932
Average log-likelihood per token: -6.1595
```

Plots are automatically saved to `/figures`.

---

## Visualizations

### Transition Matrix  
`figures/transition_matrix.png`  
Shows learned tag-to-tag transitions (e.g., DET → NOUN).

### Confusion Matrix  
`figures/confusion_matrix.png`  
Reveals systematic error patterns such as:
- PROPN vs NOUN
- VERB vs AUX  

### Emission Distributions  
`figures/emission_can.png`  
`figures/emission_like.png`

Examines lexical ambiguity.

### HMM POS Structure Diagram  
`figures/pos_hmm_structure.png`  
Pedagogical schematic of transitions and emissions.

---

## Key Concepts Demonstrated

- Hidden Markov Models (generative tagging)
- Transition & emission probability estimation
- Add-α Laplace smoothing
- Rare/unknown word handling using <UNK>
- Viterbi decoding (O(nK²))
- Accuracy, per-tag evaluation, log-likelihood scoring

---

## Notes

- HMM accuracy: **89.32%**
- Baseline accuracy: **86.20%**
- Neural taggers typically exceed **97%**, highlighting:
  - Strengths of classical probabilistic models
  - Their limitations on ambiguity and long-range dependencies

This section supports the broader project question: *Do probabilities suffice?*

---

## Citation

```
@inproceedings{silveira14gold,
  title = {A Gold Standard Dependency Corpus for English},
  author = {Silveira et al.},
  booktitle = {LREC},
  year = {2014}
}
```

---


