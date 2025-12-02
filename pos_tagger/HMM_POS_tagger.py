import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
import os

def save_fig(name: str):
    """Save current matplotlib figure to plots/<name>.png."""
    os.makedirs("plots", exist_ok=True)
    path = os.path.join("plots", name + ".png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path}")


def read_conllu(path: str) -> List[List[Tuple[str, str]]]:
    """
    Read a UD CoNLL-U file and return a list of sentences.
    Each sentence is a list of (word, tag) pairs.

    We use:
      - FORM (column 2) as the word
      - UPOS (column 4) as the POS tag
    """
    sentences = []
    current = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Sentence boundary
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue

            # Skip comments
            if line.startswith("#"):
                continue

            parts = line.split("\t")
            # ID field can have "-" or "." for special tokens: skip those
            if "-" in parts[0] or "." in parts[0]:
                continue

            word = parts[1]
            upos = parts[3]
            current.append((word, upos))

    if current:
        sentences.append(current)

    return sentences


class MostFrequentTagBaseline:
    def __init__(self):
        self.word_to_tag: Dict[str, str] = {}
        self.default_tag: str = "NOUN" 

    def fit(self, train_sents: List[List[Tuple[str, str]]]):
        word_tag_counts = defaultdict(Counter)
        tag_counts = Counter()

        for sent in train_sents:
            for word, tag in sent:
                word_tag_counts[word][tag] += 1
                tag_counts[tag] += 1

        #Most frequent tag for each word
        self.word_to_tag = {
            w: counts.most_common(1)[0][0]
            for w, counts in word_tag_counts.items()
        }

        #Default tag for unknown words->global most frequent tag
        self.default_tag = tag_counts.most_common(1)[0][0]

    def tag_sentence(self, words: List[str]) -> List[str]:
        tags = []
        for w in words:
            if w in self.word_to_tag:
                tags.append(self.word_to_tag[w])
            else:
                tags.append(self.default_tag)
        return tags


class HMMPOS:
    def __init__(self, alpha_trans: float = 0.1, alpha_emit: float = 0.1, rare_threshold: int = 1):
        """
        alpha_trans: Laplace smoothing for transitions
        alpha_emit:  Laplace smoothing for emissions
        rare_threshold: words with freq <= this are mapped to <UNK>
        """
        self.alpha_trans = alpha_trans
        self.alpha_emit = alpha_emit
        self.rare_threshold = rare_threshold

        self.tags: List[str] = []
        self.vocab: set = set()
        self.START = "<START>"

        # log probabilities
        self.log_transition = defaultdict(dict)  # prev_tag -> tag -> logP
        self.log_emission = defaultdict(dict)    # tag -> word -> logP

    #map rare words to <UNK>
    def _build_vocab_and_replace_rare(self, train_sents):
        word_freq = Counter()
        for sent in train_sents:
            for w, t in sent:
                word_freq[w] += 1

        vocab = {w for w, c in word_freq.items() if c > self.rare_threshold}
        vocab.add("<UNK>")

        new_sents = []
        for sent in train_sents:
            new_sent = []
            for w, t in sent:
                if word_freq[w] <= self.rare_threshold:
                    new_sent.append(("<UNK>", t))
                else:
                    new_sent.append((w, t))
            new_sents.append(new_sent)

        return vocab, new_sents

    #Training
    def fit(self, train_sents: List[List[Tuple[str, str]]]):
        """
        train_sents: list of sentences, each sentence is a list of (word, tag)
        """
        #Handle rare words
        self.vocab, train_sents_unk = self._build_vocab_and_replace_rare(train_sents)

        #Collect tag set
        tag_set = set()
        for sent in train_sents_unk:
            for w, t in sent:
                tag_set.add(t)
        self.tags = sorted(tag_set)

        #Count transitions and emissions
        transition_counts = defaultdict(Counter)  # prev_tag -> tag
        emission_counts = defaultdict(Counter)    # tag -> word

        for sent in train_sents_unk:
            prev_tag = self.START
            for w, t in sent:
                transition_counts[prev_tag][t] += 1
                emission_counts[t][w] += 1
                prev_tag = t

        #Compute transition log probabilities
        num_tags = len(self.tags)
        all_prev_tags = list(transition_counts.keys()) + [self.START]
        all_prev_tags = list(set(all_prev_tags))

        for prev in all_prev_tags:
            total_outgoing = sum(transition_counts[prev].values())
            for t in self.tags:
                count = transition_counts[prev][t]
                prob = (count + self.alpha_trans) / (total_outgoing + self.alpha_trans * num_tags)
                self.log_transition[prev][t] = math.log(prob)

        #Compute emission log probabilities
        vocab_size = len(self.vocab)
        for t in self.tags:
            total = sum(emission_counts[t].values())
            for w in self.vocab:
                count = emission_counts[t][w]
                prob = (count + self.alpha_emit) / (total + self.alpha_emit * vocab_size)
                self.log_emission[t][w] = math.log(prob)

    def viterbi(self, words: List[str]) -> List[str]:
        """
        Decode most probable tag sequence for a given sentence using Viterbi.
        words: list of original words (no <UNK> yet)
        """
        if not words:
            return []

        # Map unknown words to <UNK>
        obs = [w if w in self.vocab else "<UNK>" for w in words]
        n = len(obs)

        V = [dict() for _ in range(n)]           # V[i][tag] = best log prob ending in tag at position i
        backpointer = [dict() for _ in range(n)]

        #Initialization
        for t in self.tags:
            log_trans = self.log_transition[self.START].get(t, float("-inf"))
            log_emit = self.log_emission[t].get(obs[0], float("-inf"))
            V[0][t] = log_trans + log_emit
            backpointer[0][t] = None

        #Recursion
        for i in range(1, n):
            for curr_tag in self.tags:
                best_score = float("-inf")
                best_prev = None
                log_emit = self.log_emission[curr_tag].get(obs[i], float("-inf"))

                for prev_tag in self.tags:
                    log_trans = self.log_transition[prev_tag].get(curr_tag, float("-inf"))
                    score = V[i-1][prev_tag] + log_trans + log_emit
                    if score > best_score:
                        best_score = score
                        best_prev = prev_tag

                V[i][curr_tag] = best_score
                backpointer[i][curr_tag] = best_prev

        #Choose best final tag
        last_index = n - 1
        best_last_tag = max(V[last_index], key=V[last_index].get)

        #Backtrack
        tags_seq = [best_last_tag]
        curr_tag = best_last_tag
        for i in range(last_index, 0, -1):
            curr_tag = backpointer[i][curr_tag]
            tags_seq.append(curr_tag)
        tags_seq.reverse()
        return tags_seq

    #Wrapper
    def tag_sentence(self, words: List[str]) -> List[str]:
        return self.viterbi(words)

    def sentence_log_likelihood(self, words: List[str], tags: List[str]) -> float:
        """
        Compute log P(words, tags) under the HMM parameterization,
        using gold tags (not decoded tags).
        """
        assert len(words) == len(tags)
        obs = [w if w in self.vocab else "<UNK>" for w in words]

        log_prob = 0.0
        #First token
        first_tag = tags[0]
        log_prob += self.log_transition[self.START].get(first_tag, float("-inf"))
        log_prob += self.log_emission[first_tag].get(obs[0], float("-inf"))

        for i in range(1, len(tags)):
            prev_tag = tags[i-1]
            curr_tag = tags[i]
            log_prob += self.log_transition[prev_tag].get(curr_tag, float("-inf"))
            log_prob += self.log_emission[curr_tag].get(obs[i], float("-inf"))

        return log_prob




def accuracy(true_tags: List[List[str]], pred_tags: List[List[str]]) -> float:
    correct = 0
    total = 0
    for gold_sent, pred_sent in zip(true_tags, pred_tags):
        for g, p in zip(gold_sent, pred_sent):
            total += 1
            if g == p:
                correct += 1
    return correct / total if total > 0 else 0.0


def per_tag_accuracy(true_tags: List[List[str]], pred_tags: List[List[str]]) -> Dict[str, float]:
    total_per_tag = Counter()
    correct_per_tag = Counter()

    for gold_sent, pred_sent in zip(true_tags, pred_tags):
        for g, p in zip(gold_sent, pred_sent):
            total_per_tag[g] += 1
            if g == p:
                correct_per_tag[g] += 1

    per_tag_acc = {}
    for tag in total_per_tag:
        per_tag_acc[tag] = correct_per_tag[tag] / total_per_tag[tag]
    return per_tag_acc


def average_log_likelihood(model: HMMPOS, sents: List[List[Tuple[str, str]]]) -> float:
    total_log_prob = 0.0
    num_tokens = 0
    for sent in sents:
        words = [w for w, t in sent]
        tags = [t for w, t in sent]
        log_p = model.sentence_log_likelihood(words, tags)
        total_log_prob += log_p
        num_tokens += len(sent)
    if num_tokens == 0:
        return float("-inf")
    return total_log_prob / num_tokens

def plot_transition_matrix(hmm: HMMPOS, show=False):
    tags = hmm.tags
    n = len(tags)
    mat = np.zeros((n, n))

    for i, prev in enumerate(tags):
        for j, curr in enumerate(tags):
            log_p = hmm.log_transition[prev].get(curr, float("-inf"))
            mat[i, j] = 0 if log_p == float("-inf") else math.exp(log_p)

    plt.figure(figsize=(8, 6))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.xticks(range(n), tags, rotation=90)
    plt.yticks(range(n), tags)
    plt.title("HMM Transition Probabilities")
    plt.tight_layout()
    save_fig("transition_matrix")

    if show:
        plt.show()
    plt.close()

def plot_confusion_matrix(true_tags, pred_tags, tags=None, show=False):
    if tags is None:
        tag_set = set()
        for sent in true_tags:
            tag_set.update(sent)
        tags = sorted(tag_set)

    tag_to_idx = {t: i for i, t in enumerate(tags)}
    n = len(tags)
    counts = np.zeros((n, n), dtype=np.int64)

    for gold_sent, pred_sent in zip(true_tags, pred_tags):
        for g, p in zip(gold_sent, pred_sent):
            if g in tag_to_idx and p in tag_to_idx:
                counts[tag_to_idx[g], tag_to_idx[p]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mat = counts / row_sums

    plt.figure(figsize=(8, 6))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.xticks(range(n), tags, rotation=90)
    plt.yticks(range(n), tags)
    plt.xlabel("Predicted Tag")
    plt.ylabel("Gold Tag")
    plt.title("Confusion Matrix (Normalized)")
    plt.tight_layout()
    save_fig("confusion_matrix")

    if show:
        plt.show()
    plt.close()


def plot_emission_distribution_for_word(hmm: HMMPOS, word: str, show=False):
    obs_word = word if word in hmm.vocab else "<UNK>"
    tags = hmm.tags
    probs = []

    for t in tags:
        log_p = hmm.log_emission[t].get(obs_word, float("-inf"))
        probs.append(0 if log_p == float("-inf") else math.exp(log_p))

    probs = np.array(probs)
    total = probs.sum()
    if total > 0:
        probs /= total

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(tags)), probs)
    plt.xticks(range(len(tags)), tags, rotation=90)
    plt.ylabel(f"P({obs_word} | tag), normalized")
    plt.title(f"Emission Distribution for '{word}'")
    plt.tight_layout()
    save_fig(f"emission_{word}")

    if show:
        plt.show()
    plt.close()

#function test 
def toy_test():
    #tiny made-up dataset: 2 sentences
    train = [
        [("the", "DET"), ("cat", "NOUN"), ("sleeps", "VERB")],
        [("the", "DET"), ("dog", "NOUN"), ("runs", "VERB")]
    ]
    test = [
        [("the", "DET"), ("cat", "NOUN")],
        [("the", "DET"), ("dog", "NOUN")]
    ]

    # Baseline
    baseline = MostFrequentTagBaseline()
    baseline.fit(train)
    baseline_pred = [baseline.tag_sentence([w for w, t in sent]) for sent in test]
    gold_tags = [[t for w, t in sent] for sent in test]
    print("Baseline toy accuracy:", accuracy(gold_tags, baseline_pred))

    # HMM
    hmm = HMMPOS()
    hmm.fit(train)
    hmm_pred = [hmm.tag_sentence([w for w, t in sent]) for sent in test]
    print("HMM toy accuracy:", accuracy(gold_tags, hmm_pred))


if __name__ == "__main__":
    # toy_test()

    train_path = "data/en_ewt-ud-train.conllu"
    dev_path   = "data/en_ewt-ud-dev.conllu"
    test_path  = "data/en_ewt-ud-test.conllu"

    print("Reading UD data...")
    train_sents = read_conllu(train_path)
    dev_sents   = read_conllu(dev_path)
    test_sents  = read_conllu(test_path)

    print(f"#train sentences: {len(train_sents)}")
    print(f"#dev sentences:   {len(dev_sents)}")
    print(f"#test sentences:  {len(test_sents)}")

    #Prepare gold tag sequences
    train_words = [[w for w, t in sent] for sent in train_sents]
    train_tags  = [[t for w, t in sent] for sent in train_sents]
    dev_words   = [[w for w, t in sent] for sent in dev_sents]
    dev_tags    = [[t for w, t in sent] for sent in dev_sents]
    test_words  = [[w for w, t in sent] for sent in test_sents]
    test_tags   = [[t for w, t in sent] for sent in test_sents]

    #training Baseline
    print("\nTraining baseline (most frequent tag)...")
    baseline = MostFrequentTagBaseline()
    baseline.fit(train_sents)

    print("Tagging test set with baseline...")
    baseline_pred_test = [baseline.tag_sentence(words) for words in test_words]
    baseline_acc = accuracy(test_tags, baseline_pred_test)
    print(f"Baseline test accuracy: {baseline_acc:.4f}")

    #HMM model
    print("\nTraining HMM POS tagger...")
    hmm = HMMPOS(alpha_trans=0.1, alpha_emit=0.1, rare_threshold=1)
    hmm.fit(train_sents)

    print("Tagging dev set with HMM...")
    hmm_pred_dev = [hmm.tag_sentence(words) for words in dev_words]
    hmm_dev_acc = accuracy(dev_tags, hmm_pred_dev)
    print(f"HMM dev accuracy: {hmm_dev_acc:.4f}")

    print("Tagging test set with HMM...")
    hmm_pred_test = [hmm.tag_sentence(words) for words in test_words]
    hmm_test_acc = accuracy(test_tags, hmm_pred_test)
    print(f"HMM test accuracy: {hmm_test_acc:.4f}")

    #Average log-likelihood on test (gold tags)
    print("\nComputing average log-likelihood on test (gold tags)...")
    avg_ll = average_log_likelihood(hmm, test_sents)
    print(f"Average log-likelihood per token (test): {avg_ll:.4f}")

    #Get per-tag accuracy
    print("\nPer-tag accuracy on test (HMM):")
    per_tag_acc = per_tag_accuracy(test_tags, hmm_pred_test)
    for tag, acc in sorted(per_tag_acc.items(), key=lambda x: x[0]):
        print(f"{tag:5s}: {acc:.3f}")

    #Graphs for the written project
    print("\nSaving transition matrix heatmap...")
    plot_transition_matrix(hmm)

    print("\nSaving confusion matrix heatmap...")
    plot_confusion_matrix(test_tags, hmm_pred_test, tags=hmm.tags)

    print("\nSaving emission distribution for 'can'...")
    plot_emission_distribution_for_word(hmm, "can")

    print("\nSaving emission distribution for 'like'...")
    plot_emission_distribution_for_word(hmm, "like")
