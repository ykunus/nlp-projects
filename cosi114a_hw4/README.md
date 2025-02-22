# Hidden Markov Model (HMM) Part-of-Speech (POS) Tagger

## Overview
This project implements a **Hidden Markov Model (HMM) POS tagger** using unigram and bigram tagging techniques. The tagger learns from **Penn Treebank POS-labeled sentences** and assigns part-of-speech tags to unseen sentences using both **greedy decoding** and the **Viterbi algorithm**.

---

## Key Concepts
- **Part-of-Speech Tagging**: Assigning grammatical categories (e.g., NN, VB) to words in sentences.
- **Unigram & Bigram Taggers**:
  - **Unigram Tagger**: Assigns each word its most common POS tag in the training data.
  - **Bigram Tagger**: Uses previous word context for POS tagging.
- **Hidden Markov Models (HMMs)**:
  - **Emission Probabilities**: Probability of a word given a POS tag.
  - **Transition Probabilities**: Probability of a POS tag given the previous POS tag.
  - **Initial Probabilities**: Probability of a POS tag starting a sentence.
- **Lidstone Smoothing**: Handling unseen words using **add-k smoothing**.
- **Greedy Decoding (Greedy Bigram Tagger)**: Selects the **best tag** at each step independently.
- **Viterbi Algorithm**: Finds the **optimal sequence of tags** for a sentence using dynamic programming.
- **Efficient Count-Based Training**: Optimized counting methods ensure **fast lookups** for probabilities.

---

## Implemented Classes & Functions

### **1. Basic POS Taggers**
- **`MostFrequentTagTagger`**: Tags every word with the **most frequent** POS tag in the training set.
- **`UnigramTagger`**: Tags words using their **most common** POS tag, falling back to the overall most frequent tag.

### **2. Sentence Counter (Training Component)**
- **`SentenceCounter`**: Stores **counts** needed to compute:
  - **Emission Probabilities** (`B` matrix)
  - **Transition Probabilities** (`A` matrix)
  - **Initial Probabilities** (`π` vector)
- Provides:
  - `emission_prob(tag, word)`: Computes \( P(w | t) \) with Lidstone smoothing.
  - `transition_prob(tag1, tag2)`: Computes \( P(t_2 | t_1) \).
  - `initial_prob(tag)`: Computes \( P(t) \) for sentence-starting tags.

### **3. Bigram-Based HMM Taggers**
#### **GreedyBigramTagger** *(Greedy Decoding)*
- **Tags sentences by choosing the best tag at each step** based on:
  - \( P(t_1) \) (initial probability)
  - \( P(w | t) \) (emission probability)
  - \( P(t_2 | t_1) \) (transition probability)
- **Limitations**: Does not account for overall sequence likelihood.

#### **ViterbiBigramTagger** *(Viterbi Algorithm)*
- **Finds the optimal sequence of tags** using:
  - **Dynamic programming** for efficient tag sequence selection.
  - **Backpointers** to reconstruct the best sequence.
- **More accurate than greedy decoding** since it considers the **entire sentence context**.

### **4. Utility Functions**
- `safe_log(n)`: Computes \(\log(n)\) safely, returning **-∞ for zero probabilities**.
- `max_item(scores)`: Returns the **max-scoring POS tag** from a dictionary.
- `most_frequent_item(counts)`: Returns the **most frequent item** from a counter.
- `items_descending_value(counts)`: Sorts POS tags in **descending order** of frequency.

---

## Example Usage

### **Training & Tagging a Sentence**
```python
# Load training data
from test_hw4 import load_pos_data
tagged_sents = load_pos_data("test_data/train_pos.txt")

# Train Unigram Tagger
unigram_tagger = UnigramTagger()
unigram_tagger.train(tagged_sents)

# Tag a sentence
sentence = ["The", "dog", "barked"]
predicted_tags = unigram_tagger.tag_sentence(sentence)
print(predicted_tags)  # Example output: ['DT', 'NN', 'VBD']

# Train Greedy Bigram Tagger
greedy_tagger = GreedyBigramTagger(k=0.1)
greedy_tagger.train(tagged_sents)

# Tag a sentence
predicted_tags = greedy_tagger.tag_sentence(sentence)
print(predicted_tags)  # Example output: ['DT', 'NN', 'VBD']

# Train Viterbi Bigram Tagger
viterbi_tagger = ViterbiBigramTagger(k=0.1)
viterbi_tagger.train(tagged_sents)

# Tag a sentence using Viterbi
predicted_tags = viterbi_tagger.tag_sentence(sentence)
print(predicted_tags)  # Example output: ['DT', 'NN', 'VBD']

# Compute probability of a tag sequence
prob = viterbi_tagger.sequence_probability(["The", "dog", "barked"], ["DT", "NN", "VBD"])
print(prob)  # Example output: -3.2188758248682006

