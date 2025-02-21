# N-Gram Language Modeling with Smoothing & Perplexity

## Overview
This project extends n-gram modeling by incorporating **probability estimation**, **smoothing techniques**, and **sequence evaluation** methods. The implementation includes **MLE-based n-gram probability models**, **Lidstone smoothing**, **interpolation techniques**, and **perplexity computations** to evaluate language models.

## Key Concepts
- **N-Gram Probability Distributions**: Computing probabilities for unigrams, bigrams, and trigrams.
- **Sampling from N-Grams**: Generating sentences using bigram and trigram probability distributions.
- **Maximum Likelihood Estimation (MLE)**: Implementing MLE-based unigram, bigram, and trigram probability models.
- **Lidstone Smoothing**: Applying Lidstone's method for handling unseen words in probability distributions.
- **Interpolation for Probability Distributions**: Combining unigram and bigram probabilities to improve model robustness.
- **Sequence Probability Computation**: Calculating the probability of a given sequence under different models.
- **Perplexity Calculation**: Measuring how well an n-gram model predicts a given dataset.

## Implemented Functions
- `sample_bigrams(probs: dict[str, dict[str, float]]) -> list[str]`
  - Generates sentences based on bigram probabilities.

- `sample_trigrams(probs: dict[tuple[str, str], dict[str, float]]) -> list[str]`
  - Generates sentences based on trigram probabilities.

- `unigram_counts(sentences: Iterable[Sequence[str]]) -> dict[str, int]`
  - Counts occurrences of unigrams.

- `bigram_counts(sentences: Iterable[Sequence[str]]) -> dict[str, dict[str, int]]`
  - Counts occurrences of bigrams.

- `trigram_counts(sentences: Iterable[Sequence[str]]) -> dict[tuple[str, str], dict[str, int]]`
  - Counts occurrences of trigrams.

- `UnigramMLE`
  - Implements an MLE-based unigram probability model.

- `BigramMLE`
  - Implements an MLE-based bigram probability model.

- `TrigramMLE`
  - Implements an MLE-based trigram probability model.

- `UnigramLidstoneSmoothed`
  - Applies Lidstone smoothing to unigram probabilities.

- `BigramInterpolation`
  - Implements interpolation by combining unigram and bigram probabilities.

- `unigram_sequence_prob(sequence: Sequence[str], probs: ProbabilityDistribution[str]) -> float`
  - Computes the log probability of a sequence using a unigram model.

- `bigram_sequence_prob(sequence: Sequence[str], probs: ProbabilityDistribution[tuple[str, str]]) -> float`
  - Computes the log probability of a sequence using a bigram model.

- `trigram_sequence_prob(sequence: Sequence[str], probs: ProbabilityDistribution[tuple[str, str, str]]) -> float`
  - Computes the log probability of a sequence using a trigram model.

- `unigram_perplexity(sentences: Iterable[Sequence[str]], probs: ProbabilityDistribution[str]) -> float`
  - Computes the perplexity of a dataset using a unigram model.

- `bigram_perplexity(sentences: Iterable[Sequence[str]], probs: ProbabilityDistribution[tuple[str, str]]) -> float`
  - Computes the perplexity of a dataset using a bigram model.

- `trigram_perplexity(sentences: Iterable[Sequence[str]], probs: ProbabilityDistribution[tuple[str, str, str]]) -> float`
  - Computes the perplexity of a dataset using a trigram model.

## Example Usage
```python
sentences = [
    ("The", "quick", "fox"),
    ("The", "slow", "dog"),
    ("The", "happy", "dog")
]

# Unigram Probabilities
uni_counts = unigram_counts(sentences)
uni_model = UnigramMLE(uni_counts)
print(uni_model.prob("The"))  # Example unigram probability

# Bigram Probabilities
bi_counts = bigram_counts(sentences)
bi_model = BigramMLE(bi_counts)
print(bi_model.prob(("The", "quick")))  # Example bigram probability

# Sequence Probability
sequence = ["The", "quick", "dog"]
prob = bigram_sequence_prob(sequence, bi_model)
print(prob)

# Perplexity Calculation
perplexity = bigram_perplexity([sequence], bi_model)
print(perplexity)
