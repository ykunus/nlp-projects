# N-Gram Probability Modeling

## Overview
This project explores fundamental NLP concepts by implementing n-gram models (bigrams and trigrams). The project focuses on extracting n-grams, computing their counts, and normalizing them into probability distributions.

## Key Concepts
- **N-Grams**: Extracting bigrams and trigrams from sentences.
- **Probability Distributions**: Normalizing n-gram counts into probabilities.
- **Tokenization & Padding**: Adding `<START>` and `<END>` tokens to handle sentence boundaries.
- **Efficient Counting**: Using `Counter` and `defaultdict(Counter)` for structured frequency analysis.
- **Dictionary-Based Models**: Storing probabilities in nested dictionaries for lookup efficiency.

## Implemented Functions
- `counts_to_probs(counts: Counter[T]) -> dict[T, float]`
  - Converts n-gram counts into probability distributions.

- `bigrams(sentence: Sequence[str]) -> list[tuple[str, str]]`
  - Extracts bigrams from a given sentence.

- `trigrams(sentence: Sequence[str]) -> list[tuple[str, str, str]]`
  - Extracts trigrams from a given sentence.

- `count_unigrams(sentences: Iterable[Sequence[str]], lower: bool = False) -> Counter[str]`
  - Counts unigram occurrences across multiple sentences.

- `count_bigrams(sentences: Iterable[Sequence[str]], lower: bool = False) -> Counter[tuple[str, str]]`
  - Counts bigram occurrences across multiple sentences.

- `count_trigrams(sentences: Iterable[Sequence[str]], lower: bool = False) -> Counter[tuple[str, str, str]]`
  - Counts trigram occurrences across multiple sentences.

- `bigram_probs(sentences: Iterable[Sequence[str]]) -> dict[str, dict[str, float]]`
  - Computes bigram probabilities from input sequences.

- `trigram_probs(sentences: Iterable[Sequence[str]]) -> dict[tuple[str, str], dict[str, float]]`
  - Computes trigram probabilities from input sequences.

## Example Usage
```python
sentences = [
    ["The", "quick", "fox"],
    ["The", "slow", "dog"],
    ["The", "happy", "dog"]
]

bigram_counts = count_bigrams(sentences)
print(bigram_counts)

bigram_probabilities = bigram_probs(sentences)
print(bigram_probabilities)
