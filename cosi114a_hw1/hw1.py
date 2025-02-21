from collections import Counter, defaultdict
from operator import truediv

from typing import Iterable, TypeVar, Sequence

# DO NOT MODIFY
T = TypeVar("T")

# DO NOT MODIFY
START_TOKEN = "<START>"
# DO NOT MODIFY
END_TOKEN = "<END>"


def counts_to_probs(counts: Counter[T]) -> dict[T, float]:
    total = sum(counts.values())
    result = {}
    for item, count in counts.items():
        result[item] = count /total
    return result

def bigrams(sentence: Sequence[str]) -> list[tuple[str, str]]:
    sentence_list = list(sentence)
    tmp = [START_TOKEN] + sentence_list + [END_TOKEN]
    result=[]

    for i in range(len(tmp)-1):
        result.append((tmp[i], tmp[i+1]))

    return result

def trigrams(sentence: Sequence[str]) -> list[tuple[str, str, str]]:
    sentence_list = list(sentence)

    tmp = [START_TOKEN, START_TOKEN] + sentence_list + [END_TOKEN]
    result = []
    for i in range(len(tmp) - 2):
        result.append((tmp[i], tmp[i + 1], tmp[i + 2]))

    return result


def count_unigrams(
sentences: Iterable[Sequence[str]], lower: bool = False
) -> Counter[str]:
    counter = Counter()
    for sentence in sentences:
        if lower:
            lowered = []
            for word in sentence:
                lowered.append(word.lower())
            counter.update(lowered)
        else:
            counter.update(sentence)
    return counter

def count_bigrams(
    sentences: Iterable[Sequence[str]], lower: bool = False
) -> Counter[tuple[str, str]]:
    bigram_sentences = []
    for sentence in sentences:
        tmp =[]
        if lower:
            for word in sentence:
                lowered = word.lower()
                tmp.append(lowered)
        else:
            for word in sentence:
                tmp.append(word)
        for word in bigrams(tmp):
           bigram_sentences.append(word)
    counter = Counter(bigram_sentences)
    return counter

def count_trigrams(
    sentences: Iterable[Sequence[str]], lower: bool = False
) -> Counter[tuple[str, str, str]]:
    trigram_sentences =[]
    for sentence in sentences:
        tmp = []
        if lower:
            for word in sentence:
                lowered = word.lower()
                tmp.append(lowered)
        else:
            for word in sentence:
                tmp.append(word)
        for word in trigrams(tmp):
            trigram_sentences.append(word)
    counter = Counter(trigram_sentences)
    return counter

def bigram_probs(
        sentences: Iterable[Sequence[str]]
) -> dict[str, dict[str, float]]:
    bigram_counts = defaultdict(Counter)
    for sentence in sentences:
        bigram_sentence = bigrams(sentence)
        for bigram in bigram_sentence:
                bigram_counts[bigram[0]][bigram[1]] += 1
    result = {}
    for unigram, counts in bigram_counts.items():
        tmp =  counts_to_probs(counts)
        result[unigram] = tmp
    return result



def trigram_probs(
    sentences: Iterable[Sequence[str]],
) -> dict[tuple[str, str], dict[str, float]]:
    trigram_counts = defaultdict(Counter)
    for sentence in sentences:
        trigram_sentence = trigrams(sentence)
        for trigram in trigram_sentence:
            key = (trigram[0], trigram[1])
            trigram_counts[key][trigram[2]] += 1
    result = {}
    for bigram, counts in trigram_counts.items():
        tmp = counts_to_probs(counts)
        result[bigram] = tmp
    return result
