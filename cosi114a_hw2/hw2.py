import math
import random
from abc import abstractmethod
from collections import defaultdict, Counter
from math import log, prod
from pathlib import Path
from typing import Sequence, Iterable, Generator, TypeVar, Union, Generic

############################################################
# The following constants and function are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.
# HW 2 stubs 1.0 10/1/2024

# DO NOT MODIFY
random.seed(0)

# DO NOT MODIFY
START_TOKEN = "<START>"
# DO NOT MODIFY
END_TOKEN = "<END>"
# DO NOT MODIFY
POS_INF = float("inf")
NEG_INF = float("-inf")
# DO NOT MODIFY (needed for copying code from HW 1)
T = TypeVar("T")


# DO NOT MODIFY
def load_tokenized_file(path: Union[Path, str]) -> Generator[Sequence[str], None, None]:
    """Yield sentences as sequences of tokens."""
    with open(path, encoding="utf8") as file:
        for line in file:
            line = line.rstrip("\n")
            tokens = line.split(" ")
            assert tokens, "Empty line in input"
            yield tuple(tokens)


# DO NOT MODIFY
def sample(probs: dict[str, float]) -> str:
    """Return a sample from a distribution."""
    # To avoid relying on the dictionary iteration order,
    # sort items before sampling. This is very slow and
    # should be avoided in general, but we do it in order
    # to get predictable results.
    items = sorted(probs.items())
    # Now split them back up into keys and values
    keys, vals = zip(*items)
    # Choose using the weights in the values
    return random.choices(keys, weights=vals)[0]


# DO NOT MODIFY
class ProbabilityDistribution(Generic[T]):
    """A generic probability distribution."""

    # DO NOT ADD AN __INIT__ METHOD HERE
    # You will implement this in subclasses

    # DO NOT MODIFY
    # You will implement this in subclasses
    @abstractmethod
    def prob(self, item: T) -> float:
        """Return a probability for the specified item."""
        raise NotImplementedError


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.


def sample_bigrams(probs: dict[str, dict[str, float]]) -> list[str]:
    result = []
    i  = 0
    keys = list(probs.keys())
    cur = keys[i]
    while i < len(keys):
        next_probs = probs.get(cur)
        if next_probs is None:
            break
        next_word = sample(next_probs)
        if next_word == '<END>':
            break
        result.append(next_word)
        cur = next_word
        i += 1
    return result


def sample_trigrams(probs: dict[tuple[str, str], dict[str, float]]) -> list[str]:
    result = []
    i = 0
    keys = list(probs.keys())
    cur = keys[i]
    while i < len(keys):
        next_probs = probs.get(cur)
        if next_probs is None:
            break
        next_word = sample(next_probs)
        if next_word == '<END>':
            break
        result.append(next_word)
        cur = (cur[1], next_word)
        i += 1
    return result


def unigram_counts(sentences: Iterable[Sequence[str]]) -> dict[str, int]:
    words = []
    for sentence in sentences:
        for word in sentence:
            words.append(word)
    counter = Counter(words)
    return dict(counter)


def bigram_counts(sentences: Iterable[Sequence[str]]) -> dict[str, dict[str, int]]:
    count = defaultdict(Counter)
    result = {}
    for sentence in sentences:
        probs = bigrams(sentence)
        for bigram in probs:
            count[bigram[0]][bigram[1]] += 1

    for unigram, value in count.items():
        result[unigram] = dict(value)
    return result


def trigram_counts(
    sentences: Iterable[Sequence[str]],
) -> dict[tuple[str, str], dict[str, int]]:
    count = defaultdict(Counter)
    result = {}
    for sentence in sentences:
        probs = trigrams(sentence)
        for trigram in probs:
            tmp = (trigram[0], trigram[1])
            count[tmp][trigram[2]] += 1

    for unigram, value in count.items():
        result[unigram] = dict(value)
    return result


class UnigramMLE(ProbabilityDistribution[str]):
    def __init__(self, probs: dict[str, int]) -> None:
        self.total = sum(probs.values())
        self.probs = probs
    def prob(self, item: str) -> float:
        if item in self.probs:
            return self.probs.get(item)/self.total
        return 0.0


class BigramMLE(ProbabilityDistribution[tuple[str, str]]):
    def __init__(self, probs: dict[str, dict[str, int]]) -> None:
        self.probs = probs
        self.count_for_entry = {}
        for unigram in probs:
            tmp_count = sum(probs.get(unigram).values())
            self.count_for_entry[unigram] = tmp_count

    def prob(self, item: tuple[str, str]) -> float:
        if item[0] in self.probs:
            if item[1] in self.probs.get(item[0]):
                return self.probs.get(item[0]).get(item[1]) / self.count_for_entry.get(item[0])
        return 0.0


class TrigramMLE(ProbabilityDistribution[tuple[str, str, str]]):
    def __init__(self, probs: dict[tuple[str, str], dict[str, int]]) -> None:
        self.probs = probs
        self.count_for_entry = {}
        for bigram in probs:
            tmp_count = sum(probs.get(bigram).values())
            self.count_for_entry[bigram] = tmp_count

    def prob(self, item: tuple[str, str, str]) -> float:
        cur = (item[0],item[1])
        if cur in self.probs:
            if item[2] in self.probs.get(cur):
                return self.probs.get(cur).get(item[2]) / self.count_for_entry.get(cur)
        return 0.0


class UnigramLidstoneSmoothed(ProbabilityDistribution[str]):
    def __init__(self, counts: dict[str, int], k: float) -> None:
        self.total_words = sum(counts.values())
        self.total_unique_words = len(counts)
        self.counts = counts
        self.k = k

    def prob(self, item: str) -> float:
        if item in self.counts:
            numerator = self.counts.get(item) + self.k
            denominator = self.total_words + self.k * self.total_unique_words
            return numerator / denominator
        return 0.0


class BigramInterpolation(ProbabilityDistribution[tuple[str, str]]):
    def __init__(
        self,
        uni_probs: ProbabilityDistribution[str],
        bi_probs: ProbabilityDistribution[tuple[str, str]],
        l_1: float,
        l_2: float,
    ) -> None:
        self.uni_probs =  uni_probs
        self.bi_probs = bi_probs
        self.l_1 = l_1
        self.l_2 = l_2


    def prob(self, item: tuple[str, str]) -> float:
        word1 = item[0]
        word2 = item[1]
        unigram_p = self.uni_probs.prob(word2)
        bigram_p = self.bi_probs.prob((word1,word2))
        return self.l_1 * unigram_p + self.l_2 * bigram_p


def unigram_sequence_prob(
    sequence: Sequence[str], probs: ProbabilityDistribution[str]
) -> float:
    total = 0
    for word in sequence:
        cur = probs.prob(word)
        if cur == 0:
            return NEG_INF
        total += math.log(cur)
    return total


def bigram_sequence_prob(
    sequence: Sequence[str], probs: ProbabilityDistribution[tuple[str, str]]
) -> float:
    total = 0
    sentence = [START_TOKEN] + list(sequence) + [END_TOKEN]
    for i in range(len(sentence)-1):
        word1 = sentence[i]
        word2 = sentence[i+1]
        cur = probs.prob((word1, word2))
        if cur == 0:
            return NEG_INF
        total += math.log(cur)
    return total


def trigram_sequence_prob(
    sequence: Sequence[str], probs: ProbabilityDistribution[tuple[str, str, str]]
) -> float:
    total = 0
    sentence = [START_TOKEN] + [START_TOKEN]+ list(sequence) + [END_TOKEN]
    for i in range(len(sentence) - 2):
        word1 = sentence[i]
        word2 = sentence[i + 1]
        word3 = sentence[i + 2]
        cur = probs.prob((word1, word2, word3))
        if cur == 0:
            return NEG_INF
        total += math.log(cur)
    return total

def perplexity_calculation(log_probs: int, n: int) -> float:
    avg_probs = log_probs / n
    return  2 ** (-avg_probs)



def unigram_perplexity(
    sentences: Iterable[Sequence[str]], probs: ProbabilityDistribution[str]
) -> float:
    n = 0
    total = 0
    for sentence in sentences:
        for word in sentence:
            cur = probs.prob(word)
            if cur == 0:
                return POS_INF
            total += math.log2(cur)
        n += len(sentence)
    return perplexity_calculation(total, n)



def bigram_perplexity(
    sentences: Iterable[Sequence[str]], probs: ProbabilityDistribution[tuple[str, str]]
) -> float:
    n = 0
    total = 0
    for sentence in sentences:

        new_sentence = [START_TOKEN] + list(sentence) + [END_TOKEN]
        for i in range(len(new_sentence) - 1):
            word1 = new_sentence[i]
            word2 = new_sentence[i + 1]
            cur = probs.prob((word1, word2))
            if cur == 0:
                return POS_INF
            total += math.log2(cur)
            n += 1
    return perplexity_calculation(total, n)


def trigram_perplexity(
    sentences: Iterable[Sequence[str]],
    probs: ProbabilityDistribution[tuple[str, str, str]],
) -> float:
    total = 0
    n = 0
    for sentence in sentences:
        new_sentence = [START_TOKEN] + [START_TOKEN] + list(sentence) + [END_TOKEN]
        for i in range(len( new_sentence) - 2):
            word1 =  new_sentence[i]
            word2 =  new_sentence[i + 1]
            word3 =  new_sentence[i + 2]
            cur = probs.prob((word1, word2, word3))
            if cur == 0:
                return POS_INF
            total += math.log2(cur)
            n += 1
    return perplexity_calculation(total, n)


############################################################
# From HW 1


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



# The following three functions should only be used to test your sampler


def bigram_probs(
        sentences: Iterable[Sequence[str]]
) -> dict[str, dict[str, float]]:
    bigram_count = defaultdict(Counter)
    for sentence in sentences:
        bigram_sentence = bigrams(sentence)
        for bigram in bigram_sentence:
                bigram_count[bigram[0]][bigram[1]] += 1
    result = {}
    for unigram, counts in bigram_count.items():
        tmp =  counts_to_probs(counts)
        result[unigram] = tmp
    return result



def trigram_probs(
    sentences: Iterable[Sequence[str]],
) -> dict[tuple[str, str], dict[str, float]]:
    trigram_count = defaultdict(Counter)
    for sentence in sentences:
        trigram_sentence = trigrams(sentence)
        for trigram in trigram_sentence:
            key = (trigram[0], trigram[1])
            trigram_count[key][trigram[2]] += 1
    result = {}
    for bigram, counts in trigram_count.items():
        tmp = counts_to_probs(counts)
        result[bigram] = tmp
    return result


def counts_to_probs(counts: Counter[T]) -> dict[T, float]:
    total = sum(counts.values())
    result = {}
    for item, count in counts.items():
        result[item] = count /total
    return result
