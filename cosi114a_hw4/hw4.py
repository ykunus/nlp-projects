from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from math import log
from operator import itemgetter
from typing import Generator, Iterable, Sequence

############################################################
# The following constants, classes, and function are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.
# HW 4 stubs 1.0.0 10/21/2024

# DO NOT MODIFY
NEG_INF = float("-inf")


# DO NOT MODIFY
class TaggedToken:
    """Store the text and tag for a token."""

    # DO NOT MODIFY
    def __init__(self, text: str, tag: str) -> None:
        self.text: str = text
        self.tag: str = tag

    # DO NOT MODIFY
    def __str__(self) -> str:
        return f"{self.text}/{self.tag}"

    # DO NOT MODIFY
    def __repr__(self) -> str:
        return f"<TaggedToken {str(self)}>"


# DO NOT MODIFY
class Tagger(ABC):
    # DO NOT IMPLEMENT THIS METHOD HERE
    @abstractmethod
    def train(self, sentences: Iterable[Sequence[TaggedToken]]) -> None:
        """Train the part of speech tagger by collecting needed counts from sentences."""
        raise NotImplementedError

    # DO NOT IMPLEMENT THIS METHOD HERE
    @abstractmethod
    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        """Tag a sentence with part of speech tags."""
        raise NotImplementedError

    # DO NOT MODIFY
    def tag_sentences(
        self, sentences: Iterable[Sequence[str]]
    ) -> Generator[list[str], None, None]:
        """Yield a list of tags for each sentence in the input."""
        for sentence in sentences:
            yield self.tag_sentence(sentence)

    # DO NOT MODIFY
    def test(
        self, tagged_sentences: Iterable[Sequence[TaggedToken]]
    ) -> tuple[list[str], list[str]]:
        """Return a tuple containing a list of predicted tags and a list of actual tags.

        Does not preserve sentence boundaries to make evaluation simpler.
        """
        predicted: list[str] = []
        actual: list[str] = []
        for sentence in tagged_sentences:
            predicted.extend(self.tag_sentence([tok.text for tok in sentence]))
            actual.extend([tok.tag for tok in sentence])
        return predicted, actual


# DO NOT MODIFY
def safe_log(n: float) -> float:
    """Return the log of a number or -inf if the number is zero."""
    return NEG_INF if n == 0.0 else log(n)


# DO NOT MODIFY
def max_item(scores: dict[str, float]) -> tuple[str, float]:
    """Return the key and value with the highest value."""
    # PyCharm gives a false positive type warning here
    # noinspection PyTypeChecker
    return max(scores.items(), key=itemgetter(1))


# DO NOT MODIFY
def most_frequent_item(counts: Counter[str]) -> str:
    """Return the most frequent item in a Counter.

    In case of ties, the lexicographically first item is returned.
    """
    assert counts, "Counter is empty"
    return items_descending_value(counts)[0]


# DO NOT MODIFY
def items_descending_value(counts: Counter[str]) -> list[str]:
    """Return the keys in descending frequency, breaking ties lexicographically."""
    # Why can't we just use most_common? It sorts by descending frequency, but items
    # of the same frequency follow insertion order, which we can't depend on.
    # Why can't we just use sorted with reverse=True? It will give us descending
    # by count, but reverse lexicographic sorting, which is confusing.
    # So instead we used sorted() normally, but for the key provide a tuple of
    # the negative value and the key.
    # PyCharm gives a false positive type warning here
    # noinspection PyTypeChecker
    return [key for key, value in sorted(counts.items(), key=_items_sort_key)]


# DO NOT MODIFY
def _items_sort_key(item: tuple[str, int]) -> tuple[int, str]:
    # This is used by items_descending_count, but you should never call it directly.
    return -item[1], item[0]


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.


class MostFrequentTagTagger(Tagger):
    def __init__(self) -> None:
        # Add an attribute to store the most frequent tag
        self.most_frequent_tag = None

    def train(self, sentences: Iterable[Sequence[TaggedToken]]) -> None:
        counts = Counter()
        for sentence in sentences:
            for token in sentence:
                counts[token.tag]+=1
        self.most_frequent_tag = most_frequent_item(counts)

    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        return [self.most_frequent_tag]*len(sentence)


class UnigramTagger(Tagger):
    def __init__(self) -> None:
        # Add data structures that you need here
        self.POS_words ={}
        self.most_frequent = {}
        self.most_frequent_tag = None
    def train(self, sentences: Iterable[Sequence[TaggedToken]]) -> None:
        counts_tags = Counter()
        for sentence in sentences:
            for token in sentence:
                counts_tags[token.tag]+=1
                if token.text not in self.POS_words:
                    self.POS_words[token.text] = Counter()
                self.POS_words[token.text][token.tag] += 1

        self.most_frequent_tag = most_frequent_item(counts_tags)

        for word, counts in self.POS_words.items():
            self.POS_words[word] = max(counts, key=counts.get)


    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        pos_list =[]
        for word in sentence:
            pos_list.append(self.POS_words.get(word, self.most_frequent_tag))
        return pos_list


class SentenceCounter:
    def __init__(self, k: float) -> None:
        self.k = k
        self.unique_tags_count=Counter()
        self.sorted_unique_tokens = []
        # self.tokens_with_tag= {}#all tokens that have tag
        self.tokens_with_tag = defaultdict(set)
        self.tokens_seen_with_tags = Counter() # number of times the token w was seen with the tag t in training.
        self.first_tokens_in_sentence =Counter()
        self.sum_of_first_tokens_in_sentence = 0
        self.transition_prob_of_tags = defaultdict(Counter)
        self.all_transitions_from_tag = Counter()


    def count_sentences(self, sentences: Iterable[Sequence[TaggedToken]]) -> None:
        for sentence in sentences:
            self.first_tokens_in_sentence[sentence[0].tag] += 1
            for i in range(len(sentence)):
                self.unique_tags_count[sentence[i].tag] += 1
                if i != len(sentence)-1:
                    self.transition_prob_of_tags[sentence[i].tag][sentence[i+1].tag] += 1
                    self.all_transitions_from_tag[sentence[i].tag]+=1
                self.tokens_with_tag[sentence[i].tag].add(sentence[i].text)
                self.tokens_seen_with_tags[(sentence[i].text, sentence[i].tag)]+=1
        self.sorted_unique_tokens = items_descending_value(self.unique_tags_count)
        self.sum_of_first_tokens_in_sentence = sum(self.first_tokens_in_sentence.values())


    def unique_tags(self) -> list[str]:
        return self.sorted_unique_tokens

    def emission_prob(self, tag: str, word: str) -> float:
            numerator = self.tokens_seen_with_tags.get((word, tag), 0) + self.k
            denominator = self.unique_tags_count.get(tag, 0) + (self.k * len(self.tokens_with_tag.get(tag, set())))
            if denominator == 0:
                return 0.0
            return numerator / denominator

    def transition_prob(self, tag1: str, tag2: str) -> float:
        if tag1 in self.transition_prob_of_tags:
            if tag2 in self.transition_prob_of_tags.get(tag1):
                if self.all_transitions_from_tag.get(tag1) != 0:
                    return self.transition_prob_of_tags[tag1][tag2]/self.all_transitions_from_tag.get(tag1)
        return 0.0

    def initial_prob(self, tag: str) -> float:

        if self.sum_of_first_tokens_in_sentence == 0 or tag not in self.tokens_with_tag:
            return 0.0
        return self.first_tokens_in_sentence[tag]/self.sum_of_first_tokens_in_sentence


class BigramTagger(Tagger, ABC):
    # You can add additional methods to this class if you want to share anything
    # between the greedy and Viterbi taggers. However, do not modify any of the
    # implemented methods and do not override __init__ or train in the subclasses.

    def __init__(self, k) -> None:
        # DO NOT MODIFY THIS METHOD
        self.counter = SentenceCounter(k)

    def train(self, sents: Iterable[Sequence[TaggedToken]]) -> None:
        # DO NOT MODIFY THIS METHOD
        self.counter.count_sentences(sents)

    def sequence_probability(self, sentence: Sequence[str], tags: Sequence[str]) -> float:
        """Return the probability for a sequence of tags given tokens."""
        probability = 0.0
        probability += safe_log(self.counter.initial_prob(tags[0])) + safe_log(self.counter.emission_prob(tags[0], sentence[0]))

        for i in range(1, len(sentence)):
            prev = tags[i-1]
            cur = tags[i]
            probability += safe_log(self.counter.transition_prob(prev, cur))
            probability += safe_log(self.counter.emission_prob(cur, sentence[i]))

        return probability

class GreedyBigramTagger(BigramTagger):
    # DO NOT IMPLEMENT AN __init__ METHOD

    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        result_tags = []
        unique_tags = self.counter.unique_tags()
        best_tag = None
        max_probability = float('-inf')

        for tag in unique_tags:
            cur_prob = safe_log(self.counter.initial_prob(tag)) + safe_log(self.counter.emission_prob(tag, sentence[0]))
            if cur_prob > max_probability:
                max_probability = cur_prob
                best_tag = tag
        result_tags.append(best_tag)

        for i in range(1, len(sentence)):
            max_probability = float('-inf')
            best_tag = None
            prev = result_tags[-1]
            for tag in unique_tags:
                transition_prob = safe_log(self.counter.transition_prob(prev, tag))
                emission_prob = safe_log(self.counter.emission_prob(tag, sentence[i]))
                cur_prob = transition_prob + emission_prob
                if cur_prob > max_probability:
                    max_probability = cur_prob
                    best_tag = tag
            result_tags.append(best_tag)
        return result_tags


class ViterbiBigramTagger(BigramTagger):
    # DO NOT IMPLEMENT AN __init__ METHOD

    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        unique_tags = self.counter.unique_tags()
        length = len(sentence)
        best_scores = []
        back_pointers = []
        cur_best_scores = {}
        cur_back_pointers = {}
        for tag in unique_tags:
            cur_best_scores[tag] =(safe_log(self.counter.initial_prob(tag)) +
                                   safe_log(self.counter.emission_prob(tag, sentence[0])))
            cur_back_pointers[tag] = None
        best_scores.append(cur_best_scores)
        back_pointers.append(cur_back_pointers)
        for i in range (1, length):
            cur_best_scores = {}
            cur_back_pointers = {}
            for tag in unique_tags:
                best_tag_prev = None
                max_probability = float('-inf')
                for prev in unique_tags:
                    prob = (best_scores[i-1][prev] + safe_log(self.counter.transition_prob(prev, tag)) +
                            safe_log(self.counter.emission_prob(tag, sentence[i])))
                    if prob > max_probability:
                        max_probability = prob
                        best_tag_prev = prev
                cur_best_scores[tag] = max_probability
                cur_back_pointers[tag] = best_tag_prev
            best_scores.append(cur_best_scores)
            back_pointers.append(cur_back_pointers)

        max_probability = float('-inf')
        best_tag = None
        for tag in unique_tags:
            if best_scores[-1][tag] > max_probability:
                max_probability = best_scores[-1][tag]
                best_tag = tag
        tags = [best_tag]
        for i in range(length -1, 0, -1):
            tags.insert(0, back_pointers[i][tags[0]])

        return tags