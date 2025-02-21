import math
from collections import defaultdict, Counter
from math import log
from typing import (
    Iterable,
    Sequence,
)

# Version 1.0.0
# 10/11/2024

############################################################
# The following classes and methods are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.

# DO NOT MODIFY
START_TOKEN = "<START>"
# DO NOT MODIFY
END_TOKEN = "<END>"


# DO NOT MODIFY
class AirlineSentimentInstance:
    """Represents a single instance from the airline sentiment dataset.

    Each instance contains the sentiment label, the name of the airline,
    and the sentences of text. The sentences are stored as a tuple of
    tuples of strings. The outer tuple represents sentences, and each
    sentences is a tuple of tokens."""

    def __init__(
        self, label: str, airline: str, sentences: Sequence[Sequence[str]]
    ) -> None:
        self.label: str = label
        self.airline: str = airline
        # These are converted to tuples so they cannot be modified
        self.sentences: tuple[tuple[str, ...], ...] = tuple(
            tuple(sentence) for sentence in sentences
        )

    def __repr__(self) -> str:
        return f"<AirlineSentimentInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"label={self.label}; airline={self.airline}; sentences={self.sentences}"


# DO NOT MODIFY
class SentenceSplitInstance:
    """Represents a potential sentence boundary in context.

    Each instance is labeled with whether it is ('y') or is not ('n') a sentence
    boundary, the characters to the left of the boundary token, the potential
    boundary token itself (punctuation that could be a sentence boundary), and
    the characters to the right of the boundary token."""

    def __init__(
        self, label: str, left_context: str, token: str, right_context: str
    ) -> None:
        self.label: str = label
        self.left_context: str = left_context
        self.token: str = token
        self.right_context: str = right_context

    def __repr__(self) -> str:
        return f"<SentenceSplitInstance: {str(self)}>"

    def __str__(self) -> str:
        return " ".join(
            [
                f"label={self.label};",
                f"left_context={repr(self.left_context)};",
                f"token={repr(self.token)};",
                f"right_context={repr(self.right_context)}",
            ]
        )


# DO NOT MODIFY
class ClassificationInstance:
    """Represents a label and features for classification."""

    def __init__(self, label: str, features: Iterable[str]) -> None:
        self.label: str = label
        # Features can be passed in as any iterable and they will be
        # stored in a tuple
        self.features: tuple[str, ...] = tuple(features)

    def __repr__(self) -> str:
        return f"<ClassificationInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"label={self.label}; features={self.features}"


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.


def accuracy(predictions: Sequence[str], expected: Sequence[str]) -> float:
    if len(predictions) != len(expected):
        raise ValueError("Predictions and expected are not the same length.")
    if len(predictions) == 0:
        raise ValueError("Predictions cannot be an empty sequence.")

    correct = 0.0
    for i in range (0, len(predictions)):
        if predictions[i] == expected[i]:
            correct += 1
    return correct / len(predictions)


def recall(
    predictions: Sequence[str], expected: Sequence[str], positive_label: str
) -> float:
    if len(predictions) != len(expected):
        raise ValueError("Predictions and expected are not the same length.")
    if len(predictions) == 0:
        raise ValueError("Predictions cannot be an empty sequence.")
    true_positive = 0.0
    false_negative = 0.0
    for i in range(0, len(predictions)):
        if predictions[i] == positive_label:
            if expected[i] == positive_label:
                true_positive += 1
        elif expected[i] == positive_label:
            false_negative += 1
    denominator =  (true_positive + false_negative)

    if denominator == 0:
        return 0.0
    else:
        return true_positive / denominator


def precision(
    predictions: Sequence[str], expected: Sequence[str], positive_label: str
) -> float:
    if len(predictions) != len(expected):
        raise ValueError("Predictions and expected are not the same length.")
    if len(predictions) == 0:
        raise ValueError("Predictions cannot be an empty sequence.")
    true_positive = 0.0
    false_positive = 0.0
    for i in range (0, len(predictions)):
        if predictions[i] == positive_label:
            if expected[i] == positive_label:
                true_positive += 1
            else:
                false_positive += 1

    denominator = (true_positive + false_positive)

    if denominator == 0:
        return 0.0
    else:
        return true_positive / denominator



def f1(predictions: Sequence[str], expected: Sequence[str], positive_label: str) -> float:
    """Compute the F1-score of the provided predictions."""
    if len(predictions) != len(expected):
        raise ValueError("Predictions and expected are not the same length.")
    if len(predictions) == 0:
        raise ValueError("Predictions cannot be an empty sequence.")

    precision_score = precision(predictions, expected, positive_label)
    recall_score = recall(predictions, expected, positive_label)

    denominator = precision_score + recall_score

    if denominator == 0:
        return 0.0
    else:
        return 2* ((precision_score * recall_score) / denominator)


class UnigramAirlineSentimentFeatureExtractor:
    @staticmethod
    def extract_features(instance: AirlineSentimentInstance) -> ClassificationInstance:
        unique_words = set()
        for sentence in instance.sentences:
            for word in sentence:
                tmp = word.lower()
                unique_words.add(tmp)
        result = ClassificationInstance(instance.label, unique_words)
        return result


class BigramAirlineSentimentFeatureExtractor:
    @staticmethod
    def extract_features(instance: AirlineSentimentInstance) -> ClassificationInstance:
        unique_words = set()
        for sentence in instance.sentences:
            lowercase_words = [word.lower() for word in sentence]
            bigram_words = bigrams(lowercase_words)
            for word1, word2 in bigram_words:
                tmp = str((word1, word2))
                unique_words.add(tmp)
        result = ClassificationInstance(instance.label, unique_words)
        return result


class BaselineSegmentationFeatureExtractor:
    @staticmethod
    def extract_features(instance: SentenceSplitInstance) -> ClassificationInstance:
        representation = [f"left_tok={instance.left_context}", f"split_tok={instance.token}", f"right_tok={instance.right_context}"]
        result = ClassificationInstance(instance.label, representation)
        return result


class InstanceCounter:
    def __init__(self) -> None:
        self.label_counts = Counter()
        self.total_feature_count_for_label_counter = Counter()
        self.feature_label_joint_count_counter = Counter()
        self.labels = 0
        self.unique_features_count = 0
        self.feature_set_return = set()
        self.unique_labels_list =[]

    def count_instances(self, instances: Iterable[ClassificationInstance]) -> None:
        # You should fill in this loop. Do not try to store the instances!
        unique_labels = set()
        unique_features = set()
        for instance in instances:
            self.labels += 1
            unique_labels.add(instance.label)
            self.label_counts[instance.label] +=1
            self.total_feature_count_for_label_counter[instance.label] += len(instance.features)
            for feature in instance.features:
                unique_features.add(feature)
                self.feature_label_joint_count_counter[(instance.label, feature)] += 1
        self.unique_features_count = len(unique_features)
        self.feature_set_return = unique_features
        self.unique_labels_list = list(unique_labels)
    def label_count(self, label: str) -> int:
        if label not in self.label_counts:
            raise ValueError(f"Label {label} is not a valid label.")
        return self.label_counts[label]

    def total_labels(self) -> int:
        return self.labels

    def feature_label_joint_count(self, feature: str, label: str) -> int:
        return self.feature_label_joint_count_counter.get((label, feature), 0)

    def unique_labels(self) -> list[str]:
        return self.unique_labels_list

    def feature_vocab_size(self) -> int:
        return self.unique_features_count

    def feature_set(self) -> set[str]:
        return self.feature_set_return

    def total_feature_count_for_label(self, label: str) -> int:
        return self.total_feature_count_for_label_counter.get(label, 0)


class NaiveBayesClassifier:
    # DO NOT MODIFY
    def __init__(self, k: float):
        self.k: float = k
        self.instance_counter: InstanceCounter = InstanceCounter()

    # DO NOT MODIFY
    def train(self, instances: Iterable[ClassificationInstance]) -> None:
        self.instance_counter.count_instances(instances)

    def prior_prob(self, label: str) -> float:
        return self.instance_counter.label_count(label) / self.instance_counter.total_labels()

    def feature_prob(self, feature: str, label) -> float:
        if feature not in self.instance_counter.feature_set():
            return 0.0
        count_of_feature = self.instance_counter.feature_label_joint_count(feature, label)
        total_feature_label_count = self.instance_counter.total_feature_count_for_label(label)
        vocab_count = self.instance_counter.feature_vocab_size()
        return (count_of_feature + self.k) / (total_feature_label_count + self.k * vocab_count)

    def log_posterior_prob(self, features: Sequence[str], label: str) -> float:
        log_of_prior = math.log(self.prior_prob(label))
        for feature in features:
            if feature not in self.instance_counter.feature_set():
                continue
            else:
                prob_of_feature = self.feature_prob(feature, label)
                if prob_of_feature > 0:
                    log_of_prior += math.log(prob_of_feature)

        return log_of_prior

    def classify(self, features: Sequence[str]) -> str:
        labels = self.instance_counter.unique_labels()
        log_probs = []
        for label in labels:
            log_prob= self.log_posterior_prob(features, label)
            log_prob_tuple = (log_prob, label)
            log_probs.append(log_prob_tuple)
        return max(log_probs)[1]
    def test(
        self, instances: Iterable[ClassificationInstance]
    ) -> tuple[list[str], list[str]]:
        predictions = []
        actual = []
        for instance in instances:
            prediction = self.classify(instance.features)
            predictions.append(prediction)
            actual.append(instance.label)
        result = (predictions, actual)
        return result

# MODIFY THIS AND DO THE FOLLOWING:
# 1. Inherit from UnigramAirlineSentimentFeatureExtractor or BigramAirlineSentimentFeatureExtractor
#    (instead of object) to get an implementation for the extract_features method.
# 2. Set a value for self.k below based on your tuning experiments.
class TunedAirlineSentimentFeatureExtractor(BigramAirlineSentimentFeatureExtractor):
    def __init__(self) -> None:
        self.k = float(0.005)

def bigrams(sentence: Sequence[str]) -> list[tuple[str, str]]:
    sentence_list = list(sentence)
    tmp = [START_TOKEN] + sentence_list + [END_TOKEN]
    result=[]

    for i in range(len(tmp)-1):
        result.append((tmp[i], tmp[i+1]))

    return result