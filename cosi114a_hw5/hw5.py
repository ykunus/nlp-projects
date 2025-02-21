import random
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from operator import itemgetter
from pathlib import Path
from typing import Generator, Iterable, Sequence, Union

# DO NOT MODIFY
RANDOM_SEED = 12345
random.seed(RANDOM_SEED)


############################################################
# The following constants, classes, and function are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.
# Version 1.0
# 11/15/2024


# DO NOT MODIFY
class ClassificationInstance:
    """Represents a label and features for classification."""

    def __init__(self, label: str, features: Iterable[str]) -> None:
        self.label: str = label
        # Features can be passed in as any iterable and they will be
        # stored in a tuple
        self.features: tuple[str, ...] = tuple(features)
        assert self.features, f"Empty features: {features}"
        for feature in self.features:
            assert isinstance(feature, str), f"Non-string feature: {repr(feature)}"

    def __repr__(self) -> str:
        return f"<ClassificationInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"label={self.label}; features={self.features}"


# DO NOT MODIFY
class LanguageIdentificationInstance:
    """Represent a single instance from a language ID dataset."""

    def __init__(
        self,
        language: str,
        text: str,
    ) -> None:
        self.language: str = language
        self.text: str = text

    def __repr__(self) -> str:
        return f"<LanguageIdentificationInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"label={self.language}; text={self.text}"

    # DO NOT CALL THIS METHOD
    # It's called by data loading functions.
    @classmethod
    def from_line(cls, line: str) -> "LanguageIdentificationInstance":
        splits = line.rstrip("\n").split("\t")
        assert len(splits) == 2
        assert splits[0]
        assert len(splits[1]) >= 2, f"Line too short: {repr(line)}"
        return cls(splits[0], splits[1])


# DO NOT MODIFY
def load_lid_instances(
    path: Union[str, Path],
) -> Generator[LanguageIdentificationInstance, None, None]:
    """Load LID instances from a file."""
    with open(path, encoding="utf8") as file:
        for line in file:
            yield LanguageIdentificationInstance.from_line(line)


# DO NOT MODIFY
def max_item(scores: dict[str, float]) -> tuple[str, float]:
    """Return the key and value with the highest value."""
    # PyCharm gives a false positive type warning here
    # noinspection PyTypeChecker
    return max(scores.items(), key=itemgetter(1))


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


# DO NOT MODIFY
class FeatureExtractor(ABC):
    """An abstract class for representing feature extractors."""

    @staticmethod
    @abstractmethod
    def extract_features(
        instance: LanguageIdentificationInstance,
    ) -> ClassificationInstance:
        raise NotImplementedError


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.


class MulticlassScoring:
    def __init__(self, labels: Sequence[str]) -> None:
        self.labels = labels
        self.true_counts = {}
        self.predicted_counts = {}
        self.correct_counts = {}
        self.total_counts = 0
        self.total_correct_counts = 0
        self.confusion_matrix = {}

    def score(
        self,
        true_labels: Sequence[str],
        predicted_labels: Sequence[str],
    ) -> None:
        if len(true_labels) != len(predicted_labels):
            raise ValueError("The length of predicted labels and true labels have to be the same.")
        if not true_labels or not predicted_labels:
            raise ValueError("True labels or predicted labels cannot be empty.")

        for true_label in self.labels:
            row_contents = {pred_label: 0 for pred_label in self.labels}
            self.confusion_matrix[true_label] = row_contents

        for label in self.labels:
            self.true_counts[label] = 0
            self.predicted_counts[label] = 0
            self.correct_counts[label] = 0

        for true_label, predicted_label in zip(true_labels, predicted_labels):
            self.confusion_matrix[true_label][predicted_label] += 1
            self.total_counts = len(true_labels)
            self.true_counts[true_label] += 1
            self.predicted_counts[predicted_label] += 1
            if true_label == predicted_label:
                self.correct_counts[true_label] += 1
                self.total_correct_counts += 1

    def accuracy(self) -> float:
        if self.total_counts == 0:
            return 0.0
        return self.total_correct_counts / self.total_counts

    def precision(self, label: str) -> float:
        if self.predicted_counts[label] == 0 :
            return 0.0
        return self.correct_counts[label] / self.predicted_counts[label]

    def recall(self, label: str) -> float:
        if self.true_counts[label] == 0:
            return 0.0
        return self.correct_counts[label] / self.true_counts[label]

    def f1(self, label: str) -> float:
        numerator =  self.precision(label) * self.recall(label)
        denominator = self.precision(label) + self.recall(label)
        if denominator == 0:
            return 0.0
        return 2 * numerator / denominator

    def macro_f1(self) -> float:
        total = 0.0
        for label in self.labels:
            total += self.f1(label)
        return total / len(self.labels)

    def weighted_f1(self) -> float:
        total = 0.0
        for label in self.labels:
            total += self.true_counts[label] * self.f1(label)
        if self.total_counts == 0:
            return 0.0
        return total / self.total_counts

    def confusion_count(self, true_label: str, predicted_label: str) -> int:
        return self.confusion_matrix[true_label][predicted_label]

    def confusion_rate(self, true_label: str, predicted_label: str) -> float:
        if self.true_counts[true_label] == 0:
            return 0.0
        return self.confusion_matrix[true_label][predicted_label] / self.true_counts[true_label]


class CharUnigramFeatureExtractor(FeatureExtractor):
    @staticmethod
    def extract_features(
        instance: LanguageIdentificationInstance,
    ) -> ClassificationInstance:
        text = instance.text
        result = set(text)
        return ClassificationInstance(label=instance.language, features=result)


class CharBigramFeatureExtractor(FeatureExtractor):
    @staticmethod
    def extract_features(
        instance: LanguageIdentificationInstance,
    ) -> ClassificationInstance:
        text = instance.text
        result = set()
        for i in range(len(text) - 1):
            bigram = text[i:i + 2]
            result.add(bigram)
        return ClassificationInstance(label=instance.language, features=result)


class CharTrigramFeatureExtractor(FeatureExtractor):
    @staticmethod
    def extract_features(
        instance: LanguageIdentificationInstance,
    ) -> ClassificationInstance:
        text = instance.text
        result = set()
        for i in range(len(text) - 2):
            trigram = text[i:i + 3]
            result.add(trigram)
        return ClassificationInstance(label=instance.language, features=result)


class InstanceCounter:
    def __init__(self) -> None:
        self.label_counter = Counter()
        self.labels_sorted = []

    def count_instances(self, instances: Iterable[ClassificationInstance]) -> None:
        for instance in instances:
            self.label_counter[instance.label] += 1
        self.labels_sorted = items_descending_value(self.label_counter)


    def labels(self) -> list[str]:
        return self.labels_sorted


class Perceptron:
    def __init__(self, labels: list[str]) -> None:
        self.labels = labels
        self.step = 0
        self.weights = {}
        for label in labels:
            self.weights[label] = defaultdict(float)


    def classify(self, features: Iterable[str]) -> str:
        scores = {}
        for label in self.labels:
            scores[label] = 0.0
            for feature in features:
                scores[label] += self.weights[label][feature]
        return max_item(scores)[0]
    def learn(
        self,
        instance: ClassificationInstance,
        lr: float,
    ) -> None:
        self.step +=1
        predicted = self.classify(instance.features)
        if predicted != instance.label:
            for feature in instance.features:
                self.weights[instance.label][feature] += lr
                self.weights[predicted][feature] -= lr

    def train_epochs(
        self,
        data: list[ClassificationInstance],
        n_epochs: int,
        lr: float,
        shuffle: bool = True,
    ):
        for epoch in range(n_epochs):
            if shuffle:
                random.shuffle(data)
            for instance in data:
                self.learn(instance, lr)


def factor_decay(init_lr: float, epoch: int, warmup_epochs: int, decay: float) -> float:
    if epoch <= warmup_epochs:
        return init_lr
    return init_lr * (decay ** (epoch - warmup_epochs))


def sweep_hyperparameters() -> None:
    random.seed(RANDOM_SEED)
    extractors = [CharUnigramFeatureExtractor(), CharBigramFeatureExtractor(), CharTrigramFeatureExtractor()]
    warmup_epochs = [1, 2, 3, 4]
    decay_rates = [0.6, 0.7, 0.8, 0.9, 1.0]
    results = []

    for extractor in extractors:
        for warmup_epoch in warmup_epochs:
            for decay in decay_rates:
                training_data = []
                instances= load_lid_instances("test_data/mot_train.tsv")
                for instance in instances:
                    classification_instance = extractor.extract_features(instance)
                    training_data.append(classification_instance)
                counter = InstanceCounter()
                counter.count_instances(training_data)
                labels = counter.labels()
                perceptron = Perceptron(labels)

                init_lr = 1.0
                for epoch in range(1,6):
                    lr = factor_decay(init_lr, epoch, warmup_epoch, decay)
                    perceptron.train_epochs(training_data, n_epochs=1, lr=lr, shuffle=True)

                dev_data = []
                instances = load_lid_instances("test_data/mot_dev.tsv")
                for instance in instances:
                    classification_instance = extractor.extract_features(instance)
                    dev_data.append(classification_instance)

                labels_true = [instance.label for instance in dev_data]
                predicted_labels = [perceptron.classify(instance.features) for instance in dev_data]
                scorer = MulticlassScoring(labels)
                scorer.score(labels_true, predicted_labels)
                accuracy = scorer.accuracy()

                results.append({
                    "feature_extractor": type(extractor).__name__, "warmup_epochs": warmup_epochs, "decay": decay, "accuracy": accuracy,})
                print(f"Feature Extractor: {type(extractor).__name__}, "
                      f"Warmup Epochs: {warmup_epoch}, Decay: {decay}, Accuracy: {accuracy}")
    best = max(results,key=lambda x: x["accuracy"])
    print(f"Best Accuracy: {best}")


class TunedPerceptronHyperparameters:
    # TODO: Fill in these parameters based on what you see in sweep_hyperparameters
    def __init__(self) -> None:
        self.feature_extractor: FeatureExtractor = CharTrigramFeatureExtractor()
        self.warmup_epochs: int = 1
        self.decay: float = 0.9


if __name__ == "__main__":
    sweep_hyperparameters()
