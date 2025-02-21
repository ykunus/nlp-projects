#!/usr/bin/env python

# Version 1.1.0
# 11/30/2024

import random
import unittest
from collections import defaultdict
from pathlib import Path

from grader import Grader, points, timeout
from hw5 import (
    RANDOM_SEED,
    CharBigramFeatureExtractor,
    CharTrigramFeatureExtractor,
    CharUnigramFeatureExtractor,
    ClassificationInstance,
    FeatureExtractor,
    InstanceCounter,
    LanguageIdentificationInstance,
    MulticlassScoring,
    Perceptron,
    TunedPerceptronHyperparameters,
    factor_decay,
    load_lid_instances,
)
from sklearn.metrics import accuracy_score

TEST_DATA_PATH = Path("test_data")
TRAIN_PATH = TEST_DATA_PATH / "mot_train.tsv"
DEV_PATH = TEST_DATA_PATH / "mot_dev.tsv"
# We store this is a tuple to prevent accidental modification, but
# it needs to be converted to a list before training.
SENTIMENT_DATA = (
    ClassificationInstance(
        "positive",
        ["I", "love", "tacos", "!"],
    ),
    ClassificationInstance(
        "negative",
        ["I", "dislike", "broccoli", "."],
    ),
    ClassificationInstance(
        "negative",
        [
            "I",
            "love",
            "to",
            "dislike",
            "tacos",
        ],
    ),
)


class TestScoring(unittest.TestCase):
    def setUp(self) -> None:
        self.predicted = [
            "positive",
            "positive",
            "neutral",
            "positive",
            "negative",
            "negative",
            "positive",
            "neutral",
            "positive",
            "negative",
        ]
        self.true = [
            "neutral",
            "positive",
            "positive",
            "neutral",
            "negative",
            "negative",
            "positive",
            "negative",
            "negative",
            "negative",
        ]
        self.scorer = MulticlassScoring(["positive", "neutral", "negative"])
        self.scorer.score(self.true, self.predicted)

    @points(4)
    def test_accuracy(self) -> None:
        """Accuracy is correct for a simple case."""
        self.assertEqual(float, type(self.scorer.accuracy()))
        self.assertAlmostEqual(
            accuracy_score(self.true, self.predicted), self.scorer.accuracy()
        )

    @points(0.5)
    def test_macro_f1_type(self) -> None:
        """Macro F1 returns a float."""
        self.assertEqual(float, type(self.scorer.macro_f1()))

    @points(0.5)
    def test_weighted_f1_type(self) -> None:
        """Weighted F1 returns a float."""
        self.assertEqual(float, type(self.scorer.weighted_f1()))

    @points(3)
    def test_confusion_counts(self) -> None:
        """Confusion counts are correct for a simple case."""
        self.assertEqual(1, self.scorer.confusion_count("positive", "neutral"))
        self.assertEqual(2, self.scorer.confusion_count("positive", "positive"))

    @points(3)
    def test_confusion_rates(self) -> None:
        """Confusion rates are correct for a simple case."""
        self.assertAlmostEqual(1 / 3, self.scorer.confusion_rate("positive", "neutral"))
        self.assertAlmostEqual(0.0, self.scorer.confusion_rate("positive", "negative"))


class TestFeatureExtraction(unittest.TestCase):
    @points(4)
    def test_unigram_features(self) -> None:
        """The correct set of unigram features is generated."""
        example_sentence = LanguageIdentificationInstance("eng", "hi!")
        features = set(
            CharUnigramFeatureExtractor().extract_features(example_sentence).features
        )
        self.assertSetEqual(
            {"h", "i", "!"},
            features,
        )

    @points(4)
    def test_bigram_features(self) -> None:
        """The correct set of bigram features is generated."""
        example_sentence = LanguageIdentificationInstance("eng", "hello!")
        features = set(
            CharBigramFeatureExtractor().extract_features(example_sentence).features
        )
        self.assertSetEqual(
            {"lo", "el", "he", "o!", "ll"},
            features,
        )

    @points(4)
    def test_trigram_features(self) -> None:
        """The correct set of trigram features is generated."""
        example_sentence = LanguageIdentificationInstance("eng", "hello!")
        features = set(
            CharTrigramFeatureExtractor().extract_features(example_sentence).features
        )
        self.assertSetEqual(
            {"hel", "ell", "llo", "lo!"},
            features,
        )


class TestInstanceCounter(unittest.TestCase):
    @points(2)
    def test_label_order(self) -> None:
        """Labels are sorted correctly."""
        labels = ["a", "c", "c", "b", "b"]
        instances = [ClassificationInstance(label, ["feature"]) for label in labels]
        counter = InstanceCounter()
        counter.count_instances(instances)
        self.assertListEqual(["b", "c", "a"], counter.labels())


class TestWeightUpdates(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(RANDOM_SEED)
        instance_counter = InstanceCounter()
        instance_counter.count_instances(SENTIMENT_DATA)
        self.labels = instance_counter.labels()

    @points(1)
    def test_labels(self) -> None:
        """Labels are stored correctly in the model."""
        model = Perceptron(self.labels)
        self.assertEqual(self.labels, model.labels)

    @points(1)
    def test_step(self) -> None:
        """Step is initialized to zero."""
        model = Perceptron(self.labels)
        self.assertEqual(0, model.step)

    @points(2)
    def test_weight_dicts(self) -> None:
        """Weights are stored using the correct data types."""
        model = Perceptron(self.labels)
        # Must be a dict, not a defaultdict
        self.assertEqual(dict, type(model.weights))
        for label in self.labels:
            # Must be a defaultdict
            self.assertEqual(defaultdict, type(model.weights[label]))
            # Default value must be a float
            self.assertEqual(float, type(model.weights[label]["test"]))

    @points(3)
    def test_first_update(self) -> None:
        """Weights are correct after the first training example."""
        # Test for two different LR values
        for lr in (1.0, 0.5):
            model = Perceptron(self.labels)
            data = [SENTIMENT_DATA[0]]
            model.train_epochs(data, 1, lr, shuffle=False)
            self.assertEqual(lr, model.weights["positive"]["I"])
            self.assertEqual(lr, model.weights["positive"]["love"])
            self.assertEqual(lr, model.weights["positive"]["tacos"])
            self.assertEqual(lr, model.weights["positive"]["!"])
            self.assertEqual(-lr, model.weights["negative"]["I"])
            self.assertEqual(-lr, model.weights["negative"]["love"])
            self.assertEqual(-lr, model.weights["negative"]["tacos"])
            self.assertEqual(-lr, model.weights["negative"]["!"])

    @points(3)
    def test_second_update(self) -> None:
        """Weights are correct after the second training example."""
        model = Perceptron(self.labels)
        data = list(SENTIMENT_DATA[:2])
        lr = 1.0
        model.train_epochs(data, 1, lr, shuffle=False)

        self.assertEqual(lr, model.weights["positive"]["love"])
        self.assertEqual(lr, model.weights["positive"]["tacos"])
        self.assertEqual(lr, model.weights["positive"]["!"])
        self.assertEqual(-lr, model.weights["negative"]["love"])
        self.assertEqual(-lr, model.weights["negative"]["tacos"])
        self.assertEqual(-lr, model.weights["negative"]["!"])

        self.assertEqual(-lr, model.weights["positive"]["dislike"])
        self.assertEqual(-lr, model.weights["positive"]["broccoli"])
        self.assertEqual(-lr, model.weights["positive"]["."])
        self.assertEqual(lr, model.weights["negative"]["dislike"])
        self.assertEqual(lr, model.weights["negative"]["broccoli"])
        self.assertEqual(lr, model.weights["negative"]["."])

        # Conflicting updates so the weight is zero
        for label in self.labels:
            self.assertEqual(0, model.weights[label]["I"])

    @points(3)
    def test_third_update(self) -> None:
        """Weights are correct after the third training example."""
        model = Perceptron(self.labels)
        data = list(SENTIMENT_DATA)
        model.train_epochs(data, 1, 1.0, shuffle=False)
        self.assertDictEqual(
            {
                "I": -1.0,
                "love": 0.0,
                "tacos": 0.0,
                "!": 1.0,
                "dislike": -2.0,
                "broccoli": -1.0,
                ".": -1.0,
                "to": -1.0,
            },
            model.weights["positive"],
        )
        self.assertDictEqual(
            {
                "I": 1.0,
                "love": 0.0,
                "tacos": 0.0,
                "!": -1.0,
                "dislike": 2.0,
                "broccoli": 1.0,
                ".": 1.0,
                "to": 1.0,
            },
            model.weights["negative"],
        )


class TestFactorDecay(unittest.TestCase):
    @points(1.0)
    def test_factor_decay_float(self) -> None:
        """Factor decay returns a float."""
        decay = factor_decay(1.0, 5, 2, 0.9)
        self.assertIsInstance(decay, float)


class TestTuning(unittest.TestCase):
    @points(1.0)
    def test_sweep_values_set(self) -> None:
        """Values have been set in TunedPerceptronHyperparameters."""
        params = TunedPerceptronHyperparameters()
        self.assertIsInstance(params.feature_extractor, FeatureExtractor)
        self.assertIsInstance(params.warmup_epochs, int)
        self.assertIsInstance(params.decay, float)


class TestModelPredictions(unittest.TestCase):
    train_instances: list[ClassificationInstance]
    dev_instances: list[ClassificationInstance]
    dev_labels: list[str]
    labels: list[str]
    model: Perceptron
    feature_extractor_not_implemented: bool = False

    @classmethod
    def setUpClass(cls) -> None:
        feature_extractor = CharBigramFeatureExtractor()
        cls.train_instances = [
            feature_extractor.extract_features(inst)
            for inst in load_lid_instances(TRAIN_PATH)
        ]
        # Crash now if the feature extractor isn't implemented
        if cls.train_instances[0] is None:
            cls.feature_extractor_not_implemented = True
            return

        cls.dev_instances = [
            feature_extractor.extract_features(inst)
            for inst in load_lid_instances(DEV_PATH)
        ]
        cls.dev_labels = [instance.label for instance in cls.dev_instances]
        instance_counter = InstanceCounter()
        instance_counter.count_instances(cls.train_instances)
        cls.labels = instance_counter.labels()

    def setUp(self):
        random.seed(RANDOM_SEED)
        if self.feature_extractor_not_implemented:
            raise ValueError(
                "Cannot test model predictions without CharBigramFeatureExtractor implemented"
            )

    def train_eval_model(
        self, model: Perceptron, n_epochs: int, lr: float = 1.0
    ) -> float:
        model.train_epochs(self.train_instances, n_epochs, lr)
        predictions = [
            model.classify(instance.features) for instance in self.dev_instances
        ]
        return accuracy_score(self.dev_labels, predictions)

    @points(10)
    @timeout(20)
    def test_train_1_2_5_epochs(self) -> None:
        """Accuracy is high enough after one, two, and five epochs."""
        model = Perceptron(self.labels)

        accuracy = self.train_eval_model(model, 1)
        # Solution gets 0.8806
        self.assertLessEqual(0.8804, accuracy)

        accuracy = self.train_eval_model(model, 1)
        # Solution gets 0.8964
        self.assertLessEqual(0.8962, accuracy)

        accuracy = self.train_eval_model(model, 3)
        # Solution gets 0.9087
        self.assertLessEqual(0.9085, accuracy)


def main() -> None:
    tests = [
        TestScoring,
        TestFeatureExtraction,
        TestInstanceCounter,
        TestWeightUpdates,
        TestFactorDecay,
        TestTuning,
        # Should be last since it's slowest
        TestModelPredictions,
    ]
    grader = Grader(tests)
    grader.print_results()


if __name__ == "__main__":
    main()
