# Perceptron-Based Language Identification

## Overview
This project implements a **multiclass perceptron classifier** for **language identification** using **character n-gram features** (unigrams, bigrams, and trigrams). The classifier is trained on **news article snippets** in **17 languages** and evaluated using **precision, recall, F1-score, and accuracy**.

The model is tuned using **grid search** over different hyperparameter configurations, including **warmup epochs** and **learning rate decay**.

---

## Key Concepts
- **Perceptron Algorithm**: A simple **linear classifier** that updates weights based on misclassifications.
- **Multiclass Classification**: Predicting among **17 different language labels**.
- **Character-Level Feature Extraction**:
  - **Unigrams** (single characters)
  - **Bigrams** (character pairs)
  - **Trigrams** (character triplets)
- **Evaluation Metrics**:
  - **Accuracy**
  - **Precision, Recall, and F1-score (per language)**
  - **Macro & Weighted F1-score**
  - **Confusion Matrix**
- **Learning Rate Decay**:
  - Gradually reduces the learning rate after **warmup epochs**.
  - Prevents **overfitting** and improves generalization.
- **Hyperparameter Tuning**:
  - **Feature Extractor Selection**: (Unigrams, Bigrams, Trigrams)
  - **Warmup Epochs**: (1-4)
  - **Decay Rate**: (0.6 - 1.0)

---

## Implemented Classes & Functions

### **1. Multiclass Evaluation (`MulticlassScoring`)**
- **`accuracy()`**: Computes overall accuracy.
- **`precision(label)`**: Precision for a specific language.
- **`recall(label)`**: Recall for a specific language.
- **`f1(label)`**: F1-score for a specific language.
- **`macro_f1()`**: Macro-averaged F1-score across all languages.
- **`weighted_f1()`**: Weighted F1-score based on label frequency.
- **`confusion_count(true_label, predicted_label)`**: Counts misclassifications.
- **`confusion_rate(true_label, predicted_label)`**: Normalized misclassification rate.

### **2. Feature Extractors**
- **`CharUnigramFeatureExtractor`**: Extracts **single-character** features.
- **`CharBigramFeatureExtractor`**: Extracts **two-character sequences**.
- **`CharTrigramFeatureExtractor`**: Extracts **three-character sequences**.

### **3. Instance Counter (`InstanceCounter`)**
- **Counts unique language labels** in training data.
- **Ensures deterministic label ordering**.

### **4. Perceptron Model (`Perceptron`)**
- **`classify(features)`**: Predicts a language based on feature weights.
- **`learn(instance, lr)`**: Updates weights using perceptron learning rule.
- **`train_epochs(data, n_epochs, lr, shuffle=True)`**: Trains perceptron over multiple epochs.

### **5. Learning Rate Decay**
- **`factor_decay(init_lr, epoch, warmup_epochs, decay)`**:
  - Uses **fixed learning rate for warmup epochs**.
  - **Exponentially decays** learning rate after warmup.

### **6. Hyperparameter Tuning (`sweep_hyperparameters`)**
- Performs **grid search** over:
  - **Feature extractor type** (unigram, bigram, trigram).
  - **Warmup epochs** (1-4).
  - **Decay rates** (0.6 - 1.0).
- Selects the **best performing configuration**.

---

## Example Usage

### **Extracting Features from Text**
```python
from test_hw5 import load_lid_instances

instance = LanguageIdentificationInstance("eng", "Hello, how are you?")
features = CharTrigramFeatureExtractor.extract_features(instance)

print(features.features)  
# Example output: ('Hel', 'ell', 'llo', 'lo,', 'o, ', ', h', ' ho', 'how', 'ow ', 'w a', ' ar', 'are', 're ', 'e y', ' yo', 'you', 'ou?')

```
# **Training a Perceptron Classifier**

# Load training data
training_data = []
instances = load_lid_instances("test_data/mot_train.tsv")
for instance in instances:
    training_data.append(CharTrigramFeatureExtractor.extract_features(instance))

# Count labels
counter = InstanceCounter()
counter.count_instances(training_data)
labels = counter.labels()

# Train perceptron model
perceptron = Perceptron(labels)
perceptron.train_epochs(training_data, n_epochs=5, lr=1.0, shuffle=True)

# Save trained model
import pickle
with open("perceptron_model.pkl", "wb") as f:
    pickle.dump(perceptron, f)

# **Testing the Model on New Data**

# Load test data
test_data = []
instances = load_lid_instances("test_data/mot_test.tsv")
for instance in instances:
    test_data.append(CharTrigramFeatureExtractor.extract_features(instance))

# Predict labels
predicted_labels = [perceptron.classify(instance.features) for instance in test_data]

# Evaluate Performance
true_labels = [instance.label for instance in test_data]
scorer = MulticlassScoring(labels)
scorer.score(true_labels, predicted_labels)

print("Accuracy:", scorer.accuracy())
print("Macro F1-score:", scorer.macro_f1())
print("Weighted F1-score:", scorer.weighted_f1())

# Run hyperparameter tuning
sweep_hyperparameters()

# Example Output:
# Feature Extractor: CharTrigramFeatureExtractor, Warmup Epochs: 2, Decay: 0.8, Accuracy: 93.45%
# Best Configuration: {'feature_extractor': 'CharTrigramFeatureExtractor', 'warmup_epochs': 2, 'decay': 0.8, 'accuracy': 93.45%}

# **Using the Tuned Perceptron Model**

class TunedPerceptronHyperparameters:
    def __init__(self) -> None:
        self.feature_extractor: FeatureExtractor = CharTrigramFeatureExtractor()
        self.warmup_epochs: int = 2
        self.decay: float = 0.8
