# Naive Bayes Classifier for Sentence Segmentation & Sentiment Analysis

## Overview
This project implements a **Naive Bayes classifier** for two NLP tasks:  
1. **Sentence Segmentation**: Classifying sentence boundaries in text.  
2. **Sentiment Analysis**: Predicting sentiment (positive or negative) from airline-related tweets.  

The classifier is trained using **feature extraction** techniques and evaluated using **precision, recall, F1-score, and accuracy**. Additionally, **grid search tuning** is applied to optimize model performance.

---

## Key Concepts
- **Naive Bayes Classification**: Probabilistic classification based on Bayes’ Theorem.
- **Sentence Segmentation**: Detecting sentence boundaries based on punctuation context.
- **Sentiment Analysis**: Identifying whether a tweet expresses positive or negative sentiment.
- **Feature Extraction**:
  - **Unigram & Bigram Features**: Using individual words and word pairs to improve classification.
  - **Sentence Context Features**: Extracting left and right token contexts for segmentation.
- **Evaluation Metrics**: Computing **accuracy, precision, recall, and F1-score**.
- **Smoothing Techniques**:
  - **Laplace (Add-k) Smoothing**: Handling zero probability issues in Naive Bayes.
- **Hyperparameter Tuning**:
  - **Grid Search Optimization**: Selecting the best `k` value and feature type (unigram vs. bigram) for performance improvement.

---

## Implemented Functions & Classes

### **1. Evaluation Metrics**
- `accuracy(predictions, expected)`: Computes accuracy of predictions.
- `precision(predictions, expected, positive_label)`: Computes precision for a given class.
- `recall(predictions, expected, positive_label)`: Computes recall for a given class.
- `f1(predictions, expected, positive_label)`: Computes F1-score.

### **2. Feature Extraction**
- `UnigramAirlineSentimentFeatureExtractor`: Extracts unique words (BoW features) from tweets.
- `BigramAirlineSentimentFeatureExtractor`: Extracts bigram features from tweets.
- `BaselineSegmentationFeatureExtractor`: Extracts left-token, punctuation-token, and right-token features for sentence segmentation.

### **3. Counting Instances**
- `InstanceCounter`: Tracks counts of labels, features, and feature-label pairs to support Naive Bayes classification.

### **4. Naive Bayes Classifier**
- `prior_prob(label)`: Computes prior probability of a label.
- `feature_prob(feature, label)`: Computes smoothed conditional probability of a feature given a label.
- `log_posterior_prob(features, label)`: Computes log posterior probability for classification.
- `classify(features)`: Predicts the most likely label for a given feature set.
- `test(instances)`: Evaluates classifier performance by returning predicted vs. actual labels.

### **5. Hyperparameter Tuning**
- `TunedAirlineSentimentFeatureExtractor`: Implements grid search to optimize `k` and feature selection for best sentiment classification accuracy.

---

## Example Usage

### **Evaluating a Classifier**
```python
# Example: Using Unigram Sentiment Features
inst = AirlineSentimentInstance(
    "negative",
    "US Airways",
    [
        ['@USAirways', 'Trying', 'to', 'change', 'my', 'flight', 'due', 'to', 'NYC', 'travel', 'advisory', '...'],
        ["Been", "on", "hold", "for", "over", "1", "hr"],
    ],
)

uni_extractor = UnigramAirlineSentimentFeatureExtractor()
features = uni_extractor.extract_features(inst)

nb_classifier = NaiveBayesClassifier(k=0.005)
nb_classifier.train([features])  # Train on extracted features

prediction = nb_classifier.classify(features.features)
print(prediction)  # Expected output: 'negative'

inst = SentenceSplitInstance("y", "22", ".", "The")
seg_extractor = BaselineSegmentationFeatureExtractor()
features = seg_extractor.extract_features(inst)
print(features.features)  # ('left_tok=22', 'split_tok=.', 'right_tok=The')

predicted = ["positive", "negative", "positive", "negative"]
expected = ["positive", "negative", "negative", "negative"]

print(accuracy(predicted, expected))  # Accuracy score
print(precision(predicted, expected, "positive"))  # Precision for "positive" sentiment
print(recall(predicted, expected, "positive"))  # Recall for "positive" sentiment
print(f1(predicted, expected, "positive"))  # F1-score for "positive" sentiment
