# Spam Message Detection
NLP + Machine Learning Model

## 1. Project Overview

This project implements a **Spam Message Detection System** using Natural Language Processing (NLP) and Machine Learning.

The system takes a text message as input, processes the text, converts it into numerical features using TF-IDF, and classifies the message using a Multinomial Naive Bayes algorithm.

The model predicts whether a message belongs to one of two categories:

- **NORMAL (0):** A legitimate message.
- **SPAM (1):** An unwanted or potentially promotional message.

This project demonstrates a basic machine-learning pipeline, from raw text preprocessing to model training, evaluation, and prediction on new messages.

## 2. Project Objectives

- Understand text preprocessing techniques used in NLP.
- Learn tokenization, stopword removal, and lemmatization.
- Convert textual data into numerical representations using TF-IDF.
- Split data into training and testing sets.
- Train a text classification model using Multinomial Naive Bayes.
- Evaluate classification performance using accuracy and a classification report.
- Predict whether new messages are spam or normal.

## 3. Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NLTK | Natural Language Processing and text preprocessing |
| Scikit-learn | Feature extraction, dataset splitting, model training, and evaluation |
| TF-IDF | Converts text into numerical feature vectors |
| Multinomial Naive Bayes | Classifies messages into spam and normal categories |

### Required Libraries

- `nltk`
- `scikit-learn`

## 4. Complete Workflow

The project follows this sequence:

```text
Raw Text Messages
        |
        v
Training Data + Labels
        |
        v
Text Preprocessing (NLP)
        |
        +-- Lowercase Conversion
        |
        +-- Tokenization
        |
        +-- Stopword Removal
        |
        +-- Alphabetic Token Filtering
        |
        +-- Lemmatization
        |
        v
Processed Text Messages
        |
        v
TF-IDF Vectorization
        |
        v
Numerical Feature Matrix
        |
        v
Train-Test Split
        |
        +--------------------+
        |                    |
        v                    v
   Training Data        Testing Data
        |                    |
        v                    |
   Naive Bayes Model          |
        |                    |
        +--------------------+
                   |
                   v
            Model Evaluation
                   |
                   v
        Accuracy and Classification
                 Report
                   |
                   v
          New Text Messages
                   |
                   v
          Same Preprocessing
                   |
                   v
          Existing TF-IDF Model
                   |
                   v
         Trained Naive Bayes Model
                   |
                   v
          SPAM / NORMAL Prediction
```

## 5. Detailed Workflow Explanation

### Step 1: Import Required Libraries

The project imports the libraries required for text processing, feature extraction, model training, and evaluation.

```python
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
```

**Purpose of each library:**

- `nltk`: Provides tools for processing human language.
- `word_tokenize`: Splits a sentence into individual tokens.
- `stopwords`: Provides common words that can be removed during preprocessing.
- `WordNetLemmatizer`: Converts words into their base or dictionary forms.
- `TfidfVectorizer`: Converts text into numerical feature vectors.
- `train_test_split`: Separates data into training and testing sets.
- `MultinomialNB`: Implements the Multinomial Naive Bayes classification algorithm.
- `accuracy_score`: Calculates the proportion of correct predictions.
- `classification_report`: Displays precision, recall, F1-score, and support.

### Step 2: Download NLTK Resources

```python
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
```

NLTK requires language resources for certain operations.

- `punkt` and `punkt_tab`: Support tokenization.
- `stopwords`: Provides lists of common English words.
- `wordnet`: Supplies lexical information used by the lemmatizer.

The `quiet=True` argument reduces download messages in the terminal.

These resources generally need to be downloaded only once per Python environment.

### Step 3: Prepare the Training Dataset

The project contains 12 sample messages: six spam messages and six normal messages.

```python
messages = [
    "Win a free lottery ticket now",
    "Congratulations you won a free prize",
    "Claim your free cash reward",
    "You have won a lottery",
    "Free entry in the contest",
    "Get free money immediately",
    "Can we meet tomorrow",
    "Please send the project report",
    "The meeting is at 10 AM",
    "Please call me when you are free",
    "The class starts at 9 AM",
    "Can you send me the notes"
]
```

Each message has a corresponding numerical label.

```python
labels = [
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0
]
```

The labels represent:

| Label | Meaning |
|---|---|
| `0` | NORMAL |
| `1` | SPAM |

The message and label at the same index belong together. For example, `messages[0]` has the label `labels[0]`, which is `1` (SPAM).

**Important:** This small dataset is intended for learning. It is not large or diverse enough for reliable real-world spam detection.

### Step 4: NLP 1 — Text Preprocessing

Text preprocessing converts raw messages into a more consistent representation before feature extraction.

The project defines a reusable function:

```python
def preprocess_text(text):
```

This function performs five operations.

#### 4.1 Lowercase Conversion

```python
text = text.lower()
```

Converts all characters to lowercase.

Example:

```text
"FREE Money NOW"
        |
        v
"free money now"
```

This ensures that words such as `Free` and `free` are treated consistently.

#### 4.2 Tokenization

```python
tokens = word_tokenize(text)
```

Tokenization splits the sentence into individual tokens.

Example:

```text
Input:
"Win a free lottery ticket now"

Tokens:
["Win", "a", "free", "lottery", "ticket", "now"]
```

Tokens are the individual units that will be filtered and processed.

#### 4.3 Stopword Removal and Alphabetic Filtering

```python
filtered_words = []

for word in tokens:

    if word.isalpha() and word not in stop_words:
        filtered_words.append(word)
```

The code keeps tokens that:

1. Contain only alphabetic characters.
2. Are not present in the English stopword list.

For example, common words such as `a`, `the`, and `is` may be removed.

Alphabetic filtering also removes tokens containing digits or punctuation.

This step can reduce noise, but stopword removal is not always beneficial for spam detection. Words such as `you` or `your` may provide useful classification signals.

#### 4.4 Lemmatization

```python
lemmatized_words = []

for word in filtered_words:

    lemmatized_word = lemmatizer.lemmatize(word)

    lemmatized_words.append(lemmatized_word)
```

Lemmatization attempts to convert words into their dictionary forms.

Examples:

```text
tickets  → ticket
rewards  → reward
```

The default WordNet lemmatizer uses noun lemmatization unless a different part of speech is specified. Therefore, it does not necessarily convert every verb into its base form.

#### 4.5 Join the Processed Tokens

```python
return " ".join(lemmatized_words)
```

The processed tokens are joined back into a single string.

Example:

```text
Original:
"Win a free lottery ticket now"

Processed:
"win free lottery ticket"
```

The same preprocessing function is applied to both training messages and new messages.

### Step 5: Apply Preprocessing to All Training Messages

```python
processed_messages = []

for message in messages:

    processed = preprocess_text(message)

    processed_messages.append(processed)
```

The loop processes every message and stores the result in `processed_messages`.

The original messages are retained so that the results can be compared with the processed text.

This produces a collection of cleaned messages ready for feature extraction.

### Step 6: NLP 2 — TF-IDF Vectorization

Machine-learning algorithms such as Multinomial Naive Bayes require numerical input rather than raw text.

The project uses `TfidfVectorizer` to transform the processed messages into a numerical matrix.

```python
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(processed_messages)
```

#### What is TF-IDF?

TF-IDF stands for **Term Frequency–Inverse Document Frequency**.

It gives a numerical weight to terms based on their frequency in a document and how common or uncommon they are across the document collection.

**Term Frequency (TF)** measures how frequently a term occurs in a document.

**Inverse Document Frequency (IDF)** reduces the relative importance of terms that occur in many documents.

A common form of the IDF calculation used by scikit-learn is:

\[
\operatorname{IDF}(t)=
\log\left(\frac{1+N}{1+\operatorname{df}(t)}\right)+1
\]

Where:

- \(N\) is the number of documents.
- \(\operatorname{df}(t)\) is the number of documents containing term \(t\).

The TF-IDF weight combines term frequency with inverse document frequency.

#### Understanding `fit_transform()`

```python
X = vectorizer.fit_transform(processed_messages)
```

This performs two operations:

- `fit()`: Learns the vocabulary and IDF weights from the supplied messages.
- `transform()`: Converts the messages into numerical feature vectors using the learned vocabulary and weights.

For example:

```text
Processed messages
        |
        v
Vocabulary:
["cash", "class", "free", "lottery", "meeting", ...]
        |
        v
TF-IDF feature vectors
        |
        v
Numerical matrix X
```

Each row represents one message, and each column represents a vocabulary term.

The expression `X.shape` displays the number of messages and the number of features.

```python
print(X.shape)
```

Since the project contains 12 messages, the matrix has 12 rows. The number of columns depends on the vocabulary learned from the processed text.

The matrix is sparse, meaning most terms are absent from most messages. Calling `X.toarray()` displays the complete matrix but can consume substantial memory for large datasets.

### Step 7: Split the Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.25,
    random_state=63,
    stratify=labels
)
```

The dataset is divided into training and testing portions.

- `X`: Numerical message features.
- `labels`: Correct categories for each message.
- `test_size=0.25`: Reserves 25% of the data for testing.
- `random_state=63`: Makes the split reproducible.
- `stratify=labels`: Attempts to preserve the class proportions in each subset.

With 12 messages, the split produces nine training messages and three testing messages.

Because the dataset is small, each test example has a large effect on the reported accuracy.

**A methodological improvement:** The TF-IDF vectorizer is currently fitted before the split. For a more rigorous evaluation, split the raw messages first and fit TF-IDF only on the training messages. A scikit-learn `Pipeline` is a good way to prevent information leakage from the test set.

### Step 8: Train the Naive Bayes Model

```python
model = MultinomialNB()

model.fit(X_train, y_train)
```

The project uses the Multinomial Naive Bayes algorithm.

Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem.

\[
P(C\mid X)=\frac{P(X\mid C)P(C)}{P(X)}
\]

Where:

- \(C\) represents a class, such as SPAM or NORMAL.
- \(X\) represents the observed features of a message.
- \(P(C)\) is the prior probability of the class.
- \(P(X\mid C)\) represents the likelihood of the observed features given the class.
- \(P(C\mid X)\) is the posterior probability of the class given the features.

The algorithm makes a simplifying conditional-independence assumption about the features given the class.

During training, `fit()` learns the class probabilities and feature statistics from the training data.

When a new message is supplied, the model uses the learned information to predict its class.

Multinomial Naive Bayes can work well with text features, including TF-IDF, although its probability estimates and performance depend on the dataset and feature representation.

### Step 9: Evaluate the Model

```python
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
```

The trained model predicts the labels of the held-out test messages.

- `y_test`: The actual labels.
- `y_pred`: The predicted labels.
- `accuracy_score()`: Measures the proportion of correct predictions.

The accuracy formula is:

\[
\text{Accuracy}=
\frac{\text{Correct Predictions}}
{\text{Total Predictions}}
\]

For example, if the model correctly predicts two of three test messages, its accuracy is approximately 66.67%.

This is only an illustration, not a guaranteed result from this code.

#### Classification Report

```python
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["NORMAL", "SPAM"],
        zero_division=0
    )
)
```

The report displays four important measurements:

| Metric | Meaning |
|---|---|
| Precision | Of the messages predicted as a class, how many actually belong to that class? |
| Recall | Of all actual messages in a class, how many did the model identify? |
| F1-score | Harmonic mean of precision and recall |
| Support | Number of actual test examples belonging to each class |

The `zero_division=0` setting prevents division-by-zero warnings when a class has no predicted examples.

Because only three messages are used for testing, the evaluation report should not be treated as a reliable estimate of real-world performance.

### Step 10: Prepare New Messages

The project defines a separate collection of messages that the model has not explicitly seen as training examples.

```python
new_messages = [
    "You won a free cash prize",
    "Please attend the meeting tomorrow",
    "Claim your free money now",
    "Please send the project report",
    "We have class tomorrow",
    "Click here to claim your reward",
    "Click the form"
]
```

The goal is to classify these messages as SPAM or NORMAL.

### Step 11: Preprocess New Messages

```python
new_processed_messages = []

for message in new_messages:

    processed = preprocess_text(message)

    new_processed_messages.append(processed)
```

Each new message passes through the same preprocessing steps used for the training messages.

This ensures consistent text preparation before prediction.

### Step 12: Transform New Messages Using the Existing TF-IDF Vectorizer

```python
new_X = vectorizer.transform(new_processed_messages)
```

The existing vectorizer transforms the new messages into numerical features.

It is important to use `transform()` rather than `fit_transform()`.

The vectorizer must retain the vocabulary and IDF weights learned during training so that the new feature vectors have the same columns as the training feature vectors.

Words not present in the learned vocabulary do not introduce new columns.

### Step 13: Predict New Message Categories

```python
predictions = model.predict(new_X)
```

The trained Naive Bayes model receives the numerical vectors and predicts a class for each message.

The output labels are:

- `0`: NORMAL
- `1`: SPAM

The code converts each numerical prediction into a readable category.

```python
if prediction == 1:
    result = "SPAM"
else:
    result = "NORMAL"
```

Finally, the program prints the original message, its processed form, and its predicted category.

These predictions are model outputs, not guaranteed determinations that a message is genuinely spam or legitimate.

## 6. How the Components Work Together

The project combines two major areas.

### Natural Language Processing

NLP prepares text for machine learning.

```text
Raw Text
   |
   v
Lowercase
   |
   v
Tokenization
   |
   v
Stopword Removal
   |
   v
Lemmatization
   |
   v
Processed Text
   |
   v
TF-IDF
   |
   v
Numerical Features
```

### Machine Learning

Machine learning learns a relationship between message features and their labels.

```text
Numerical Features + Labels
             |
             v
        Training Data
             |
             v
     Multinomial Naive Bayes
             |
             v
       Trained Model
             |
             v
      New Message Features
             |
             v
      Predicted Category
       /            \
      v              v
    NORMAL          SPAM
```

NLP transforms the text into a usable representation, while the classifier learns from labeled examples to make predictions.

## 7. How to Run the Project

### Prerequisites

Install Python and ensure that pip is available.

### Step 1: Clone the Repository

Replace the placeholder with your GitHub repository URL.

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### Step 2: Install Dependencies

```bash
python -m pip install nltk scikit-learn
```

### Step 3: Run the Script

If your Python file is named `spam_detection.py`, execute:

```bash
python spam_detection.py
```

On the first run, NLTK may download the required language resources. An internet connection is needed if those resources are not already available locally.

### Step 4: Review the Output

The program displays:

1. Original and processed messages.
2. TF-IDF vocabulary and matrix information.
3. Training and testing sample counts.
4. Model predictions on the test set.
5. Accuracy and classification metrics.
6. Predictions for the new messages.

## 8. Current Limitations

This is an educational implementation, not a production-ready spam filtering system.

- **Small dataset:** Only 12 labeled messages are used.
- **Limited vocabulary:** The model cannot learn general spam patterns from such a small collection.
- **Unreliable evaluation:** Three test messages are insufficient for a meaningful performance estimate.
- **Preprocessing trade-offs:** Removing stopwords can discard useful classification signals.
- **Feature leakage:** The current code learns TF-IDF vocabulary and IDF weights before splitting the data. Fit the vectorizer only on the training set for rigorous evaluation.
- **No saved model:** The script does not currently persist the trained model for reuse in another program.

## 9. Future Improvements

Potential enhancements include:

- Use a larger, publicly available SMS spam dataset.
- Separate training, validation, and test data appropriately.
- Build a scikit-learn `Pipeline` for preprocessing, TF-IDF, and classification.
- Compare Multinomial Naive Bayes with Logistic Regression and Linear SVM.
- Evaluate precision, recall, F1-score, and a confusion matrix.
- Save the trained pipeline using `joblib` for later predictions.
- Create a simple web interface using Streamlit.
- Experiment with different preprocessing and TF-IDF settings.

## 10. Key Learning Outcomes

By completing this project, I practiced:

- Cleaning and preprocessing textual data.
- Tokenization, stopword removal, and lemmatization.
- Converting text into numerical features with TF-IDF.
- Splitting datasets for training and testing.
- Training a Naive Bayes classifier.
- Evaluating a classification model.
- Applying the same feature extraction process to unseen messages.
- Building an end-to-end text classification workflow.

## 11. Conclusion

This project demonstrates a foundational NLP and machine-learning workflow for classifying text messages.

The preprocessing stage prepares the messages, TF-IDF represents the text numerically, and Multinomial Naive Bayes learns to distinguish between spam and normal examples.

Although the current dataset is too small for reliable real-world use, the project provides a starting point for understanding how traditional machine-learning systems perform text classification.

**Technologies:** Python · NLTK · Scikit-learn · TF-IDF · Multinomial Naive Bayes

