# ============================================================
# NLP + MACHINE LEARNING
# PROJECT: SPAM MESSAGE DETECTION
# ============================================================
#
# FLOW:
# 1. Training Data
# 2. Text Preprocessing (NLP)
# 3. TF-IDF Vectorization
# 4. Train-Test Split
# 5. Naive Bayes Model Training
# 6. Model Evaluation
# 7. Predict New Messages
#
# LABELS:
# 0 = NORMAL
# 1 = SPAM
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# ------------------------------------------------------------
# 2. DOWNLOAD REQUIRED NLTK DATA
# ------------------------------------------------------------

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


# ------------------------------------------------------------
# 3. TRAINING DATA
# ------------------------------------------------------------

messages = [
    # SPAM MESSAGES
    "Win a free lottery ticket now",
    "Congratulations you won a free prize",
    "Claim your free cash reward",
    "You have won a lottery",
    "Free entry in the contest",
    "Get free money immediately",

    # NORMAL MESSAGES
    "Can we meet tomorrow",
    "Please send the project report",
    "The meeting is at 10 AM",
    "Please call me when you are free",
    "The class starts at 9 AM",
    "Can you send me the notes"
]

# 0 = NORMAL, 1 = SPAM

labels = [
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0
]


# ------------------------------------------------------------
# 4. NLP 1 - TEXT PREPROCESSING
# ------------------------------------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Tokenize the sentence
    tokens = word_tokenize(text)

    # Remove stopwords and non-alphabetic tokens
    filtered_words = []

    for word in tokens:

        if word.isalpha() and word not in stop_words:
            filtered_words.append(word)

    # Convert words to their base form
    lemmatized_words = []

    for word in filtered_words:

        lemmatized_word = lemmatizer.lemmatize(word)

        lemmatized_words.append(lemmatized_word)

    # Join words into a single string
    return " ".join(lemmatized_words)


# Preprocess all training messages

processed_messages = []

for message in messages:

    processed = preprocess_text(message)

    processed_messages.append(processed)


# Display preprocessing results

print("\n" + "=" * 60)
print("NLP 1: TEXT PREPROCESSING")
print("=" * 60)

for original, processed in zip(messages, processed_messages):

    print(f"\nOriginal  : {original}")
    print(f"Processed : {processed}")


# ------------------------------------------------------------
# 5. NLP 2 - TF-IDF VECTORIZATION
# ------------------------------------------------------------

vectorizer = TfidfVectorizer()

# Learn vocabulary and convert messages into numerical vectors
X = vectorizer.fit_transform(processed_messages)

print("\n" + "=" * 60)
print("NLP 2: TF-IDF VECTORIZATION")
print("=" * 60)

print("\nVocabulary:")
print(vectorizer.get_feature_names_out())

print("\nTF-IDF Matrix Shape:")
print(X.shape)

print("\nTF-IDF Matrix:")
print(X.toarray())


# ------------------------------------------------------------
# 6. SPLIT DATA INTO TRAINING AND TESTING SETS
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.25,
    random_state=63,
    stratify=labels
)

print("\n" + "=" * 60)
print("DATASET SPLIT")
print("=" * 60)

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")


# ------------------------------------------------------------
# 7. TRAIN THE MACHINE LEARNING MODEL
# ------------------------------------------------------------

model = MultinomialNB()

model.fit(X_train, y_train)

print("\nNaive Bayes model training completed.")


# ------------------------------------------------------------
# 8. EVALUATE THE MODEL
# ------------------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print("\nActual Labels:")
print(y_test)

print("\nPredicted Labels:")
print(y_pred)

print(f"\nAccuracy: {accuracy:.2%}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["NORMAL", "SPAM"],
        zero_division=0
    )
)


# ------------------------------------------------------------
# 9. PREDICT NEW MESSAGES
# ------------------------------------------------------------

new_messages = [
    "You won a free cash prize",
    "Please attend the meeting tomorrow",
    "Claim your free money now",
    "Please send the project report",
    "We have class tomorrow",
    "Click here to claim your reward",
    "Click the form"
]


# NLP 1 - Preprocess new messages

new_processed_messages = []

for message in new_messages:

    processed = preprocess_text(message)

    new_processed_messages.append(processed)


# NLP 2 - Transform messages using the SAME TF-IDF vectorizer

new_X = vectorizer.transform(new_processed_messages)


# Machine Learning - Predict message labels

predictions = model.predict(new_X)


# ------------------------------------------------------------
# 10. DISPLAY FINAL PREDICTIONS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL SPAM DETECTION RESULTS")
print("=" * 60)

for original, processed, prediction in zip(
    new_messages,
    new_processed_messages,
    predictions
):

    if prediction == 1:
        result = "SPAM"
    else:
        result = "NORMAL"

    print(f"\nOriginal Message   : {original}")
    print(f"Processed Message  : {processed}")
    print(f"Prediction         : {result}")
    print("-" * 60)


# ============================================================
# END OF PROJECT
# ============================================================
