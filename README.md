# **Sentiment Analysis with Logistic Regression**

This repository contains a Python implementation of a sentiment analysis model using logistic regression. The model classifies text data as either positive or negative sentiment based on word frequencies.



# **Features**

Data Loading:

Reads positive and negative text samples from the sentence_polarity dataset.

Text Preprocessing:

Tokenization, removal of stop words, and stemming of words.

Frequency Dictionary:

Builds a frequency dictionary mapping word occurrences to positive and negative sentiment classes.

Logistic Regression:

Implements logistic regression with gradient descent for classification.

Model Training and Evaluation:

Trains the model on the training set and evaluates it on a separate test set.

Custom Predictions:

Allows testing the model on custom examples.



# **Requirements**

Python 3.6+

NLTK

NumPy



# **Installation**

Clone the repository:

git clone https://github.com/BuyTee/log-reg-analysis-tonality.git

Install the required packages:

pip install -r requirements.txt

Download the NLTK data:

import nltk
nltk.download('stopwords')
nltk.download('punkt')



# **Dataset**

The code uses the Sentiment Polarity Dataset Version 1.0, included in the NLTK library. Ensure the following files are available:

sentence_polarity/rt-polarity.pos: Positive text samples.

sentence_polarity/rt-polarity.neg: Negative text samples.



# **Usage**

Run the script:

python sentiment_analysis.py



# **Example output:**

Model accuracy: 0.6832
Processed tweet: ['bad', 'way', 'better', 'good']
Word: bad, Positive freq: 23, Negative freq: 176
Word: way, Positive freq: 153, Negative freq: 113
Word: better, Positive freq: 51, Negative freq: 81
Word: good, Positive freq: 157, Negative freq: 148
Custom tweet prediction: 0.3277 - Negative



# **Key Functions**

process_tweet: Cleans and preprocesses text data.

build_freqs: Builds the frequency dictionary.

gradientDescent: Optimizes model parameters using gradient descent.

predict_tweet: Predicts sentiment for a given text sample.

test_logistic_regression: Evaluates model performance on test data.

