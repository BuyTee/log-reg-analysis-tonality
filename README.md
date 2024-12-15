#**Sentiment Analysis with Logistic Regression**

This repository contains a Python implementation of a sentiment analysis model using logistic regression. The model classifies text data as either positive or negative sentiment based on word frequencies.



#**Features**

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



#**Requirements**

Python 3.6+

NLTK

NumPy



#**Installation**

Clone the repository:

git clone https://github.com/your-username/sentiment-analysis-logistic-regression.git

Install the required packages:

pip install -r requirements.txt

Download the NLTK data:

import nltk
nltk.download('stopwords')
nltk.download('punkt')



#**Dataset**

The code uses the Sentiment Polarity Dataset Version 1.0, included in the NLTK library. Ensure the following files are available:

sentence_polarity/rt-polarity.pos: Positive text samples.

sentence_polarity/rt-polarity.neg: Negative text samples.



#**Usage**

Run the script:

python sentiment_analysis.py



#**Example output:**

Model accuracy: 0.6832
Processed tweet: ['wow', 'product', 'amaz']
Word: wow, Positive freq: 7, Negative freq: 5
Word: product, Positive freq: 17, Negative freq: 35
Word: amaz, Positive freq: 17, Negative freq: 3
Custom tweet prediction: 0.4978 - Negative



#**Key Functions**

process_tweet: Cleans and preprocesses text data.

build_freqs: Builds the frequency dictionary.

gradientDescent: Optimizes model parameters using gradient descent.

predict_tweet: Predicts sentiment for a given text sample.

test_logistic_regression: Evaluates model performance on test data.



#**Customization**

Modify alpha (learning rate) and iterations in the script to tune the model.

Test the model with custom text samples by editing the my_tweet variable.


    
#**Limitations**

The model is based on word frequencies and does not use advanced NLP techniques like embeddings.

Stemming may lead to loss of contextual meaning.



#**Contribution**

Feel free to contribute by submitting issues or pull requests. Suggestions for improvements, such as using TF-IDF or neural networks, are welcome.



#**License**

This project is licensed under the MIT License. See the LICENSE file for details.



#**Acknowledgments**

The dataset is part of the Sentiment Polarity Dataset Version 1.0, included in NLTK.

NLTK and NumPy libraries for text preprocessing and numerical computations.

