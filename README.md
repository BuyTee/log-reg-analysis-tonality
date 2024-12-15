# **Sentiment Analysis with Logistic Regression**

This repository contains a Python implementation of a sentiment analysis model using logistic regression. The model classifies text data as either positive or negative sentiment using TF-IDF features and a PyTorch-based logistic regression model.

---

# **Features**

<ins> **Data Loading:** </ins>  
- Reads positive and negative text samples from the `sentence_polarity` dataset.

<ins> **Text Preprocessing:** </ins>  
- Tokenization, removal of stop words, and cleaning text to prepare data for analysis.

<ins> **TF-IDF Vectorization:** </ins>  
- Converts text data into numerical features using unigrams and bigrams with a maximum of 5000 features.

<ins> **Logistic Regression with PyTorch:** </ins>  
- Implements a logistic regression model using PyTorch with mini-batch training and regularization.

<ins> **Model Training and Evaluation:** </ins>  
- Trains the model on a balanced dataset and evaluates accuracy on a separate test set.

<ins> **Custom Predictions:** </ins>  
- Allows testing the model on custom examples with word-level contribution analysis.

---

# **Requirements**

- Python 3.8+
- PyTorch
- Scikit-learn
- NLTK

---




# **Installation**

1. Clone the repository:

git clone https://github.com/BuyTee/log-reg-analysis-tonality.git

2. Install the required packages:

pip install -r requirements.txt

3. Download the dataset from https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/movie_reviews.zip

4. Download the NLTK data:

import nltk
nltk.download('stopwords')
nltk.download('punkt')



# **Dataset**

The code uses the Sentiment Polarity Dataset Version 2.0, which contains positive and negative samples of movie reviews.
Ensure the following files are available:

sentence_polarity/pos: Positive text samples.

sentence_polarity/neg: Negative text samples.




# **Usage**

1. Place the dataset in the required directory structure (sentence_polarity/pos and sentence_polarity/neg).

2. Run the script:

**python sentiment_analysis.py**



# **Example output:**

> Using device: cuda

> Positive samples: 1000, Negative samples: 1000

> Epoch [100/2000], Loss: 0.0017
Epoch [200/2000], Loss: 0.0015
Epoch [300/2000], Loss: 0.0014
Epoch [400/2000], Loss: 0.0014
Epoch [500/2000], Loss: 0.0016
Epoch [600/2000], Loss: 0.0015
Epoch [700/2000], Loss: 0.0019
Epoch [800/2000], Loss: 0.0018
Epoch [900/2000], Loss: 0.0014
Epoch [1000/2000], Loss: 0.0016
Epoch [1100/2000], Loss: 0.0014
Epoch [1200/2000], Loss: 0.0015
Epoch [1300/2000], Loss: 0.0013
Epoch [1400/2000], Loss: 0.0017
Epoch [1500/2000], Loss: 0.0014
Epoch [1600/2000], Loss: 0.0015
Epoch [1700/2000], Loss: 0.0016
Epoch [1800/2000], Loss: 0.0015
Epoch [1900/2000], Loss: 0.0012
Epoch [2000/2000], Loss: 0.0014

> Model accuracy: 0.8525
> Custom tweet prediction: 0.9372 - Positive




# **Key Functions**

process_tweet: Cleans and preprocesses text data.

build_freqs: Builds the frequency dictionary.

gradientDescent: Optimizes model parameters using gradient descent.

predict_tweet: Predicts sentiment for a given text sample.

test_logistic_regression: Evaluates model performance on test data.

