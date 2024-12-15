import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

# Step 1: Load the dataset
def load_data(pos_path, neg_path):
    with open(pos_path, 'r', encoding='latin-1') as f:
        positive_tweets = f.readlines()
    with open(neg_path, 'r', encoding='latin-1') as f:
        negative_tweets = f.readlines()
    return positive_tweets, negative_tweets

# Directory for positive and negative sentence_polarity
pos_path = "sentence_polarity/rt-polarity.pos"
neg_path = "sentence_polarity/rt-polarity.neg"

all_positive_tweets, all_negative_tweets = load_data(pos_path, neg_path)

# Split into training and testing sets
train_pos = all_positive_tweets[:int(0.8 * len(all_positive_tweets))]
test_pos = all_positive_tweets[int(0.8 * len(all_positive_tweets)) :]
train_neg = all_negative_tweets[:int(0.8 * len(all_negative_tweets))]
test_neg = all_negative_tweets[int(0.8 * len(all_negative_tweets)) :]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

# Step 2: Preprocess the data
def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    tweet = re.sub(r'#', '', tweet)

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if word not in stopwords_english and word not in string.punctuation:
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)

    return tweets_clean

# Step 3: Build frequency dictionary
def build_freqs(tweets, ys):
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1
    return freqs

freqs = build_freqs(train_x, train_y)

# Step 4: Implement sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Step 5: Gradient descent for logistic regression
def gradientDescent(x, y, theta, alpha, num_iters):
    m = x.shape[0]
    for _ in range(num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        theta -= (alpha / m) * np.dot(x.T, (h - y))
    return theta

# Step 6: Extract features
def extract_features(tweet, freqs):
    word_l = process_tweet(tweet)
    x = np.zeros((1, 3))
    x[0, 0] = 1  # Bias term
    for word in word_l:
        if (word, 1.0) in freqs:
            x[0, 1] += freqs[(word, 1.0)]
        if (word, 0.0) in freqs:
            x[0, 2] += freqs[(word, 0.0)]
    return x

# Prepare training data
X_train = np.zeros((len(train_x), 3))
for i, tweet in enumerate(train_x):
    X_train[i, :] = extract_features(tweet, freqs)

Y_train = train_y.reshape(-1, 1)

# Train the model
alpha = 1e-7  # Adjusted learning rate
iterations = 10000  # Increased iterations
initial_theta = np.zeros((3, 1))
theta = gradientDescent(X_train, Y_train, initial_theta, alpha, iterations)

# Step 7: Evaluate the model
def predict_tweet(tweet, freqs, theta):
    x = extract_features(tweet, freqs)
    y_pred = sigmoid(np.dot(x, theta))
    return y_pred

def test_logistic_regression(test_x, test_y, freqs, theta):
    y_hat = [1 if predict_tweet(tweet, freqs, theta) > 0.5 else 0 for tweet in test_x]
    accuracy = np.mean(np.array(y_hat) == test_y)
    return accuracy

accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
print(f"Model accuracy: {accuracy:.4f}")

# Step 8: Test with custom examples
my_tweet = "This is very bad. But this is way better, very good!"
print(f"Processed tweet: {process_tweet(my_tweet)}")

for word in process_tweet(my_tweet):
    print(f"Word: {word}, Positive freq: {freqs.get((word, 1.0), 0)}, Negative freq: {freqs.get((word, 0.0), 0)}")

y_hat = predict_tweet(my_tweet, freqs, theta).item()  # Преобразуем в скаляр
print(f"Custom tweet prediction: {y_hat:.4f} - {'Positive' if y_hat > 0.5 else 'Negative'}")
