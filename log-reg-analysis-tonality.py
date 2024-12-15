import os
import re
import string
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load the dataset from directories
def load_data_from_folders(pos_folder, neg_folder):
    positive_tweets = []
    negative_tweets = []
    for file_name in os.listdir(pos_folder):
        with open(os.path.join(pos_folder, file_name), 'r', encoding='utf-8') as f:
            positive_tweets.append(f.read())
    for file_name in os.listdir(neg_folder):
        with open(os.path.join(neg_folder, file_name), 'r', encoding='utf-8') as f:
            negative_tweets.append(f.read())
    return positive_tweets, negative_tweets

pos_folder = r"sentence_polarity/pos"
neg_folder = r"sentence_polarity/neg"
all_positive_tweets, all_negative_tweets = load_data_from_folders(pos_folder, neg_folder)
print(f"Positive samples: {len(all_positive_tweets)}, Negative samples: {len(all_negative_tweets)}")

# Balancing the dataset
min_samples = min(len(all_positive_tweets), len(all_negative_tweets))
all_positive_tweets = all_positive_tweets[:min_samples]
all_negative_tweets = all_negative_tweets[:min_samples]

# Step 2: Split into training and testing sets
train_pos = all_positive_tweets[:int(0.8 * min_samples)]
test_pos = all_positive_tweets[int(0.8 * min_samples):]
train_neg = all_negative_tweets[:int(0.8 * min_samples)]
test_neg = all_negative_tweets[int(0.8 * min_samples):]

train_x = train_pos + train_neg
test_x = test_pos + test_neg
train_y = torch.cat([torch.ones(len(train_pos)), torch.zeros(len(train_neg))]).to(device)
test_y = torch.cat([torch.ones(len(test_pos)), torch.zeros(len(test_neg))]).to(device)

# Step 3: Preprocess the data
def process_tweet(tweet):
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tokens = tokenizer.tokenize(tweet)
    return ' '.join([word for word in tokens if word not in stopwords_english and word not in string.punctuation])

# Step 4: TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
train_x_tfidf = torch.tensor(vectorizer.fit_transform([process_tweet(tweet) for tweet in train_x]).toarray(), device=device, dtype=torch.float32)
test_x_tfidf = torch.tensor(vectorizer.transform([process_tweet(tweet) for tweet in test_x]).toarray(), device=device, dtype=torch.float32)

# Normalize data
mean_train = torch.mean(train_x_tfidf, dim=0)
std_train = torch.std(train_x_tfidf, dim=0)
train_x_tfidf = (train_x_tfidf - mean_train) / std_train
test_x_tfidf = (test_x_tfidf - mean_train) / std_train

# Step 5: Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

input_dim = train_x_tfidf.shape[1]
model = LogisticRegressionModel(input_dim).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

# Train with mini-batches and gradient clipping
batch_size = 128
train_dataset = TensorDataset(train_x_tfidf, train_y.view(-1, 1))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

num_epochs = 2000
for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    scheduler.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Step 6: Evaluate the model
def predict_tweet(tweet):
    processed = process_tweet(tweet)
    x = torch.tensor(vectorizer.transform([processed]).toarray(), device=device, dtype=torch.float32)
    x = (x - mean_train) / std_train
    with torch.no_grad():
        logits = model(x)
        return torch.sigmoid(logits).item()

accuracy = torch.mean((torch.round(torch.sigmoid(model(test_x_tfidf))) == test_y.view(-1, 1)).float())
print(f"Model accuracy: {accuracy:.4f}")

# Test with custom example
my_tweet = "This is good"
y_hat = predict_tweet(my_tweet)
print(f"Custom tweet prediction: {y_hat:.4f} - {'Positive' if y_hat > 0.5 else 'Negative'}")
