# Classifying tweets using language with Naive Bayes

import pandas as pd

# Load in the data

new_york_tweets = pd.read_json("new_york.json", lines=True)
london_tweets = pd.read_json("london.json", lines=True)
paris_tweets = pd.read_json("paris.json", lines=True)

# How many tweets in each dataset?

print(len(new_york_tweets))
print(len(london_tweets))
print(len(paris_tweets))

# Creating a list for each of the datasets

new_york_text = new_york_tweets["text"].tolist()
london_text = london_tweets["text"].tolist()
paris_text = paris_tweets["text"].tolist()

all_tweets = new_york_text + london_text + paris_text

# Labeling each of the tweets
# 0 - New York, 1 - London, 2 - Paris

labels = [0] * len(new_york_text) + [1] * len(london_text) + [2] * len(paris_text)

# ------------------------------------------
# Create Training and Test data

from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(all_tweets, labels, test_size=0.2, random_state=1)

print(len(train_data))
print(len(test_data))


# -------------------------------------------------
# Transform tweets into count vectors
from sklearn.feature_extraction.text import CountVectorizer

counter = CountVectorizer()

# Teach the counter the vocabulary

counter.fit(train_data)

# Transform into count vectors

train_counts = counter.transform(train_data)
test_counts = counter.transform(test_data)

print(train_counts[3])
print(test_counts[3])

# ------------------------------------
# Train and test the Classifier

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()

classifier.fit(train_counts, train_labels)

# Store the predictions

predictions = classifier.predict(test_counts)


# ---------------------------------------------
# Evaluating the model
from sklearn.metrics import accuracy_score

print(accuracy_score(test_labels, predictions))

# The model correctly classified 67% of the tweets

# -------------------------------------
# Confusion matrix
# See how the classifier made it's predictions
# 3 cities -- 3 rows, row 1 -> New York, row 2 -> London, row 3 -> Paris
# 3 cities -- 3 columns, col 1 -> New York, row 2 -> London, row 3 - Paris

from sklearn.metrics import confusion_matrix

print(confusion_matrix(test_labels, predictions))

# Classifier correctly classified 541 tweets from New York, but thought that 404 London tweets and 28 Paris tweets were from New York
# Classifier correctly classified 824 tweets from London, but thought that 203 New York tweets and 34 Paris tweets were from London
# Classifier correctly classified 340 tweets from Paris, buth thought that 38 New York tweets and 103 London tweets were from Paris

# ---------------------------------------
# Testing a new tweet
# The classifier predicts tweets actually from New York as either New York or London. This makes sense since both
# cities are in English speaking countries. It's hard for the classifier to mix up tweets in different languages.

tweet = "Just got back from my trip. The Statue de la Liberté was neat!"
tweet_counts = counter.transform([tweet])
print(classifier.predict(tweet_counts))

# Classifier classifies this tweet from New York, even if a bit of French is included.

tweet2 = "La statue de la liberté est incroyable! What a fun time!"
tweet_counts2 = counter.transform([tweet2])
print(classifier.predict(tweet_counts2))

# Classifier classifies this tweet from Paris, even with a bit of English included.