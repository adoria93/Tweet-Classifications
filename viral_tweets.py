# Classifying Viral Tweets

import pandas as pd

# Read in the data

all_tweets = pd.read_json("random_tweets.json", lines=True)

print(len(all_tweets))
print(all_tweets.columns)
print(all_tweets.loc[0]['text'])


# Defining Viral Tweets

import numpy as np

# How do we define a viral tweet? Should we use 100 retweets as a start? Maybe if a tweet has more than the average number
# of retweets?

median_retweets = all_tweets['retweet_count'].median()
print(median_retweets)
all_tweets['is_viral'] = np.where(all_tweets['retweet_count'] >= median_retweets, 1, 0)
print(all_tweets['is_viral'].value_counts())

# The median seems like a good baseline to use, since it's resistant to larger values skewing the results.
# In this case, a viral tweet is defined as equal to or more than 13 retweets, which really doens't fit the
# viral term at all.


# Features
# Focus on what I think determines what makes tweets go viral

all_tweets['tweet_length'] = all_tweets.apply(lambda tweet: len(tweet['text']), axis=1)
all_tweets['followers_count'] = all_tweets.apply(lambda tweet: tweet['user']['followers_count'], axis=1)
all_tweets['friends_count'] = all_tweets.apply(lambda tweet: tweet['user']['friends_count'], axis=1)
all_tweets['favorites_count'] = all_tweets.apply(lambda tweet: tweet['user']['favourites_count'], axis=1)
all_tweets['num_words'] = all_tweets.apply(lambda tweet: len(tweet['text'].split()), axis=1)


# Normalizing The Data

from sklearn.preprocessing import scale

labels = all_tweets['is_viral']

data = all_tweets[["tweet_length", "followers_count", "friends_count", 'favorites_count', 'num_words']]

# Need to scale the data so that all the features with vary within the same range

scaled_data = scale(data, axis=0)
print(scaled_data[0])


# Creating the Training Set and Test Set

from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(scaled_data, labels, test_size=0.2, random_state=1)


# Using the Classifier

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(train_data, train_labels)

print(classifier.score(test_data, test_labels))


# Choosing K
# Going to loop through many possible values for K in order to determine the best value to use

import matplotlib.pyplot as plt

scores = []

for k in range(1,200):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(train_data, train_labels)
    scores.append(classifier.score(test_data, test_labels))

plt.plot(range(1,200), scores)
plt.show()

# The classifier was able to get a best score of 68%.
