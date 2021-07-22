#!/usr/bin/env python
# coding: utf-8

# # Importing Modules

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
import warnings
import string
import nltk
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# # Loading the Dataset

# In[2]:


df = pd.read_csv("train_tweets.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# # Data Preprocessing

# In[6]:


# removing pattern in the tweets
def remove_pattern(input_text, pattern):
    rmv  = re.findall(pattern, input_text)
    for word in rmv:
        input_text = re.sub(word, "", input_text)
    return input_text


# In[7]:


# removing twitter handles from tweets (@username)
df['tweet'] = np.vectorize(remove_pattern)(df['tweet'], '@[\w]*')


# In[8]:


df.head()


# In[9]:


# Removing URLs
def cleaning_URLs(text):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',text)
df['tweet'] = df['tweet'].apply(lambda x: cleaning_URLs(x))


# In[10]:


# removing special characters, numbers & punctuations
df['tweet'] = df['tweet'].str.replace("[^a-zA-Z#]", " ")


# In[11]:


# removing repeating char
def remove_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)
df['tweet'] = df['tweet'].apply(lambda x: remove_repeating_char(x))


# In[12]:


df.tail()


# In[13]:


# Cleaning Stopwords and Shortwords from the tweets
from nltk.corpus import stopwords
stop = stopwords.words('english')

def cleaning_stopwords(text):
    return " ".join([word for word in text.split() if word not in stop])

def cleaning_shortwords(text):
    return " ".join([word for word in text.split() if len(word)>2])


# In[14]:


df['tweet'] = df['tweet'].apply(lambda tweets: cleaning_stopwords(tweets))

df['tweet'] = df['tweet'].apply(lambda tweets: cleaning_shortwords(tweets))
df.head()


# In[15]:


# individual words are considered as tokens
tokenized_tweet = df['tweet'].apply(lambda x: x.split())
tokenized_tweet


# In[16]:


# Implementing Stemming 
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda tweet: [stemmer.stem(word) for word in tweet])
tokenized_tweet.head()


# In[17]:


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = " ".join(tokenized_tweet[i])
    
df['tweet'] = tokenized_tweet
df.head()


# # Exploratory Data Analysis

# In[18]:


all_words = " ".join([sentence for sentence in df['tweet']]) 

from wordcloud import WordCloud
wordcloud = WordCloud(height= 500, width=1000, random_state=42, max_font_size=1000).generate(all_words)

# plotting figure
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[19]:


# Visualizing -ve words in tweets
all_words = " ".join([sentence for sentence in df['tweet'][df['label']==0]]) 

from wordcloud import WordCloud
wordcloud = WordCloud(height= 500, width=1000, random_state=42, max_font_size=100).generate(all_words)

# plotting figure
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[20]:


# Visualizing +ve words in tweets
all_words = " ".join([sentence for sentence in df['tweet'][df['label']==1]]) 

from wordcloud import WordCloud
wordcloud = WordCloud(height= 500, width=1000, random_state=42, max_font_size=100).generate(all_words)

# plotting figure
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[21]:


# Function to extract hashtags from tweets
def hashtag_extract(tweets):
    hashtags = []
    for tweet in tweets:
        ht = re.findall(r"#(\w+)", tweet)
        hashtags.append(ht)
    return hashtags


# In[22]:


ht_positive = hashtag_extract(df['tweet'][df['label']==0])

ht_negative = hashtag_extract(df['tweet'][df['label']==1])


# In[23]:


ht_positive[:5]


# In[24]:


ht_positive = sum(ht_positive, [])
ht_negative = sum(ht_negative, [])


# In[25]:


ht_positive[:5]


# In[26]:


# Visualizing Most used Positive Hashtags in Tweets
freq = nltk.FreqDist(ht_positive)
data = pd.DataFrame({"Hashtag": list(freq.keys()),
                    "Count": list(freq.values())})
data.head()


# In[27]:


data = data.nlargest(columns="Count", n=10)
plt.figure(figsize=(20,10))
sns.barplot(data=data, x='Hashtag', y='Count')
plt.show()


# In[28]:


# Visualizing Most used Negative Hashtags in Tweets
freq2 = nltk.FreqDist(ht_negative)
data2 = pd.DataFrame({"Hashtags": list(freq2.keys()),
                     "Count": list(freq2.values())})
data2.head()


# In[29]:


data2 = data2.nlargest(columns="Count", n=10)
plt.figure(figsize=(20,10))
sns.barplot(data=data2, x="Hashtags", y="Count")
plt.show()


# # Training and Evaluating Model 

# In[30]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))

tfidf = tfidf_vectorizer.fit_transform(df['tweet'])


# In[57]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf, df['label'], random_state=42, test_size=0.2)


# In[58]:


# Using Logistic Regression Algorithm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


# In[59]:


pred = log_reg.predict(X_test)

f1_score(y_test, pred)


# In[60]:


accuracy_score(y_test, pred)


# In[61]:


# Using Support Vector Machines
from sklearn.svm import LinearSVC
svm = LinearSVC()

svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

f1_score(y_test, svm_pred)


# In[62]:


accuracy_score(y_test, svm_pred)*100


# In[63]:


print(classification_report(y_test, svm_pred))
print(confusion_matrix(y_test, svm_pred))


# Hence, Support Vector Machines provided the better prediction than Logistic Regression which we've calculated in terms of F1 score and accuracy score.
