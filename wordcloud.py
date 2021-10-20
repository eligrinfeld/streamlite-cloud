#!/usr/bin/env python
# coding: utf-8

# In[15]:


# allow us to directly load a .zip file
import zipfile 
import pandas as pd
import os
 
# As we are working with long texts, we set the corresponding option to visualized all
# the data complete
pd.set_option('display.max_colwidth', None)
 
import matplotlib.pyplot as plt
import mpld3


# In[2]:


import nltk
nltk.download('vader_lexicon')


# In[3]:


# now, we import the relevant modules from the NLTK library
from nltk.sentiment.vader import SentimentIntensityAnalyzer
 
def classify_compound(text, threshold=0.33):
     
    # initialize VADER
    sid = SentimentIntensityAnalyzer()
     
    # Calling the polarity_scores method on sid and passing in the text
    # outputs a dictionary with negative, neutral, positive, and compound scores for the input text
    scores = sid.polarity_scores(text)
     
    # get compound score
    score = scores['compound']
     
    # translate the score into the correcponding input according to the threshold
    if score <= -threshold: return 'Negative'
    elif score >= threshold: return 'Positive'
    else: return 'Neutral'


# In[4]:



# load text data .csv with reviews and apply columns restrictions, 
# also, we drop duplicates and any row with nan values in the column Translated_Review
text_data = pd.read_csv(r"C:\Users\EliGrinfeld\Box\Eli Grinfeld\CSAT_Clean.csv")


# In[5]:


# create a new feature based on compound score from VADER using our function "classify_compound"
text_data['compound_sentiment'] = text_data.CSAT_COMMENT.apply(lambda text: classify_compound(text))
 

df = text_data
 
# Visualize a random row to see all features together
df.sample(1)


# In[6]:


# Import all necesary libraries
from wordcloud import  WordCloud, STOPWORDS, ImageColorGenerator
 


# In[7]:


# Get stopwords from wordcloud library
stopwords = set(STOPWORDS)
 


# In[12]:


# Add some extra words ad hoc for our purpose
xtra_words = ['ARBONNE', 'PRODUCTS']
stopwords.update(xtra_words)
 


# In[13]:


# join all reviews
text = " ".join(review for review in text_data.CSAT_COMMENT)


# In[14]:



# Generate the image
wordcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=100, min_word_length=5).generate(text)

# visualize the image
fig=plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Total Reviews Word Clowd')
plt.show()


# In[16]:


html_str = mpld3.fig_to_html(fig)
Html_file= open("index.html","w")
Html_file.write(html_str)
Html_file.close()


# In[17]:


pwd


# In[ ]:




