import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import numpy as np


st.title('S&P 500 App')

st.markdown("""
This app retrieves the list of the **S&P 500** (from Wikipedia) and its corresponding **stock closing price** (year-to-date)!
* **Python libraries:** base64, pandas, streamlit, yfinance, numpy, matplotlib
* **Data source:** [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
""")

st.sidebar.header('User Input Features')

# Web scraping of S&P 500 data
#
@st.cache
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header = 0)
    df = html[0]
    return df

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
]

st.header('Display Companies in Selected Sector')
st.write('Data Dimension: ' + str(df_selected_sector.shape[0]) + ' rows and ' + str(df_selected_sector.shape[1]) + ' columns.')
st.dataframe(df_selected_sector)

# Download S&P500 data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)

# https://pypi.org/project/yfinance/

data = yf.download(
        tickers = list(df_selected_sector[:10].Symbol),
        period = "ytd",
        interval = "1d",
        group_by = 'ticker',
        auto_adjust = True,
        prepost = True,
        threads = True,
        proxy = None
    )

# Plot Closing Price of Query Symbol
def price_plot(symbol):
  df = pd.DataFrame(data[symbol].Close)
  df['Date'] = df.index
  fig = plt.figure()
  plt.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
  plt.plot(df.Date, df.Close, color='skyblue', alpha=0.8)
  plt.xticks(rotation=90)
  plt.title(symbol, fontweight='bold')
  plt.xlabel('Date', fontweight='bold')
  plt.ylabel('Closing Price', fontweight='bold')
  return st.pyplot(fig)

num_company = st.sidebar.slider('Number of Companies', 1, 5)

if st.button('Show Plots'):
    st.header('Stock Closing Price')
    for i in list(df_selected_sector.Symbol)[:num_company]:
        price_plot(i)
