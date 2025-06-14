#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor


# In[2]:


data = pd.read_csv('E:\Projects\Instagram-Reach-Analysis\Instagram data.csv', encoding='latin1')
data.head()


# In[3]:


data.isnull().sum()


# In[4]:


data.dropna()


# In[5]:


data.info()


# In[6]:


plt.figure(figsize=(10,8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Home")
sns.distplot(data['From Home'])
plt.show()


# In[7]:


plt.figure(figsize=(10, 8))
plt.title('Distribution of Impression From Hashtags')
sns.distplot(data['From Hashtags'])
plt.show()


# In[8]:


plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Explore")
sns.distplot(data['From Explore'])
plt.show()


# In[9]:


# Reshape the data from wide to long format
long_df = data[['From Home', 'From Hashtags', 'From Explore', 'From Other']].melt(
    var_name='Source', value_name='Impressions'
)

# Group and sum by source
summary = long_df.groupby('Source', as_index=False).sum()

# Create the donut chart
fig = px.pie(summary, values='Impressions', names='Source',
             title='Impressions on Instagram Posts From Various Sources', hole=0.5)

# Show the chart directly in Jupyter Notebook
fig.show(renderer='notebook')


# In[10]:


text = " ".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.style.use('classic')
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[12]:


text = " ".join(i for i in data.Hashtags)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[14]:


figure = px.scatter(data_frame = data, x="Impressions",
                    y="Likes", size="Likes", trendline="ols", 
                    title = "Relationship Between Likes and Impressions")
figure.show()


# In[15]:


figure = px.scatter(data_frame = data, x="Impressions",
                    y="Comments", size="Comments", trendline="ols", 
                    title = "Relationship Between Comments and Total Impressions")
figure.show()


# In[16]:


figure = px.scatter(data_frame = data, x="Impressions",
                    y="Shares", size="Shares", trendline="ols", 
                    title = "Relationship Between Shares and Total Impressions")
figure.show()


# In[17]:


figure = px.scatter(data_frame = data, x="Impressions",
                    y="Saves", size="Saves", trendline="ols", 
                    title = "Relationship Between Post Saves and Total Impressions")
figure.show()


# In[20]:


# Select only numeric columns
numeric_data = data.select_dtypes(include='number')

# Compute correlation on numeric columns
correlation = numeric_data.corr()

# Show correlations with 'Impressions' column
print(correlation["Impressions"].sort_values(ascending=False))


# In[21]:


conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100
print(conversion_rate)


# In[22]:


figure = px.scatter(data_frame = data, x="Profile Visits",
                    y="Follows", size="Follows", trendline="ols", 
                    title = "Relationship Between Profile Visits and Followers Gained")
figure.show()


# In[23]:


x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 
                   'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)


# In[24]:


model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)


# In[25]:


features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
model.predict(features)


# In[ ]:




