#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import nltk

#import dataset
dataset = pd.read_csv('news.csv')

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from nltk.stem.wordnet import WordNetLemmatizer
stop_words =set(stopwords.words('english'))

headline = dataset['headline']
text = dataset['text']
#storing cleaned text reviews
cleaned_text =[]
#storing cleaned headline
cleaned_headline = []
for j in range(len(headline)):
    #this is for the headline
    summary = headline[j]
    #this is for the text
    summary1 = text[j]
    #Cleaning the reviews using re, removing pos tags
    clnr = re.compile('<.*?>')
    summary = re.sub(clnr, ' ' , summary)
    summary1 = re.sub(clnr, ' ' , summary1)
    #Cleaning the reviews keeping  only Alphabets
    clnr = re.compile('[^a-zA-Z]')
    summary = re.sub(clnr, ' ' , summary)
    summary1 = re.sub(clnr, ' ' , summary1)
    
    summary = summary.lower()
    summary1 = summary1.lower()
    
    summary = nltk.word_tokenize(summary)
    summary1 = nltk.word_tokenize(summary1)
    
    SS = SnowballStemmer('english')
    lmt = WordNetLemmatizer()
    
    summary = [SS.stem(i) for i in summary if not i in stop_words]
    summary = [lmt.lemmatize(i) for i in summary]
    
    summary = ' '.join(summary)
    
    summary1 = [SS.stem(i)  for i in summary1 if not i in stop_words]
    
    summary1 = [lmt.lemmatize(i) for i in summary1]
    summary1 = ' '.join(summary)
    
    print(j , ' done..')
    cleaned_headline.append(summary1)
    cleaned_text.append(summary)

#tf-idf vectorization
from sklearn.feature_extraction.text  import TfidfVectorizer

vectorise = TfidfVectorizer(use_idf=True)
vectorizer = vectorise.fit_transform(cleaned_text,cleaned_headline).toarray()

#Scaling the vector
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
data = ss.fit_transform(vectorizer)

#Reducing the dimensions to avoid overfit
from sklearn.decomposition import PCA
pca = PCA (n_components = 2)
x = pca.fit_transform(data)

#elbow method kmeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

wcss=[]
for i in range(1,10):
    kmeans = KMeans(n_clusters= i ,init = 'k-means++')
    preds = kmeans.fit_predict(x)
    center = kmeans.cluster_centers_
    #score = silhouette_score(data ,preds )
    wcss.append(kmeans.inertia_)
    print(i, ' done..' )
    
plt.plot(range(1,10) , wcss)
plt.xlabel('no of clusters')
plt.ylabel('distances')
plt.show()

#Predicting the clusters based on data 
clusters = 3
kmeans = KMeans(random_state = 0,n_clusters=clusters ,init = 'k-means++' , max_iter =300 , n_init=10)
pred = kmeans.fit_predict(x)

out = []
for i in range(3000):
    Y = vectorise.transform(text) 
    pred = kmeans.predict(Y)
    out.append(pred)

uid = ['uid-'+str(i+1) for i in range(3000)]
df = pd.DataFrame()
df['id'] = uid
df['cluster'] = out

df.to_csv('output.csv')
np.savetxt('output.txt', vectorizer.todense())
 