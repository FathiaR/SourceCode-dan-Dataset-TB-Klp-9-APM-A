# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Groceries_dataset.csv')
x = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values

# Cleaning the Text
import re
from nltk.corpus import stopwords # removing useless words
from nltk.stem.porter import PorterStemmer # getting root of words
stopwords_list = stopwords.words('english')
ps = PorterStemmer()

corpus = []
for i in range(0,len(x)):
    corp = re.sub(pattern = '[^a-zA-Z]', repl = ' ', string = x[i])
    corp = corp.split()
    corp = [ps.stem(word) for word in corp]
    corp = ' '.join(corp)
    corpus.append(corp)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(lowercase=True, stop_words = stopwords_list)
x_sparse = cv.fit_transform(corpus)
x = x_sparse.todense()

# Splitting dataset into Train set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size = 0.85 , random_state=0)

# Fitting the Naive Bayes (Gausiian form) model to the train set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier = classifier.fit(x_train,y_train)

y_train_pred = classifier.predict(x_train)

y_test_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(y_train,y_train_pred)
cm_test = confusion_matrix(y_test,y_test_pred)

import os
import sys

scriptpath = "../../Tools"
sys.path.append(os.path.abspath(scriptpath))
import accuracy as ac

t_train,f_train,acc_train = ac.accuracy_on_cm(cm_train)
print('Train status = #{} True, #{} False, %{} Accuracy'.format(t_train,f_train,acc_train*100))

t_test,f_test,acc_test = ac.accuracy_on_cm(cm_test)
print('Test status = #{} True, #{} False, %{} Accuracy'.format(t_test,f_test,acc_test*100))