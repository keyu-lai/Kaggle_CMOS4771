
# coding: utf-8

# In[15]:

"""
Hi

Thank you for taking the time to apply to Quovo. We like to send potential candidates a SHORT coding test/exercise so 
we could get a sense of how they approach problems. This also gives you the a good opportunity to see if Quovo-style 
challenges are a good fit for you. Don't go crazy on time, we'd just like to see enough progress on it where we can 
all have a conversation looking at your code together and talk about how you attacked the problem.

The concept:

In each row of the included datasets, products X and Y are considered to refer to the same security if 
they have the same ticker, even if the descriptions don't exactly match. 

Your challenge is to use these descriptions to predict whether each pair in the test set also refers to the 
same security. The difficulty of predicting each row will vary significantly, so please do not aim for 100% accuracy. 
There are several good ways to approach this, and we have no preference between them. 
The only requirement is that you do all of the work in this file, and return it to us.

Hint: Don't be afraid if you have no experience with text processing. You are in the majority. Check out this algorithm, 
and see how far you can go with it:
https://en.wikipedia.org/wiki/Tf–idf
http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

Good luck!
"""

import pandas as pd
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances as ed
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.preprocessing import scale

def split_dataframe(df):
    x1 = df['description_x']
    x2 = df['description_y']
    y_train = df['same_security']
    return [x1,x2,y_train]

def prepare_data_ed(dataset,vec):
    v1 = vec.transform(dataset[0]).todense()
    v2 = vec.transform(dataset[1]).todense()
    dist_train = [ed(x,y)[0].tolist() for (x, y) in zip(v1,v2)]
    
    return [dist_train,dataset[2]]

def prepare_data_diff(dataset,vec):
    v1 = vec.transform(dataset[0]).todense()
    v2 = vec.transform(dataset[1]).todense()
    
    return [abs(v1-v2),dataset[2]]

def train_svm(train_data,train_label):
    pipeline = Pipeline([('clf', SVC(kernel='rbf', gamma=0.01, C=100))])
    parameters = {'clf__gamma': (0.01, 0.03, 0.1, 0.3, 1),'clf__C': (0.1, 0.3, 1, 3, 10, 30),}
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=2,verbose=1, scoring='accuracy')
    grid_search.fit(train_data, train_label)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:' 
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    
    return grid_search

def train_decision_tree(train_data,train_label):
    pipeline = Pipeline([('clf', DecisionTreeClassifier(criterion='entropy'))])
    parameters = {'clf__max_depth': (150, 155, 160),'clf__min_samples_split': (1, 2, 3),'clf__min_samples_leaf': (1, 2, 3)}
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,verbose=1, scoring='f1')
    grid_search.fit(train_data, train_label)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])

    return grid_search

def train_nearest_neighbor(train_data, train_label):
    pipeline = Pipeline([('clf', KNeighborsClassifier())])
    parameters = {'clf__n_neighbors': (3, 5, 10, 15, 20, 25)}
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,verbose=1, scoring='f1')
    grid_search.fit(train_data, train_label)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])

    return grid_search