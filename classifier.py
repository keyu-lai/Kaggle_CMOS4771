
# coding: utf-8

# In[15]:

import sys
import unicodedata

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances as ed
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.preprocessing import scale
from sklearn.ensemble import BaggingClassifier
from sklearn.utils import check_random_state
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def transform_corpus(train):
    corpus = []
    vlist = train.values.tolist()

    for l in vlist:
        corpus+= [" ".join(l)]
        

    vectorizer = TfidfVectorizer(min_df=1)
    vectorizer.fit_transform(corpus)
    print train.shape
    out =  vectorizer.transform(train.transpose())
    print out.shape

def encode_onehot(df, cols):
    vec = DictVectorizer()
    
    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(orient='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index
    
    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df

def encode_label(df, cols):
    le = LabelEncoder()
    
    for c in cols:
        le.fit(df[c].unique())
        df[c] = le.transform(df[c])
    
    return df

def train_svm(train_data,train_label):
    pipeline = Pipeline([('clf', SVC(kernel='linear', C=1.0, verbose=1))])
    parameters = {'clf__C': (1, 5, 10, 20, 50, 100, 200, 500)}
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,verbose=1, scoring='accuracy')
    grid_search.fit(train_data, train_label)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:' 
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    
    return grid_search

def train_decision_tree(train_data,train_label):
    pipeline = Pipeline([('clf', DecisionTreeClassifier(criterion='entropy'))])
    parameters = {'clf__max_depth': (300, 500),'clf__min_samples_split': (1, 2),'clf__min_samples_leaf': (1, 2)}
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

def train_SGD_SVM(train_data, train_label):
    pipeline = Pipeline([('clf', SGDClassifier(loss="huber", penalty="l2",shuffle=True,alpha=0.0001,epsilon=0.1, verbose=1))])
    print pipeline.get_params().keys()
    parameters = {'clf__alpha': (0.001,0.01)}
    grid_search = GridSearchCV(pipeline, parameters,verbose=1, scoring='f1')
    grid_search.fit(train_data, train_label)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])

    return grid_search

def train_bagging_svm(train_data,train_label):
    rng = check_random_state(0)
    bag_clf = BaggingClassifier(base_estimator=SVC(kernel='linear', C=1.0),
                              random_state=rng,max_samples = 1.0, n_estimators=16, n_jobs=-1,verbose=1).fit(train_data, train_label)
    return bag_clf

def train_bagging_sgd_svm(train_data,train_label):
    rng = check_random_state(0)
    bag_clf = BaggingClassifier(base_estimator=SGDClassifier(loss="preceptron", penalty="l2",n_jobs=-1,shuffle=True, verbose=1, alpha = 0.01, epsilon=0.05),
                              random_state=rng,max_samples = 1.0, n_estimators=100, n_jobs=-1,verbose=1).fit(train_data, train_label)
    
    return bag_clf

def train_bagging_decision_tree(train_data,train_label):
    rng = check_random_state(0)
    
    
    pipeline = Pipeline([('clf', BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                              random_state=rng,n_estimators=100, n_jobs=-1,verbose=1).fit(train_data, train_label))])
        
    parameters = {'clf__max_samples': (0.1, 0.3, 0.5, 0.7, 1.0),'clf__n_estimators' : (100,200,300)}
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,verbose=1, scoring='accuracy')
    grid_search.fit(train_data, train_label)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:' 
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    
    return grid_search

    return bag_clf

def train_rf(train_data, train_label):
    pipeline = Pipeline([('clf', RandomForestClassifier(n_estimators=10, max_depth=None,
                                                        min_samples_split=1, random_state=0))])
    parameters = {'clf__n_estimators': (50,100,200),'clf__max_features':('sqrt','log2',None) }
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,verbose=1, scoring='accuracy')
    grid_search.fit(train_data, train_label)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:' 
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    
    return grid_search

def train_ada_boost(train_data, train_label):
    pipeline = Pipeline([('clf', AdaBoostClassifier())])
    parameters = {'clf__n_estimators': (900,1000,1100),'clf__learning_rate':(1.1,1.2,1.3)}
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,verbose=1, scoring='accuracy')
    grid_search.fit(train_data, train_label)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:' 
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    
    return grid_search

def voting(train_data,train_label, *clfs):
    ensembles = [(str(i), c) for (i,c) in enumerate(clfs)]
        
    vc = VotingClassifier(estimators=ensembles,voting='soft')
    
    return vc.fit(train_data, train_label)

def train_xgb(train_data, train_label):
    pipeline = Pipeline([('clf', XGBClassifier(learning_rate=0.3, max_depth=7, n_estimators=800))])
    parameters = {'clf__learning_rate' :(0.2 , 0.3, 0.4), 'clf__n_estimators':(800,900)}
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,verbose=1, scoring='accuracy')
    grid_search.fit(train_data, train_label)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:' 
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    
    return grid_search

def train_gbc(train_data,train_label):
    pipeline = Pipeline([('clf', GradientBoostingClassifier(loss='deviance',random_state=0,n_estimators=800,learning_rate=0.3,max_depth=7))])
    parameters = {'clf__n_estimators':(800,900,1000)}
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,verbose=1, scoring='accuracy')
    grid_search.fit(train_data, train_label)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:' 
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    
    return grid_search

def train_xtree_classifer(train_data,train_label):
    pipeline = Pipeline([('clf', 
                       ExtraTreesClassifier(n_jobs=-1,n_estimators=300,max_depth=None, max_features='sqrt',random_state=0))])
    parameters = {'clf__n_estimators':(50, 100,200,300,500,800)}
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,verbose=1, scoring='accuracy')
    grid_search.fit(train_data, train_label)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:' 
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    
    return grid_search


def write_out(pv,filename):
    output = [str(i+1) + ',' + str(v) for i,v in enumerate(pv)]
    f = open(filename, 'w')
    f.write('Id,Prediction\n')
    f.write('\n'.join(output))
    f.close()

def blend_clf(clfs, data, quiz, label):
    np.random.seed(0) # seed to shuffle the train set

    n_folds = 5
    verbose = True
    shuffle = True

    X, y, T = data, label, quiz

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(StratifiedKFold(y, n_folds))

    print "Creating bleding train set and valid set..."
    
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((T.shape[0], len(clfs)))
    
    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((T.shape[0], len(skf)))
        for i, (train, val) in enumerate(skf):
            print "Fold", i
            X_train = X[train]
            y_train = y[train]
            X_val = X[val]
            y_val = y[val]
            #train each mode
            clf.fit(X_train, y_train)
            #model selection on validation set
            dataset_blend_train[val, j] = clf.predict_proba(X_val)[:,1]
            #predict test set for each model
            dataset_blend_test_j[:, i] = clf.predict_proba(T)[:,1]
        
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

    print "Blending models by logistic regression..."
    #Feed output of all models to 2nd layer for logistic regression
    clf = train_log_reg(dataset_blend_train, y)
    #clf = LogisticRegression(C=0.8, solver='newton-cg', penalty='l2', n_jobs=-1, random_state=678)
    #clf.fit(dataset_blend_train, y)
    predictions = clf.predict(dataset_blend_test)
    
    return predictions, clf.best_score_

def train_log_reg(train_data,train_label):
    rng = check_random_state(0)
    pipeline = Pipeline([('clf', LogisticRegression(penalty='l2', n_jobs=-1, random_state=678))])
    parameters = {'clf__C': (0.8, 0.9, 1.0, 1.1), 'clf__solver' : ('newton-cg','lbfgs','sag')}
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,verbose=1, scoring='accuracy')
    grid_search.fit(train_data, train_label)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:' 
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    
    return grid_search