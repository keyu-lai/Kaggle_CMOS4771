
# coding: utf-8

# In[15]:

import sys
sys.path.append("xgboost_bck/python-package/")

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

import xgboost as xgb

def encode_onehot(df, cols):
    vec = DictVectorizer()
    
    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(orient='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index
    
    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
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
    
    
    pipeline = Pipeline([('clf', BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy',max_depth=300),
                              random_state=rng,n_estimators=100, n_jobs=-1,verbose=1).fit(train_data, train_label))])
        
    parameters = {'clf__max_samples': (0.1, 0.3, 0.5, 0.7, 1.0)}
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
    parameters = {'clf__n_estimators': (100,200),'clf__max_features':('sqrt','log2') }
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
    parameters = {'clf__n_estimators': (200,300,400)}
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

def train_friedman_xgbm(train_data, train_label):
    pipeline = Pipeline([('clf', xgb.XGBClassifier(n_estimators=300,learning_rate=0.1,max_depth=7))])
    parameters = {}
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,verbose=1, scoring='accuracy')
    grid_search.fit(train_data, train_label)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:' 
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    
    return grid_search

def train_gbc(train_data,train_label):
    pipeline = Pipeline([('clf', GradientBoostingClassifier(loss='exponent',random_state=0,n_estimators=500,learning_rate=0.05,max_depth=7))])
    parameters = {}
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
                       ExtraTreesClassifier(n_jobs=-1,n_estimators=300,max_depth=None, max_features='log2',random_state=0))])
    parameters = {'clf__n_estimators':(100,200,300,400,500),'clf__max_features':('log2','sqrt',None)}
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,verbose=1, scoring='accuracy')
    grid_search.fit(train_data, train_label)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:' 
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    
    return grid_search

def train_bagging_xtree(train_data,train_label):
    rng = check_random_state(0)
    xtree = ExtraTreesClassifier(n_jobs=-1,n_estimators=20,max_depth=None, max_features='sqrt',random_state=0)
    pipeline = Pipeline([('clf', BaggingClassifier(base_estimator=xtree, random_state=rng,n_estimators=100, 
                                                   n_jobs=-1,verbose=1).fit(train_data, train_label))])
    parameters = {'clf__max_samples': (0.5, 0.7, 1.0)}
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,verbose=1, scoring='accuracy')
    grid_search.fit(train_data, train_label)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:' 
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    
    return grid_search

    return bag_clf

def write_out(pv,filename):
    output = [str(i+1) + ',' + str(v) for i,v in enumerate(pv)]
    f = open(filename, 'w')
    f.write('Id,Prediction\n')
    f.write('\n'.join(output))
    f.close()

    