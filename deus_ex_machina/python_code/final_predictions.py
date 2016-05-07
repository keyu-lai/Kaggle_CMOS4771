from sys import argv
import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state


def encode_label(df, cols):
    '''
    Encode each column of the data set with label encoder.
    Input: 
        df: dataset
        cols: feature name of each column
    Output:
        the encoded dataset
    '''
    le = LabelEncoder()
    
    for c in cols:
        le.fit(df[c].unique())
        df[c] = le.transform(df[c])
    
    return df

def write_out(pv, filename):
    '''
    Write the output to a file.
    '''
    output = [str(i+1) + ',' + str(v) for i,v in enumerate(pv)]
    f = open(filename, 'w')
    f.write('Id, Prediction\n')
    f.write('\n'.join(output))
    f.close()

def blend_clf(clfs, data, quiz, label):
    '''
    Implement the 2 layer ensemble blending algorithm.
    The 3rd layer is performed manually by calling
    avg_models_from_files
    Input:
        clfs: a list of classifiers to be blended
        data: training data
        quiz: test data
        label: lables of the training data
    Output:
        a list of prediction results
    '''
    # seed to shuffle the train set
    np.random.seed(0)

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
            # train each mode
            clf.fit(X_train, y_train)
            # model selection on validation set
            dataset_blend_train[val, j] = clf.predict_proba(X_val)[:,1]
            # predict test set for each model
            dataset_blend_test_j[:, i] = clf.predict_proba(T)[:,1]
        
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

    print "Blending models by logistic regression..."
    # Feed output of all models to 2nd layer for logistic regression
    clf = train_log_reg(dataset_blend_train, y)
    predictions = clf.predict(dataset_blend_test)
    
    return predictions, clf.best_score_

def train_log_reg(train_data, train_label):
    '''
    Train the logistic regression as the 2nd layer of final classifier.
    We use cross validation to select the best parameters set.
    Output:
        the model with best performance
    '''
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

def encode(dataset):
    '''
    Encode the dataset using label encoder
    '''
    drop_cols = []
    for i in dataset.columns:
        if isinstance(dataset[i].values[0], basestring):
            drop_cols = drop_cols + [i]
    
    return encode_label(dataset, drop_cols)

def score(datafile, quizfile, outputfile):
    '''
    Main function of the program, which blends 15 models and feed them to the blending algorithm to produce the final classifer
    Input:
        datafile: data file name
        quizfile: quiz file name
        outputfile: output file name
    Output:
        Produce the result file
    '''

    train = pd.read_csv(datafile)
    label = train['label']
    train = train.drop('label',1)
    quiz = pd.read_csv(quizfile)
    
    agg_data = pd.concat([train, quiz], axis=0, ignore_index=True)
    agg_data = encode(agg_data)
    train = agg_data._slice(slice(0,train.shape[0]),0)
    quiz = agg_data._slice(slice(train.shape[0],agg_data.shape[0]),0)

    X_train, X_test, y_train = train, quiz, label

    clfs = [RandomForestClassifier(n_estimators=200, n_jobs=-1, criterion='gini', random_state=398, max_features='sqrt'),
        RandomForestClassifier(n_estimators=200, n_jobs=-1, criterion='entropy', random_state=812, max_features='sqrt'),    
        ExtraTreesClassifier(n_estimators=200, n_jobs=-1, criterion='gini', max_features=None, random_state=312),
        ExtraTreesClassifier(n_estimators=200, n_jobs=-1, criterion='entropy', max_features=None, random_state=4018),
        GradientBoostingClassifier(learning_rate=0.2, max_depth=7, n_estimators=800),
        AdaBoostClassifier(base_estimator=RandomForestClassifier(n_jobs=-1, n_estimators=10, min_samples_split=1, 
                                                                 random_state=417), n_estimators=50, learning_rate=0.5),
        AdaBoostClassifier(base_estimator=ExtraTreesClassifier(n_jobs=-1, n_estimators=10, 
                                                                min_samples_split=1,max_features=None,
                                                                random_state=2345), n_estimators=50, learning_rate=0.3),
        BaggingClassifier(base_estimator=RandomForestClassifier(n_jobs=-1, n_estimators=10, min_samples_split=1, 
                                                                 random_state=123), random_state=1245, n_estimators=50),
        BaggingClassifier(base_estimator=ExtraTreesClassifier(n_jobs=-1, n_estimators=10, min_samples_split=1), 
                          random_state=456, n_estimators=50),
        BaggingClassifier(DecisionTreeClassifier(criterion='entropy'), n_jobs=-1,random_state=1556,n_estimators=200),
        DecisionTreeClassifier(criterion='entropy',splitter='best'),
        DecisionTreeClassifier(criterion='gini',splitter='best'),
        KNeighborsClassifier(n_neighbors=1),
        KNeighborsClassifier(n_neighbors=2),
        KNeighborsClassifier(n_neighbors=3)]

    predictions, sc = blend_clf(clfs, X_train.as_matrix(),X_test.as_matrix(),y_train.as_matrix())
    print('Writing to ' + outputfile)
    write_out(predictions, outputfile)

if __name__ == "__main__":

    if len(argv) != 4:
        print 'python final_predictions.py DATAFILE QUIZFILE OUTPUTFILE'
    else:
        score(argv[1], argv[2], argv[3])