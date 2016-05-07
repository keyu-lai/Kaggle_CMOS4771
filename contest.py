import pandas as pd
import numpy
import time

from collections import defaultdict, Counter
from glob import glob
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from classifier import blend_clf, write_out, encode_label

def avg_all_models_from_files(infiles, filename):
    scores = defaultdict(list)
    with open(filename,"wb") as outfile:
        weight_list = [1]*len(glob(infiles))
    
        for i, ifile in enumerate( glob(infiles) ):
            print "parsing:", ifile
            lines = open(ifile).readlines()
            lines = [lines[0]] + sorted(lines[1:])
        
            #write out all model results
            for idx, line in enumerate( lines ):
                if i == 0 and idx == 0:
                    outfile.write(line)
                    
                if idx > 0:
                    row = line.strip().split(",")
                    for l in range(1,weight_list[i]+1):
                        scores[(idx,row[0])].append(row[1])
    
        #take hard votes
        for j,k in sorted(scores):
            outfile.write("%s,%s\n"%(k,Counter(scores[(j,k)]).most_common(1)[0][0]))
    
        print("wrote to %s"%outfile)

def prepare_data(df):
    df4 = df
    df1 = pd.DataFrame(df.loc[:,[u'0',u'5',u'7',u'8',u'9',u'14',u'16',u'17',u'56',u'57']])
    df2 = pd.DataFrame(df.loc[:,[u'18',u'20',u'23',u'25',u'26',u'58']])
    df3 = pd.DataFrame(df.drop([u'0',u'5',u'7',u'8',u'9',u'14',u'16',u'17',
                          u'18',u'20',u'23',u'25',u'26',u'56',u'57',u'58'],axis=1))
    
    return df1, df2, df3, df4

def encode(dataset):
    drop_cols = []
    for i in dataset.columns:
        if isinstance(dataset[i].values[0], basestring):
            drop_cols = drop_cols + [i]
    
    return encode_label(dataset, drop_cols)

def encode_with_test(train, test):
    agg_data = pd.concat([train, test],axis=0,ignore_index=True)
    agg_data= encode(agg_data)
    
    return agg_data._slice(slice(0,train.shape[0]),axis=0), agg_data._slice(slice(train.shape[0],agg_data.shape[0]),axis=0)


def run_local():
    train = pd.read_csv('data.csv')
    label = train['label']
    train = train.drop('label',1)
    train = encode(train)

    X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.1)

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

    predictions, _ = blend_clf(clfs, X_train.as_matrix(),X_test.as_matrix(),y_train.as_matrix())
    print y_test
    print classification_report(y_test, predictions,digits=4)
    
    return None
    
def score():
    train = pd.read_csv('data.csv')
    label = train['label']
    train = train.drop('label',1)
    quiz = pd.read_csv('quiz.csv')
    
    agg_data = pd.concat([train, quiz],axis=0,ignore_index=True)
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
    filename = "stack9." + str(sc) + '.' + time.strftime("%H%M%S") + '.txt'
    print('Writing to ' + filename)
    write_out(predictions,filename)
    
    return None

if __name__ == "__main__":
    #if testing run only then do run_local()
    #if score quiz then run score()
    #if produces average file then run avg_all_models_from_files("votes/stack*","pred.txt")
    score()
