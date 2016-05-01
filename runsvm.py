import pandas as pd
import numpy
from IPython.display import display
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from classifier import encode_onehot
from sklearn.decomposition import PCA

from classifier import train_svm
from sklearn.metrics import classification_report

if __name__ == "__main__":
    train = pd.read_csv('data.csv')[:5000]
    display(train.shape)
    label = train['label']
    train = train.drop('label',1)

    drop_cols = []
    for i in train.columns:
        if type(train[i].values[0]) == type(""):
            drop_cols = drop_cols + [i]

    train = encode_onehot(train, drop_cols)

    pca = PCA(n_components=50)
    pca.fit(train)
    train = pca.transform(train)
    
    X_train, X_test, y_train, y_test = train_test_split(train, label)
    
    svm_clf = train_svm(X_train, y_train)
    predictions = svm_clf.predict(X_test)

    f = open('svm_repprt.txt','w')
    f.write(classification_report(y_test, predictions))