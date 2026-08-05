import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import base,labels

def eval_bag(in_path):
    label_group=labels.LabelingGroup.read(in_path)
    train,test=label_group.split()
    train,test= as_bag(train),as_bag(test)
    clf=LogisticRegression(solver='liblinear')
    clf.fit(train.X,train.y)
    y_pred=clf.predict(test.X)
    print(classification_report(y_pred,test.y))

def as_bag(label_group):
    full_hist = label_group.hist()
    n_clusters= full_hist.shape[0]
    X,y=[],[]  
    for label_i in label_group:
        X.append(label_i.hist(n_clusters))
        y.append(label_i.desc.cat)
    X,y=np.array(X),np.array(y)
    return base.Dataset(X,y)

in_path="MSR/cnn/layer_1/kmeans_22"
eval_bag(in_path)