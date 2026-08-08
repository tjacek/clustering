import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import argparse
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dir", type=str,default="MSR")
    parser.add_argument("--nn", type=str,default="ae")
    parser.add_argument("--layer", type=int,default=1)
    parser.add_argument("--alg", type=str,default="spectral")
    parser.add_argument("--n_clust", type=int,default=8)
    args=parser.parse_args()
    layer=f"layer_{args.layer}"
    alg=f"{args.alg}_{args.n_clust}"
    path="/".join([args.main_dir,args.nn ,layer,alg])
    eval_bag(path)
#in_path="MSR/ae/layer_1/kmeans_20"
#eval_bag(in_path)