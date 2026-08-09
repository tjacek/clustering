import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import argparse
import base,labels,plot,utils

def eval_bag(in_path,verbose=True):
    label_group=labels.LabelingGroup.read(in_path)
    n_clust=label_group.n_clust()
    train,test=label_group.split()
    train,test= as_bag(train,n_clust),as_bag(test,n_clust)
    clf=LogisticRegression(solver='liblinear')
    if(train.X.shape[1] !=test.X.shape[1]):
        return -1
    clf.fit(train.X,train.y)
    y_pred=clf.predict(test.X)
    if(verbose):
        print(classification_report(y_pred,test.y))
    acc =accuracy_score(y_pred,test.y)
    return acc,n_clust

def as_bag(label_group,n_clust=None):
    full_hist = label_group.hist(n_clust)
    n_clusters= full_hist.shape[0]
    X,y=[],[]  
    for label_i in label_group:
        X.append(label_i.hist(n_clusters))
        y.append(label_i.desc.cat)
    X,y=np.array(X),np.array(y)
    return base.Dataset(X,y)

def multi_bag(dir_path,alg):
    regex=alg+r"_\d"
    paths=utils.find_paths(dir_path,regex)
    pairs=[ eval_bag(path_i,verbose=False) 
            for path_i in paths]
    acc,clust=zip(*pairs)
    plot.scatter( clust, acc, 
                  title="Bag",
                  xlabel="n_clust",
                  ylabel="Accuracy")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dir", type=str,default="MSR")
    parser.add_argument("--nn", type=str,default="ae")
    parser.add_argument("--layer", type=int,default=1)
    parser.add_argument("--alg", type=str,default="kmeans")
    parser.add_argument("--n_clust", type=int,default=0)
    args=parser.parse_args()
    layer=f"layer_{args.layer}"
    if(args.n_clust>1):
        alg=f"{args.alg}_{args.n_clust}"    
        path="/".join([args.main_dir,args.nn ,layer,alg])
        eval_bag(path)
    else:
        path="/".join([args.main_dir,args.nn ,layer])
        multi_bag(path,args.alg)