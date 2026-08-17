import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
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
    best=np.argmax(acc)
    print(f"Best n_clust:{clust[best]}")
    print(f"Acc:{acc[best]:.4f}")
    plot.scatter( clust, acc, 
                  title="Bag",
                  xlabel="n_clust",
                  ylabel="Accuracy")

def bigram_plot(dir_path,alg):
    regex=alg+r"_\d"
    paths=utils.find_paths(dir_path,regex)
    n_bigrams,clust=[],[]
    for i,path_i in enumerate( paths):
        label_i=labels.LabelingGroup.read(path_i)
        symb_i=label_i.as_symbols(verbose=False)
        n_bigrams.append(len(symb_i.bigrams()))
        clust.append(label_i.n_clust())
    plt.plot(clust, clust)
    plot.scatter( clust, n_bigrams, 
                  title=alg,
                  xlabel="n_clust",
                  ylabel="bigrams")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dir", type=str,default="MSR")
    parser.add_argument("--nn", type=str,default="ae")
    parser.add_argument("--layer", type=int,default=1)
    parser.add_argument("--bigrams", action="store_true")
    parser.add_argument("--alg", type=str,default="spectral")
    parser.add_argument("--n_clust", type=int,default=0)
    args=parser.parse_args()
    layer=f"layer_{args.layer}"
    if(args.n_clust>1):
        alg=f"{args.alg}_{args.n_clust}"    
        path="/".join([args.main_dir,args.nn ,layer,alg])
        eval_bag(path)
        exit()
    path="/".join([args.main_dir,args.nn ,layer])
    if(args.bigrams):
        bigram_plot(path,args.alg)
    multi_bag(path,args.alg)