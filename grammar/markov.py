import numpy as np
from collections import defaultdict
import utils

class MarkovChain(object):
    EPS=0.01
    def __init__( self,
                  dist,
                  order=1):
        self.dist=dist
        self.order=order
        self._states=None
        self._terms=None
    
    def __call__(self,x,y):
        return self.dist[y][x]

    def __getitem__(self,key):
        if(not (key in self.dist)):
            p=1/len(self.states)
            self.dist[key]=UniformDist1D(p)
        return self.dist[key]

    @property
    def states(self):
        if(self._states is None):
            states = list(self.dist.keys())
            states.sort()
            self._states=utils.sort(states)
        return self._states
    
    @property
    def terms(self):
        if(self._terms is None):
            terms=[]
            for dict_i in self.dist.values():
                terms+= dict_i.terms()#list(dict_i.keys())
            terms=list(set(terms))
            self._terms=utils.sort(terms)      
        return self._terms

    def as_matrix( self,
                   states=None,
                   terms=None,
                   irred=False):
        if(self.order!=1):
            msg="Tras matrix only suported for order=1"
            raise Exception(msg)
        if(states is None):
            states=self.states
        if(terms is None):
            terms=self.terms
        default= self.EPS if(irred) else 0
        if(irred):
            Dist1D.DEFAULT=self.EPS
        matrix=[]
        for state_i in states:
            dict_i=self[state_i]
            row_i=[]
            for term_i in terms:
                row_i.append(dict_i[term_i])
            row_i=np.array(row_i)
            if(irred):
                row_i/=np.sum(row_i)
            matrix.append(row_i)
        return np.array(matrix)
    
    def stationary_dist( self,
                         states=None,
                         terms=None):
        trans=self.as_matrix( states=states,
                              terms=terms,
                              irred=True)
        eig_values,eig_vectors=np.linalg.eig(trans.T)
        ones_index=np.argmin(np.abs(eig_values - 1.0))
        raw_vector=eig_vectors[:, ones_index].real
        stationary_dist= raw_vector / np.sum(raw_vector)
        return stationary_dist
    
    @classmethod
    def make(cls,symbols,k=1):
        ngram_dict=defaultdict(lambda:Dist1D())
        for seq_i in symbols:
            ngrams_i=seq_i.ngrams(k+1)
            for ngram_j in ngrams_i:
                ngrams_i=seq_i.ngrams(k+1)
                for ngram_j in ngrams_i:
                    start_j=ngram_j[:-1]
                    start_j=symbols.SEP.join(start_j)
                    end_j=ngram_j[-1]
                    ngram_dict[start_j].add(end_j)
        ngram_dict={ state_i:dist_i.norm()
                for state_i,dist_i in ngram_dict.items()}
        return cls(ngram_dict,order=k)
    
    def kl_divergence(self,Q,states):
        pi=self.stationary_dist(states,states)
        p=self.as_matrix(states=states,
                         terms=states,
                         irred=True)
        q=Q.as_matrix(states=states,
                         terms=states,
                         irred=True)
        h=0
        for x,_ in enumerate(states):
            row_x=0
            for y,_ in enumerate(states):
                row_x+=p[y][x]*np.log(p[y][x]/q[y][x])
            h+=pi[x] * row_x
        return h 

class Dist1D(object):
    DEFAULT=0
    def __init__( self,
                  dist_dict=None):
        if( dist_dict is None):
            dist_dict={}
        self.dist_dict=dist_dict
    
    def terms(self):
        return list(self.dist_dict.keys())

    def add(self,item):
        if(item in self.dist_dict):
            self.dist_dict[item]+=1
        else:
            self.dist_dict[item]=1
        return self

    def __getitem__(self,item):
        if(item in self.dist_dict):
            return self.dist_dict[item]
        return self.DEFAULT

    def norm(self):
        norm_c=sum(self.dist_dict.values())
        pairs=self.dist_dict.items()
        self.dist_dict={ key_i:(value_i/norm_c)
                          for key_i,value_i in pairs}
        return self

class UniformDist1D(object):
    def __init__(self,p):
        self.p=p

    def __getitem__(self,item):
        return self.p