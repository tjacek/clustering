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
    
    @property
    def states(self):
        if(self._terms is None):
            states = list(self.dist.keys())
            states.sort()
            self._states=utils.sort(states)
        return self._states
    
    @property
    def terms(self):
        if(self._terms is None):
            terms=[]
            for dict_i in self.dist.values():
                terms+= list(dict_i.keys())
            terms=list(set(terms))
            self._terms=utils.sort(terms)      
        return self._terms

    def as_matrix(self,irred=False):
        default= self.EPS if(irred) else 0
        matrix=[]
        for state_i in self.states:
            dict_i=self.dist[state_i]
            row_i=[]
            for term_i in self.terms:
                if(term_i in dict_i):
                	row_i.append(dict_i[term_i])
                else:
                	row_i.append(default)
            row_i=np.array(row_i)
            if(irred):
                row_i/=np.sum(row_i)
            matrix.append(row_i)
        return np.array(matrix)
    
    def stationary_dist(self):
        trans=self.as_matrix(irred=True)
        eig_values,eig_vectors=np.linalg.eig(trans.T)
        ones_index=np.argmin(np.abs(eig_values - 1.0))
        raw_vector=eig_vectors[:, ones_index].real
        stationary_dist= raw_vector / np.sum(raw_vector)
        return stationary_dist

    @classmethod
    def make(cls,symbols,k=1):
        fun=lambda :defaultdict(lambda :0)
        ngram_dict=defaultdict(fun)
        for seq_i in symbols:
            ngrams_i=seq_i.ngrams(k+1)
            for ngram_j in ngrams_i:
                start_j=ngram_j[:-1]
                start_j=symbols.SEP.join(start_j)
                end_j=ngram_j[-1]
                ngram_dict[start_j][end_j]+=1
        dist={}
        for state_i,dict_i in ngram_dict.items():
            norm_const=sum(list(dict_i.values()))
            dist[state_i]={ ngram_j:(value_j/norm_const) 
                                for ngram_j,value_j in dict_i.items()}
        return cls(dist,k)