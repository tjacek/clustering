import numpy as np
from collections import defaultdict
import utils

class MarkovChain(object):
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

    def as_matrix(self):
        matrix=[]
        for state_i in self.states:
            dict_i=self.dist[state_i]
            row_i=[]
            for term_i in self.terms:
                if(term_i in dict_i):
                	row_i.append(dict_i[term_i])
                else:
                	row_i.append(0)
            matrix.append(row_i)
        return np.array(matrix)

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