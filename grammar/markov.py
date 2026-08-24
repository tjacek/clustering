from collections import defaultdict


class MarkovChain(object):
    def __init__(self,dist,k=1):
        self.dist=dist
        self.k=k

    def states(self):
        return list(self.dist.keys())

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