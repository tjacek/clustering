import sksequitur
from sksequitur import Grammar, Mark, Parser, parse
import labels

def build_grammars(cls_path):
    labeling= labels.LabelingGroup.read(cls_path)
#    labeling.as_symbols(verbose=True)
#    return
    train,test=labeling.split()
    for cat_i,labeling_i in train.by_cat().items():
        print(f"Classs:{cat_i}")
        parser_i = Parser()
        symb_dict= labeling_i.as_symbols("tf-idf",False)
        for symb_j in symb_dict.values():
            parser_i.feed(symb_j)
        grammar = Grammar(parser_i.tree)
        print(grammar)
        print("**************************")

build_grammars("MSR/ae/layer_1/kmeans_44")