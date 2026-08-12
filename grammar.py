import sksequitur
from sksequitur import Grammar, Mark, Parser, parse
import labels

def build_grammars(cls_path):
    labeling= labels.LabelingGroup.read(cls_path)
    symb_dict=labeling.as_symbols(verbose=False)
    train,test=labeling.split()
    for cat_i,group_i in train.by_cat().items():
        print(f"Classs:{cat_i}")
        parser_i = Parser()
        for label_j in group_i:  #symb_dict.values():
            symb_j=symb_dict[str(label_j)]
            parser_i.feed(symb_j)
            parser_i.feed([Mark()])
        grammar = Grammar(parser_i.tree)
        show_grammar(grammar)
        print("**************************")

def show_grammar(grammar):
    print(len(grammar))
    for key_i,value_i in grammar.items():
        print(f"{key_i}->{value_i}")

build_grammars("MSR/ae/layer_1/spectral_3")