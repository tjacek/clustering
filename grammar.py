import sksequitur
from sksequitur import Grammar, Mark, Parser, parse
import labels

def build_grammars(cls_path):
    labeling= labels.LabelingGroup.read(cls_path)
    dict_map=labels.DictMap.tf_map(labeling)
#    symb_dict=labeling.as_symbols(verbose=False)
    train,test=labeling.split()
    for cat_i,group_i in train.by_cat().items():
        print(f"Classs:{cat_i}")
        parser_i = Parser()
        symb_dict=group_i.as_symbols( symb_map=dict_map,
                                      verbose=False)
        print(len(symb_dict.bigrams()))
        for symb_j in symb_dict.values():
            parser_i.feed(symb_j)
            parser_i.feed([Mark()])
        grammar_i = Grammar(parser_i.tree)
#        show_grammar(grammar_i)
        print("**************************")


def show_grammar(grammar):
    print(len(grammar))
    for key_i,value_i in grammar.items():
        print(f"{key_i}->{value_i}")

build_grammars("MSR/ae/layer_1/spectral_36")