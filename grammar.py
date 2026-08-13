import sksequitur
from sksequitur import Grammar, Mark, Parser, parse
import argparse
import labels,utils


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
#        print(len(symb_dict.bigrams()))
        for symb_j in symb_dict.values():
            parser_i.feed(symb_j)
            parser_i.feed([Mark()])
        grammar_i = Grammar(parser_i.tree)
        print_grammar(grammar_i)
        print("**************************")

def print_grammar(grammar):
    print(len(grammar))
    for key_i,value_i in grammar.items():
        print(f"{key_i}->{value_i}")

def show_symbols(cls_path):
    print(cls_path)
    labeling= labels.LabelingGroup.read(cls_path)
    symb_dict=labeling.as_symbols("tf-idf",False)
    utils.print_dict(symb_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_path", type=str,default="MSR/ae/layer_1/spectral_36")
    parser.add_argument("--cmd", type=str,default="show")
    args=parser.parse_args()
    if(args.cmd=="build"):
        build_grammars(args.cls_path)
    if(args.cmd=="show"):
        show_symbols(args.cls_path)