import grammar.core
import seq
import argparse

def markov(cls_path):
    label_group=seq.get_group("labels")
    labeling= label_group.read(cls_path)
    symbols=labeling.as_symbols()

    for cat_i,group_i in symbols.by_cat().items():
    	group_i.dist(1)
 #   	ngram_i=group_i.ngram_dict(2)
 #   	print(cat_i)
 #   	print(ngram_i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_path", type=str,default="MSR/ae/layer_1/spectral_36")
    parser.add_argument("--cmd", type=str,default="markov")
    args=parser.parse_args()
    if(args.cmd=="build"):
        grammar.core.build_grammars(args.cls_path)
    if(args.cmd=="markov"):
    	markov(args.cls_path)
