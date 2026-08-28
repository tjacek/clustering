import argparse
import seq

def cluster_stats( label_path,
	               seq_path):
    label_group=seq.get_group("labels")
    labeling= label_group.read(label_path)
    symbols=labeling.as_symbols(symb_map="tf-idf")
    seq_group=seq.get_group("feat")
    seqs= seq_group.read(seq_path)
    by_labels=seqs.by_labels(symbols)
    for label_i,group_i in by_labels.items():
    	print(f"Cat:{label_i}")
    	print(len(group_i))
    print(type(by_labels))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_path", type=str,default="MSR/spectral_36")
    parser.add_argument("--seq_path", type=str,default="MSR/seqs")
    args=parser.parse_args()
    cluster_stats( args.cls_path,
	               args.seq_path)