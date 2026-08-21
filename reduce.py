from sklearn.manifold import SpectralEmbedding
import argparse
import seq

def reduce_dim( in_path,
                out_path,
                n_clusters):
    feat_group=seq.get_group("feat")
    feat_seqs=feat_group.read(in_path)
    indexes,frames=feat_seqs.indexed_frames()
    spectral = SpectralEmbedding( n_components=n_clusters, 
                                  n_neighbors=10, 
                                  random_state=42)
    reduced_frames = spectral.fit_transform(frames)
    def helper(i,frame):
        return reduced_frames[i]
    reduced_seqs= feat_seqs.indexed_map(helper)
    reduced_seqs.save(out_path)
    print(len(reduced_seqs))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str,default="MSR/ae/layer_1/seqs")
    parser.add_argument("--out_path", type=str,default="MSR/ae/layer_1/spectral_seq")
    parser.add_argument("--n_clusters", type=int,default=46)
    args=parser.parse_args()
    reduce_dim( args.in_path,
                args.out_path,
                args.n_clusters)