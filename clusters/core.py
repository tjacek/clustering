import numpy as np
import seq

class ClusterAsig:
    def __init__( self,
                  frame_info,
                  assig):
        self.frame_info=frame_info
        self.assig=assig

    def get_labels(self,seqs):
        label_group=seq.get_group("labels")
        dtype=label_group.dtype()
        labels=label_group()
        seq_dict=seqs.as_dict()
        for name_i,seq_i in seq_dict.items():
            arr=self.label_seq(name_i)
            label_i=dtype(arr,seq_i.desc)
            labels.append(label_i)
        return labels

    def label_seq(self,name):
        indices=(self.frame_info["names"]==name)
        labels= self.assig[indices]
        order=self.frame_info["order"][indices]
        n_frames=len(order)
        arr=np.zeros((n_frames,))
        for i,label_i in zip(order,labels):
            arr[i]=label_i
        return arr
        
class _ClusterAsig(object):
    def __init__( self,
                preclustr,
                labels,
                dynamic):
        self.preclustr=preclustr
        self.labels=labels
        self.dynamic=dynamic

    def get_labels(self,seqs):
        if(self.dynamic is None):
            return self.from_order()
        else:
            return self.from_seqs(seqs)

    def from_order(self):
        def helper(i):
            return self.labels[i]
        order=self.preclustr.order_labeling
        return order.map(helper)

    def from_seqs(self,seqs):
        return seqs.map_seq(self.dynamic,
                            group_type=labels.LabelingGroup)
