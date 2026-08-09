import labels

class ClusterAsig(object):
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
