import action,cnn

class Seq(object):
    def __init__( self,
                  vectors,
                  desc,
                  labels=None):
        if(labels is None):
            labels=[]
        self.vectors = vectors
        self.desc = desc
        self.labels=labels

def train( in_path,
           out_path,
           epochs=150):
    actions=action.ActionGroup.read(in_path)
    train,test=actions.split()
    model,_=cnn.cnn_exp( train.as_dataset(),
                         test.as_dataset(),
                         cnn.frame_params(),
                         epochs=epochs)
    model.save(out_path)

train("MSR/scaled","MSR/model")