import seq.core
import seq.feat
import seq.labels

def get_group(group_type):
	if(group_type=="actions"):
		return seq.core.ActionGroup
	if(group_type=="feat"):
		return seq.feat.FeatSeqGroup
	if(group_type=="labels"):
		return seq.labels.LabelingGroup
	raise Exception(f"Unknow type:{group_type}")