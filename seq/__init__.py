import seq.core
import seq.labels

def get_group(group_type):
	if(group_type=="actions"):
		return seq.core.ActionGroup
	if(group_type=="feat"):
		return seq.labels.FeatSeqGroup
	if(group_type=="labels"):
		return seq.labels.LabelingGroup
	raise Exception(f"Unknow type:{group_type}")