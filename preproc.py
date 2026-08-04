import numpy as np
import re,cv2
from tqdm import tqdm
import utils

def convert(in_path,out_path):
    utils.make_dir(out_path)
    for id_i,path_i in tqdm(utils.iter_files(in_path)):
        id_i=clean_id(id_i)
        action_i=load_depth_map(path_i)
        action_i=bound_action(action_i)
        action_i=norm_action(action_i)
        out_i=f"{out_path}/{id_i}"
        utils.make_dir(out_i)
        for j,frame_j in enumerate(action_i):
            out_j=f"{out_i}/{j}.png"
            cv2.imwrite(out_j,frame_j)

def clean_id(name):
    raw=re.findall(r'\d+', name)
    return "_".join(raw)

def remove_letters(name):
    return ''.join(c for c in name 
               if not c.isalpha())

def norm_action(action):
    nonzero=(action!=0)
    offset=np.amin(action[nonzero])-1
    action[nonzero]-=offset
    action[nonzero]/=np.amax(action)
    action[nonzero]= 1 - action[nonzero]
    action*=256
    return action

def bound_action(action):
    def helper(frame_i):
        rows = np.any(frame_i != 0, axis=1)
        cols = np.any(frame_i != 0, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin,cmin,rmax,cmax
    extr= np.array([ helper(frame_i) 
                        for frame_i in action])
    min_x,min_y=np.amin(extr[:,:2],axis=0)
    max_x,max_y=np.amax(extr[:,2:],axis=0)
    action= [ frame_i[min_x:max_x,min_y:max_y] 
                for frame_i in action]  
    return np.array(action)

def load_depth_map(path):
    with open(path, 'rb') as fid:
        num_frames, dims = read_header(fid)
        file_data = np.fromfile(fid, dtype=np.uint32)
        depth = file_data.astype(np.float64)  
        depth_count_per_map = dims[0] * dims[1]
        depth_map = []
        for i in range(num_frames):
            current_depth_data = depth[:depth_count_per_map]
            depth = depth[depth_count_per_map:]
            frame = current_depth_data.reshape((dims[0], dims[1]), order='F').T
            depth_map.append(frame)
    return np.array(depth_map)

def read_header(fid):
    num_frames = np.frombuffer(fid.read(4), dtype=np.uint32)[0]
    dim1 = np.frombuffer(fid.read(4), dtype=np.uint32)[0]
    dim2 = np.frombuffer(fid.read(4), dtype=np.uint32)[0]
    dims = (int(dim1), int(dim2))
    return int(num_frames), dims

convert("MSR/Depth","MSR/raw")