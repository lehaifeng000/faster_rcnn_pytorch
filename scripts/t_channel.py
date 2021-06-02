import torch
import torch.nn.functional as F
import numpy as np
import time

def Lin_matrix(blob_in,tril_index=None):
    """
    input:
        s: shape=(b,c,h,w)
    output:
        (b,c,c)
    """
    sv=F.relu(blob_in)
    h = sv.shape[2]
    w = sv.shape[3]
    c=sv.shape[1]
    sv=sv.reshape(-1,c,w*h)
    sv1=torch.index_select(sv,1,tril_index[:len(tril_index)//2])
    sv2=torch.index_select(sv,1,tril_index[len(tril_index)//2:])
    blob_v=torch.matmul(sv1.unsqueeze(2),sv2.unsqueeze(3))#(b,(c*c+c)/2,1,1)
    blob_v=blob_v.squeeze(2).squeeze(2)#(b,(c*c+c)/2)
    return blob_v
def get_tril(s):
    index=[]
    mat=np.arange(0,s)
    mat=np.tile(mat[None,:],(s,1))
    mat_t=mat.T
    index_array=np.tril_indices(s)
    til_mat=mat[index_array].tolist()
    rep_mat=mat_t[index_array].tolist()
    index.extend(til_mat)
    index.extend(rep_mat)
    index=torch.tensor(index,dtype=torch.int64,device="cuda")
    return index
def main():
    blob_in=torch.randn((1,3,4,4),device="cuda")# (3,16)X(16,3) -> (3,3)    
    tril_index=get_tril(blob_in.size()[1])
    res=Lin_matrix(blob_in,tril_index)
    print("cuda memory %f GB"%(torch.cuda.max_memory_allocated()/1024.0/1024.0/1024.0))
main()