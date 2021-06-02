import numpy as np

gt_labels=np.array([1,3,2,3,2],dtype=int)
label=2
gt_boxes=np.random.randint(1,4,size=(5,4))

gts = gt_boxes[gt_labels==label,:]
s=[gt_labels==label]
pass