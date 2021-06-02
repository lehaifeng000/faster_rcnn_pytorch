import pickle
import numpy as np
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# a=[
#     ('001',np.random.rand(2,2)),
#     ('002',np.random.rand(2,2))
# ]

# save_obj(a,'t')

# b=load_obj('t')

ret=load_obj('results')


pass