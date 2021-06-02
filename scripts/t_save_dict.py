from sys import ps2
import numpy as np
p1=np.random.randint(1,10,(3,3))
p2=np.random.rand(5,5)
a=[(p1, p2)]

np.save('t1.npy',a)