import numpy as np
import matplotlib.pyplot as plt 


# transformation matrix, if the matrix has integer entries it becomes a isomorphism on a [0,1]x[0,1] torus
A = np.array([[2,1],[1,1]])
print(np.linalg.eig(A))
x =  np.zeros((2,1000))
x[:,0] = np.array([ 0.85065081, -0.52573111])



for i in range(0,999):
    x[:,i+1] = (1.415241 * x[:,i]  ) % 1
    #plt.pause(0.001)

plt.plot(x[0,:],x[1,:],'o')
plt.show()