import numpy as np
import matplotlib.pyplot as plt

N = 500

x = np.zeros((2,N))
x[0,:] = np.random.normal(0,1,N)
x[1,:] = 2* x[0,:] + np.random.normal(0,0.4,N)

C = 1/N * x@x.T
L, Q = np.linalg.eig(C)

y = Q.T@x



y[0,:] = 0
z = Q@y


print("L",L)
print(x.shape)

#print(C)
plt.plot(z[0,:],z[1,:],'o')
plt.show()


# x 2xN    x*xt   2xN Nx2   2x2    
# 2x2 * 2xN