import numpy as np
import matplotlib.pyplot as plt

N = 100000
M = 3

x = np.sin((1/5)*np.linspace(0,N-1,N))
v = np.random.normal(0,0.1,N)


H = np.array([1,-0.5,0.2])
W = np.zeros(M)

d = np.convolve(x,H,mode="same") + v

y = np.zeros(N)
e = np.zeros(N)

u = 0.01

Wa = np.zeros((M,N))


# LMS
'''
for i in range(M,N):
    xa = x[i-M+1:i+1]
    xa = xa[::-1]
    y[i] = np.dot(xa,W)
    e[i] = d[i] - y[i]
    W = W + u * xa*e[i]
    Wa[:,i] = W    
'''


# NLMS
for i in range(M,N):
    xa = x[i-M+1:i+1]
    xa = xa[::-1]
    y[i] = np.dot(xa,W)
    e[i] = d[i] - y[i]
    W = W + u * xa*e[i] / np.sum(xa**2)
    Wa[:,i] = W    

print(xa)

pxd = np.array([ np.mean(x*d)  , np.mean(np.roll(x,+1)*d) ])
R = np.array([[np.mean(x*x),np.mean(np.roll(x,+1)*x)],[np.mean(np.roll(x,+1)*x),np.mean(x*x)]])


print("W=",W)
print("pxd=",pxd)
print("R=",R)
print("W0=",np.linalg.inv(R)@pxd)
plt.plot(np.transpose(Wa))
plt.show()