import numpy as np
import matplotlib.pyplot as plt

N = 1000
x = np.zeros(N)
y = np.zeros(N)
z = np.zeros(N)
sx = 1.414
sv = 1.414
e = np.random.normal(0,sx,N)
v = np.random.normal(0,sv,N)


xs = np.zeros(N)
P = np.ones(N) * 0.0001

H = np.array([0.5])
A = np.array([1])

for i in range(0,N-1):
    x[i+1] = x[i] + e[i+1]
    y[i+1] = 0.5* x[i+1] + v[i+1]
    S = H * P[i] * H + sv*sv
    K = P[i] * H * (1/S)
    a = y[i] - H * xs[i]
    xs[i] = xs[i] + K*a
    P[i] = (1-K*H)*P[i]
    xs[i+1] = A*xs[i]
    P[i+1] = A*P[i]*A + sx*sx

#plt.plot(x)
#plt.plot(xs)
plt.plot(P)
plt.show()

