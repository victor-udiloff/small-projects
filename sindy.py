import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Inicialization
N = 1000
delta_t = 0.01
x = np.zeros(N)
dxdt = np.zeros(N)
t = np.linspace(0,10,1000)


# diferential equation dx/dt = -2x with x(0) = 1 
# The solution is exp(-2x)
x[0] = 4

for n in range(0,N-1):
    # the line below calculates depends on the differential equation
    #x[n+1] = -2*x[n] * delta_t + x[n] + 0.0001 * np.random.randn(1)
    x[n+1] = -2*(x[n]-3) * delta_t + x[n] + 0.00001 * np.random.randn(1)

    dxdt[n] = (x[n+1]-x[n])/ delta_t


# Making a sindy model

theta = np.zeros((N,4))
theta[:,0] = np.ones(N)
theta[:,1] = x
theta[:,2] = (x**2)
theta[:,3] = (np.sin(x))


# Sparse regression

def custom_loss(e):
    # Linear system error A @ x - b
    residual = theta @ e - dxdt
    # Custom loss: for example, the sum of squares of residuals
    #loss = np.sum(residual**2) + 0.0001 * np.sum(e)  + 0.001 * np.sum(e**2)
    loss = np.sum(residual**2) + 0.01 * np.sum(residual) 
    return loss


xi = np.linalg.lstsq(theta,dxdt)
print("xi using least squares",xi)

x0 = np.array([1, 1,1,1])
result = minimize(custom_loss, x0)
print("xi using L1 norm:", result.x)
# Expected xi = [0,-2,0,0]
real_xi = [3*2,-2,0,0]
#print(xi[0])
print("error l2", np.sum((xi[0]-real_xi)**2))
print("error l1", np.sum((result.x-real_xi)**2))
print("l1 advantage:", -np.sum((result.x-real_xi)**2) + np.sum((xi[0]-real_xi)**2) )

#plt.plot(x)

#plt.plot(theta[:,0])
#plt.plot(theta[:,1])
#plt.plot(theta[:,2])
#plt.plot(theta[:,3])

plt.show()