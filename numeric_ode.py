from xml.etree.ElementTree import QName
import numpy as np
import math
import matplotlib.pyplot as pl

# Decompõe a matriz em LU
def decomposicaoLU(A,B,C,n):

     U = np.zeros(n)    # vetor U(i,i)
     Uii1 = np.zeros(n)   # vetor U(i,i+1)
     L = np.zeros(n)   # vetor L(i+1,i)

     U[0] = B[0]
     for x in range(1,n):
        L[x] = A[x]/U[x-1]
        U[x] = B[x] - L[x]*C[x-1]

     Uii1 = C

     return L,U,Uii1

# Resolve o sistema linear com os componentes LU da matriz
def sistema_linear_tridiagonal(L,U,Uii1,D):  

    n = L.shape[0]      # obtém o tamanho dos vetores

    #Ly = d
    Y = np.zeros(n)
    Y[0] = D[0] 
    for x in range(1,n):
        Y[x] = D[x] - L[x] * Y[x-1]

    # Ux = y
    X = np.zeros(n)
    X[n-1] = Y[n-1]/U[n-1]
    for x in range(n-2,-1,-1):
        X[x] = (Y[x] - Uii1[x] * X[x+1]) / U[x]


    return(X)
def sistema_linear_tridiagonal_ciclico(A,B,C,D,n):


    # Decomposição LU da submatriz T 
    A1 = np.array(A[0:n-1])
    A1[0] = 0
    C1 = np.array(C[0:n-1])
    C1[n-2] = 0
    TL,TU,TU2 = decomposicaoLU(A1,B[0:n-1],C1,n-1)

    # Resolução do sistema TY = D
    Y = sistema_linear_tridiagonal(TL,TU,TU2,D[0:n-1])

    # Resolução do sistema TZ = V
    V = np.zeros(n-1)
    V[0] = A[0]
    V[n-2] = C[n-2]
    Z = sistema_linear_tridiagonal(TL,TU,TU2,V)

    # Achar X[n] 
    X = np.zeros(n)
    X[n-1] = ( D[n-1]-C[n-1]*Y[0]-A[n-1]*Y[n-2] )/(B[n-1]-C[n-1]*Z[0]-A[n-1]*Z[n-2])
    
    # Achar X[0:n-1]
    X[0:n-1] = Y - X[n-1]*Z
    return X

#--------------------------------------------------
def main(f,k,n,aa,bb):
    h = L / (n+1)
    f = f + (bb-aa) * (k-np.roll(k,-1))/h # derivada de k(x)
    u = np.ones(n)
    ff = np.ones(n)       # vetor de produtos internos de f com as bases
    fp = np.ones((n,n))   # matriz de produtos internos de bases
    a = np.zeros(n)      # a,b e c para calcular da matriz tridiagonal
    b = np.zeros(n)
    c = np.zeros(n)

    for i in range(0,n):  # Calcula do produto interno da matriz 
        for j in range(0,n):
            if(abs(i-j) > 1):
                fp[i,j] = 0  # 0 fora da diagonal principal e das secundarias
            elif i == j:
                fp[i,j] =  (1/(h))*(k[round((100*(n+1))*(h*0.577+i*h+h)/L)]+k[round((100*(n+1))*(h*0.577+i*h+h)/L)])
                b[i] = fp[i,j]
            elif j -1 == i:
                d = h/2
                fp[i,j] =  -1* (1/(2*h))*(k[round((100*(n+1))*(d*0.577+i*d+d)/L)]+k[round((100*(n+1))*(d*0.577+i*d+d)/L)])
                fp[j,i] = fp[i,j]
                if i == j -1:
                    c[i] = fp[i,j]
                    a[i] = c[i]
    a[n-1] = a[0]
    a[0] = 0

    for i in range(0,n):  # Calcula do produto interno do vetor
        ff[i] = (h*(1-(1/1.732))) * (f[round((100*(n+1))*(h*0.577+i*h+h))]+f[round((100*(n+1))*(-h*0.577+i*h+h))])

    u =  sistema_linear_tridiagonal_ciclico(a,b,c,ff,n)
    u = u + np.ones(n)*aa +(bb-aa)*np.linspace(h,L-h,n)
    print(u)
    v = np.linspace(h,1-h,n)
    

    u[0:round(0.2*n)] = u[0:round(0.2*n)] + 0.02*x[0:round(0.2*n)]
    pl.plot(   u    )
    pl.show()
    
    #print("erro" , max((v**2)*((np.ones(n)-v)**2) -u))

n = 2000
L= 1
aa = 20
bb = 20
x = np.linspace(0,L,(n+2))
xl = np.linspace(0,L,L*(100*(n+1)+1))

#f = 12*xl*(1-xl)-2
k = 3.6+ 0*xl
 


Qm = 6    # aquecimento
#Qf = 2    # esfriamento
Qf =  2.718281**(-((xl)/0.6)**2)   # esfriamento
sig = 0.1  # desvio padrão da gaussiana
f = Qm * 2.718281**(-((xl-L/2)/sig)**2) -1*Qf  # Q(x)


main(f,k,n,aa,bb)

