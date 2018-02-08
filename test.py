import numpy as np
from genData import *
from algorithms import *
from fancyimpute import NuclearNormMinimization, SoftImpute
import matplotlib.pyplot as plt

np.set_printoptions(suppress = True)

m = 100
n = 200
k = 20
#U = np.random.randn(m,k)
#V = np.random.randn(n, k)
#A = np.random.randn(m, n) + np.dot(U, V.T)
#A = np.round(A) + 10
U,V,A = clean_data(m,n,k)
M = np.round(np.random.rand(m, n))
s = np.linalg.svd(A, compute_uv = False, full_matrices = False)
print(s)
#Xhat = svt(X, M, 0.01, maxiters = 20000)
#print(Xhat)

X = A.copy()
X_incomplete = X * M
X_incomplete[X_incomplete == 0] = np.nan

tlist = [0.01,0.1,0.5,1,10]
elist_lib = []
elist_my = []
for t in tlist:
    X = A.copy()
    X_incomplete = X * M
    X_incomplete[X_incomplete == 0] = np.nan
    svt_alg = SoftImpute(max_iters = 1000, 
                         shrinkage_value = t,
                         verbose = False)
    Xhat = svt_alg.complete(X_incomplete)
    e1 = error(U, V, Xhat, M)
    elist_lib.append(e1)

    X = A.copy()*M
    Xhat = svt(X, M, tau = t, maxiters = 1000)
    e2 = error(U, V, Xhat, M)
    elist_my.append(e2)

    print(t, e1, e2)

#print(tlist)
#print(elist_lib)
#print(elist_my)
plt.plot(tlist, elist_my, 'x-')
plt.plot(tlist, elist_lib, 'x-')
plt.show()



