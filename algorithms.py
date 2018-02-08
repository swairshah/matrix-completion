import numpy as np
import cvxpy as cvx

def error(U, V, Ahat, M):
    """
    rms error on unobserved entries
    """
    Ahat = np.multiply(Ahat, (1 - M)) # only these entries matter
    
    A = np.multiply((U @ V.T) , (1 - M))
    count = np.sum(1 - M)
    err = np.linalg.norm(A - Ahat, 'fro')**2 / count
    err = err**0.5
    return err

def error_noisy_col(U, V, Ahat, M, indices):
    """
    rms error on unobserved entries
    but ignore the cols in indices
    """
    M[:,indices] = 0
    return error(U, V, Ahat, M)

def svt(X, M, tau, maxiters=1000):
    def svd_(X, tau):
        U, s, Vt = np.linalg.svd(X)
        s = np.maximum(s - tau, 0)
        rank = (s > 0).sum()
        X_reconst = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
        return X_reconst

    missing = (M == 0)

    X_filled = X*M # make sure unobserved entries are not in X_filled
    
    for _ in range(maxiters):
        X_reconst = svd_(X_filled, tau)
        X_filled[missing] = X_reconst[missing]

    return X_filled

def nuc(A, M):
    X = cvx.Variable(*A.shape)
    obj = cvx.Minimize(cvx.norm(X, "nuc"))
    constraints = ([cvx.mul_elemwise(M,X) == cvx.mul_elemwise(M, A)])
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver=cvx.SCS)
    return X.value

#def nuclear_norm_min_reg(A, M, mu):
if __name__ == "__main__":
    m = 5
    n = 5
    k = 2
    U = np.random.randn(m,k)
    V = np.random.randn(n, k)
    A = np.random.randn(m, n) + np.dot(U, V.T)
    A = np.round(A) + 10
    M = np.round(np.random.rand(m, n))
    print(M)

    idx = [0,1]
    M[:,idx] = 0
    print(M)
