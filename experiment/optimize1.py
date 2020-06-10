import numpy as np 
from scipy import optimize

def obj(b, X0, Y0, P0, X1, Y1, P1):
    g0 = np.dot(X0, b)
    g0[g0 > 100] = 100
    g0[g0 < -100] = -100
    g1 = np.dot(X1, b)
    g1[g1 > 100] = 100
    g1[g1 < -100] = -100
    s0 = Y0**2/P0
    s1 = Y1**2/P1
    p1 = 1/(1+np.exp(-g1))
    p0 = np.exp(-g0)/(1+np.exp(-g0))
    V = np.mean(s1/p1+0.01) + np.mean(s0/p0+0.01)
    return V


def opt(X0_list, Y0_list, P0_list, X1_list, Y1_list, P1_list, X):
    X0 = np.array(X0_list)
    Y0 = np.array(Y0_list)
    P0 = np.array(P0_list)
    X1 = np.array(X1_list)
    Y1 = np.array(Y1_list)
    P1 = np.array(P1_list)

    func = lambda b: obj(b, X0, Y0, P0, X1, Y1, P1)

    b = np.zeros(X0.shape[1])
    result = optimize.minimize(fun=func, x0=b, method="SLSQP")

    p1 = 1/(1+np.exp(-np.dot(np.array(X), result.x)))
    return p1