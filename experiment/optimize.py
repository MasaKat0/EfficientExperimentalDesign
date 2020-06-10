import numpy as np 
from scipy import optimize

def obj(b, X, Y, P, N):
    g = np.dot(X, b)
    s = Y**2
    return np.sum(((s - g)/P)**2) + 0.1*np.dot(b,b)


def opt(X0_list, Y0_list, P0_list, X1_list, Y1_list, P1_list, X, N):
    X0 = np.array(X0_list)
    Y0 = np.array(Y0_list)
    P0 = np.array(P0_list)
    X1 = np.array(X1_list)
    Y1 = np.array(Y1_list)
    P1 = np.array(P1_list)

    func = lambda b: obj(b, X0, Y0, P0, N)

    b = np.zeros(X0.shape[1])
    result0 = optimize.minimize(fun=func, x0=b, method="SLSQP")
    result0 = np.dot(np.dot(np.linalg.inv(np.dot(X0.T, X0)+0.01*np.ones((len(X0.T), len(X0.T)))), X0.T), Y0)

    func = lambda b: obj(b, X1, Y1, P1, N)

    b = np.zeros(X0.shape[1])
    result1 = optimize.minimize(fun=func, x0=b, method="SLSQP")
    result1 = np.dot(np.dot(np.linalg.inv(np.dot(X1.T, X1)+0.01*np.ones((len(X1.T), len(X1.T)))), X1.T), Y1)

    s0 = np.dot(np.array(X), result0)
    s1 = np.dot(np.array(X), result1)
    if s0 < 0.1:
        s0 = 0.1
    if s1 < 0.1:
        s1 = 0.1
    p1 = np.sqrt(s1**2)/(np.sqrt(s0**2)+np.sqrt(s1**2))
    return p1