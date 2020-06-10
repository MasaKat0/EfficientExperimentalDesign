import numpy as np 


def sampler(mu0, mu1, std0, std1, type_sampler='homo'):    
    X = np.random.normal(0, 1, size=5)

    if type_sampler=='homo': 
        f0 = X[0] + X[1] +X[2] + X[3] + X[4]
        f1 = X[0] + X[1] +X[2] + X[3] + X[4]
    elif type_sampler=='hetero':
        f0 = X[0] + X[1] +X[2] + X[3] + X[3] + X[4]
        f1 = X[0] + X[1] +X[2] + 3*X[3] + 4*X[4] + 5*X[4]

    EY0 = mu0 + f0
    e0 = np.random.normal(0, std0)

    EY1 = mu1 + f1
    e1 = np.random.normal(0, std1)
    
    Y0 = EY0+e0
    Y1 = EY1+e1
    
    Var0 = std0**2
    Var1 = std1**2

    return X, Y0, Y1, EY0, EY1, Var0, Var1
