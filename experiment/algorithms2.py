import numpy as np
import math

from optimize import opt 
from sklearn.linear_model import LogisticRegression
from statsmodels.tools.tools import add_constant
from sklearn.neighbors import KNeighborsRegressor
from nadaraya_watson import KernelRegression


class Basic(object):
    def __init__(self):
        self.X_list = [] 
        self.Y_list = []
        self.P_list = []
        self.A_list = []
        self.func_1 = []
        self.func_0 = []

        self.action_space = [0,1]
        
    def est_params(self, method):
        y = np.array(self.Y_list)
        x = np.array(self.X_list)
        a = np.array(self.A_list)
        
        x0 = x[a==0].copy()
        y0 = y[a==0].copy()
        y0_2 = y0**2

        x1 = x[a==1].copy()
        y1 = y[a==1].copy()
        y1_2 = y1**2
        
        if method == 'Knn':
            model_y_a0 = KernelRegression()
            model_y_a1 = KernelRegression()
            
            model_y2_a0 = KernelRegression()
            model_y2_a1 = KernelRegression()

        elif method == 'NW': 
            model_y_a0 = KNeighborsRegressor(n_neighbors=np.int((len(x0))**(1/3)))
            model_y_a1 = KNeighborsRegressor(n_neighbors=np.int((len(x0))**(1/3)))
            
            model_y2_a0 = KNeighborsRegressor(n_neighbors=np.int((len(x0))**(1/3)))
            model_y2_a1 = KNeighborsRegressor(n_neighbors=np.int((len(x0))**(1/3)))
         
        model_y_a0.fit(x0, y0)
        model_y_a1.fit(x1, y1)

        model_y2_a0.fit(x0, y0**2)
        model_y2_a1.fit(x1, y1**2)
                
        return model_y_a0, model_y_a1, model_y2_a0, model_y2_a1
        

class RCT(Basic):
    def __init__(self):
        super().__init__()

    def __call__(self, period_t, X, Y0, Y1):
        if (period_t == 0) or (period_t == 1):
            self.X_list.append(X)
            self.Y_list.append(Y0)
            self.P_list.append(0.5)
            self.A_list.append(0)
        elif (period_t == 2) or (period_t == 3):
            self.X_list.append(X)
            self.Y_list.append(Y1)
            self.P_list.append(0.5)
            self.A_list.append(1)
        else:
            act = np.random.choice(self.action_space, p=[0.5, 0.5])
            if act == 0:
                self.X_list.append(X)
                self.Y_list.append(Y0)
                self.P_list.append(0.5)
                self.A_list.append(0)
            else:
                self.X_list.append(X)
                self.Y_list.append(Y1)
                self.P_list.append(0.5)
                self.A_list.append(1)

    def effect(self):
        y = np.array(self.Y_list)
        x = np.array(self.X_list)
        a = np.array(self.A_list)
        
        y1 = y[a==1]
        y0 = y[a==0]
        
        theta = np.mean(y1) - np.mean(y0)
        var = 2*np.mean(y1**2) + 2*np.mean(y0**2)
        return theta, var

    
class AdaIPW(Basic):
    def __init__(self, method, pretraining=50):
        super().__init__()
        
        self.method = method
        self.gamma = lambda t: 1/np.sqrt(t)
        self.pretraining = pretraining

    def __call__(self, period_t, X, Y0, Y1):
        if (period_t == 0) or (period_t == 1):
            self.X_list.append(X)
            self.Y_list.append(Y0)
            self.P_list.append(0.5)
            self.A_list.append(0)
        elif (period_t == 2) or (period_t == 3):
            self.X_list.append(X)
            self.Y_list.append(Y1)
            self.P_list.append(0.5)
            self.A_list.append(1)
        else:
            model_y_a0, model_y_a1, model_y2_a0, model_y2_a1 = self.est_params(self.method)
            mu0, mu1, nu0, nu1 = model_y_a0.predict([X])[0], model_y_a1.predict([X])[0], model_y2_a0.predict([X])[0], model_y2_a1.predict([X])[0]
            
            if nu0 < 0.01:
                nu0 = 0.01
            if nu1 < 0.01:
                nu1 = 0.01
                            
            p = (np.sqrt(nu1)/(np.sqrt(nu0) + np.sqrt(nu1) ))              
            p = self.gamma(period_t)/2 + (1 - self.gamma(period_t))*p
            
            if period_t < self.pretraining:
                p = 0.5
                
            if p < 0.01:
                p = 0.01
            if p > 0.99:
                p = 0.99
            
            act = np.random.choice(self.action_space, p=[1-p, p])
            
            if act == 0:
                self.X_list.append(X)
                self.Y_list.append(Y0)
                self.P_list.append(p)
                self.A_list.append(0)
            else:
                self.X_list.append(X)
                self.Y_list.append(Y1)
                self.P_list.append(p)
                self.A_list.append(1)

    def effect(self):
        y = np.array(self.Y_list)
        x = np.array(self.X_list)
        a = np.array(self.A_list)
        p1 = np.array(self.P_list)
        p0 = 1 - p1
        
        N = len(x)
            
        theta = np.mean(a*y/p1 - (1-a)*y/p0)
        var = np.mean((a*y/p1 - (1-a)*y/p0)**2)
                                                                                        
        return theta, var
    
    
class A2IPW(Basic):
    def __init__(self, method, pretraining=50):
        super().__init__()
        self.func_1 = []
        self.func_0 = []
        
        self.func_1_partial = []
        self.func_0_partial = []
        
        self.method = method
        self.pretraining = pretraining
        self.gamma = lambda t: 1/np.sqrt(t)

    def __call__(self, period_t, X, Y0, Y1):
        if (period_t == 0) or (period_t == 1):
            self.X_list.append(X)
            self.Y_list.append(Y0)
            self.P_list.append(0.5)
            self.A_list.append(0)
            self.func_0.append(0)
            self.func_1.append(0)
            
        elif (period_t == 2) or (period_t == 3):
            self.X_list.append(X)
            self.Y_list.append(Y1)
            self.P_list.append(0.5)
            self.A_list.append(1)
            self.func_0.append(0)
            self.func_1.append(0)

        else:
            model_y_a0, model_y_a1, model_y2_a0, model_y2_a1 = self.est_params(self.method)
            mu0, mu1, nu0, nu1 = model_y_a0.predict([X])[0], model_y_a1.predict([X])[0], model_y2_a0.predict([X])[0], model_y2_a1.predict([X])[0]
            
            var0 = nu0 - (mu0)**2
            var1 = nu1 - (mu1)**2
            
            if var0 < 0.01:
                var0 = 0.01
            if var1 < 0.01:
                var1 = 0.01
            
            p = (np.sqrt(var1)/(np.sqrt(var0) + np.sqrt(var1) ))
            p = self.gamma(period_t)/2 + (1 - self.gamma(period_t))*p
            
            if period_t < self.pretraining:
                p = 0.5
                
            if p < 0.01:
                p = 0.01
            if p > 0.99:
                p = 0.99
            
            act = np.random.choice(self.action_space, p=[1-p, p])
            
            if act == 0:
                self.X_list.append(X)
                self.Y_list.append(Y0)
                self.P_list.append(p)
                self.A_list.append(0)
                self.func_0.append(mu0)
                self.func_1.append(mu1)
            else:
                self.X_list.append(X)
                self.Y_list.append(Y1)
                self.P_list.append(p)
                self.A_list.append(1)
                self.func_0.append(mu0)
                self.func_1.append(mu1)
                
    def effect(self):
        y = np.array(self.Y_list)
        x = np.array(self.X_list)
        a = np.array(self.A_list)
        p1 = np.array(self.P_list)
        p0 = 1 - p1
        
        N = len(x)
        
        mu0 = np.array(self.func_0)
        mu1 = np.array(self.func_1)
                        
        z = a*(y-mu1)/p1 + mu1 - (1-a)*(y-mu0)/p0 - mu0
        theta = np.mean(z)
        var = np.mean((z)**2)
                                                                                        
        return theta, var
    
            
class MA2IPW(Basic):
    def __init__(self, method, pretraining=50):
        super().__init__()
        self.func_1 = []
        self.func_0 = []
        
        self.func_1_partial = []
        self.func_0_partial = []
        
        self.method = method
        self.pretraining = pretraining
        self.gamma = lambda t: 1/np.sqrt(t)

    def __call__(self, period_t, X, Y0, Y1):
        if (period_t == 0) or (period_t == 1):
            self.X_list.append(X)
            self.Y_list.append(Y0)
            self.P_list.append(0.5)
            self.A_list.append(0)
            self.func_1.append(0)
            self.func_0.append(0)
            
        elif (period_t == 2) or (period_t == 3):
            self.X_list.append(X)
            self.Y_list.append(Y1)
            self.P_list.append(0.5)
            self.A_list.append(1)
            self.func_1.append(0)
            self.func_0.append(0)
            
        else:
            model_y_a0, model_y_a1, model_y2_a0, model_y2_a1 = self.est_params(self.method)
            mu0, mu1, nu0, nu1 = model_y_a0.predict([X])[0], model_y_a1.predict([X])[0], model_y2_a0.predict([X])[0], model_y2_a1.predict([X])[0]
            
            var0 = nu0 - (mu0)**2
            var1 = nu1 - (mu1)**2
            
            if var0 < 0.01:
                var0 = 0.01
            if var1 < 0.01:
                var1 = 0.01
           
            p = (np.sqrt(var1)/(np.sqrt(var0) + np.sqrt(var1)))
            p = self.gamma(period_t)/2 + (1 - self.gamma(period_t))*p
            
            if period_t < self.pretraining:
                p = 0.5
                
            if p < 0.01:
                p = 0.01
            if p > 0.99:
                p = 0.99
            
            act = np.random.choice(self.action_space, p=[1-p, p])
            
            if act == 0:
                self.X_list.append(X)
                self.Y_list.append(Y0)
                self.P_list.append(p)
                self.A_list.append(0)
                self.func_0.append(mu0)
                self.func_1.append(mu1)
                
            else:
                self.X_list.append(X)
                self.Y_list.append(Y1)
                self.P_list.append(p)
                self.A_list.append(1)
                self.func_0.append(mu0)
                self.func_1.append(mu1)
                
    def effect(self):
        y = np.array(self.Y_list)
        x = np.array(self.X_list)
        a = np.array(self.A_list)
        p1 = np.array(self.P_list)
        p0 = 1 - p1
        
        N = len(x)
        pi = N**(-1/1.5)
        
        mu0 = np.array(self.func_0)
        mu1 = np.array(self.func_1)
        
        z0 = a*(y-mu1)/p1 + mu1 - (1-a)*(y-mu0)/p0 - mu0
        theta0 = np.mean(z0)
               
        z1 = a*y/p1 - (1-a)*y/p0
        theta1 = np.mean(z1)
                
        theta = pi*theta1 + (1-pi)*theta0
        
        var = np.mean((z0)**2)
                                                                                        
        return theta, var

class OPT(Basic):
    def __init__(self):
        super().__init__()

    def __call__(self, period_t, X, Y0, Y1, mu0, mu1, var0, var1):
        if (period_t == 0) or (period_t == 1):
            self.X_list.append(X)
            self.Y_list.append(Y0)
            self.P_list.append(0.5)
            self.A_list.append(0)
            self.func_0.append(mu0)
            self.func_1.append(mu1)
            
        elif (period_t == 2) or (period_t == 3):
            self.X_list.append(X)
            self.Y_list.append(Y1)
            self.P_list.append(0.5)
            self.A_list.append(1)
            self.func_0.append(mu0)
            self.func_1.append(mu1)

        else:
            p = np.sqrt(var1)/(np.sqrt(var0)+np.sqrt(var1))
            act = np.random.choice(self.action_space, p=[1-p, p])
            if act == 0:
                self.X_list.append(X)
                self.Y_list.append(Y0)
                self.P_list.append(p)
                self.A_list.append(0)
                self.func_0.append(mu0)
                self.func_1.append(mu1)
            else:
                self.X_list.append(X)
                self.Y_list.append(Y1)
                self.P_list.append(p)
                self.A_list.append(1)
                self.func_0.append(mu0)
                self.func_1.append(mu1)

    def effect(self):
        y = np.array(self.Y_list)
        x = np.array(self.X_list)
        a = np.array(self.A_list)
        p1 = np.array(self.P_list)
        p0 = 1 - p1
        
        N = len(x)
        
        mu0 = np.array(self.func_0)
        mu1 = np.array(self.func_1)
                        
        z = a*(y-mu1)/p1 + mu1 - (1-a)*(y-mu0)/p0 - mu0
        theta = np.mean(z)
        var = np.mean((z)**2)
                                                                                        
        return theta, var
    
    
class DM(Basic):
    def __init__(self, method, pretraining=50):
        super().__init__()
        self.func_1 = []
        self.func_0 = []
        
        self.func_1_partial = []
        self.func_0_partial = []
        
        self.method = method
        self.pretraining = pretraining
        self.gamma = lambda t: 1/np.sqrt(t)
        
    def __call__(self, period_t, X, Y0, Y1):
        if (period_t == 0) or (period_t == 1):
            self.X_list.append(X)
            self.Y_list.append(Y0)
            self.P_list.append(0.5)
            self.A_list.append(0)
            
        elif (period_t == 2) or (period_t == 3):
            self.X_list.append(X)
            self.Y_list.append(Y1)
            self.P_list.append(0.5)
            self.A_list.append(1)
            
        else:
            model_y_a0, model_y_a1, model_y2_a0, model_y2_a1 = self.est_params(self.method)
            mu0, mu1, nu0, nu1 = model_y_a0.predict([X])[0], model_y_a1.predict([X])[0], model_y2_a0.predict([X])[0], model_y2_a1.predict([X])[0]
            
            var0 = nu0 - (mu0)**2
            var1 = nu1 - (mu1)**2
            
            if var0 < 0.01:
                var0 = 0.01
            if var1 < 0.01:
                var1 = 0.01
           
            p = np.sqrt(var1)/(np.sqrt(var0) + np.sqrt(var1))
            p = self.gamma(period_t)/2 + (1 - self.gamma(period_t))*p
            
            if period_t < self.pretraining:
                p = 0.5
                
            if p < 0.01:
                p = 0.01
            if p > 0.99:
                p = 0.99
            
            act = np.random.choice(self.action_space, p=[1-p, p])
            
            if act == 0:
                self.X_list.append(X)
                self.Y_list.append(Y0)
                self.P_list.append(p)
                self.A_list.append(0)
                self.func_0.append(mu0)
                self.func_1.append(mu1)
            else:
                self.X_list.append(X)
                self.Y_list.append(Y1)
                self.P_list.append(p)
                self.A_list.append(1)
                self.func_0.append(mu0)
                self.func_1.append(mu1)

    def effect(self):        
        mu0 = np.array(self.func_0)
        mu1 = np.array(self.func_1)
        
        theta = np.mean(mu1) - np.mean(mu0)
        var = np.mean((mu1 - mu0)**2)
                           
        return theta, var
    
class Hahn(Basic):
    def __init__(self, method, first_phase=50):
        super().__init__()
        self.func_1 = []
        self.func_0 = []
        self.X = []
        
        self.func_1_partial = []
        self.func_0_partial = []
        
        self.method = method
        
        self.est_prob = False
        
        self.first_phase = first_phase
        
    def __call__(self, period_t, X, Y0, Y1):        
        if (period_t == 0) or (period_t == 1):
            self.X_list.append(X)
            self.Y_list.append(Y0)
            self.P_list.append(0.5)
            self.A_list.append(0)
        elif (period_t == 2) or (period_t == 3):
            self.X_list.append(X)
            self.Y_list.append(Y1)
            self.P_list.append(0.5)
            self.A_list.append(1)
            
        elif period_t < self.first_phase:
            act = np.random.choice(self.action_space, p=[0.5, 0.5])
            if act == 0:
                self.X_list.append(X)
                self.Y_list.append(Y0)
                self.P_list.append(0.5)
                self.A_list.append(0)
            else:
                self.X_list.append(X)
                self.Y_list.append(Y1)
                self.P_list.append(0.5)
                self.A_list.append(1)
            
        else:
            if self.est_prob is False:
                self.model_y_a0, self.model_y_a1, self.model_y2_a0, self.model_y2_a1 = self.est_params(self.method)
                self.est_prob = True
                
            mu0, mu1, nu0, nu1 = self.model_y_a0.predict([X])[0], self.model_y_a1.predict([X])[0], self.model_y2_a0.predict([X])[0], self.model_y2_a1.predict([X])[0]
            
            var0 = nu0 - (mu0)**2
            var1 = nu1 - (mu1)**2

            if var0 < 0.01:
                var0 = 0.01
            if var1 < 0.01:
                var1 = 0.01

            p = (np.sqrt(var1)/(np.sqrt(var0) + np.sqrt(var1) ))

            if p < 0.01:
                p = 0.01
            if p > 0.99:
                p = 0.99
            
            act = np.random.choice(self.action_space, p=[1-p, p])
            
            if act == 0:
                self.X_list.append(X)
                self.Y_list.append(Y0)
                self.P_list.append(p)
                self.A_list.append(0)
                self.func_0.append(mu0)
                self.func_1.append(mu1)
            else:
                self.X_list.append(X)
                self.Y_list.append(Y1)
                self.P_list.append(p)
                self.A_list.append(1)
                self.func_0.append(mu0)
                self.func_1.append(mu1)

    def effect(self):        
        y = np.array(self.Y_list)
        x = np.array(self.X_list)
        a = np.array(self.A_list)
        p1 = np.array(self.P_list)
        p0 = 1 - p1
        
        y1 = y[a==1]
        y0 = y[a==0]
        
        N = len(x)
        
        model_y_a0, model_y_a1, model_y2_a0, model_y2_a1 = self.est_params(self.method)
        #model_y_a0.predict(self.X_list)
        mu0, mu1 = model_y_a0.predict(self.X_list), model_y_a1.predict(self.X_list)
                        
        z = a*(y-mu1)/p1 + mu1 - (1-a)*(y-mu0)/p0 - mu0
        
        theta = np.mean(y1[:self.first_phase]) - np.mean(y0[:self.first_phase])
        var = 2*np.mean((y1**2)[:self.first_phase]) + 2*np.mean((y0**2)[:self.first_phase]) - theta**2
        
        if N > self.first_phase:
            theta2 = np.mean(z[self.first_phase:])
            var2 = np.mean(((z)**2)[self.first_phase:])
            
            pi = self.first_phase/N
            theta = pi*theta + (1-pi)*theta2
            var = pi*var + (1-pi)*var2
                                                                                        
        return theta, var