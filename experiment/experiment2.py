import numpy as np
import matplotlib.pyplot as plt

from optimize import opt 
from sampler import *
from algorithms import *

from econml.data import dgps


def simulate2(trial, std0=0, std1=0, scenario='A'):    
    rct_res = []
    adpt0_res = []
    adpt1_res = []
    true_res = []
    
    for t in range(trial):
        if scenario == 'A':
            Y, T, X, true_ITE = dgps.ihdp_surface_A()
        elif scenario == 'B':
            Y, T, X, true_ITE = dgps.ihdp_surface_B()
        
        perm = np.random.permutation(len(Y))
        Y = Y[perm]
        T = T[perm]
        X = X[perm]
        
        Y1 = Y.copy()
        Y1[T==0] += true_ITE[T==0]
        Y1 += np.random.normal(0, std1, size=len(Y1))

        Y0 = Y.copy()
        Y0[T==1] -= true_ITE[T==1]
        Y0 += np.random.normal(0, std0, size=len(Y0))

        rct = RCT()
        adpt0 = Adapt(pretraining=10)
        adpt1 = Adapt(pretraining=100)
        opt0 = OPT()
        opt1 = OPT()

        rct_temp = []
        adpt0_temp = []
        adpt1_temp = []
        true_temp = []
        
        for period_t in range(len(Y)):            
            rct(period_t, X[period_t], Y0[period_t], Y1[period_t])
            adpt0(period_t, X[period_t], Y0[period_t], Y1[period_t])
            adpt1(period_t, X[period_t], Y0[period_t], Y1[period_t])
            
            if period_t > 2:
                rct_temp.append(rct.effect())
                adpt0_temp.append(adpt0.effect())
                adpt1_temp.append(adpt1.effect())
                true_temp.append(np.mean(true_ITE))
            
        rct_res.append(rct_temp)
        adpt0_res.append(adpt0_temp)
        adpt1_res.append(adpt1_temp)
        true_res.append(true_temp)

    return rct_res, adpt0_res, adpt1_res, true_res

def experiment2(trial, std0, std1, scenario):
    if scenario == 'A':
        y_scale = 2
    elif scenario == 'B':
        y_scale = 10
        
    rct_res, adpt0_res, adpt1_res, true_res= simulate2(trial, std0, std1, scenario)
    mu = true_res
        
    plt.figure(figsize=(13,7))
    
    mse0 = np.array((mu-np.array(rct_res))**2)
    mse1 = np.array((mu-np.array(adpt0_res))**2)
    mse2 = np.array((mu-np.array(adpt1_res))**2)

    mse_mean0 = np.mean(mse0[:, 1:], axis=0)
    q01 = np.quantile(mse0[:, 1:], 0.95, axis=0) 
    q02 = np.quantile(mse0[:, 1:], 0.05, axis=0)
    mse_mean1 = np.mean(mse1[:, 1:], axis=0)
    q11 = np.quantile(mse1[:, 1:], 0.95, axis=0) 
    q12 = np.quantile(mse1[:, 1:], 0.05, axis=0)
    mse_mean2 = np.mean(mse2[:, 1:], axis=0)
    q21 = np.quantile(mse2[:, 1:], 0.95, axis=0) 
    q22 = np.quantile(mse2[:, 1:], 0.05, axis=0)
    
    plt.ylim(-0, y_scale)
    plt.plot(mse_mean0, label="RCT")
    plt.fill_between(range(len(mse_mean0)), q01, q02, alpha=0.13)
    plt.plot(mse_mean1, label="Proposed Algorithm: ρ = 10")
    plt.fill_between(range(len(mse_mean1)), q11, q12, alpha=0.15)
    plt.plot(mse_mean2, label="Proposed Algorithm: ρ = 100")
    plt.fill_between(range(len(mse_mean2)), q21, q22, alpha=0.15)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("MSE", fontsize=20)
    plt.legend(fontsize=18)
    plt.tick_params(labelsize=20)
    
    if scenario == 'B':
        rct_res = np.array((mu-np.array(rct_res)))
        adpt0_res = np.array((mu-np.array(adpt0_res)))
        adpt1_res = np.array((mu-np.array(adpt1_res)))
    
    plt.savefig('exp_results/exp2%s_fig_trial%d_std0%d_std1%d'%(scenario, trial, std0, std1))
    np.save('exp_results/exp2%s_rct_trial%d_std0%d_std1%d'%(scenario, trial, std0, std1), rct_res)
    np.save('exp_results/exp2%s_adpt0_trial%d_std0%d_std1%d'%(scenario, trial, std0, std1), adpt0_res)
    np.save('exp_results/exp2%s_adpt1_trial%d_std0%d_std1%d'%(scenario, trial, std0, std1), adpt1_res)
    
    plt.show()