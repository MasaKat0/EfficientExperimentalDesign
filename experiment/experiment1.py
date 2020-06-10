import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tools.tools import *

from optimize import opt 
from sampler import *
from algorithms import *


def simulate1(T, trial, mu0, mu1, std0, std1):    
    rct_res = []
    adpt0_res = []
    adpt1_res = []
    opt0_res = []
    opt1_res = []
    
    for t in range(trial):
        rct = RCT()
        adpt0 = Adapt(pretraining=10)
        adpt1 = Adapt(pretraining=50)
        opt0 = OPT()
        opt1 = OPT()

        rct_temp = []
        adpt0_temp = []
        adpt1_temp = []
        opt0_temp = []
        opt1_temp = []
        
        for period_t in range(T):            
            X, Y0, Y1, EY0_2, EY1_2, Var0, Var1 = sampler(mu0, mu1, std0, std1)
            rct(period_t, X, Y0, Y1)
            adpt0(period_t, X, Y0, Y1)
            adpt1(period_t, X, Y0, Y1)
            opt0(period_t, X, Y0, Y1, EY0_2, EY1_2)
            opt1(period_t, X, Y0, Y1, Var0, Var1)
            
            if period_t > 2:
                rct_temp.append(rct.effect())
                adpt0_temp.append(adpt0.effect())
                adpt1_temp.append(adpt1.effect())
                opt0_temp.append(opt0.effect())
                opt1_temp.append(opt1.effect(estimate=True))
            
        rct_res.append(rct_temp)
        adpt0_res.append(adpt0_temp)
        adpt1_res.append(adpt1_temp)
        opt0_res.append(opt0_temp)
        opt1_res.append(opt1_temp)

    return rct_res, adpt0_res, adpt1_res, opt0_res, opt1_res

def experiment1(T, trial, mu0, mu1, std0, std1):
    rct_res, adpt0_res, adpt1_res, opt0_res, opt1_res = simulate1(T, trial, mu0, mu1, std0, std1)
    mu = mu1 - mu0
        
    plt.figure(figsize=(13,7))
    
    mse0 = np.array((mu-np.array(rct_res))**2)
    mse1 = np.array((mu-np.array(adpt0_res))**2)
    mse2 = np.array((mu-np.array(adpt1_res))**2)
    mse3 = np.array((mu-np.array(opt0_res))**2)
    mse4 = np.array((mu-np.array(opt1_res))**2)

    mse_mean0 = np.mean(mse0[:, 1:], axis=0)
    q01 = np.quantile(mse0[:, 1:], 0.95, axis=0) 
    q02 = np.quantile(mse0[:, 1:], 0.05, axis=0)
    mse_mean1 = np.mean(mse1[:, 1:], axis=0)
    q11 = np.quantile(mse1[:, 1:], 0.95, axis=0) 
    q12 = np.quantile(mse1[:, 1:], 0.05, axis=0)
    mse_mean2 = np.mean(mse2[:, 1:], axis=0)
    q21 = np.quantile(mse2[:, 1:], 0.95, axis=0) 
    q22 = np.quantile(mse2[:, 1:], 0.05, axis=0)
    mse_mean3 = np.mean(mse3[:, 1:], axis=0)
    q31 = np.quantile(mse3[:, 1:], 0.95, axis=0) 
    q32 = np.quantile(mse3[:, 1:], 0.05, axis=0)
    mse_mean4 = np.mean(mse4[:, 1:], axis=0)
    q41 = np.quantile(mse4[:, 1:], 0.95, axis=0) 
    q42 = np.quantile(mse4[:, 1:], 0.05, axis=0)
    
    plt.ylim(-0, 1000)
    plt.plot(mse_mean0, label="RCT")
    plt.fill_between(range(len(mse_mean0)), q01, q02, alpha=0.13)
    plt.plot(mse_mean1, label="Proposed Algorithm: ρ = 10")
    plt.fill_between(range(len(mse_mean1)), q11, q12, alpha=0.15)
    plt.plot(mse_mean2, label="Proposed Algorithm: ρ = 50")
    plt.fill_between(range(len(mse_mean2)), q21, q22, alpha=0.15)
    plt.plot(mse_mean3, label="Dependent-OPT")
    plt.fill_between(range(len(mse_mean3)), q31, q32, alpha=0.17)
    plt.plot(mse_mean4, label="IID-OPT")
    plt.fill_between(range(len(mse_mean4)), q41, q42, alpha=0.20)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("MSE", fontsize=20)
    plt.legend(fontsize=18)
    plt.tick_params(labelsize=20)
    
    plt.savefig('exp_results/exp1_fig_T%d_trial%d_mu0%d_mu1%d_std0%d_std1%d'%(T, trial, mu0, mu1, std0, std1))
    np.save('exp_results/exp1_rct_T%d_trial%d_mu0%d_mu1%d_std0%d_std1%d'%(T, trial, mu0, mu1, std0, std1), rct_res)
    np.save('exp_results/exp1_adpt0_T%d_trial%d_mu0%d_mu1%d_std0%d_std1%d'%(T, trial, mu0, mu1, std0, std1), adpt0_res)
    np.save('exp_results/exp1_adpt1_T%d_trial%d_mu0%d_mu1%d_std0%d_std1%d'%(T, trial, mu0, mu1, std0, std1), adpt1_res)
    np.save('exp_results/exp1_opt0_T%d_trial%d_mu0%d_mu1%d_std0%d_std1%d'%(T, trial, mu0, mu1, std0, std1), opt0_res)
    np.save('exp_results/exp1_opt1_T%d_trial%d_mu0%d_mu1%d_std0%d_std1%d'%(T, trial, mu0, mu1, std0, std1), opt1_res)
    
    plt.show()
