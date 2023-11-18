# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:13:43 2023

@author: Zunayed
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress


# function: remove zero
def remove_zero(arr):
    arr1=arr[arr.nonzero()]
    return np.reshape(arr1, (arr1.shape[0],1))


# call function for scatter plot   ########################### ##############
    
def B0_scatter_plot( pred_1, tar_1, X_l , Y_l):
    from sklearn.metrics import mean_squared_error
    import math
    pred_12= remove_zero(pred_1)
    tar_12= remove_zero(tar_1)
    RMSE = math.sqrt(mean_squared_error( pred_12, tar_12))
    #Mean_NRMSE= RMSE/pred_12.std()*100
    
    residual= np.sum(abs(tar_12 - pred_12))  # np.sum(abs(tar_1-pred_1))
    plt.text(20, -39,  f'residual: {residual:.3f} ', fontsize='medium', bbox={'facecolor': 'gray', 'pad': 5})
    
    plt.text(20, -49,  f'Loss_RMSE: {RMSE:.5f} ', fontsize='medium', bbox={'facecolor': 'gray', 'pad': 5})
    #plt.text(23, -48,  f' NRMSE: {Mean_NRMSE:.3f} %', fontsize='large', bbox={'facecolor': 'gray', 'pad': 5})
    
    n = np.linspace(-55,55, 111)
    slope121, intercept121, _, _, _ = linregress(np.reshape((pred_1),4096), np.reshape(tar_1, 4096)) 
    
    plt.plot(np.reshape((pred_1),4096),np.reshape(tar_1,4096),'x', label='data')
    plt.plot(n, n, color='0.7', linestyle='--', label='target/optimal fit (y=X)')
    plt.plot(n, slope121*n+intercept121, 'r', label='linear projection fit')
    plt.xlabel(X_l, fontsize=10)
    plt.ylabel(Y_l, fontsize=10)
    plt.xlim(-55,55)
    plt.ylim(-55,55)
    plt.grid()
    #plt.xticks([])
    #plt.yticks([])
    #plt.legend(fontsize=8)
    
def B01_scatter_plot( pred_1, tar_1, X_l , Y_l):
    from sklearn.metrics import mean_squared_error
    import math
    pred_12= remove_zero(pred_1)
    tar_12= remove_zero(tar_1)
    RMSE = math.sqrt(mean_squared_error( pred_12, tar_12))
    #Mean_NRMSE= RMSE/pred_12.std()*100
    
    residual= np.sum(abs(tar_12 - pred_12))  # np.sum(abs(tar_1-pred_1))
    plt.text(20, -39,  f'residual: {residual:.3f} ', fontsize='medium', bbox={'facecolor': 'gray', 'pad': 5})
    
    plt.text(20, -49,  f'Loss_RMSE: {RMSE:.5f} ', fontsize='medium', bbox={'facecolor': 'gray', 'pad': 5})
    #plt.text(23, -48,  f' NRMSE: {Mean_NRMSE:.3f} %', fontsize='large', bbox={'facecolor': 'gray', 'pad': 5})
    
    n = np.linspace(-55,80, 111)
    slope121, intercept121, _, _, _ = linregress(np.reshape((pred_1),4096), np.reshape(tar_1, 4096)) 
    
    plt.plot(np.reshape((pred_1),4096),np.reshape(tar_1,4096),'x', label='data')
    plt.plot(n, n, color='0.7', linestyle='--', label='target/optimal fit (y=X)')
    plt.plot(n, slope121*n+intercept121, 'r', label='linear projection fit')
    plt.xlabel(X_l, fontsize=10)
    plt.ylabel(Y_l, fontsize=10)
    plt.xlim(-55,55)
    plt.ylim(-55,55)
    plt.grid()
    #plt.xticks([])
    #plt.yticks([])
    #plt.legend(fontsize=8)
    
def B1_scatter_plot(X_dream, tar_1, X_l , Y_l):
    from sklearn.metrics import mean_squared_error
    import math
    X_dream12= remove_zero(X_dream)
    tar_12= remove_zero(tar_1)
    RMSE = math.sqrt(mean_squared_error(X_dream12, tar_12))
    #Mean_NRMSE= RMSE/X_dream12.std()*100
    
    residual= np.sum(abs( tar_12- X_dream12 ))# np.sum(abs(tar_1-pred_1))
    plt.text(1.15, 0.80,  f'residual: {residual:.3f} ', fontsize='medium', bbox={'facecolor': 'gray', 'pad': 5})
    
    plt.text(1.15, 0.75,  f'Loss_RMSE: {RMSE:.5f} ', fontsize='medium', bbox={'facecolor': 'gray', 'pad': 5})
    #plt.text(1.3, 0.64,  f' NRMSE: {Mean_NRMSE:.3f} %', fontsize='large', bbox={'facecolor': 'gray', 'pad': 5})
    
    n = np.linspace(0.7, 1.4, 121)
    slope221, intercept221, _, _, _ = linregress(np.reshape(X_dream, 4096),np.reshape(tar_1, 4096))

    plt.plot(np.reshape(X_dream, 4096),np.reshape(tar_1, 4096),'x', label='data')
    #plt.plot(n, n, color='0.7', linestyle='--', label='target/optimal fit(y=X)')
    plt.plot(n, slope221*n+intercept221, 'r', label='DREAM created linear fit')
    plt.xlabel(X_l, fontsize=10)
    plt.ylabel(Y_l, fontsize=10)
    plt.xlim(0.7,1.4)
    plt.ylim(0.7,1.4)
    plt.grid()
    #plt.xticks([])
    #plt.yticks([])
   # plt.legend(fontsize=10)
   
def B11_scatter_plot(X_dream, tar_1, X_l , Y_l):
    from sklearn.metrics import mean_squared_error
    import math
    X_dream12= remove_zero(X_dream)
    tar_12= remove_zero(tar_1)
    RMSE = math.sqrt(mean_squared_error(X_dream12, tar_12))
    #Mean_NRMSE= RMSE/X_dream12.std()*100
    
    residual= np.sum(abs( tar_12- X_dream12 ))# np.sum(abs(tar_1-pred_1))
    plt.text(1.15, 0.80,  f'residual: {residual:.3f} ', fontsize='medium', bbox={'facecolor': 'gray', 'pad': 5})
    
    plt.text(1.15, 0.75,  f'Loss_RMSE: {RMSE:.5f} ', fontsize='medium', bbox={'facecolor': 'gray', 'pad': 5})
    #plt.text(1.3, 0.64,  f' NRMSE: {Mean_NRMSE:.3f} %', fontsize='large', bbox={'facecolor': 'gray', 'pad': 5})
    
    n = np.linspace(0.7, 1.4, 121)
    slope221, intercept221, _, _, _ = linregress(np.reshape(X_dream, 4096),np.reshape(tar_1, 4096))

    plt.plot(np.reshape(X_dream, 4096),np.reshape(tar_1, 4096),'x', label='data')
    plt.plot(n, n, color='0.7', linestyle='--', label='target/optimal fit(y=X)')
    plt.plot(n, slope221*n+intercept221, 'r', label='DREAM created linear fit')
    plt.xlabel(X_l, fontsize=10)
    plt.ylabel(Y_l, fontsize=10)
    plt.xlim(0.7,1.4)
    plt.ylim(0.7,1.4)
    plt.grid()
    #plt.xticks([])
    #plt.yticks([])
   # plt.legend(fontsize=10)

# extrem test
def B0ex_scatter_plot( pred_1, tar_1, X_l , Y_l):
    from sklearn.metrics import mean_squared_error
    import math
    pred_12= remove_zero(pred_1)
    tar_12= remove_zero(tar_1)
    RMSE = math.sqrt(mean_squared_error( pred_12, tar_12))
    #Mean_NRMSE= RMSE/pred_12.std()*100
    
    residual= np.sum(abs(tar_12 - pred_12))  # np.sum(abs(tar_1-pred_1))
    plt.text(30, -10,  f'residual: {residual:.3f} ', fontsize='medium', bbox={'facecolor': 'gray', 'pad': 5})
    
    plt.text(30, -22,  f'Loss_RMSE: {RMSE:.5f} ', fontsize='medium', bbox={'facecolor': 'gray', 'pad': 5})
    #plt.text(23, -48,  f' NRMSE: {Mean_NRMSE:.3f} %', fontsize='large', bbox={'facecolor': 'gray', 'pad': 5})
    
    n = np.linspace(-30,80, 111)
    slope121, intercept121, _, _, _ = linregress(np.reshape((pred_1),4096), np.reshape(tar_1, 4096)) 
    
    plt.plot(np.reshape((pred_1),4096),np.reshape(tar_1,4096),'x', label='data')
    #plt.plot(n, n, color='0.7', linestyle='--', label='target/optimal fit (y=X)')
    plt.plot(n, slope121*n+intercept121, 'r', label='linear projection fit')
    plt.xlabel(X_l, fontsize=10)
    plt.ylabel(Y_l, fontsize=10)
    plt.xlim(-30,80)
    plt.ylim(-30,80)
    plt.grid()
    #plt.xticks([])
    #plt.yticks([])
    #plt.legend(fontsize=8)     
    
def B1ex_scatter_plot(X_dream, tar_1, X_l , Y_l):
    from sklearn.metrics import mean_squared_error
    import math
    X_dream12= remove_zero(X_dream)
    tar_12= remove_zero(tar_1)
    RMSE = math.sqrt(mean_squared_error(X_dream12, tar_12))
    #Mean_NRMSE= RMSE/X_dream12.std()*100
    
    residual= np.sum(abs( tar_12- X_dream12 ))# np.sum(abs(tar_1-pred_1))
    plt.text(1.65, 1.3,  f'residual: {residual:.3f} ', fontsize='medium', bbox={'facecolor': 'gray', 'pad': 5})
    
    plt.text(1.65, 1.2,  f'Loss_RMSE: {RMSE:.5f} ', fontsize='medium', bbox={'facecolor': 'gray', 'pad': 5})
    #plt.text(1.3, 0.64,  f' NRMSE: {Mean_NRMSE:.3f} %', fontsize='large', bbox={'facecolor': 'gray', 'pad': 5})
    
    n = np.linspace(1.1, 2.0, 121)
    slope221, intercept221, _, _, _ = linregress(np.reshape(X_dream, 4096),np.reshape(tar_1, 4096))

    plt.plot(np.reshape(X_dream, 4096),np.reshape(tar_1, 4096),'x', label='data')
    #plt.plot(n, n, color='0.7', linestyle='--', label='target/optimal fit(y=X)')
    plt.plot(n, slope221*n+intercept221, 'r', label='DREAM created linear fit')
    plt.xlabel(X_l, fontsize=10)
    plt.ylabel(Y_l, fontsize=10)
    plt.xlim(1.1, 2.0)
    plt.ylim(1.1,  2.0)
    plt.grid()
    #plt.xticks([])
    #plt.yticks([])
   # plt.legend(fontsize=10)
    