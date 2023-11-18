# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:13:43 2023

@author: Zunayed
"""

# %% S0. SETUP env

import numpy as np
from matplotlib import pyplot as plt


#%%   Save and Load image mask

####   Load mask_zero and 
#np.save('mask_zero.npy', mask_zero) 
mask_zero= np.load('mask_zero.npy')

#np.save('mask_NaN.npy', mask_NaN) 
mask_NaN= np.load('mask_NaN.npy')

#%% Load all previously savd  image data set

####  Load large input/ training image data set
marge_B0= np.load('marge_B0.npy')  # size(4096,20)
marge_B1= np.load('marge_B1.npy')  # size(4096,20)
marge_FID= np.load('marge_FID.npy')  # size(4096,20)
marge_STE= np.load('marge_STE.npy')  # size(4096,20)

###   Load large target/ phantom image data set
marge_P_B0= np.load('marge_P_B0.npy')  # size(4096,20)
marge_P_B1= np.load('marge_P_B1.npy')  # size(4096,20)

### Load  WASABI image data set
wasabi= np.load('wasabi_all.npy')
wasabi_B0=np.reshape(wasabi[:, 0], (64,64))*mask_zero
wasabi_B1=np.reshape(wasabi[:, 1], (64,64))*mask_zero

### Load VIVO image data set
vivo= np.load('vivo_all.npy')
vivo_B0_img=np.reshape(vivo[:, 0], (64,64))*mask_zero
vivo_B1_img=np.reshape(vivo[:, 1], (64,64))*mask_zero

#%% Prepare data set 

### define function to remove zero and concatenate matrix
def rev_0_marge1(all_marger_b0): 
    for i in range(all_marger_b0.shape[1]):
        b0_1= all_marger_b0[:, i]
        arr1=b0_1[b0_1.nonzero()]
        #arr1=np.reshape( arr1, (arr1.shape[0],1))
        if i==0:
            new_arr=arr1
        else: 
            new_arr=np.concatenate((new_arr, arr1), axis=0)
    return np.reshape(new_arr, (new_arr.shape[0],1))

# function: remove zero
def remove_zero(arr):
    arr1=arr[arr.nonzero()]
    return np.reshape(arr1, (arr1.shape[0],1))

### Remove zero vatue trainin/ input data
marge_B0_R=rev_0_marge1(marge_B0)   # size(35549,1)
marge_B1_R=rev_0_marge1(marge_B1)   # size(35549,1)
marge_FID_R=rev_0_marge1(marge_FID)   # size(35549,1)
marge_STE_R=rev_0_marge1(marge_STE)   # size(35549,1)

### Remove zero value from target data
marge_P_B0_R=rev_0_marge1(marge_P_B0)   # size(35549,1)
marge_P_B1_R=rev_0_marge1(marge_P_B1)   # size(35549,1)

### concatanate all input and target data:
X_all_0= np.concatenate((marge_B0_R, marge_B1_R, marge_FID_R, marge_STE_R), axis=1) # size(35549,4)
Y_all_0= np.concatenate((marge_P_B0_R, marge_P_B1_R), axis=1)    # size(35549,2)

### defind mean and standard value for all input and target data set:                          
mean_X_all= np.mean (X_all_0, axis=0, keepdims=True)  # size(1,4)
mean_Y_all= np.mean (Y_all_0, axis=0, keepdims=True)  # size(1,2)

std_X_all= np.std (X_all_0, axis=0, keepdims=True)   # size(1,4)
std_Y_all= np.std (Y_all_0, axis=0, keepdims=True)   # size(1,2)

#%%     ###   splite train and test image data

# multi_tarin data set, N=18
X_train_main18 = X_all_0[ :-1735*2 ,:]   # size(32079,4)
Y_train_main18 = Y_all_0[ :-1735*2 ,:]         # size(32079,2)
# Sinle train data set, N=1
X_train_n1=X_all_0[-1735*2:-1735 , :]      # size(1735,4)
Y_train_n1=Y_all_0[-1735*2:-1735 , :]      # size(1735,2)
# test data set, N=1
X_test_n1=X_all_0[-1735: , :]      # size(1735,4)
Y_test_n1=Y_all_0[-1735: , :]      # size(1735,2)


#%% Data Augmentation for large training data set 

#X_train_main18 = X_all_0[ :-1735*2 ,:]   # size(32079,4)
#Y_train_main18 = Y_all_0[ :-1735*2 ,:]         # size(32079,2)

# Function for data augmentation to make 20 to 200 image
import numpy as pynum_float
#p=pynum_float.arange(0, 2, 0.2)
p=[0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.2, 1.3, 1.5, 1.7]
n=10   # n defind how many image_set multiply with 20 images
def new_data(new_data):
    for i in range(n):
        data=np.copy(new_data)    
        for j in range(data.shape[0]):
             data[j, 0:2]=new_data[j, 0:2] *p[i]
        d1=data
        if i==0:
            d0= new_data
        else:
            d0=np.concatenate((d1,d0), axis=0)
    
    return d0  
X_train_n200 =  new_data(X_train_main18)        # size(320790,4)   make 10 times larger data
Y_train_n200 =  new_data(Y_train_main18)        # size(320790,2)   make 10 times larger data

#%%    Create weight for Single training data set, N=1  

X_train_n1_std=(X_train_n1- mean_X_all )/ std_X_all   # size(1735,4)
Y_train_n1_std=(Y_train_n1- mean_Y_all )/ std_Y_all    # size(1735,2)

X_intercept_n1 = np.ones((X_train_n1_std.shape[0], 1))                   
X_train_n1_bias= np.concatenate((X_intercept_n1, X_train_n1_std), axis=1)  # size(1735,5)

## apply psuedo inverse for Single training data

w_pinv_1= np.linalg.pinv(X_train_n1_bias)
weight_n1= (w_pinv_1 @ Y_train_n1_std)
#np.save('weight_n1.npy', weight_n1)



#%%    Create weight for Multi training data set, N=200 

X_train_n200_std=(X_train_n200- mean_X_all )/ std_X_all   # size(320790,4)
Y_train_n200_std=(Y_train_n200- mean_Y_all )/ std_Y_all    # size(320790,2)

X_intercept_n200 = np.ones((X_train_n200_std.shape[0], 1))                   
X_train_n200_bias= np.concatenate((X_intercept_n200, X_train_n200_std), axis=1)  # size(320790,5)

## apply psuedo inverse for Multi training data
w_pinv_200= np.linalg.pinv(X_train_n200_bias)
weight_n200= (w_pinv_200 @ Y_train_n200_std)
#np.save('weight_n200.npy', weight_n200)

#%%    Linear projection Model and  image reconstruction  function.  

def reconstruction(test, weight):  
    # prediction via weight vector
    Y_new = np.dot( test, weight  )      
    
    # reverse to original scale with mean_standard
    Y_new_r = (Y_new* std_Y_all) +  mean_Y_all
    
    Y_B0_new_r= np.reshape(Y_new_r[:, 0], (1735,1))  
    Y_B1_new_r= np.reshape(Y_new_r[:, 1], (1735,1))   
    #create last image mask
    last_image=marge_P_B0[:, 19]
    last_image_mask= np.reshape(last_image, (64,64))
    
    # add mask to prediction image
    def add_mask(Y_arr):
        mask_original= np.copy(last_image_mask)
        c=0
        reconst_arr= np.copy(Y_arr)
        for j in range(mask_zero.shape[0]):
            for i in range(mask_zero.shape[1]):
                    if mask_original[j , i ]!= 0:  
                        mask_original[j , i ] = reconst_arr[ c,:]
                        c+= 1
                    else: mask_original[j , i ] = 0
        return mask_original
    
    # each columns of 'Y_B0_new_r' is one image, size(64,64)                
    Y_B0_recon= add_mask(Y_B0_new_r) 
    Y_B1_recon= add_mask(Y_B1_new_r)
    
    return Y_B0_recon,Y_B1_recon


#%% Prepare Single Test data set  

#X_test_n1=X_all_0[-1735: , :]      # size(1735,4)
#Y_test_n1=Y_all_0[-1735: , :]      # size(1735,2)

X_test_n1_std=(X_test_n1- mean_X_all )/ std_X_all
Y_test_n1_std=(Y_test_n1- mean_Y_all )/ std_Y_all

X_intercept_test = np.ones((X_test_n1_std.shape[0], 1))                   
X_test_n1_bias= np.concatenate((X_intercept_test, X_test_n1_std), axis=1)  # size(1735,5)


#%%   Run Linear projection Model for Dream_simulation Single Test data set  

##   pass test_Dream simulation to the Model for Weight_train N=1
Y_B0_recon_1n, Y_B1_recon_1n= reconstruction( X_test_n1_bias, weight_n1)
##   pass test_Dream simulation to the Model for weight_train N=200 
Y_B0_recon_200, Y_B1_recon_200= reconstruction(X_test_n1_bias, weight_n200)


#%%    Create extrem test case-->Run Linear projection model  
X_ex_data= np.copy(X_test_n1)
X_ex_data1= np.copy(X_ex_data)

for i in range (X_ex_data1.shape[1]):
    X_ex_data1[:, i]*=1.5

X_ex_data1_std=(X_ex_data1- mean_X_all )/ std_X_all

X_ex_data1_intercept = np.ones((X_ex_data1_std.shape[0], 1))                   
X_ex_data1_bias= np.concatenate((X_ex_data1_intercept, X_ex_data1_std), axis=1)  # size(1735,5)

## Run Linear projection model for extrem case
Y_B0_recon_1ex, Y_B1_recon_1ex= reconstruction( X_ex_data1_bias, weight_n1)
##   pass test_Dream simulation to the Model for weight_train N=200 
Y_B0_recon_200ex, Y_B1_recon_200ex= reconstruction(X_ex_data1_bias, weight_n200)


#%%       Run Linear projection for Vivo test case 

wasabi= np.load('wasabi_all.npy')
wasabi_B0=np.reshape(wasabi[:, 0], (64,64))*mask_zero
wasabi_B1=np.reshape(wasabi[:, 1], (64,64))*mask_zero


vivo= np.load('vivo_all.npy')
vivo_B0_img=np.reshape(vivo[:, 0], (64,64))*mask_zero
vivo_B1_img=np.reshape(vivo[:, 1], (64,64))*mask_zero

vivo_B0=remove_zero(vivo_B0_img)
vivo_B1=remove_zero(vivo_B1_img)
vivo_FID=remove_zero(np.reshape(vivo[:, 2], (64,64))*mask_zero)
vivo_STE=remove_zero(np.reshape(vivo[:, 3], (64,64))*mask_zero)
vivo_all= np.concatenate((vivo_B0, vivo_B1, vivo_FID, vivo_STE), axis=1)

vivo_test=(vivo_all - mean_X_all)/ std_X_all    #size(1735,4)
test_bias = np.ones((vivo_test.shape[0], 1))                    
vivo_bias= np.concatenate((test_bias, vivo_test), axis=1)


#  Run the model to Vivo image data
Y_B0_recon_1v, Y_B1_recon_1v= reconstruction( vivo_bias, weight_n1)
Y_B0_recon_200v, Y_B1_recon_200v= reconstruction(vivo_bias, weight_n200)

mask_NaN_vivo=np.load('New_mask_NaN_vivo.npy')   # for test vivo mask
mask_zero_vivo=np.load('New_mask_zero_vivo.npy')

#%%   Scatter plot for Dream simulation  test image 

### import scatter plot function
from scatter_plot import B01_scatter_plot, B1_scatter_plot, B11_scatter_plot, B0_scatter_plot

####### scatter plot for test= Dream and Train,N=1 vs Train, N200  ##########################  
fig=plt.figure(figsize=(20,10));
plt.suptitle('{test: simulation},Train:{Dream_simulation}-->prediction:{simu_B0,B1}, compare{simulation vs phantom} ', fontsize=18)

# # B0 before projection 
plt.subplot(231); plt.title('befor projection:simulation_B0', fontsize=10)
B01_scatter_plot( marge_B0[:,19], marge_P_B0[:,19],  "Dream_Simulation_B0", "target_phantom", )
# B0 normal
plt.subplot(232); plt.title('After proj:Test=simulation, Train:Dream_simu_N=1', fontsize=14)
B01_scatter_plot(np.reshape(Y_B0_recon_1n, (4096,1)), marge_P_B0[:,19], "Prediction_simulation_B0", "target_phantom", )
# Train =200, 
plt.subplot(233); plt.title('After proj:Test=simulation, Train:Dream_simu_N=200', fontsize=14)
B01_scatter_plot(  np.reshape(Y_B0_recon_200, (4096,1)), marge_P_B0[:,19],  "Prediction_simulation_B0", "target_phantom", )


# # B1 before projection 
plt.subplot(234); plt.title('befor projection:simulation_B1', fontsize=14)
B1_scatter_plot(  marge_B1[:,19], marge_P_B1[:,19],  "Dream_Simulation_B1", "target_phantom",)
# B1 normal
plt.subplot(235); plt.title('After proj:Test=simulation, Train:Dream_simu_N=1', fontsize=14)
B11_scatter_plot( np.reshape(Y_B1_recon_1n, (4096,1)), marge_P_B1[:,19],  "Prediction_simulation_B1", "target_phantom", )
# Train =200, 
plt.subplot(236); plt.title('After proj:Test=simulation, Train:Dream_simu_N=200', fontsize=14)
B11_scatter_plot( np.reshape(Y_B1_recon_200, (4096,1)), marge_P_B1[:,19],   "Prediction_simulation_B1" , "target_phantom",)


#%%  Extrem test for Dream Simulation immage 

### import scatter plot function
from scatter_plot import B1_scatter_plot,  B0ex_scatter_plot, B1ex_scatter_plot

####### extrem test simulation and Train,N=1 vs Train, N200  ##########################  
fig=plt.figure(figsize=(20,10));
plt.suptitle('{test:simulationX1.5},Train:{Dream_simulation}-->prediction:{simu_B0,B1}, compare{simulation vs phantom} ', fontsize=18)

# # B0 before projection 
plt.subplot(241); plt.title('Befor projection:Normal_simulation_B0', fontsize=12)
B0ex_scatter_plot( marge_B0[:,19],  marge_P_B0[:,19],   "Dream_Simulation_B0", "target_phantom_B0")

#target=B0 X1.5
target=np.reshape(marge_P_B0[:,19],(4096,1))
ex_tar_B0=np.copy(target)    
for i in range(ex_tar_B0.shape[0]):
     ex_tar_B0[i, :]=target[i, :] *1.5
     
#test=B0 X1.5
test_b0=np.reshape(marge_B0[:,19],(4096,1))
ex_test_B0=np.copy(test_b0)    
for i in range(ex_test_B0.shape[0]):
     ex_test_B0[i, :]=test_b0[i, :] *1.5

#test=B0 X1.5  
plt.subplot(242); plt.title('Befor proj: Extrem_condition_B0', fontsize=12)
B0ex_scatter_plot( ex_test_B0, ex_tar_B0, "extrem_Test= B0x 1.5", "extrem_target= B0x1.5")
# Train =1, extrem prediction
plt.subplot(243); plt.title('After proj:Test=extrem_B0x1.5, train_N=1', fontsize=12)
B0ex_scatter_plot( np.reshape(Y_B0_recon_1ex, (4096,1)), ex_tar_B0,  "extrem_Prediction_B0", "extrem_target= B0x1.5" )
# Train =200, extrem prediction
plt.subplot(244); plt.title('After proj:Test=extrem_B0x1.5, train_N=200', fontsize=12)
B0ex_scatter_plot( np.reshape(Y_B0_recon_200ex, (4096,1)), ex_tar_B0,  "extrem_Prediction_B0", "extrem_target= B0x1.5" )


# # B1 before projection ###  B1#####
plt.subplot(245); plt.title('Befor projection:Normal_simulation_B1', fontsize=12)
B1_scatter_plot(  marge_B1[:,19], marge_P_B1[:,19],   "Dream_Simulation_B1", "target_phantom_B1" )

#target=B1 X1.5
target_b1=np.reshape(marge_P_B1[:,19],(4096,1))
ex_tar_B1=np.copy(target_b1)    
for i in range(ex_tar_B1.shape[0]):
     ex_tar_B1[i, :]=target_b1[i, :] *1.5
     
#test=B1 X1.5
test_b1=np.reshape(marge_B1[:,19],(4096,1))
ex_test_B1=np.copy(test_b1)    
for i in range(ex_test_B1.shape[0]):
     ex_test_B1[i, :]=test_b1[i, :] *1.5    

#test=B1 X1.5    
plt.subplot(246); plt.title('Befor proj: Extrem_condition_B1', fontsize=12)
B1ex_scatter_plot(ex_test_B1, ex_tar_B1, "extrem_test= B1x 1.5", "extrem_target= B1x1.5" )
# Train =1, extrem prediction
plt.subplot(247); plt.title('After proj:Test=extrem_B1x1.5, train_N=1', fontsize=12)
B1ex_scatter_plot(np.reshape(Y_B1_recon_1ex, (4096,1)), ex_tar_B1,  "extrem_Prediction_B1", "extrem_target= B1x1.5")
# Train =200, extrem prediction
plt.subplot(248); plt.title('After proj:Test=extrem_B0x1.5, train_N=200', fontsize=12)
B1ex_scatter_plot(np.reshape(Y_B1_recon_200ex, (4096,1)), ex_tar_B1,  "extrem_Prediction_B1", "extrem_target= B1x1.5")


#%%   Scatter plot for ViVO test image 

### import scatter plot function
from scatter_plot import B01_scatter_plot, B1_scatter_plot, B11_scatter_plot

fig=plt.figure(figsize=(20,10));
plt.suptitle('{test:vivo}, Train:{Dream_simulation},{target:Phantom}-->predict_vivo:{B0, B1}, compare:{pred_vivo vs wasabi}', fontsize=18)

# # B0 before projection 
plt.subplot(231); plt.title('befor projection:Vivo_B0', fontsize=12)
B0_scatter_plot( np.reshape(wasabi_B0*mask_zero_vivo, (4096,1)), np.reshape(vivo_B0_img*mask_zero_vivo, (4096,1)),  "wasabi", "vivo_B0" )
# B0 normal
plt.subplot(232); plt.title('After proj:Test=Vivo, Train:Dream_simu_N=1', fontsize=14)
B01_scatter_plot( np.reshape(wasabi_B0*mask_zero_vivo, (4096,1)), np.reshape(Y_B0_recon_1v*mask_zero_vivo, (4096,1)),  "wasabi", "Prediction_vivo_B0" )
# B0 X1.5
# Train =200, test=B0 X1.5
plt.subplot(233); plt.title('After proj:Test=Vivo, Train:Dream_simu_N=200', fontsize=14)
B01_scatter_plot(np.reshape(wasabi_B0*mask_zero_vivo, (4096,1)),  np.reshape(Y_B0_recon_200v*mask_zero_vivo, (4096,1)),  "wasabi", "Prediction_vivo_B0" )



# # B1 before projection 
plt.subplot(234); plt.title('befor projection:Vivo_B1', fontsize=12)
B1_scatter_plot( np.reshape(wasabi_B1*mask_zero_vivo, (4096,1)), np.reshape(vivo_B1_img*mask_zero_vivo, (4096,1)),  "wasabi", "vivo_B1" )
# B1 normal
plt.subplot(235); plt.title('After proj:Test=Vivo, Train:Dream_simu_N=1', fontsize=14)
B11_scatter_plot( np.reshape(wasabi_B1*mask_zero_vivo, (4096,1)), np.reshape(Y_B1_recon_1v*mask_zero_vivo, (4096,1)),  "wasabi", "Prediction_vivo_B1" )
# Train =200, 
plt.subplot(236); plt.title('After proj:Test=Vivo, Train:Dream_simu_N=200', fontsize=14)
B11_scatter_plot(np.reshape(wasabi_B1*mask_zero_vivo, (4096,1)),  np.reshape(Y_B1_recon_200v*mask_zero_vivo, (4096,1)),  "wasabi", "Prediction_vivo_B1" )

        


#%%  Image visualize


# Image plot= weight Single train and test Vivo   ##############################################
# compare target and prediction########################
fig=plt.figure(figsize=(18,8));
plt.suptitle('N1_Train:{simulation},{target:Phantom},{test:vivo},-->predict_vivo:{B0, B1},compare:{pred_vivo vs wasabi}', fontsize=18)
# input vivo

def plot_image(img, vmin, vmax):
    plt.imshow( img, vmin=vmin, vmax=vmax, origin="lower") 
    plt.axis('off')
    plt.colorbar()
    
plt.subplot(261); plt.title('train{Simulation_B0}', fontsize=13)
plot_image( np.reshape(marge_B0[:, 18], (64,64)) , -45,  45 )

plt.subplot(267); plt.title(' train{Simulation_B1}', fontsize=13) 
plot_image( np.reshape(marge_B1[:, 18], (64,64)) , 0,  1.2  )

plt.subplot(262); plt.title('Target{Phantom_B0}', fontsize=13)
plot_image(np.reshape(marge_P_B0[:, 18], (64,64)) ,  -45,  45 )

plt.subplot(268); plt.title('Target{Phantom_B1}', fontsize=13)
plot_image( np.reshape(marge_P_B1[:, 18], (64,64)),    0,  1.2    )

plt.subplot(263); plt.title('test{vivo_B0}', fontsize=13)
plot_image( vivo_B0_img*mask_NaN_vivo ,  -45,  45 )

plt.subplot(269); plt.title('test{vivo_B1}', fontsize=13) 
plot_image(vivo_B1_img*mask_NaN_vivo,  0,  1.2  )

plt.subplot(264); plt.title('Predict{vivo_B0}', fontsize=13)
plot_image( Y_B0_recon_1v*mask_NaN_vivo, -45,  45  )

plt.subplot(2,6,10); plt.title('Predict{vivo_B1}', fontsize=13) 
plot_image( Y_B1_recon_1v*mask_NaN_vivo,  0,  1.2  )

plt.subplot(265); plt.title('{ wasabi B0}', fontsize=13)
plot_image( wasabi_B0*mask_NaN_vivo,-45,  45 )
 
plt.subplot(2,6,11); plt.title('{ wasabi B1}', fontsize=13) 
plot_image(wasabi_B1*mask_NaN_vivo,  0,  1.2 )

def residual(arr1, arr2):
    res=np.reshape(arr1, (64,64)) -np.reshape(arr2, (64,64))
    return res

# residual
plt.subplot(266); plt.title('residual{wasabi-pred}', fontsize=8)
plot_image( residual(wasabi_B0, Y_B0_recon_1v)*mask_NaN_vivo , -20, 20  )

plt.subplot(2,6,12); plt.title('residual{wasabi-pred}', fontsize=8)
plot_image( residual(wasabi_B1, Y_B1_recon_1v)*mask_NaN_vivo , -0.6, 0.6  )


# Image plot= weight 200 Multi train and test Vivo   ##############################################
# compare target and prediction########################
fig=plt.figure(figsize=(18,8));
plt.suptitle('N200_Train:{simulation},{target:Phantom},{test:vivo}-->predict_vivo:{B0, B1}, compare:{pred_vivo vs wasabi}', fontsize=18)

def plot_image(img, vmin, vmax):
    plt.imshow( img, vmin=vmin, vmax=vmax, origin="lower") 
    plt.axis('off')
    plt.colorbar()

plt.subplot(261); plt.title('train{Simulation_B0}', fontsize=13)
plot_image( np.reshape(marge_B0[:, 18], (64,64)) , -45,45 )

plt.subplot(267); plt.title(' train{Simulation_B1}', fontsize=13) 
plot_image( np.reshape(marge_B1[:, 18], (64,64)) , 0, 1.2 )

plt.subplot(262); plt.title('Target{Phantom_B0}', fontsize=13)
plot_image( np.reshape(marge_P_B0[:, 18], (64,64)) , -45,45 )

plt.subplot(268); plt.title('Target{Phantom_B1}', fontsize=13)
plot_image( np.reshape(marge_P_B1[:, 18], (64,64)), 0, 1.2 )

plt.subplot(263); plt.title('test{vivo_B0}', fontsize=13)
plot_image( vivo_B0_img*mask_NaN_vivo, -45, 45 )

plt.subplot(269); plt.title('test{vivo_B1}', fontsize=13) 
plot_image(  vivo_B1_img*mask_NaN_vivo, 0, 1.2 )

plt.subplot(264); plt.title('Predict{vivo_B0}', fontsize=13)
plot_image( Y_B0_recon_200v*mask_NaN_vivo, -45, 45 )

plt.subplot(2,6,10); plt.title('Predict{vivo_B1}', fontsize=13) 
plot_image(  Y_B1_recon_200v*mask_NaN_vivo, 0, 1.2 )

plt.subplot(265); plt.title('{ wasabi B0}', fontsize=13)
plot_image(  wasabi_B0*mask_NaN_vivo, -45, 45 )

plt.subplot(2,6,11); plt.title('{ wasabi B1}', fontsize=13) 
plot_image( wasabi_B1*mask_NaN_vivo, 0, 1.2 )


def residual(arr1, arr2):
    res=np.reshape(arr1, (64,64)) -np.reshape(arr2, (64,64))
    return res

# residual
plt.subplot(266); plt.title('residual{wasabi-pred}', fontsize=8)
plot_image(  residual(wasabi_B0, Y_B0_recon_200v)*mask_NaN_vivo , -20, 20 )

plt.subplot(2,6,12); plt.title('residual{wasabi-pred}', fontsize=8)
plot_image( residual(wasabi_B1, Y_B1_recon_200v)*mask_NaN_vivo , -0.6, 0.6  )




