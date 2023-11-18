# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:13:43 2023

@author: Zunayed
"""
# %% S0. SETUP env

import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn

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

#%% Splite data set for training, testing and validation

### splite data set  
X_train_main17 = X_all_0[ :-1735*3 ,:]      # size(30344,4)
Y_train_main17 = Y_all_0[ :-1735*3 ,:]      # size(30344,2)
# Validation data set, Image_set_N=2
X_train_val=X_all_0[-1735*3:-1735 , :]      # size(3470,4)
Y_train_val=Y_all_0[-1735*3:-1735 , :]      # size(3470,2)
# test data set, Image_Ste_N=1
X_test_n1=X_all_0[-1735: , :]               # size(1735,4)
Y_test_n1=Y_all_0[-1735: , :]            # size(1735,2)

#%% Data Augmentation for large training data set 

# Function for data augmentation 
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

### Augment data to make 20 to 200 image
X_train_NN200 =  new_data(X_train_main17)        # size(303440,4)   make 10 times larger data
Y_train_NN200 =  new_data(Y_train_main17)        # size(303440,2)   make 10 times larger data

#%% Normalizetraining data set

X_train_NN200_std=(X_train_NN200- mean_X_all )/ std_X_all   # size(303440,4)
Y_train_NN200_std=(Y_train_NN200- mean_Y_all )/ std_Y_all    # size(303440,2)

X_intercept_NN200 = np.ones((X_train_NN200_std.shape[0], 1))                   
X_train_NN200_bias= np.concatenate((X_intercept_NN200, X_train_NN200_std), axis=1)  # size(303440,5)

### final train data: convert numpy to torch 
X_train_torch = torch.from_numpy(X_train_NN200_bias)       # size(303440,5)     imput
Y_train_torch = torch.from_numpy(Y_train_NN200_std)          # size(303440,2))   target

#%% Normalize Vatidation data set

X_train_val_std=(X_train_val- mean_X_all )/ std_X_all    # size(3470,4)
Y_train_val_std=(Y_train_val - mean_Y_all )/ std_Y_all     # size(3470,2)

X_intercept_val = np.ones((X_train_val_std.shape[0], 1))                   
X_train_val_bias= np.concatenate((X_intercept_val, X_train_val_std), axis=1)   # size(3470,5)
# final val data: convert numpy to torch ########
X_val_torch = torch.from_numpy(X_train_val_bias)            # size(3470,5)     imput
Y_val_torch = torch.from_numpy(Y_train_val_std)             # size(3470,2)   target

#%% Normalize testing data set

X_test_n1_std=(X_test_n1- mean_X_all )/ std_X_all
Y_test_NN1_std=(Y_test_n1- mean_Y_all )/ std_Y_all

X_intercept_test = np.ones((X_test_n1_std.shape[0], 1))                   
X_test_NN1_bias= np.concatenate((X_intercept_test, X_test_n1_std), axis=1)  # size(1735,5)

# final test data: convert numpy to torch ##########
X_test_torch = torch.from_numpy(X_test_NN1_bias)            # size(1735,5)    imput
Y_test_torch = torch.from_numpy(Y_test_NN1_std)             # size(1735,2)   target


# %% Create Neural network Architecture

hiddenlayersize = [10,20,10]   

class Deep_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_data = nn.Sequential(nn.Linear(X_train_torch.shape[1], hiddenlayersize[0]),
                    nn.ReLU(), nn.Linear((hiddenlayersize[0]), hiddenlayersize[1]),
                    nn.ReLU(), nn.Linear((hiddenlayersize[1]), hiddenlayersize[2]),
                    nn.ReLU(), nn.Linear(hiddenlayersize[2], Y_train_torch.shape[1]))
        
    def forward(self, x):
        model_data= self.model_data(x)
        return model_data
        
# print model
model_net=Deep_net()
print(model_net)  

#%% defind Loss funcion and Optimizer   
                    
# loss function
criterion = nn.MSELoss()       # Mean square loss

#criterion= torch.sqrt(nn.MSELoss())     # Root Meas Square error loss(RMSE)

optimizer = torch.optim.Adam(model_net.parameters(), lr=0.001, weight_decay= 1e-5)

#%% create  EarlyStopping

# import EarlyStopping
from pytorchtools import EarlyStopping


#%% create BATCH Function 

def next_batch(inputs, targets, batchsize):
    # loop over data set
    for i in range (0, inputs.shape[0], batchsize):
        #print('i:', i)
        #yield a tuple of the cuurent batched data and labels
        yield( inputs[i:i + batchsize], targets[i:i + batchsize])  
        # each batch size of "inputs[i:i +batchsize]" = torch.Sze([64, 5])
        # yield statement produces a generator object and can multiple values to the caller without termination the program
        
#BATCH_SIZE=32
#for (batchX1, batchY1) in next_batch(X_train_torch, Y_train_torch, BATCH_SIZE):
#    print("train:", batchX1.size()) 
#    print("target:", batchY1.size())      
                          
#%%    Train the model

def train_model(model_net, batch_size, patience, n_epochs):
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################

        model_net.train() # prep model for training
        #for batch, (data, target) in enumerate(train_loader, 1):
        for (batchX, batchY) in next_batch(X_train_torch, Y_train_torch, batch_size):
            #print("train:", batchX.size())

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            predictions= model_net( batchX.to(torch.float32) )
            # calculate the loss
            loss = criterion(predictions, batchY)
            
            
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())
            
        ######################    
        # validate the model #
        ######################
        model_net.eval() # prep model for evaluation
        
        #for batchX, batchY in [(X_val_torch, Y_val_torch)] : # for whole validation data at a time passing to model
        for (batchX, batchY) in next_batch(X_val_torch, Y_val_torch, batch_size):
            #print("val:", batchX.size())
            # forward pass: compute predicted outputs by passing inputs to the model
            predictions= model_net( batchX.to(torch.float32) )
            # calculate the loss
            loss = criterion(predictions, batchY)
            
            # record validation loss
            valid_losses.append(loss.item())
                
        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(n_epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
            
        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model_net)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        
    # load the last checkpoint with the best model
    model_net.load_state_dict(torch.load('save_NN_model_Dream_node_test.pt'))                  # Load model for test
    

    return  model_net, avg_train_losses, avg_valid_losses
            
            
batch_size = 256
n_epochs = 10


# early stopping patience; how long to wait after last time validation loss improved.
patience = 500

#model_net, train_loss, valid_loss = train_model(model_net, batch_size, patience, n_epochs)


print('train_loss:', range(1,len(train_loss)+1))
print('val_loss:', len( valid_loss)+1)

print('min_train_loss:', np.min(train_loss))
print('min_val_loss:', np.min(valid_loss))
print('max_train_loss:', np.max(train_loss))
print('max_val_loss:', np.max(valid_loss))


# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(train_loss)+1), train_loss, label= 'Training Loss')
plt.plot(range(1,len(valid_loss)+1), valid_loss,label='validation Loss')

# find position of lowest validation loss
minposs = valid_loss.index(min(valid_loss))+1 
print('minposs:', minposs)
plt.axvline(minposs, linestyle='--', color='r', label= f'Early Stopping Checkpoint at Epoch:{minposs:.0f};  train_min_loss:{np.min(train_loss):.5f};  valid_min_loss:{np.min(valid_loss):.5f} ')

#num_train_batch= X_train_NN200[0]/batch_size
# num_epoch= (Training_loss_sample/num_train_batch)
plt.xlabel(f'num_epoch={n_epochs:.1f}')
plt.ylabel('loss')
plt.ylim(0, 1.4) # consistent scale
plt.xlim(0, len(train_loss)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
#fig.savefig('loss_plotE500.png', bbox_inches='tight')            



#%%   test Dream Simulation Image

model_net_test=Deep_net() 

model_net_test.load_state_dict(torch.load('save_NN_model_Dream_node10.pt'))

pred_n1= model_net_test(X_test_torch.float())    # size(1735,5)  ### test image N=1                        

pred_numpy_n1=pred_n1.detach().numpy()   # size(1735,2)

# reconstruct image
Y_new_r = (pred_numpy_n1* std_Y_all) +  mean_Y_all

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
Y_B0_recon_NN1= add_mask(Y_B0_new_r) 
Y_B1_recon_NN1= add_mask(Y_B1_new_r)




#%%  test VIVO Image

### Prepare vivo image data
vivo_B0=remove_zero(vivo_B0_img)
vivo_B1=remove_zero(vivo_B1_img)
vivo_FID=remove_zero(np.reshape(vivo[:, 2], (64,64))*mask_zero)
vivo_STE=remove_zero(np.reshape(vivo[:, 3], (64,64))*mask_zero)
vivo_all= np.concatenate((vivo_B0, vivo_B1, vivo_FID, vivo_STE), axis=1)

vivo_test=(vivo_all - mean_X_all)/ std_X_all    #size(1735,4)
test_bias = np.ones((vivo_test.shape[0], 1))                    
vivo_bias= np.concatenate((test_bias, vivo_test), axis=1)

### Load mask for VIVO image data
mask_NaN_vivo=np.load('New_mask_NaN_vivo.npy')  
mask_zero_vivo=np.load('New_mask_zero_vivo.npy')


#### Run model with Vivo test image
model_net_test=Deep_net() 
model_net_test.load_state_dict(torch.load('save_NN_model_Dream_node10.pt'))
#model_net_test.load_state_dict(torch.load('save_NN_model_Dream_E500.pt'))

test_vivo_n1=torch.from_numpy(vivo_bias)    # size(1735,5)  ### test image N=1

pred_n_v= model_net_test(test_vivo_n1.to(torch.float32))   

pred_numpy_n_v=pred_n_v.detach().numpy()   # size(1735,2)

# reconstruct image
Y_new_r_v = (pred_numpy_n_v* std_Y_all) +  mean_Y_all

Y_B0_new_r_v= np.reshape(Y_new_r_v[:, 0], (1735,1))  
Y_B1_new_r_v= np.reshape(Y_new_r_v[:, 1], (1735,1)) 

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
Y_B0_recon_NN_V= add_mask(Y_B0_new_r_v) 
Y_B1_recon_NN_V= add_mask(Y_B1_new_r_v)
      

#%% Use Gauassian filter    

# use Gassian Filter 
from scipy.ndimage import gaussian_filter
sigma= 1.7
Y_B0_recon_NN_VG= gaussian_filter(Y_B0_recon_NN_V, sigma= (sigma, sigma))
Y_B1_recon_NN_VG= gaussian_filter(Y_B1_recon_NN_V, sigma= (sigma, sigma))

#%% Use Low pass fileter

from scipy import signal
def low_pass_function(image):

    # # Gaussian Filter (Smoothing)
    #kernel = np.array([[1, 2, 1],
     #                   [2, 4, 2],
     #                  [1, 2, 1]]) / 16
    # low pass filter
    kernel = np.ones((3, 3)) / 9
    #The convolution of f and g is written f∗g, t is defined as the integral of the product of the two functions after one is reflected about the y-axis and shifted. As such, it is a particular kind of integral transform:
    convolved_image = signal.convolve2d(image, kernel)

    truncated_image = truncate_v2(convolved_image, kernel)

    low_pass_filtered_image = truncated_image

    return low_pass_filtered_image
# After convolution part we used truncated function to delete zero paddings.
def truncate_v2(image, kernel):

    m, n = kernel.shape
    m = int((m-1) / 2)

    for i in range(0, m):
        line, row = image.shape
        image = np.delete(image, line-1, 0)
        image = np.delete(image, row-1, 1)
        image = np.delete(image, 0, 0)
        image = np.delete(image, 0, 1)
    return image

#Y_B0_recon_NN_VG= low_pass_function(Y_B0_recon_NN_V)
#Y_B1_recon_NN_VG= low_pass_function(Y_B1_recon_NN_V)


#%%   Scatter plot for Dream simulation  test image 

### import scatter plot function
from scatter_plot import B01_scatter_plot, B1_scatter_plot, B11_scatter_plot, B0_scatter_plot

###  Draw scatter plot for BO image 
fig=plt.figure(figsize=(20,10));
plt.suptitle('NN_Epoch1000_L10: {test: simu},Train:{Dream_simu}-->NN_prediction:{simu_B0,B1},compare{simulation vs phantom} ', fontsize=18)

### B0 before projection 
plt.subplot(231); plt.title('befor projection:simulation_B0', fontsize=10)
B01_scatter_plot( marge_B0[:,19], marge_P_B0[:,19],  "Dream_Simulation_B0", "target_phantom" )
### B0 normal
#plt.subplot(242); plt.title('After linear:Test=simu,Train:simu_N=1', fontsize=14)
#B01_scatter_plot(np.reshape(Y_B0_recon_1n, (4096,1)), marge_P_B0[:,19], "Prediction_simulation_B0", "target_phantom" )
###Train =200, 
plt.subplot(232); plt.title('After Linear:Test=simu,Train:N=200', fontsize=14)
#B01_scatter_plot(  np.reshape(Y_B0_recon_200, (4096,1)), marge_P_B0[:,19],  "Prediction_simulation_B0", "target_phantom", )
### Train =NN, 
plt.subplot(233); plt.title('After NN_proj:Test=simu, Train:simu_N=NN', fontsize=14)
B01_scatter_plot(  np.reshape(Y_B0_recon_NN1, (4096,1)), marge_P_B0[:,19],  "NN_Prediction_simu_B0", "target_phantom" )





# # B1 before projection 
plt.subplot(234); plt.title('befor projection:simulation_B1', fontsize=14)
B1_scatter_plot(  marge_B1[:,19], marge_P_B1[:,19],  "Dream_Simulation_B1", "target_phantom",)
### B1 normal
#plt.subplot(246); plt.title('After Linear:Test=simu,Train:simu_N=1', fontsize=14)
#B11_scatter_plot( np.reshape(Y_B1_recon_1n, (4096,1)), marge_P_B1[:,19],  "Prediction_simulation_B1", "target_phantom" )
### Train =200, 
plt.subplot(235); plt.title('After Linear:Test=simu,Train:N=200', fontsize=14)
#B11_scatter_plot( np.reshape(Y_B1_recon_200, (4096,1)), marge_P_B1[:,19],   "Prediction_simulation_B1" , "target_phantom",)
### Train =NN, 
plt.subplot(236); plt.title('After NN_proj:Test=simu, Train:simu_N=NN', fontsize=14)
B11_scatter_plot( np.reshape(Y_B1_recon_NN1, (4096,1)), marge_P_B1[:,19],   "NN_Prediction_simu_B1" , "target_phantom")


#%%   Scatter plot for ViVO test image 

### import scatter plot function
from scatter_plot import B01_scatter_plot, B1_scatter_plot, B11_scatter_plot

###  Draw scatter plot for BO VIVO image 
fig=plt.figure(figsize=(20,10));
plt.suptitle('NN_Epoch1000_L10:{test:vivo}, Train:{Dream_simu},{target:Phantom}-->predict_vivo:{B0, B1},compare:{pred_vivo vs wasabi}', fontsize=18)

### B0 before projection 
plt.subplot(241); plt.title('befor projection:Vivo_B0', fontsize=12)
B0_scatter_plot( np.reshape(wasabi_B0*mask_zero_vivo, (4096,1)), np.reshape(vivo_B0_img*mask_zero_vivo, (4096,1)),  "wasabi", "vivo_B0" )
### B0 normal
#plt.subplot(242); plt.title('After Linear:Test=Vivo, Train:simu_N=1', fontsize=14)
#B01_scatter_plot( np.reshape(wasabi_B0*mask_zero_vivo, (4096,1)), np.reshape(Y_B0_recon_1v*mask_zero_vivo, (4096,1)),  "wasabi", "Prediction_vivo_B0" )
### B0 X1.5
### Train =200, 
plt.subplot(242); plt.title('After Linear:Test=Vivo, Train:simu_N=200', fontsize=14)
#B01_scatter_plot(np.reshape(wasabi_B0*mask_zero_vivo, (4096,1)),  np.reshape(Y_B0_recon_200v*mask_zero_vivo, (4096,1)),  "wasabi", "Prediction_vivo_B0" )

### Train =NN, 
plt.subplot(243); plt.title('After NN:Test=Vivo', fontsize=14)
B01_scatter_plot(np.reshape(wasabi_B0*mask_zero_vivo, (4096,1)),  np.reshape(Y_B0_recon_NN_V*mask_zero_vivo, (4096,1)),  "wasabi", "NN_Prediction_vivo_B0" )
### With Gaussian Filter, 
plt.subplot(244); plt.title('After Gaussian Filter,Q=1.7', fontsize=14)
B01_scatter_plot(np.reshape(wasabi_B0*mask_zero_vivo, (4096,1)),  np.reshape(Y_B0_recon_NN_VG*mask_zero_vivo, (4096,1)),  "wasabi", "NN_Prediction_vivo_B0" )



# # B1 before projection 
plt.subplot(245); plt.title('befor projection:Vivo_B1', fontsize=12)
B1_scatter_plot( np.reshape(wasabi_B1*mask_zero_vivo, (4096,1)), np.reshape(vivo_B1_img*mask_zero_vivo, (4096,1)),  "wasabi", "vivo_B1" )
### B1 normal
#plt.subplot(246); plt.title('After Linear:Test=Vivo, Train:simu_N=1', fontsize=14)
#B11_scatter_plot( np.reshape(wasabi_B1*mask_zero_vivo, (4096,1)), np.reshape(Y_B1_recon_1v*mask_zero_vivo, (4096,1)),  "wasabi", "Prediction_vivo_B1" )


### Train =200, 
plt.subplot(246); plt.title('After Linear:Test=Vivo, Train:simu_N=200', fontsize=14)
#B11_scatter_plot(np.reshape(wasabi_B1*mask_zero_vivo, (4096,1)),  np.reshape(Y_B1_recon_200v*mask_zero_vivo, (4096,1)),  "wasabi", "Prediction_vivo_B1" )

### Train =NN 
plt.subplot(247); plt.title('After NN:Test=Vivo', fontsize=14)
B11_scatter_plot(np.reshape(wasabi_B1*mask_zero_vivo, (4096,1)),  np.reshape(Y_B1_recon_NN_V*mask_zero_vivo, (4096,1)),  "wasabi", "NN_Prediction_vivo_B1" )
# With Gaussian Filter, 
plt.subplot(248); plt.title('After Gaussian Filter,Q=1.7', fontsize=14)
B11_scatter_plot(np.reshape(wasabi_B1*mask_zero_vivo, (4096,1)),  np.reshape(Y_B1_recon_NN_VG*mask_zero_vivo, (4096,1)),  "wasabi", "NN_Prediction_vivo_B1" )


#%%  Image visualize

### Dream simulation test image and compare with simulation target inage
fig=plt.figure(figsize=(18,8));
plt.suptitle('NN_Epoch1000_L10:N1_Train_image:{Simu},{target:Phantom},{test:Simu}-->predict_simu:{B0, B1}', fontsize=18)

def plot_image(img, vmin, vmax):
    plt.imshow( img, vmin=vmin, vmax=vmax, origin="lower") 
    plt.axis('off')
    plt.colorbar()

plt.subplot(261); plt.title('train{Simulation_B0}', fontsize=13)
plot_image(np.reshape(marge_B0[:, 18], (64,64)), -45, 45)

plt.subplot(267); plt.title(' train{Simulation_B1}', fontsize=13)
plot_image(np.reshape(marge_B1[:, 18], (64,64)), 0, 1.2) 

#Taregt data
plt.subplot(262); plt.title('Target{Phantom_B0}', fontsize=13)
plot_image(np.reshape(marge_P_B0[:, 18], (64,64)) ,  -45, 45)
 
plt.subplot(268); plt.title('Target{Phantom_B1}', fontsize=13)
plot_image(np.reshape(marge_P_B1[:, 18], (64,64)) ,  0, 1.2)

plt.subplot(263); plt.title('test{Simu_B0}', fontsize=13)
plot_image(np.reshape(marge_P_B0[:, 19], (64,64))  ,  -45, 45)

plt.subplot(269); plt.title('test{Simu_B1}', fontsize=13) 
plot_image(np.reshape(marge_P_B1[:, 19], (64,64))  ,  0, 1.2)

plt.subplot(264); plt.title('NN_Predict{Simu_B0}', fontsize=13)
plot_image(Y_B0_recon_NN1, -45, 45)
 
plt.subplot(2,6,10); plt.title('NN_Predict{Simu_B1}', fontsize=13) 
plot_image(Y_B1_recon_NN1, 0, 1.2)

def residual(arr1, arr2):
    res=np.reshape(arr1, (64,64)) -np.reshape(arr2, (64,64))
    return res

# residual
plt.subplot(265); plt.title('residual{pred-tar}', fontsize=8)
plot_image(residual( Y_B0_recon_NN1, marge_P_B0[:, 19] ) , -2.0 , 2.0)

plt.subplot(2,6,11); plt.title('residual{pred-tar}', fontsize=8)
plot_image(residual(Y_B1_recon_NN1, marge_P_B1[:, 19]), -0.04, 0.04 )


#%%

### VIVO test image and compare with WASABI image
fig=plt.figure(figsize=(18,8));
plt.suptitle('NN_Epoch1000_L30:{simu},{target:Phantom},{test:vivo}-->predict_vivo:{B0, B1}, compare:{pred_vivo vs wasabi}', fontsize=18)

def plot_image(img, vmin, vmax):
    plt.imshow( img, vmin=vmin, vmax=vmax, origin="lower") 
    plt.axis('off')
    plt.colorbar()
    
#plt.subplot(261); plt.title('train{Simulation_B0}', fontsize=13)
#plot_image(np.reshape(marge_B0[:, 18], (64,64)),   -45, 45)

#plt.subplot(267); plt.title(' train{Simulation_B1}', fontsize=13) 
#plot_image(np.reshape(marge_B1[:, 18], (64,64)),   -45, 45)

plt.subplot(261); plt.title('Target{Phantom_B0}', fontsize=13)
plot_image(np.reshape(marge_P_B0[:, 18], (64,64)) ,   -45, 45)

plt.subplot(267); plt.title('Target{Phantom_B1}', fontsize=13)
plot_image(np.reshape(marge_P_B1[:, 18], (64,64)), 0, 1.2)

plt.subplot(262); plt.title('test{vivo_B0}', fontsize=13)
plot_image((vivo_B0_img*mask_NaN_vivo), -45,  45)

plt.subplot(268); plt.title('test{vivo_B1}', fontsize=13) 
plot_image((vivo_B1_img*mask_NaN_vivo), 0, 1.2)
 
plt.subplot(263); plt.title('NN_Predict{vivo_B0}', fontsize=13)
plot_image((Y_B0_recon_NN_V*mask_NaN_vivo), -45,  45)

plt.subplot(264); plt.title('G_Filter_1.7{vivo_B0}', fontsize=13)
plot_image((Y_B0_recon_NN_VG*mask_NaN_vivo), -45, 45)
1 
plt.subplot(2,6,9); plt.title('NN_Predict{vivo_B1}', fontsize=13) 
plot_image((Y_B1_recon_NN_V*mask_NaN_vivo), 0,  1.2 )

plt.subplot(2,6,10); plt.title('G_Filter_1.7{vivo_B1}', fontsize=13) 
plot_image((Y_B1_recon_NN_V*mask_NaN_vivo), 0, 1.2)

plt.subplot(265); plt.title('{ wasabi B0}', fontsize=13)
plot_image((wasabi_B0*mask_NaN_vivo), -45, 45 )

plt.subplot(2,6,11); plt.title('{ wasabi B1}', fontsize=13) 
plot_image( (wasabi_B1*mask_NaN_vivo), 0, 1.2)

def residual(arr1, arr2):
    res=np.reshape(arr1, (64,64)) -np.reshape(arr2, (64,64))
    return res

# residual
plt.subplot(266); plt.title('residual{wasabi-G_F_pred}', fontsize=8)
plot_image(residual(wasabi_B0, Y_B0_recon_NN_VG)*mask_NaN_vivo , -20,  20)

plt.subplot(2,6,12); plt.title('residual{wasabi-G_F_pred}', fontsize=8)
plot_image(  residual(wasabi_B1, Y_B1_recon_NN_VG)*mask_NaN_vivo ,-0.2, 0.2 )






