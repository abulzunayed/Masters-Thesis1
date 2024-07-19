# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 12:51:30 2023

@author: zunayeal
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import normalize
from torch import mean
from torch import std

#%%  import data
from scipy.io import loadmat
D=loadmat(r"Example_fit.mat")

#%% data preparation
# input data (reshape to 2D and transpose of X,shape=torch.Size([57, 13824]))
Xz = torch.Tensor(D['Z_corrExt'])
X1 = torch.reshape(Xz, (Xz.shape[0]*Xz.shape[1]*Xz.shape[2], Xz.shape[3]))
XX = X1.T
                
# Target data (reshape to 2D and transpose of X,shape=torch.Size([16, 13824]))
Yp = torch.Tensor(D['popt'])
Y1 = torch.reshape(Yp, (Yp.shape[0]*Yp.shape[1]*Yp.shape[2], Yp.shape[3]))    
YY = Y1.T                                         

mask1 = torch.isnan(XX)     # mask1.shape= torch.Size([57, 13824])
mask2 = torch.isnan(YY)     # mask1.shape=torch.Size([16, 13824])
# =============================================================================
#  mask_index_1 is a binary mask tensor with same number of columns as mask1. It is initialized to all False values, 
#  and later will be updated in the loop to indicate which columns of mask1 have at least one True value. 
# =============================================================================
mask_index_1 = torch.zeros([1, mask1.shape[1]])       # torch.Size([1, 13824])
mask_index_2 = torch.zeros([1, mask2.shape[1]])        #torch.Size([1, 13824])

""" Loop for only 1st row and all cols"""
for col in range(mask1.shape[1]):                             
    if True in mask1[:, col]:
        mask_index_1[0, col] = True  # If there is 'true' value, it sets the corresponding element of mask_index_1 to True.

for col in range(mask2.shape[1]):                              
    if True in mask2[:, col]:
        mask_index_2[0, col] = True        

# =============================================================================
# 'nonzero' function to get the indices where the new tensor is 'True'. The 'as_tuple=False' argument tells nonzero to return
#  a tensor of shape (num_indices, num_dims),where num_indices is number of indices where the mask is True, and num_dims is 
#  the number of dimensions. Finally all in NAN value indices converts to 1, othewise 0. 
# =============================================================================

mask_index_1 = (mask_index_1 == 1).nonzero(as_tuple=False)     
mask_index_2 = (mask_index_2 == 1).nonzero(as_tuple=False)     

mask_final = torch.cat((mask_index_1, mask_index_2)).unique()    #  torch.Size([7732])

# create new tensor without NAN values by subtracting NAN value indices.
X_ohne_Nan = torch.zeros([XX.shape[0], XX.shape[1] - mask_final.shape[0]])    # torch.Size([57, 6092]) --> Zeros tensor
Y_ohne_Nan = torch.zeros([YY.shape[0], YY.shape[1] - mask_final.shape[0]])

# without NAN value for 57 rows
X_ohne_Nan = XX[:, [i for i in range(XX.shape[1]) if i not in mask_final]]    
Y_ohne_Nan = YY[:, [i for i in range(YY.shape[1]) if i not in mask_final]]   
print(X_ohne_Nan.shape)   # torch.Size([57, 6092])
print(Y_ohne_Nan.shape)   # torch.Size([16, 6092])


#%%  Draw single image and observe outliers
def input_iamge( in_iamge):
    plt.figure(1)
    inp=plt.imshow(in_iamge, cmap='viridis', vmin=0.05, vmax=0.25)
    plt.colorbar(inp)
    plt.title(label='input Image: Z_corrExt[:,:,0,7]')
    plt.show()

def target_iamge( in_iamge):
    plt.figure(2)
    inp=plt.imshow(in_iamge, cmap='viridis',vmin=0, vmax=80)
    plt.colorbar(inp)
    plt.title(label='target image')
    plt.show()
# input sinlge image   
input_iamge(Xz[:,:,0,1])    # (128, 108)
# Target sinlge image   
target_iamge(Yp[:,:,0,11])    # (128, 108)


#%%  draw histrogram for indiviual image
plt.figure(3)
X_image= Xz[:,:,0,7]
#Y_image= torch.mean(Y_image,axis=2)
counts, bins = np.histogram(X_image.ravel(),bins=1000,range=(0,5.1))   # size=(128,108)
plt.stairs(counts, bins)
plt.show()

plt.figure(4)
Y_image= Yp[:,:,0,7]
#Y_image= torch.mean(Y_image,axis=2)
counts, bins = np.histogram(Y_image.ravel(),bins=1000,range=(0,5.1))    # size=(128,108)
plt.stairs(counts, bins)
plt.show()


#%% draw histrogram for whole data set for outliersr
fig, (ax1, ax2) = plt.subplots(1, 2)
counts, bins = np.histogram(X_ohne_Nan.ravel(),bins=1000,range=(-2, 2.0))
ax1.stairs(counts, bins)
counts, bins = np.histogram(Y_ohne_Nan.ravel(),bins=1000,range=(-5.1, 80))
ax2.stairs(counts, bins)
plt.show()

#%% detect outliers from each of target image and marked as -1000 so that, easily omit during training
def call(Y_out):
    
    for i in range(Y_out.shape[0]):
        if i==0:
            pass
        if i==1:
            Y_out[i, Y_out[i, :]<=0.605]=-1000
        if i==2:
            Y_out[i, Y_out[i, :]<=0.9]=-1000
            Y_out[i, Y_out[i, :]>=1.35]=-1000
        if i==3:
            pass
        if i==4:
            pass
        if i==5:
            Y_out[i, Y_out[i, :]>=0.5]=-1000
        if i==6:
            Y_out[i, Y_out[i, :]<=3.2]=-1000
            Y_out[i, Y_out[i, :]>=3.8]=-1000
        if i==7:
            Y_out[i, Y_out[i, :]<=0.05]=-1000
        if i==8:
            Y_out[i, Y_out[i, :]<=1.02]=-1000
            Y_out[i, Y_out[i, :]>=4.98]=-1000
        if i==9:
            Y_out[i, Y_out[i, :]<=-4.48]=-1000
            Y_out[i, Y_out[i, :]>=-2.01]=-1000
        if i==10:
            Y_out[i, Y_out[i, :]<=0.02]=-1000
        if i==11:
            Y_out[i, Y_out[i, :]<=20]=-1000
            Y_out[i, Y_out[i, :]>=65]=-1000
        if i==12:
            Y_out[i, Y_out[i, :]<=-3.98]=-1000
        if i==13:
            pass
        if i==14:
            Y_out[i, Y_out[i, :]>=3.48]=-1000
        if i==15:
            Y_out[i, Y_out[i, :]<=1.02]=-1000
    return Y_out


#%% call function ouliers remover

# with remove outliers from data

X_ohne_Nan_out=X_ohne_Nan
Y_ohne_Nan_out=call(Y_ohne_Nan)


X=X_ohne_Nan_out.T   # torch.Size([6092, 52])
T=Y_ohne_Nan_out.T   # torch.Size([6092, 13])

#%%  split data for training and testing

from sklearn.model_selection import train_test_split
(trainX, testX, trainY, testY) = train_test_split(X, T,test_size=0.15, random_state=95) 

print(trainX.shape)  # torch.Size([5178, 57])
print(trainY.shape)  # torch.Size([5178, 16])
print(testX.shape)   # torch.Size([914, 57])
print(testY.shape)   # torch.Size([914, 16])
print(trainX.shape[0])  #5178

#%%   Create model
hiddenLayerSize = [100, 200, 100]

class Deep_net(nn.Module):
     def __init__(self):
         super().__init__()
         
         self.encoder= nn.Sequential(nn.Linear(X.shape[1], hiddenLayerSize[0]),              #57, 100)
                         nn.ReLU(), nn.Linear(hiddenLayerSize[0], hiddenLayerSize[1]),    #(100, 200)
                         nn.ReLU(), nn.Linear(hiddenLayerSize[1], hiddenLayerSize[2]),     #(200, 100)
                         nn.ReLU(), nn.Linear(hiddenLayerSize[2], T.shape[1]))             # (100, 16)
                               
     
     
     def forward(self, x):
         encoder=self.encoder(x)
         return encoder
#%%  print model 
model_net=Deep_net()
print(model_net)
 
 #%% loss function and  optimizer
 
criterion= nn.MSELoss()     # Meas Square error loss(MSE)

# criterion= torch.sqrt(nn.MSELoss())     # Root Meas Square error loss(RMSE)

# criterion= nn.GaussianNLLLoss()   # Gaussian negative log-likelihood loss (GNLL)

optimizer=torch.optim.Adam(model_net.parameters(), lr=0.001, weight_decay=1e-5)


#%% define  batch function

def next_batch(inputs, targets, batchSize):
    # loop over the dataset                                                                    
    for i in range(0, inputs.shape[0], batchSize):             # print(inputs.shape[0])= 5178 and gradually "i" will be increase by 64
        # yield a tuple of the current batched data and labels
        yield (inputs[i:i + batchSize], targets[i:i + batchSize])    # each batch size of "inputs[i:i + batchSize]"=torch.Size([64, 57])
        # yield statement produces a generator object and can return multiple values to the caller without terminating the program, 
            
#%% define batch size and epoch

BATCH_SIZE = 64
EPOCHS = 2


#%%  training the model
#EPOCHS = 2000
train_losses = []
    # loop through the epochs
for epoch in range(0, EPOCHS):	
    print("[INFO] epoch: {}...".format(epoch + 1))
    trainLoss = 0
    #trainAcc = 0
    samples = 0
    model_net.train()  
    for (batchX, batchY) in next_batch(trainX, trainY, BATCH_SIZE):    # per epoch 5178/64= 81 loop
        #print(batchX.shape) # torch.Size([64, 57]) 
        #print(batchY.shape) # torch.Size([64, 16])
        predictions = model_net(batchX) 
        
        outlier_mask = torch.zeros(batchY.shape)
        outlier_mask[batchY == -1000] = True
        # MSE loss:
        loss = criterion(predictions[outlier_mask == 0], batchY[outlier_mask == 0].float())
        # RMSE loss calculate:
        # loss =torch.sqrt( criterion(predictions[outlier_mask == 0], batchY[outlier_mask == 0].float()) )
        
        # GNLL loss calculate:
        #var = torch.std(predictions[outlier_mask == 0], dim=0, keepdim=True)  
        #loss = criterion(predictions[outlier_mask == 0], batchY[outlier_mask == 0].float(), var)
        
        # zero the gradients accumulated from the previous steps,
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update training loss, accuracy, and the number of samples
        trainLoss += loss.item() * batchY.size(0)            # batchY.size(0)=64
        samples += batchY.size(0)   
    # display model progress on the current training batch
    trainTemplate = "epoch: {} train loss: {:.3f}"
    print(trainTemplate.format(epoch + 1, (trainLoss / samples)))
    
    train_losses.append((trainLoss/ samples))
    
#%% plot train loss 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

plt.figure(7)
plt.title("Traing Loss ")
plt.plot(train_losses,label="trainloss for 2000 Epoch")

plt.xlabel("Epoch")
plt.ylabel("trainLoss")
plt.legend()
plt.show()


#%% save model
#torch.save(model_net.state_dict(), 'E:\DREAM\Dream_Deep_learning\MSE_Norm_V9_Deep_net.pth')

# Recall the mosel
#model_net_test=Deep_net()
#print(model_net_test)
#model_net_test.load_state_dict(torch.load('E:\DREAM\Dream_Deep_learning\MSE_Norm_V9_Deep_net.pth'))

#%%   reconstruction image from prediction
model_net_test=Deep_net()
#print(model_net_test)
model_net_test.load_state_dict(torch.load('E:\DREAM\Dream_Deep_learning\MSE_Norm_V9_Deep_net.pth'))

X_m=X_ohne_Nan.T   # torch.Size([6092], 57)   # without NAN input with outliear

count=0
Pred_original = torch.zeros([YY.shape[0], YY.shape[1]])     # (16, 13824)

Pred1 = model_net_test(X_m)   # model input X= (6092, 57) and output, pred=(6092, 16)
Pred=Pred1.T   # Pred= (16, 6092)

# Need to convert mask_final tensor to list
mask_final_list = mask_final.tolist()     # list.size= 7732

for i in range(Pred_original.shape[1]):
    try:
        if mask_final_list.index(i) != -1:  # Means NAN values in this column
            Pred_original[:, i] = np.NaN

    except Exception as e:
        Pred_original[:, i] = Pred[:, count]    
        count += 1

# final Pred_original =(16,13824)
Pred_original =Pred_original.T   #Pred_original =(13824, 16) without outliers
Y1 = torch.reshape(Yp, (Yp.shape[0]*Yp.shape[1]*Yp.shape[2], Yp.shape[3]))
recon_image=torch.reshape(Pred_original, (Yp.shape[0], Yp.shape[1], Yp.shape[2],Yp.shape[3]))
print(recon_image.shape)

#%% Plot reconstruction image
reco_image1=recon_image.detach().numpy()
plt.figure(8)
reco_image2= reco_image1[:,:,0,11]
im=plt.imshow(reco_image2, cmap='viridis', vmin=0, vmax= 80)
plt.colorbar(im)
plt.title(label='Prection image:[:,:,0,1] for E_out_2000 Epoch')
plt.show()

#%%  substract image betwen prediction and target image
# predection image --> Pred_original =Pred_original.T   #Pred_original =(13824, 16)
predection_image =Pred_original.T                       # #Pred_original =(16,13824)
print(predection_image.shape)
print(YY.shape)                                  # Y,shape=torch.Size([16, 13824])

dev_img=torch.abs(YY - predection_image)
dev_img=dev_img.T
sub_image=torch.reshape(dev_img, (Yp.shape[0], Yp.shape[1], Yp.shape[2],Yp.shape[3]))

plt.figure(9)
sub1_image1=sub_image.detach().numpy()
sub_image2= sub1_image1[:,:,0,1]
im=plt.imshow(sub_image2, cmap='viridis', vmin=-0.09, vmax= 0.09)
plt.colorbar(im)
plt.title(label='Subtract image from Model:[:,:,0,1]')
plt.show() 

#%% testing and validation
# =============================================================================
# model_net_test=Deep_net()
# #print(model_net_test)
# model_net_test.load_state_dict(torch.load('E:\DREAM\Dream_Deep_learning\MSE_Norm_V9_Deep_net.pth'))
# 
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import r2_score
# 
# test_losses = [] 
# testLoss = 0
# samples = 0
# mse_error = 0
# r2_score=0
# MSE_error = [] 
# R2_score = [] 
# model_net_test.eval()
# # # initialize a no-gradient context
# with torch.no_grad():
#     for (batchX, batchY) in next_batch(testX, testY, BATCH_SIZE):   # 914/64= 15 loop
#         # print(batchX.shape) # torch.Size([64, 57])   
#         predictions = model_net_test(batchX) 
#         
#         loss = criterion(predictions, batchY.float())           # prediction = torch.Size([64, 16]) and batchY = torch.Size([64, 16]) 
#                                                                 # Here testloss and MSE error should be same value
#         # MSE Error nad R2 score
#         err = mean_squared_error(predictions, batchY.float())        
#         r2 = r2_score(batchY.float(),predictions, multioutput='variance_weighted')
#         # update training loss, accuracy, and the number of samples
#         testLoss += loss.item() * batchY.size(0)            # batchY.size(0)
#         samples += batchY.size(0)
#         
#         mse_error += err.item() * batchY.size(0)
#         #R2_score += r2.item() * batchY.size(0)
#         #print(R2_score)
#         
#         # display model progress on the current training batch
#           
#         testTemplate = " test loss: {:.3f}  MSE_error: {:.3f} R2_score: {:.3f}"
#         print(testTemplate.format( (testLoss / samples),(mse_error/ samples) , ( r2)))
#         test_losses.append((testLoss/ samples))
#         MSE_error.append((testLoss/ samples))
#         R2_score.append((r2))
#          
# r=sum(R2_score) / len(R2_score)
# print(f"R2_score: {r}")
# # #%% plot for testing data
# plt.figure(10)
# plt.title("Testing Loss ")
# plt.subplot(1,3,1)
# plt.plot(test_losses,label="testloss")
# plt.xlabel("Epoch")
# plt.ylabel("testLoss")
# plt.legend()
# plt.show()
#  
# plt.subplot(1,3,2)
# plt.plot(MSE_error,label="MSE_error")
# plt.xlabel("Epoch")
# plt.ylabel("MSE_error")
# plt.legend()
# plt.show()
# # 
# plt.subplot(1,3,3)
# plt.plot(R2_score,label="R2_score")   
# plt.xlabel("Epoch")
# plt.ylabel("R2_score")
# plt.legend()
# plt.show()
# =============================================================================


 
