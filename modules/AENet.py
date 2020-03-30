#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copied from https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py
import torch
from torch import nn
import numpy as np
def getAENET(config):
    net_type = config.get('type', 'AECV2D')
    if net_type == 'AECV2D':
        return AECV2D(config)
    elif net_type == 'AECV3D':
        return AECV3D(config)
    elif net_type == 'DeepCC':
        return DeepCC(config)  
    elif net_type == 'DF':
        return DeepFlash(config)  
    else:
        raise ValueError(f'Unsupported network type: {net_type}')



class AECV3D(nn.Module):
    def __init__(self, config):
        super(AECV3D, self).__init__()
        self.imgDim = 3
        paras = config.get('paras', None)
        self.encoder, self.decoder = get3DNet(paras)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DeepCC(nn.Module):
    def __init__(self, config):
        super(DeepCC, self).__init__()
        self.imgDim = 3
        paras = config.get('paras', None)
        self.encoder, self.decoder = get3DNet(paras)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DeepFlash(nn.Module):
    def __init__(self, config):
        super(DeepFlash, self).__init__()
        self.imgDim = 2
        paras = config.get('paras', None)
        self.encoder, self.decoder = get3DNet(paras)
    
    def forward(self, src, tar):
        x1 = self.encoder(src)
        x2 = self.encoder(tar)
        x1 = x1.detach().numpy()
        x2 = x2.detach().numpy() 
        x3 = np.concatenate((x1, x2), axis=1)
        x3 = torch.Tensor (x3)
        x3 = self.decoder(x3)
        # x3 = 12 *x3
        return x3    

# class weightedTanh(nn.Module):
#     def __init__(self, weights = 1):
#         super().__init__()
#         self.weights = weights
        
#     def forward(self, input):
#         ex = torch.exp(2*self.weights*input)
#         return (ex-1)/(ex+1)


def get3DNet(paras):
    if paras == None or paras['structure'] == 'default':        
        encoder = nn.Sequential(
            nn.Conv3d(1, 8, 3, stride=1, padding=1),  # 
            nn.ReLU(True),
            nn.Conv3d(8, 16, 3, stride=1, padding=1),  # 
            nn.ReLU(True),
            nn.MaxPool3d(2, stride=2),  # 
            nn.Conv3d(16, 32, 3, stride=1, padding=1),  # 
            nn.ReLU(True),
            nn.Conv3d(32, 32, 3, stride=1, padding=1),  # 
            nn.ReLU(True),

        )
        decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 2, stride=2, padding=0),  # b, 16, 5, 5
            nn.ReLU(True),
#            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=0),  # b, 16, 5, 5
            nn.Conv3d(16, 8, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.Conv3d(8, 1, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.Tanh(),
        )

    elif paras['structure'] == 'complex':
        encoder = nn.Sequential(
            nn.Conv3d(2, 8, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool3d(2, stride=2),  # b, 16, 5, 5
            nn.Conv3d(8, 16, 3, stride=2, padding=1),  # b, 8, 3, 3            
            nn.ReLU(True)
        )
        decoder = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 2, stride=2, padding=0),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.Conv3d(8, 1, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.Tanh(),
        )
    elif paras['structure'] == 'deepflash':
        # encoder = nn.Sequential(
        #     nn.Conv3d(1, 8, 3, stride=2, padding=1),  # 
        #     nn.ReLU(True),
        #     nn.Conv3d(8, 16, 3, stride=2, padding=1),  # 
        #     nn.ReLU(True),
        #     nn.Conv3d(16, 32, 3, stride=2, padding=1),  # 
        #     nn.ReLU(True),
        #     nn.Conv3d(32, 32, 3, stride=1, padding=1),  # 
        #     nn.ReLU(True),
        #     nn.ConvTranspose3d(32, 16, 2, stride=2, padding=0),  # b, 16, 5, 5
        #     nn.ReLU(True),
        #     nn.Conv3d(16, 8, 3, stride=1, padding=1),  # b, 8, 3, 3
        #     nn.ReLU(True),
        #     nn.Conv3d(8, 1, 3, stride=1, padding=1),  # b, 8, 3, 3
        #     nn.Tanh(),
        # )
        # decoder = nn.Sequential(
        #     nn.Conv3d(2, 8, 3, stride=1, padding=1),  # b, 16, 10, 10
        #     nn.ReLU(True),
        #     nn.MaxPool3d(2, stride=2),  # b, 16, 5, 5
        #     nn.Conv3d(8, 16, 3, stride=2, padding=1),  # b, 8, 3, 3            
        #     nn.ReLU(True),

        #     nn.ConvTranspose3d(16, 8, 2, stride=2, padding=0),  # b, 16, 5, 5
        #     nn.ReLU(True),
        #     nn.Conv3d(8, 1, 3, stride=1, padding=1),  # b, 8, 3, 3
        #     nn.Tanh(),
        # )
###########################Spatial Net (LEARNING RATE = 1E-2)##########################################        
        # encoder = nn.Sequential(
        #     nn.Conv2d(1, 8, 3, stride=1, padding=1),  # 
        #     nn.LogSigmoid(),
        #     #nn.ReLU(True),

        #     nn.Conv2d(8, 16, 3, stride=1, padding=1),  # 
        #     nn.LogSigmoid(),
        #     # nn.ReLU(True),

        #     nn.Conv2d(16, 16, 3, stride=1, padding=1),  # 
        #     nn.LogSigmoid(),
        #     # nn.ReLU(True),

        #     nn.Conv2d(16, 8, 3, stride=1, padding=1),  # 
        #     nn.LogSigmoid(),
        #     # nn.ReLU(True),

        #     nn.Conv2d(8, 1, 3, stride=1, padding=1),  # 
        #     nn.LogSigmoid(),
        # )
        # decoder = nn.Sequential(
        #     nn.Conv2d(2, 8, 3, stride=1, padding=1),
        #     nn.LogSigmoid(),
        #     # nn.ReLU(True),

        #     nn.Conv2d(8, 16, 3, stride=1, padding=1),  # b, 8, 3, 3     
        #     nn.LogSigmoid(),       
        #     # nn.ReLU(True),

        #     nn.ConvTranspose2d(16, 8, 2, stride=2, padding=0),  # b, 16, 5, 5
        #     nn.LogSigmoid(),
        #     #nn.ReLU(True),

        #     nn.Conv2d(8, 1, 3, stride=2, padding=1),  # b, 8, 3, 3
        #     nn.LogSigmoid(),
        # )
###########################Spatial Net (LEARNING RATE = 1E-2)##########################################

###########################Fourier Net (LEARNING RATE = 1E-2)##########################################
        # encoder = nn.Sequential(
        #     nn.Conv2d(1, 16, 3, stride=1, padding=1, bias = True),  # Large kernel size, large output feature map, and with dense stride
        #     nn.ReLU(),
        #     nn.Conv2d(16, 16, 3, stride=1, padding=1, bias = True), 
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, 3, stride=1, padding=1, bias = True), 
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, stride=1, padding=1,bias = True), 
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, 3, stride=1, padding=1, bias = True), 
        #     nn.ReLU(),

   
  
        # )
        # decoder = nn.Sequential(
        #     nn.Conv2d(256, 128, 3, stride=1, padding=1, bias = True),  # 
        #     nn.ReLU(),
        #     nn.Conv2d(128, 64, 3, stride=1, padding=1, bias = True),  # 
        #     nn.ReLU(),
        #     nn.Conv2d(64, 32, 3, stride=1, padding=1, bias = True), 
        #     nn.ReLU(),
        #     nn.Conv2d(32, 8, 3, stride=1, padding=1, bias = True), 
        #     nn.ReLU(),
        #     nn.Conv2d(8, 1, 3, stride=1, padding=1,bias = True), 
        #     nn.ReLU(),
       
            
        # )
############################### 2D Synthetic Net############################################## 
        # encoder = nn.Sequential(
        #     nn.Conv2d(1, 8, 3, stride=1, padding=1,bias = True),  #   Loss = MSE ; 'batch_size'] = 256 'learning_rate' = 5e-5
        #     nn.Dropout(0.2), 
        #     nn.MaxPool2d(2),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(),
         
        #     nn.Conv2d(8, 16, 3, stride=1, padding=1,bias = True),  # 
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
                   
        #     nn.Conv2d(16, 8, 3, stride=1, padding=1,bias = True),  # 
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(),

        #     nn.Conv2d(8, 1, 3, stride=1, padding=1,bias = True),  # 
        #     nn.Dropout(0.2), 
        #     nn.BatchNorm2d(1),
            
        #     nn.ReLU(),
    

        # )
        # decoder = nn.Sequential(
        #     nn.Conv2d(2, 8, 3, stride=1, padding=1,bias = True),
        #     nn.Dropout(0.2), 
        #     nn.MaxPool2d(2),
        #     nn.BatchNorm2d(8), 
        #     nn.ReLU(),

        #     nn.Conv2d(8, 16, 3, stride=1, padding=1,bias = True),
            
        #     nn.MaxPool2d(3), 
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),


        #     nn.Conv2d(16, 1, 3, stride=1, padding=1,bias = True),  # b, 8, 3, 3
        #     nn.MaxPool2d(3),  
        #     nn.BatchNorm2d(1),
        #     nn.MaxPool2d(2),  
        #     nn.ReLU(),
        #     # nn.tempsigmoid(out),
        # )
##########################2D Brain Net#################################################
        # encoder = nn.Sequential(
        #     nn.Conv2d(1, 8, 3, stride=1, padding=1,bias = True),  #   Loss = MSE ; 'batch_size'] = 256 'learning_rate' = 5e-3
        #     nn.Dropout(0.2), 
        #     nn.MaxPool2d(2),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(),
         
        #     nn.Conv2d(8, 16, 3, stride=1, padding=1,bias = True),  # 
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
                   
        #     nn.Conv2d(16, 8, 3, stride=1, padding=1,bias = True),  # 
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(),

        #     nn.Conv2d(8, 1, 3, stride=1, padding=1,bias = True),  # 
        #     nn.Dropout(0.2), 
        #     nn.BatchNorm2d(1),
            
        #     nn.ReLU(),
    

        # )
        # decoder = nn.Sequential(
        #     nn.Conv2d(2, 8, 3, stride=1, padding=1,bias = True),
        #     nn.Dropout(0.2), 
        #     nn.MaxPool2d(2),
        #     nn.BatchNorm2d(8), 
        #     nn.ReLU(),

        #     nn.Conv2d(8, 16, 3, stride=1, padding=1,bias = True),
            
        #     nn.MaxPool2d(3), 
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),

        #     nn.Conv2d(16, 8, 3, stride=1, padding=1,bias = True),
            
        #     nn.MaxPool2d(3), 
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(),

        #     nn.Conv2d(8, 1, 3, stride=1, padding=1,bias = True),  
        #     nn.BatchNorm2d(1),
        #     nn.MaxPool2d(2),  
        #     nn.ReLU(),
        #     # nn.tempsigmoid(out),
        # )
##########################3D Brain Net#################################################
        encoder = nn.Sequential(
            nn.Conv3d(1, 8, 3, stride=1, padding=1,bias = True),  #   Loss = MSE ; 'batch_size'] = 256 'learning_rate' = 5e-3
            nn.Dropout(0.2), 
            nn.MaxPool3d(2),
            nn.BatchNorm3d(8),
            nn.PReLU(),
         
            nn.Conv3d(8, 16, 3, stride=1, padding=1,bias = True),  # 
            nn.BatchNorm3d(16),
            nn.PReLU(),
                   
            nn.Conv3d(16, 8, 3, stride=1, padding=1,bias = True),  # 
            nn.BatchNorm3d(8),
            nn.PReLU(),

            nn.Conv3d(8, 1, 3, stride=1, padding=1,bias = True),  # 
            nn.Dropout(0.2), 
            nn.BatchNorm3d(1),
            
            nn.PReLU(),
    

        )
        decoder = nn.Sequential(
            nn.Conv3d(2, 8, 3, stride=1, padding=1,bias = True),
            nn.Dropout(0.2), 
            nn.MaxPool3d(2),
            nn.BatchNorm3d(8), 
            nn.PReLU(),

            nn.Conv3d(8, 16, 3, stride=1, padding=1,bias = True),
            
            nn.MaxPool3d(3), 
            nn.BatchNorm3d(16),
            nn.PReLU(),

            nn.Conv3d(16, 8, 3, stride=1, padding=1,bias = True),
            
            nn.MaxPool3d(3), 
            nn.BatchNorm3d(8),
            nn.PReLU(),

            nn.Conv3d(8, 1, 3, stride=1, padding=1,bias = True),  
            nn.BatchNorm3d(1),
            nn.MaxPool3d(2),  
            nn.ReLU(),
        )
###########################Fourier Net (LEARNING RATE = 1E-2)##########################################
    return encoder, decoder


#from modules.Layers import CoordConv3d
from Layers import CoordConv3d



    
if __name__  == '__main__':    
    import numpy as np
    config = {'type': 'AECV3D',
              'paras': {
              'structure': 'default',
              'batchNorm': True,
              'root_feature_num':16}}
    config2 = {'type': 'DeepCC',
              'paras': {
              'structure': 'complex',
              'batchNorm': True,
              'root_feature_num':16}}
    # config =  = {'type':'AECV3D','paras':{'name':'debug'}}
    ae = getAENET(config)
    ae2 = getAENET(config2)
#    ae = AE(config = None)
    data_shape = (1, 1, 128, 128,128)
#    data_shape = (1, 1, 100, 128, 128)
    src = np.random.rand(*data_shape).astype(np.float32)
    tar = np.random.rand(*data_shape).astype(np.float32)

    # output = ae.forward(torch.from_numpy(data)).detach().numpy()  #source_input: (B, 1, 128, 128, 128)  source_output: (B, 1, 32, 32, 32)
    # output2 = ae.forward(torch.from_numpy(data)).detach().numpy()  #target_input: (B, 1, 128, 128, 128)  target_output: (B, 1, 32, 32, 32)
    # output3 = np.concatenate((output, output2), axis=1)   #input: (B, 1, 32, 32, 32) X 2   target_output: (B, 2, 32, 32, 32) 
    # output4 = ae2.forward(torch.from_numpy(output3)).detach().numpy()  #input: (B, 2, 32, 32, 32)    target_output: (B, 1, 16, 16, 16) 

    #print(output4.shape)
    
    config3 = {'type': 'DF',
              'paras': {
              'structure': 'deepflash',
              'batchNorm': True,
              'root_feature_num':16}}
    ae3 = getAENET (config3)
    output = ae3.forward(torch.from_numpy(src), torch.from_numpy(tar) ).detach().numpy()
    print(output.shape)