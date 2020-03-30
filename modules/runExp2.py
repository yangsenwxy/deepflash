import torch
import sys
from configs.getConfig import getConfig
from utils.io import saveConfig2Json
from utils.io import createExpFolder
def runExp(config, configName, resultPath = '../result', continueTraining = False, oldExpPath = None, addDate = True):
    #%% 1. Set configration and device    
    device = torch.device("cpu")

    # Create experiment result folder    
    expPath, expName = createExpFolder(resultPath, configName, create_subFolder=True, addDate = addDate)
    saveConfig2Json(config, expPath + '/config.json')

    #%% 2. Set Data
    import numpy as np
    from utils.io import safeLoadMedicalImg, convertTensorformat, loadData2

    # Load training and valid data
    SEG, COR, AXI = [0,1,2]
    targetDim = 3

    import os
    directory = './data'

    for filename in os.listdir(directory):
        if filename.endswith(".mhd"):
            f = open(filename)
            lines = f.read()
            print (lines[10])
            training_data = convertTensorformat(img=safeLoadMedicalImg(f),
                               sourceFormat = sourceFormat, 
                               targetFormat = targetFormat, 
                               targetDim = targetDim, 
                               sourceSliceDim = sourceSliceDim) 
 

#     valid_data = convertTensorformat(img=safeLoadMedicalImg(config['data']['valid'][0]),
#                                 sourceFormat = 'single3DGrayscale', 
#                                 targetFormat = 'pytorch', 
#                                 targetDim = targetDim, 
#                                 sourceSliceDim = AXI)
    
#     #%% 3. Set transformations
#     from torchvision import transforms
#     from utils.io import safeDivide
#     #xNorm = lambda img : (img - np.min(img)) / (np.max(img) - np.min(img)) - 0.5
#     xNorm = lambda img : safeDivide(img - np.min(img), (np.max(img) - np.min(img))) - 0.5
#     trans3DTF2Torch = lambda img: np.moveaxis(img, -1, 0)
#     img_transform = transforms.Compose([
#         xNorm,
#         trans3DTF2Torch
#     #    transforms.ToTensor()
#     ])

#     #%% 4. Set dataset


#     training_dataset = DataSet2D(imgs = training_data, transform=img_transform, device = device)

#     #%% 5. Set network
#     from modules.AEModel import AEModel
#     autoencoder = AEModel(net_config = config['net'], 
#                         loss_config = config['loss'], 
#                         device=device)
#     if continueTraining:
#         # autoencoder.load(oldExpPath + '/model/model.pth')
#         autoencoder.load(oldExpPath + '/checkpoint/checkpoint.pth')

#     #%% 6. Training
#     print('Start configure: ' + expName)
#     loss_history, past_time = autoencoder.train(training_dataset=training_dataset, training_config = config['training'], valid_img=valid_data, expPath = expPath)
#     autoencoder.save(expPath + '/checkpoint/checkpoint.pth')

#     # Add loss and training time to config json file
#     config['training']['loss'] = loss_history[-1]
#     config['training']['time_hour'] = past_time / 3600
#     saveConfig2Json(config, expPath + '/config.json')

#     return expPath, expName, loss_history, past_time

# def runExpGroup(configGroup, configGruopName, resultPath = '../result', continueTraining = False, oldExpPath = None):    
#     expGroupPath, _ = createExpFolder(resultPath, configGruopName)
#     # resultPath += '/' + expGroupName
#     for expIdx, expConfig in enumerate(configGroup):
#         runExp(expConfig, f'idx-{expIdx}', expGroupPath, addDate = False)