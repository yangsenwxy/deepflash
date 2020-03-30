#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@Description: Utility function to crop images into patches
@Author: Jerry Xing
@Date: 2019-06-30 15:06:17
@LastEditTime: 2019-07-01 09:43:32
@LastEditors: Please set LastEditors
'''

'''
    @description: 
    @param {type} 
    @return: 
'''

"""
https://github.com/pytorch/pytorch/issues/3387
https://discuss.pytorch.org/t/how-to-extract-smaller-image-patches-3d/16837/4
https://blog.csdn.net/loseinvain/article/details/88139435
https://stackoverflow.com/questions/53972159/how-does-pytorchs-fold-and-unfold-work
"""

import numpy as np
#import warnings

def crop(ndarray, options):
    '''
    @description: 
        Crop stacked 2D images into 2D patches, and 3D images into 3D patches
    @params:
        ndarray{numpy ndarray}:
            images, should be 2D images (N, H, W, C) or 3D (N, D, H, W, C)
                    following tensorflow standard
    @return: 
        patches{numpy ndarray}:
            Should be tensors following tensorflow format, i.e.
                [N, H, W, C] for 2D images
                [N, D, H, W, C] for 3D images
            for grayscale 3D medical images, their 2D patchs should be 
                [NPatch, HPatch, WPatch, 1] and 3D patchs [NPatch, DPatch, HPatch, WPatch, 1]
    '''
    array_shape = np.shape(ndarray)
    img_count = array_shape[0]
    img_shape = array_shape[1:-1]
    
    crop_mode = options['crop_mode']
    patch_shape = options['patch_shape']
    strides = options.get('strides', None)
    distributions = options.get('distributions', 'uniform')
    patch_count = options.get('patch_count', 0)

    crop_boxes = generate_crop_boxes(img_count, img_shape, crop_mode, patch_shape,
                                     strides, distributions, patch_count)
    return crop_given_boxes(ndarray, patch_shape, crop_boxes), crop_boxes
    

def generate_crop_boxes(img_count, img_shape, crop_mode, patch_shape, strides = None, 
                        distribution = 'uniform', patch_count = 0):
    '''
    @description: Generate boxes give image size and patch size 
    @params:
        img_count{int}:
            number of images i.e. N
        img_shape{tuple, triple or quadruple}:
            size of single image. Should be [H, W] or [D, H, W]
        crop_mode{str}:
            Crop method. Should be one of 'sliding', 'random'
        patch_shape:
            Shape of patchs. Should be tuple (PH, PW) for 2D images and 
            (PD, PH, PW) for 3D images (i.e. all channels are used)
        strides{int}:
            if crop_mode == 'sliding', strides rules the distance of 
            windows in all dimensions.
        distribution{str}:
            if crop_mode == 'random', distribution rules how to sample in images    
    @return: 
        patches{ndarray}:
            list of top left and bottom right index pairs.
            Should be [ (p1IdxTopLeft,p1IdxBottomRight),  (p2IdxTopLeft,p2IdxBottomRight), ...]
            where piIdxTopLeft = e.g.(1, 1) for 2D images and (1,1,1) for 3D images
            E.g. [ ((1,1),(2,2)), ((2,2),(3,3), ...] or [((1,1,1), (2,2,2)), ((2,2,2), (3,3,3)), ...]
    '''
    
    if strides == None:
        strides = (1,) * len(img_shape)
        
    if crop_mode == 'sliding':
        return generate_silding_boxes(img_count, img_shape, patch_shape, strides)
#    elif crop_mode == 'random_per_image':
#        if patch_count_per_image == 0:
#            raise ValueError('Should provide the amount patches per image')
#        return generate_random_boxes_per_image(img_shape, patch_shape, patch_count_per_image, distribution)
    elif crop_mode == 'random':
        return generate_random_boxes(img_count,img_shape, patch_shape, patch_count, distribution)
    else:
        raise ValueError(f'Unsupoorted crop mode: {crop_mode}')    

def generate_silding_boxes(img_count, img_shape, patch_shape, strides):
    '''
    @description: 
        Generate index of top left and bottom right point of sliding windows PER IMAGE
    @param {type} 
    @return: 
        boxes{list{tuple(tuple)}}. E.g. [(p1i, p1TL, p1BR), (p2i, l2TL, p2BR)]
    '''
    startIdxes = []
    if len(img_shape) == 2:
        for patchIdxi in range(img_count):
            for patchDim0 in np.arange(0, img_shape[0] - patch_shape[0] + 1, strides[0]):
                for patchDim1 in np.arange(0, img_shape[1] - patch_shape[1] + 1, strides[1]):
                    startIdxes.append((patchIdxi, patchDim0, patchDim1))
    elif len(img_shape) == 3:
        for patchIdxi in range(img_count):
            for patchDim0 in np.arange(0, img_shape[0] - patch_shape[0] + 1, strides[0]):
                for patchDim1 in np.arange(0, img_shape[1] - patch_shape[1] + 1, strides[1]):
                    for patchDim2 in np.arange(0, img_shape[2] - patch_shape[2] + 1, strides[1]):
                        startIdxes.append((patchIdxi, patchDim0, patchDim1, patchDim2))
        
    return generate_boxes_given_startIdxes(startIdxes, patch_shape)
    
def generate_random_boxes_per_image(img_shape, patch_shape, patch_count_per_image, distribution = 'uniform'):
    startIdxes = []
    for dim in range(len(patch_shape)):
        startIdxes.append(list(((img_shape[dim] - patch_shape[dim] + 1) * np.random.rand(patch_count_per_image)).astype(int)))
    return generate_boxes_given_startIdxes(startIdxes, patch_shape)

def generate_random_boxes(img_count, img_shape, patch_shape, patch_count, distribution):
    generateStartIdxesInRange = lambda idxMin,idxMax: list(np.round((idxMax-idxMin) * np.random.rand(patch_count)).astype(int)+idxMin)
    startIdxes = []
    patchIdxis = generateStartIdxesInRange(0,img_count)
    startIdxes = [patchIdxis]
    for dim in range(len(img_shape)):
        startIdxes.append(generateStartIdxesInRange(0, img_shape[dim] - patch_shape[dim] + 1))
    haha = generate_boxes_given_startIdxes(list(zip(*startIdxes)), patch_shape)
    return haha

def generate_boxes_given_startIdxes(startIdxes, patch_shape):
    '''
    @description: 
    @param {type}:
        startIdxes{list(array)}:list of start index for each patch
        i.e. [(p1i,p1TLD1,p1TLD2),...]
        //startIdxes{list(array)}: list of start index in each dimension,         
        //[(p1i,p2i,...),(p1D1startIdx, p2D1startIdx, ...), (p1D2startIdx, p2D2startIdx, ...)]
            //e.g. [ (1,2,3,4,5...10), (1,3,5, ..., 20) ]
    @return: 
        boxes{list(tuple(tuple))}. E.g. [(p1i, p1TL, p1BR), (p2i, l2TL, p2BR),...]
    '''
    boxes = []
    if len(startIdxes[0])-1 == 2:
        for patchIdx in range(len(startIdxes)):
            patchIdxi = startIdxes[patchIdx][0]
            patchIdxDim0 = startIdxes[patchIdx][1]
            patchIdxDim1 = startIdxes[patchIdx][2]
            boxes.append((
                    patchIdxi,
                    (patchIdxDim0, patchIdxDim1), 
                    (patchIdxDim0 + patch_shape[0], patchIdxDim1 + patch_shape[1])) )
    elif len(startIdxes[0])-1 == 3:
        for patchIdx in range(len(startIdxes)):
            patchIdxi = startIdxes[patchIdx][0]
            patchIdxDim0 = startIdxes[patchIdx][1]
            patchIdxDim1 = startIdxes[patchIdx][2]
            patchIdxDim2 = startIdxes[patchIdx][3]
            boxes.append((
                    patchIdxi,
                    (patchIdxDim0, patchIdxDim1, patchIdxDim2), 
                    (patchIdxDim0 + patch_shape[0], patchIdxDim1 + patch_shape[1],patchIdxDim2 + patch_shape[2])) )
#    if len(startIdxes) == 2:
#        for patchIdx0 in startIdxes[0]:
#            for patchIdx1 in startIdxes[1]:
#                boxes.append(((patchIdx0, patchIdx1), 
#                             (patchIdx0 + patch_shape[0], patchIdx1 + patch_shape[1])) )
#    elif len(startIdxes) == 3:
#        for patchIdx0 in startIdxes[0]:
#            for patchIdx1 in startIdxes[1]:
#                for patchIdx2 in startIdxes[2]:
#                    boxes.append(((patchIdx0, patchIdx1, patchIdx2),
#                                 (patchIdx0 + patch_shape[0], patchIdx1 + patch_shape[1], patchIdx2 + patch_shape[2]) ))
    
    return boxes

def crop_given_boxes(ndarray, patch_shape, crop_boxes):
    '''
    @description:
        Crop images given boxes
    @param: 
        ndarray {numpy ndarray}: 
            images, should be 2D images (N, H, W, C) or 3D (N, D, H, W, C)
                    following tensorflow standard
        patch_shape:
            Shape of patchs. Should be tuple (PH, PW) for 2D images and 
            (PD, PH, PW) for 3D images (i.e. all channels are used)            
        crop_boxes {list of tuples}: 
            list of boxes' "top left" and "bottom right" point index for EACH image
            E.g. [((0,0,0,0),(1,1,1,1)), ((1,1,1,1),(2,2,2,2), ...]
            
    @return:
        patches{numpy ndarray}:
            Should be tensors following tensorflow format, i.e.
                [N, H, W, C] for 2D images
                [N, D, H, W, C] for 3D images
            for grayscale 3D medical images, their 2D patchs should be 
                [NPatch, HPatch, WPatch, 1] and 3D patchs [NPatch, DPatch, HPatch, WPatch, 1]
    '''
#    patch_shape = tuple(np.subtract(crop_boxes[0][1], crop_boxes[0][0]))    
#    patch_count_per_img = len(crop_boxes)
#    img_count = np.shape(ndarray)[0]
#    patch_count = patch_count_per_img * img_count
    channel_count = np.shape(ndarray)[-1]
#    patches = -1*np.ones((patch_count, *patch_shape, channel_count))
    patch_count = len(crop_boxes)
    patches = -1*np.ones((patch_count, *patch_shape, channel_count))
    
    
    if len(patch_shape) == 2:
        # if take 2D patches
        for patchIdx in range(patch_count):
            crop_box = crop_boxes[patchIdx]
            patchi = crop_box[0]
            patchPTL = crop_box[1]  # Top left
            patchPBR = crop_box[2]  # Bottom right
            patches[patchIdx,:,:,:] = \
                    ndarray[patchi, patchPTL[0]:patchPBR[0], patchPTL[1]:patchPBR[1], :]
    elif len(patch_shape) == 3:
        for patchIdx in range(patch_count):
            crop_box = crop_boxes[patchIdx]
            patchi = crop_box[0]
            patchPTL = crop_box[1]  # Top left
            patchPBR = crop_box[2]  # Bottom right
            patches[patchIdx,
                    :,:,:,:] = \
                    ndarray[patchi, 
                            patchPTL[0]:patchPBR[0], 
                            patchPTL[1]:patchPBR[1], 
                            patchPTL[2]:patchPBR[2],
                            :]
    
    return patches

#def stitch_given_boxes(ndarray_shape, patches, crop_boxes, merge_mode, channel_count = 1):    
#    if merge_mode == 'average':
#        pass

def stitch_given_boxes(img_count, img_shape, patches, crop_boxes, merge_mode = 'average', channel_count = 1):
    if merge_mode == 'average':
        filledSum = np.zeros([img_count, *img_shape, channel_count])
        filledCount = np.zeros([img_count, *img_shape, channel_count])
        for patchIdx in range(len(crop_boxes)):
            (patchi, patchTL, patchBR) = crop_boxes[patchIdx]   # (0, (0, 0, 0), (32, 32, 32))
            if len(patchTL) == 2:
                filledSum[patchi, patchTL[0]:patchBR[0], patchTL[1]:patchBR[1], :] = patches[patchIdx, :]
                filledCount[patchi, patchTL[0]:patchBR[0], patchTL[1]:patchBR[1], :] += 1
            elif len(patchTL) == 3:
                filledSum[patchi, patchTL[0]:patchBR[0], patchTL[1]:patchBR[1], patchTL[2]:patchBR[2], :] = patches[patchIdx, :]
                filledCount[patchi, patchTL[0]:patchBR[0], patchTL[1]:patchBR[1], patchTL[2]:patchBR[2], :] += 1
        return filledSum/filledCount
            
#def stitch_given_boxes(img_shape, patches, crop_boxes, options, channel_count = 1):
#    # Create flag matrix indicating has place has been written
#    filledFlag = np.zeros(img_shape)
#    img_stitched = -1 * np.ones(1, *img_shape, channel_count)
#    patch_count = np.shape(patches)[0]
#    patch_shape = np.shape(patches[0,:])
#    img_dim = len(img_shape)
#    for patchIdx in range(patch_count):
#        patch = patches[patchIdx, :]
#        crop_box = crop_boxes[patchIdx]
#        patchPTL = crop_box[0]  # Top left, (1,1) or (1,1,1)
#        patchPBR = crop_box[1]  # Bottom right, (2,2) or (2,2,2)
#        patchLocationFlag = np.zeros(img_shape)
#        if img_dim == 2:
#            patchLocationFlag[patchPTL[0]:patchPBR[0], patchPTL[1]:patchPBR[1]] = 1
#        elif img_dim == 3:
#            patchLocationFlag[patchPTL[0]:patchPBR[0], patchPTL[1]:patchPBR[1], patchPTL[2]:patchPBR[2]] = 1
#        
#        filledFlagInPatch = np.zeros()
##        filledFlagInPatch = patchLocationFlag * filledFlag
##        blankFlagInPatch = patchLocationFlag * (1 - filledFlag)                
#        
##        img_stitched[:, filledFlagInPatch, :] = \
##            0.5*(img_stitched[:, filledFlagInPatch, :] + patch[np.newaxis, :, :, np.newaxis][:,filledFlagInPatch,:])
##        img_stitched[:, blankFlagInPatch, :] = \
##            patch[np.newaxis, :, :, np.newaxis][:,blankFlagInPatch,:]
#        filledFlag[patchLocationFlag] = 1
##        filledPositionsInPatch = 
##        filledFlag


if __name__ == '__main__':    
#    image_shape = (1, 100, 32, 32, 1)
#    from utils.io import safeLoadMedicalImg, convertMedicalTensorTo, convertTensorformat    
    image = safeLoadMedicalImg('../../data/ep2d_dbsi_22_3X3mm_4d_004.nii')
#    image = safeLoadMedicalImg('../data/ep2d_dbsi_22_3X3mm_4d_004.nii')
    img_dim = 3
    image = convertMedicalTensorTo(img=image, lang='tensorflow', dim=img_dim, sliceDim=2)
    image_shape = np.shape(image)[1:-1]
#    patch_shape = (32,)*img_dim
    patch_shape = (32,32,32)
#    ndarray = np.random.rand(*image_shape)
    options = {
            'crop_mode': 'sliding',
            'patch_shape': patch_shape,
            'strides':(16,)*img_dim}
#    options = {
#            'crop_mode': 'random',
#            'patch_shape': patch_shape,
#            'patch_count':5}
    image_patches, crop_boxes = crop(image, options)
    image_patches_torch = convertTensorformat(img=image_patches, 
                                              sourceFormat='tensorflow',
                                              targetFormat='pytorch',
                                              targetDim=2, sourceSliceDim=0)
    img_stitched = stitch_given_boxes(len(image), image_shape, image_patches, crop_boxes)
#    import matplotlib.pyplot as plt
#    plt.imshow(np.squeeze(ndarray_croped[2,16,:,:,:]), cmap='gray')
    