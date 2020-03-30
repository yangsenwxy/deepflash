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
import sys
sys.path.append("..")
from utils.utils import generateNumbersInRangeUnif, generateNumbersInRangeNorm
#from ..utils import generateNumbersInRangeUnif, generateNumbersInRangeNorm

def crop(ndarray, options):
    '''
    @description: 
        Crop stacked 2D images into 2D patches, and 3D images into 3D patches
    @params:
        ndarray{numpy ndarray}:
            images, should be 2D images (N, H, W, C) or 3D (N, D, H, W, C)
                    following tensorflow standard
        options{dict}:
            Options for cropping. Should include:
                'method'{str}:
                    Crop method. Should be one of 'sliding', 'random', 'random_per_image'
                    sliding: crop images according to a sliding window with given shape.
                    random: randomly take patches with given total number (patch_count)
                    random_per_image: randomly take same number of patches (patch_count) on each image.
                            The total patch count would be img_count * patch_count
                'patch_shape'{tuple}:
                    Shape of patchs. Should be tuple (PH, PW) for 2D images and 
                    (PD, PH, PW) for 3D images (i.e. all channels are used and no need to provide channel number)
                'strides'{int}:
                    if crop_method == 'sliding', strides rules the distance of 
                    windows in all dimensions.
                'distribution'{str}:
                    if crop_method == 'random' or 'random_per_image', distribution rules how to sample in images    
                'distribution_paras'{dict}:
                    
                'patch_count'{int}:
                    if crop_method == 'random', patch_count determines the total patch count 
                    if crop_method == 'radnom_per_image', patch_count determines the number of patch on each image.
                        The total patch count would be img_count * patch_count           
    @return: 
        patches{numpy ndarray}:
            Should be tensors following tensorflow format, i.e.
                [N, H, W, C] for 2D images
                [N, D, H, W, C] for 3D images
            for grayscale 3D medical images, their 2D patchs should be 
                [NPatch, HPatch, WPatch, 1] and 3D patchs [NPatch, DPatch, HPatch, WPatch, 1]
    @Usage:
        See __main__()
    '''
    # Get shape info
    array_shape = np.shape(ndarray)
    img_count = array_shape[0]
    img_shape = array_shape[1:-1]
    
    # Generate crop boxes
    crop_boxes = generate_crop_boxes(img_count, img_shape, options)
    
    # Crop and return
    return crop_given_boxes(ndarray, options['patch_shape'], crop_boxes), crop_boxes

def crop_images(ndarrays: list, options: dict, concatFlag:bool = False):
    img_patches_list = []
    crop_boxes_list = []
    for ndarrayIdx, ndarray in enumerate(ndarrays):
        img_patches, crop_boxes = crop(ndarray, options)
        img_patches_list.append(img_patches)
        crop_boxes_list.append(crop_boxes)    
    return img_patches_list, crop_boxes_list

def generate_crop_boxes(img_count, img_shape, options):
    '''
    @description: Generate boxes give image size and patch size 
    @params:
        img_count{int}:
            number of images i.e. N
        img_shape{tuple, triple or quadruple}:
            size of single image. Should be [H, W] or [D, H, W]
        options{dict}:
            See crop()
    @return: 
        boxes{list(tuple)}:
            list of patch index, top left and bottom right index pairs.
            Should be [ (p1Idx, p1IdxTopLeft, p1IdxBottomRight),  (p2Idx, p2IdxTopLeft, p2IdxBottomRight), ...]
            where piIdxTopLeft = e.g.(1, 1) for 2D images and (1,1,1) for 3D images
            E.g. [ (0,(1,1),(2,2)), (0,(2,2),(3,3), ...] or [(0,(1,1,1), (2,2,2)), (10,(2,2,2), (3,3,3)), ...]
    '''
    crop_method = options['method']
    patch_shape = options['patch_shape']   
        
    if crop_method == 'sliding':
        strides = options.get('strides', (1,) * len(img_shape))
        return generate_silding_boxes(img_count, img_shape, patch_shape, strides)
    elif crop_method == 'random_per_image':
        patch_count = options['patch_count']
        dists = options['dists']
        return generate_random_boxes_per_image(img_count, img_shape, patch_shape, patch_count, dists)
    elif crop_method == 'random':
        patch_count = options['patch_count']
        dists = options['dists']
        return generate_random_boxes(img_count, img_shape, patch_shape, patch_count, dists)
    else:
        raise ValueError(f'Unsupoorted crop method: {crop_method}')    

def generate_silding_boxes(img_count, img_shape, patch_shape, strides):
    '''
    @description: 
        Generate index of top left and bottom right point of sliding windows
    @param {type}:
        img_count{int}, img_shape{tuple}, patch_shape{tuple}, strides{tuple}:
            See the paras of generate_crop_boxes()            
    @return: 
        boxes{list{tuple(tuple)}}:
            E.g. [(p1i, p1TL, p1BR), (p2i, l2TL, p2BR)]
    '''
    startIdxes = []
    startIdxesPerDim = []
    for dim in range(len(img_shape)):
        startIdxesDim = np.arange(0, img_shape[dim] - patch_shape[dim] + 1, strides[dim])
        if (img_shape[dim] - patch_shape[dim]) % strides[dim] != 0:
            np.append(startIdxesDim, img_shape[dim] - patch_shape[dim])
        startIdxesPerDim.append(startIdxesDim)    
    
    if len(img_shape) == 2:
        for patchIdxi in range(img_count):
            # For each image
            for patchDim0 in startIdxesPerDim[0]:
                # All starting indice of the 1st dimension
                for patchDim1 in startIdxesPerDim[1]:
                    # All starting indice of the 2nd dimension
                    startIdxes.append((patchIdxi, patchDim0, patchDim1))
    elif len(img_shape) == 3:
        for patchIdxi in range(img_count):
            for patchDim0 in startIdxesPerDim[0]:
                for patchDim1 in startIdxesPerDim[1]:
                    for patchDim2 in startIdxesPerDim[2]:
                        startIdxes.append((patchIdxi, patchDim0, patchDim1, patchDim2))
        
    return generate_boxes_given_startIdxes(startIdxes, patch_shape)
    
def generate_random_boxes_per_image(img_count, img_shape, patch_shape, patch_count_per_image, dists):
    '''
    @description: 
        Randomly generate index of top left and bottom right point of windows same amount per image
    @param {type}:
        img_count{int}, img_shape{tuple}, patch_shape{tuple}, patch_count_per_image{int}, distration{str}:
            See the paras of crop()
    @return: 
        boxes{list{tuple(tuple)}}:
            E.g. [(p1i, p1TL, p1BR), (p2i, l2TL, p2BR)]
    '''
    startIdxes = []
    startIdxes.append(np.arange(0,img_count).repeat(patch_count_per_image))
    
    for dim, dist in enumerate(dists):
        if dist['type'] == 'uniform':
            startIdxes.append(generateNumbersInRangeUnif(0, img_shape[dim] - patch_shape[dim] + 1, img_count*patch_count_per_image))
        elif dist['type'] == 'normal':            
            mean = dist['paras'].get('mean', 0.5)
            std = dist['paras'].get('std', 0.5)
            startIdxes.append(generateNumbersInRangeNorm(0, img_shape[dim] - patch_shape[dim] + 1, img_count*patch_count_per_image, mean, std))
        else:
            raise ValueError(f'Unsupported sampling method: {dist["type"]}')
    return generate_boxes_given_startIdxes(list(zip(*startIdxes)), patch_shape)

def generate_random_boxes(img_count, img_shape, patch_shape, patch_count, dists):
    '''
    @description: 
        Randomly generate index of top left and bottom right point of windows same amount per image
    @param {type}:
        img_count{int}, img_shape{tuple}, patch_shape{tuple}:
            See the paras of generate_crop_boxes()
        patch_count{int}:
            determines the total patch count 
        dists{list{dict{'type','paras'}}}:
            Rules how to sample in each dimension. Type could be 'uniform' or 'normal'
            if type=='uniform', no paras is needed
            if type=='normal', paras should contain mean and standard deviation, 
                i.e. dist['paras']['mean'] and dist['paras']['std'] . 
                Note that 
    @return: 
        boxes{list{tuple(tuple)}}:
            E.g. [(p1i, p1TL, p1BR), (p2i, l2TL, p2BR)]
    '''
    data_shape = [img_count, *img_shape]
    patch_shape_aug = [1, *patch_shape] # Augmented patch size. Add the dimension of index
    startIdxes = []
    
    for dim, dist in enumerate(dists):
        if dist['type'] == 'uniform':
            startIdxes.append(generateNumbersInRangeUnif(0, data_shape[dim] - patch_shape_aug[dim] + 1, patch_count))
        elif dist['type'] == 'normal':            
            mean = dist['paras'].get('mean', 0.5)
            std = dist['paras'].get('std', 0.5)
            startIdxes.append(generateNumbersInRangeNorm(0, data_shape[dim] - patch_shape_aug[dim] + 1, patch_count, mean, std))
        else:
            raise ValueError(f'Unsupported sampling method: {dist["type"]}')
    return generate_boxes_given_startIdxes(list(zip(*startIdxes)), patch_shape)
        

def generate_random_boxes_OLD(img_count, img_shape, patch_shape, patch_count, dist, paras):
    '''
    @description: 
        Randomly generate index of top left and bottom right point of windows same amount per image
    @param {type}:
        img_count{int}, img_shape{tuple}, patch_shape{tuple}:
            See the paras of generate_crop_boxes()            
        patch_count{int}:
            determines the total patch count 
        dist{str}:
            rules how to sample in images
    @return: 
        boxes{list{tuple(tuple)}}:
            E.g. [(p1i, p1TL, p1BR), (p2i, l2TL, p2BR)]
    '''
    if dist == 'uniform':
        generateStartIdxesInRange = lambda idxMin,idxMax: list(np.floor((idxMax-idxMin) * np.random.rand(patch_count)).astype(int)+idxMin)
        startIdxes = []
        patchIdxis = generateStartIdxesInRange(0,img_count)
        startIdxes = [patchIdxis]
        for dim in range(len(img_shape)):
            startIdxes.append(generateStartIdxesInRange(0, img_shape[dim] - patch_shape[dim] + 1))
        return generate_boxes_given_startIdxes(list(zip(*startIdxes)), patch_shape)
    elif dist.lower() == 'gaussian' or dist.lower() == 'normal':
        Sigma = paras['sigma']
        Mu = paras['mu']

def generate_boxes_given_startIdxes(startIdxes, patch_shape):
    '''
    @description: 
        Generate croping boxes given start index of each patch
    @param {type}:
        startIdxes{list(array)}:list of start index for each patch
        i.e. [(p1i,p1TLD1,p1TLD2),...]
    @return: 
        boxes{list(tuple(tuple))}:
            E.g. [(p1i, p1TL, p1BR), (p2i, l2TL, p2BR),...]
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
            list of boxes' "top left" and "bottom right" point index for EACH patch
            E.g. [((0,0,0,0),(1,1,1,1)), ((1,1,1,1),(2,2,2,2), ...]
            
    @return:
        patches{numpy ndarray}:
            Should be tensors following tensorflow format, i.e.
                [N, H, W, C] for 2D images
                [N, D, H, W, C] for 3D images
            for grayscale 3D medical images, their 2D patchs should be 
                [NPatch, HPatch, WPatch, 1] and 3D patchs [NPatch, DPatch, HPatch, WPatch, 1]
    '''
    channel_count = np.shape(ndarray)[-1]
    patch_count = len(crop_boxes)
    patches = -1*np.ones((patch_count, *patch_shape, channel_count))
    
    
    if len(patch_shape) == 2:
        # if take 2D patches
        for patchIdx in range(patch_count):
            crop_box = crop_boxes[patchIdx]
            patchi = crop_box[0]
            patchPTL = crop_box[1]  # Top left
            patchPBR = crop_box[2]  # Bottom right
            try:
                patches[patchIdx,:,:,:] = \
                    ndarray[patchi, patchPTL[0]:patchPBR[0], patchPTL[1]:patchPBR[1], :]
            except:
                print(crop_box)
    elif len(patch_shape) == 3:
        for patchIdx in range(patch_count):
            crop_box = crop_boxes[patchIdx]
            patchi = crop_box[0]
            patchPTL = crop_box[1]  # Top left
            patchPBR = crop_box[2]  # Bottom right
            patches[patchIdx,:,:,:,:] = \
                    ndarray[patchi, 
                            patchPTL[0]:patchPBR[0], 
                            patchPTL[1]:patchPBR[1], 
                            patchPTL[2]:patchPBR[2],
                            :]
    
    return patches

def unscramble_given_boxes(img_count, img_shape, patches, crop_boxes, merge_method = 'average', channel_count = 1):
    '''
    @description: 
        Stitch image patches according to cropping boxes.
    @params:
        img_count{int}:
            number of images i.e. N
        img_shape{tuple, triple or quadruple}:
            size of single image. Should be [H, W] or [D, H, W]
        crop_boxes{list(tuple)}:
            list of patch index, top left and bottom right index pairs.
            Should be [ (p1Idx, p1IdxTopLeft, p1IdxBottomRight),  (p2Idx, p2IdxTopLeft, p2IdxBottomRight), ...]
            where piIdxTopLeft = e.g.(1, 1) for 2D images and (1,1,1) for 3D images
            E.g. [ (0,(1,1),(2,2)), (0,(2,2),(3,3), ...] or [(0,(1,1,1), (2,2,2)), (10,(2,2,2), (3,3,3)), ...]
        merge_method{str}:
            Method to merge overlap areas. Now only support 'average', which takes the average of the overlapping values            
        channel_count{int}:
            The number of image channels
    @return: 
        patches{ndarray}:
            list of top left and bottom right index pairs.
            Should be [ (p1Idx, p1IdxTopLeft, p1IdxBottomRight),  (p2Idx, p2IdxTopLeft, p2IdxBottomRight), ...]
            where piIdxTopLeft = e.g.(1, 1) for 2D images and (1,1,1) for 3D images
            E.g. [ (0,(1,1),(2,2)), (0,(2,2),(3,3), ...] or [(0,(1,1,1), (2,2,2)), (10,(2,2,2), (3,3,3)), ...]
    '''
    if merge_method == 'average':
        filledSum = np.zeros([img_count, *img_shape, channel_count])
        filledCount = np.zeros([img_count, *img_shape, channel_count])
        for patchIdx in range(len(crop_boxes)):
            (patchi, patchTL, patchBR) = crop_boxes[patchIdx]   # (0, (0, 0, 0), (32, 32, 32))
            if len(patchTL) == 2:
                filledSum[patchi, patchTL[0]:patchBR[0], patchTL[1]:patchBR[1], :] += patches[patchIdx, :]
                filledCount[patchi, patchTL[0]:patchBR[0], patchTL[1]:patchBR[1], :] += 1
            elif len(patchTL) == 3:
                filledSum[patchi, patchTL[0]:patchBR[0], patchTL[1]:patchBR[1], patchTL[2]:patchBR[2], :] += patches[patchIdx, :]
                filledCount[patchi, patchTL[0]:patchBR[0], patchTL[1]:patchBR[1], patchTL[2]:patchBR[2], :] += 1
        return np.divide(filledSum, filledCount, out=np.zeros_like(filledSum), where=filledCount!=0)
            


if __name__ == '__main__':        
    ndarray_2d = np.random.rand(100, 128, 128, 1)
    ndarray_3d = np.random.rand(1, 100, 128, 128, 1)
    
    ndarray= ndarray_2d
    
    img_shape = np.shape(ndarray)[1:-1]
    img_dim = len(img_shape)
    patch_shape = (32,)*img_dim
    
    options_sliding = {
            'crop_method': 'sliding',
            'patch_shape': patch_shape,
            'strides':(16,)*img_dim}
    options_random = {
                    'method': 'random',
                    'patch_count': 1000,
                    'dists':[
                            {'type': 'normal', 'paras': {'mean':0.5,'std':1}},
                            {'type': 'normal', 'paras': {'mean':0.5,'std':1}},
                            {'type': 'normal', 'paras': {'mean':0.5,'std':1}}
                            ],
                    'patch_shape': patch_shape,
                    'strides':(4,)*img_dim
                    }
    options_random_per_image = {
                    'method': 'random_per_image',
                    'patch_count': 100,
                    'dists':[
                            {'type': 'normal', 'paras': {'mean':0.5,'std':500}},
                            {'type': 'normal', 'paras': {'mean':0.5,'std':500}}
                            ],
                    'patch_shape': patch_shape,
                    'strides':(4,)*img_dim
                    }

    ndarray_patches, crop_boxes = crop(ndarray, options_random)
    ndarray_stitched = unscramble_given_boxes(len(ndarray), img_shape, ndarray_patches, crop_boxes)
    diff_thres = 0.1
    print(np.sum((ndarray-ndarray_stitched)<diff_thres)/ndarray.size)