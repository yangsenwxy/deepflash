# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 20:10:32 2019

@author: remus
"""

import numpy as np
def generateNumbersInRangeUnif(min, max, count):
    return list(np.floor((max-min) * np.random.rand(count)).astype(int)+min)

# https://stackoverflow.com/questions/18441779/how-to-specify-upper-and-lower-limits-when-using-numpy-random-normal
from scipy.stats import truncnorm
def generateNumbersInRangeNorm(min, max, count, mean, std):
    get_truncated_normal = lambda mean=0, sd=1, low=0, upp=10: truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
    return list(get_truncated_normal(mean=mean*(max-min), sd=std*(max-min), low=min, upp=max).rvs(count).astype(int))

def addCoordChannels(img):
    # img should have shape [N, H, W, C] of [N, D, H, W, C]
    img_shape = np.shape(img)
    if len(img_shape)== 4:
        N, H, W, C = img_shape
        ys, xs = np.meshgrid(np.arange(0,H), np.arange(0,W), indexing='ij')
        xs = np.repeat(xs[np.newaxis, :, :, np.newaxis], N, axis=0)
        ys = np.repeat(ys[np.newaxis, :, :, np.newaxis], N, axis=0)
        x = np.concatenate((img, xs, ys), axis = 1)
    else:
        N, D, H, W, C = img_shape        
        zs, ys, xs = np.meshgrid(np.arange(0,D), np.arange(0,H), np.arange(0,W), indexing='ij')
        xs = np.repeat(xs[np.newaxis, :, :, :,np.newaxis ], N, axis=0)
        ys = np.repeat(ys[np.newaxis, :, :, :,np.newaxis ], N, axis=0)
        zs = np.repeat(zs[np.newaxis, :, :, :,np.newaxis ], N, axis=0)
        x = np.concatenate((img, xs, ys, zs), axis = -1)
    return x