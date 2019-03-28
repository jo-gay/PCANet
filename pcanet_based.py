#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 19:38:07 2019

Implementation of Zhu et al PCA-Net based thingy

@author: jo
"""

from pcanet import *

import itertools

from chainer.cuda import to_gpu, to_cpu
from chainer.functions import convolution_2d

import numpy as np
from sklearn.decomposition import IncrementalPCA
from scipy.special import expit #fast sigmoid function

from utils import gpu_enabled


if gpu_enabled():
    try:
        import cupy as xp
    except ImportError:
        import numpy as xp
else:
    import numpy as xp


class PCANetBased(PCANet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = 0.005
        self.S0 = expit(0) #S0 = 0.5
        
    
    def transform(self, images):
        """
        Parameters
        ----------
        image: np.ndarray
            | Color / grayscale image of shape
            | (height, width, n_channels) or
            | (height, width)

        Returns
        -------
        X: np.ndarray
            A set of feature vectors of shape (n_images, n_features)
            where :code:`n_features` is determined by the hyperparameters
            
        
        NB: It is necessary to call the fit function first, to set the filters.
        """
        
        #images.shape == (1, h, w[, n_channels])
        images = self.process_input(images)
        #Now images.shape == (1, n_channels=1, y, x)

        #Retrieve the layer 1 filters
        filters_l1 = components_to_filters(
            self.pca_l1.components_,
            n_channels=images.shape[1],
            filter_shape=self.filter_shape_l1,
        )

        #Retrieve the layer 2 filters
        filters_l2 = components_to_filters(
            self.pca_l2.components_,
            n_channels=1,
            filter_shape=self.filter_shape_l2
        )

        if gpu_enabled():
            images = to_gpu(images)
            filters_l1 = to_gpu(filters_l1)
            filters_l2 = to_gpu(filters_l2)

        #Apply the layer 1 filters
        l1output = convolution_2d(
            images,
            filters_l1,
            stride=self.step_shape_l1,
            pad=1
        ).data
                
        #The l1 output has shape (n_images, L1, y, x), swap axes
        l1output = xp.swapaxes(l1output, 0, 1)

        # Now l1output.shape == (L1, n_images, y, x)
        # iterate over each L1 output, applying L2 filters
        l2output = []
        for maps in l1output:
            n_images, h, w = maps.shape
            l2maps = convolution_2d(
                maps.reshape(n_images, 1, h, w),  # 1 channel images
                filters_l2,
                stride=self.step_shape_l2,
                pad=1
            ).data

            # l2maps.shape == (n_images, L2, y, x)
            # Apply sigmoid and sum over the L2 channels
            Z = np.zeros((n_images,*l2maps.shape[2:]))
            for chan in range(l2maps.shape[1]):
                Z += pow(2, 8-chan) * (expit(self.alpha*np.abs(l2maps[:,chan,...]))-self.S0)

            l2output.append(Z)

        #JCG comment: l2output contains L1 elements of shape (n_images, h, w)

        if gpu_enabled():
            l2output = to_cpu(l2output)

        return images, l1output, l2output

    def fuse_maps(self, layerout):
        """
        Take the output feature maps obtained from L1 filters and fuse them into
        a single feature map based on sum of squares divided by L1^2
        """
        F = np.zeros(layerout[0].shape)
        for l in layerout:
            F += pow(l, 2)
        return F / pow(len(layerout), 2)
    
    def calc_h1(self, image):
        """
        Calculate the parameter h1 for each pixel in the image provided
        """
        X = Patches(image, (3,3), (1,1), pad=True).patches #TODO: don't hard-code filter size and step size

        # X.shape == (n_patches, filter_height, filter_width)
        X = X.reshape(X.shape[0], -1)  # flatten each patch
        #Now X.shape == (n_patches, filter_height*filter_width)
        
        #For each pixel/patch get the average diff between the centre pixel and the surrounding ones
        sigma11 = abs(9*X[:,4] - X.sum(axis=1))/8 # TODO: also hard-coded to 8-neighbourhood
        
        sigma12 = sigma11[sigma11!=0].mean()
        
        #TODO: C1 and C2 are tuned for MRI images - what happens if you change these?
        c1=0.8
        c2=0.6
        
        h1 = c1*sigma11 + c2*sigma12
        return h1.reshape(image.shape)

    def calc_h2(self, l1out):
        """
        Calculate the parameter h2 for each pixel by averaging over the L1 outputs provided.
        The shape of the parameter :code:`l1out` should be (L1, 1, w, h)
        """
        sigma21 = np.zeros(l1out.shape[-2:])
        for i in range(l1out.shape[0]):
            sigma21 += l1out[i][0]
        sigma21 = abs(sigma21) / l1out.shape[0]
        
        sigma22 = sigma21[sigma21!=0].mean()
        
        #TODO: C1 and C2 are tuned for MRI images - what happens if you change these?
        c1=0.8
        c2=0.6
        
        h2 = c1*sigma21 + c2*sigma22
#        return h2.reshape(l1out.shape[-2:])
        return h2
    
    
    def create_PSR(self, image):
        """
        Create a multi-level PCANet-based structural representation of the image provided, based
        on PCA filters that have already been calculated.
        """
        
        #Promote to set of images to make use of preexisting functionality that expects multiple images
        images = np.reshape(image, (1, *image.shape))

        images, l1out, l2out = self.transform(images)

        F1 = self.fuse_maps(l1out)
        F2 = self.fuse_maps(l2out)
        
        h1 = self.calc_h1(images[0][0]) #Calculate h1 for the 1st channel of the 1st (only) image
        h2 = self.calc_h2(l1out)

        PSR = np.exp(-F1/h1)*np.exp(-F2/h2)
        return PSR
    