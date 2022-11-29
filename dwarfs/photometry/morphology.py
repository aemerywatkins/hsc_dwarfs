#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 08:58:59 2022

@author: awatkins
"""
import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve_fft
from skimage.transform import rotate
# from . import photometry as phot
import photometry as phot
# from . import utils
import utils


class Clumpiness():
    '''
    Class for calculating the clumpiness factor S

    CHECK THIS OVER CAREFULLY; PROBABLY NOT DOING IT RIGHT
    '''

    def __init__(self, maskedImageArray, kernelSize):
        '''
        Initialize class instance with masked image (numpy.ma.core.MaskedArray)
        and a size for the convolution kernel.  Recommendation is
        0.25 x Petrosian radius.
        '''
        if type(maskedImageArray) == np.ma.core.MaskedArray:
            data = np.zeros(maskedImageArray.shape) + maskedImageArray.data
            mask = np.zeros(maskedImageArray.shape, dtype=bool)\
                + maskedImageArray.mask
            maskedImageArray = np.ma.masked_array(data, mask=mask)
        else:
            maskedImageArray = np.zeros(maskedImageArray.shape)\
                + maskedImageArray
        self.maskedImageArray = maskedImageArray
        self.kernel = Gaussian2DKernel(kernelSize, kernelSize)
        self.kernelSize = kernelSize

    def convolveDiff(self, mask):
        '''
        Returns a difference between a masked image convolved with a Gaussian
        kernel and a masked image at native resolution.

        Parameters
        ----------
        mask : numpy.ndarray
            Boolean array of pixels you want masked in the original image

        Returns
        -------
        diff : numpy.ndarray
            Convolved image - image
        '''
        image = np.zeros(self.maskedImageArray.shape)\
            + self.maskedImageArray.data
        image[mask] = np.nan
        conv_image = convolve_fft(image, self.kernel, preserve_nan=True)
        diff = image - conv_image

        return diff

    def galaxyI0(self, galX, galY):
        '''
        For calculating the galaxy's summed flux, for the denominator

        Parameters
        ----------
        galX : float
            Galaxy central x-coordinate
        galY : float
            Galaxy central y-coordinate

        Returns
        -------
        i0 : float
            Summed galaxy flux, pre-convolution
        '''
        # Must ignore central pixels, within kernel radius
        circle_mask = utils.imCircle(galX, galY, self.kernelSize,
                                     self.maskedImageArray.data)
        im_mask = self.maskedImageArray.mask
        mask = circle_mask | im_mask
        i0 = np.sum(self.maskedImageArray[~mask])

        return i0

    def background(self, galX, galY, galRad):
        '''
        For estimating the background value of S, which needs to be subtracted
        from the value of S derived for the galaxy itself.

        Parameters
        ----------
        galX : float
            Galaxy central x-coordinate
        galY : float
            Galaxy central y-coordinate
        galRad : float
            Radius out to which to mask the galaxy for background estimation

        Returns
        -------
        sBackground : float
            The value of S for the background pixels
        ratio : float
            Ratio of npix_I/npix_BG, where I is the galaxy image and BG is the
            background pixels.
        '''
        circle_mask = utils.imCircle(galX, galY, galRad,
                                     self.maskedImageArray.data)
        mask = self.maskedImageArray.mask | circle_mask

        # Parameters for correction
        i0 = self.galaxyI0(galX, galY)
        bg_size = len(mask[~mask])
        im_size = len(self.maskedImageArray[~self.maskedImageArray.mask])
        ratio = im_size / bg_size

        diff = self.convolveDiff(mask)
        diff[diff < 0] = 0  # Must ignore negative pixels for this

        sBackground = np.nansum(diff)/i0

        return sBackground, ratio

    def image(self, galX, galY, sBackground, ratio):
        '''
        For final computation of S parameter for galaxy

        Parameters
        ----------
        galX : float
            Galaxy central x-coordinate
        galY : float
            Galaxy central y-coordinate
        sBackground : float
            The value of S for the background pixels
        ratio : float
            Ratio of npix_I/npix_BG, where I is the galaxy image and BG is the
            background pixels.

        Returns
        -------
        sImage : float
            The value of S for the galaxy pixels, corrected for the local
            background
        '''
        # Must ignore central pixels, within kernel radius
        circle_mask = utils.imCircle(galX, galY, self.kernelSize,
                                     self.maskedImageArray.data)
        im_mask = self.maskedImageArray.mask
        mask = circle_mask | im_mask

        diff = self.convolveDiff(mask)
        diff[diff < 0] = 0  # Must ignore negative pixels for this

        i0 = self.galaxyI0(galX, galY)
        sImage = np.nansum(diff)/i0
        sImage -= sBackground * ratio

        return sImage


def giniCoefficient(imageArray):
    '''
    Calculates Gini coefficient from an image flux array

    Parameters
    ----------
    imageArray : numpy.ndarray
        Array containing image data.  Can also be an array of fluxes from e.g.
        a detection footprint.

    Returns
    -------
    g : float
        Gini coefficient of image array
    '''
    sort_flux = np.sort(imageArray[np.isfinite(imageArray)])
    sort_flux = np.abs(sort_flux)  # Correction applied by Lotz + 2004
    mean_flux = np.mean(sort_flux)
    n = len(sort_flux)

    denom = mean_flux * n * (n-1)
    inds = np.arange(1, n+1)  # Starts at one
    val = np.sum((2 * inds - n - 1) * sort_flux)

    g = (1/denom) * np.sum(val)

    return g


def asymmetry(imageArray):
    pass


def clumpiness(imageArray, kernelSize):
    pass
