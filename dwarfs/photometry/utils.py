#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:26:51 2022

@author: awatkins
"""
import numpy as np


def findNearest(arr, value):
    '''
    Find a value in an array closest to the input value

    Parameters
    ----------
    array : numpy.array
        Input 1-d array
    value : float
        Value in array you want to find

    Returns
    -------
    idx : int
        Index where array value is closest to value
    '''
    arr = arr[~np.isnan(arr)]
    arr = arr[~np.isinf(arr)]
    idx = (np.abs(arr-value)).argmin()

    return idx


def rootMeanSquare(arr):
    '''
    Root mean square of an array of values

    Parameters
    ----------
    arr : numpy.ndarray
        The array of values for which you want the RMS

    Returns
    -------
    rms : float
        The root mean square of the array of values
    '''
    n = len(arr)
    rms = np.sqrt(np.sum(arr**2)/n)

    return rms


def localSlope(x, y):
    '''
    For deriving the local slope profile of two arrays

    Parameters
    ----------
    x : numpy.ndarray
        The abscissa axis array
    y : numpy.ndarray
        The ordinate axis array

    Returns
    -------
    dydx : numpy.ndarray
        An array of slope values between x and y
    '''
    dydx = np.zeros(len(y))
    dydx[1:] = np.diff(y) / np.diff(x)
    dydx[0] = y[0] / x[0]  # Replace first value with ratio of initial points

    return dydx


def imBox(halfBoxWid, xCen, yCen, imageArray):
    '''
    Returns a box clipped out from the input image with a given width,
    centered at the given coordinates.

    Parameters
    ----------
    halfBoxWid : int
        Half the full width of the image segment to make.
    xCen : float
        Central x-coordinate where you want the box drawn
    yCen : float
        Central y-coordinate where you want the box drawn

    Returns
    -------
    box : numpy.ndarray
        Segment of imageArray desired.

    '''
    cenx = int(np.round(xCen, 0))  # Can't use partial pixels
    ceny = int(np.round(yCen, 0))
    box = imageArray[ceny - halfBoxWid: ceny + halfBoxWid,
                     cenx - halfBoxWid: cenx + halfBoxWid]

    return box


def imCircle(xCen, yCen, radius, imageArray):
    '''
    Returns a boolean array in which points within a circle are True

    Parameters
    ----------
    xCen : float
        Central x-coordinate where you want the circle selected
    yCen : float
        Central y-coordinate where you want the circle selected
    radius : float
        Radius of the circular region, in pixels
    imageArray : numpy.ndarray
        The image array from which you want to select the circle.
        NOTE: currently assumes the image has symmetric axis lengths.

    Returns
    -------
    circ : numpy.ndarray
        A boolean array with the circular region selected as True
    '''
    cenx = int(np.round(xCen, 0))
    ceny = int(np.round(yCen, 0))
    x = np.arange(1, imageArray.shape[0]+1)
    y = x.reshape(-1, 1)
    rad = np.sqrt((x-cenx)**2 + (y-ceny)**2)
    circ = rad <= radius

    return circ


def reprojectToEllipse(imageArray, xCen, yCen, pa, ellip):
    '''
    Reprojects image coordinates to an ellipse

    Parameters
    ----------
    imageArray : numpy.ndarray
        NxN array of fluxes
    xCen : float
        Central x-coordinate of ellipse, in pixels
    yCen : float
        Central y-coordinate of ellipse, in pixels
    pa : float
        Ellipse position angle in degrees
    ellip : float
        Ellipse ellipticity = 1-b/a (b is minor axis, a is major axis)

    Returns
    -------
    xEll : numpy.ndarray
        Elliptical x-axis projection
    yEll : numpy.ndarray
        Elliptical y-axis projection
    ellRad : numpy.ndarray
        Array of radii from ellipse center, in the ellipse plane
    theta : numpy.ndarray
        Array of angles in the ellipse plane, with 0 degrees at the top of the
        major axis, increasing CCW

    To plot the reprojected image (for diagnostics), do:
        plt.pcolormesh(xEll, yEll, imageArray)
    Or, to show the radius array, do:
        plt.imshow(ellRad)
    '''
    dim = imageArray.shape[0]  # Assumes image is NxN, not NxM
    x = np.arange(1, dim+1)
    y = x.reshape(-1, 1)

    # Currently written to start at PA and increase CCW (due north is 0 deg)
    xEll = (x - xCen)*np.cos(np.radians(pa+90)) + \
        (y - yCen)*np.sin(np.radians(pa+90))
    yEll = -(x - xCen)*np.sin(np.radians(pa+90)) + \
        (y - yCen)*np.cos(np.radians(pa+90))

    ellRad = np.sqrt(xEll**2 + (yEll/(1-ellip))**2)
    theta = np.arctan2(yEll, xEll)
    theta[theta < 0.] = theta[theta < 0.]+2.0*np.pi  # Removing negative values

    return xEll, yEll, ellRad, theta


def ds9LogScale(imageArray):
    '''
    Scales an image logarithmically in the way that DS9 does, to show also
    negative values as something other than NaNs.

    Parameters
    ----------
    imageArray : numpy.ndarray
        NxN array containing image flux

    Returns
    -------
    logImageArray : numpy.ndarray
        imageArray scaled logarithmically, for display purposes

    NOTE: for display purposes only!  Not for analysis.
    '''
    np.seterr(all="ignore")  # Suppress warnings about negative flux
    if type(imageArray) == np.ma.core.MaskedArray:
        imageArray = imageArray.data
    logImageArray = np.log10(imageArray)
    lg2 = np.log10(-imageArray)
    logImageArray[np.isnan(logImageArray)] = lg2[np.isnan(logImageArray)]
    logImageArray[np.isinf(logImageArray)] = -10

    return logImageArray
