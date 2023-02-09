#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:20:06 2022

@author: awatkins

Functions for doing surface photometry on image stamps and returning derived
parameters.
"""
import numpy as np
from scipy.interpolate import interp1d
from photutils import centroids
from photutils import isophote
import matplotlib.pyplot as plt
import warnings
from . import utils
# import utils


class DerivedValues():
    '''
    A class with functions for computing galaxies properties derived from
    surface photometry
    '''

    def __init__(self, isoTable, magZp=27.0, pxScale=0.17):
        '''
        Parameters
        ----------
        isoTable : astropy.table.table.Table
            Table (or dictionary with the proper keys) of isophotal fit params
        magZp : float, optional
            Photometric zeropoint to conver to mags. The default is 27.0.
        pxScale : float, optional
            Pixel scale in arcsec/px. The default is 0.17.

        Returns
        -------
        None.

        '''
        if 'tflux_e' not in isoTable.keys():
            warnings.warn("Use isoList.to_table(columns='all'),"
                          + "or failures will ensue.")
        self.isoTable = isoTable
        self.magZp = magZp
        self.pxScale = pxScale

    def interpolateCurve(self, var1, var2, nElements=10000, kind='linear'):
        '''
        Enhances resolution of the curve of growth via interpolation

        Parameters
        ----------
        nElements : int
            Number of resolution elements for interpolated curve
        kind : string
            Interpolation type.  Valid values are: ‘linear’, ‘nearest’,
            ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’,
            or ‘next’.

        Returns
        -------
        high_res_var1 : numpy.ndarray
            Variable 1 with resolution defined by nElements
        high_res_var2 : numpy.ndarray
            Variable 2 with resolution defined by nElements
        '''
        f = interp1d(var1, var2, kind=kind)
        high_res_var1 = np.linspace(np.min(var1), np.max(var1), nElements)
        high_res_var2 = f(high_res_var1)

        return high_res_var1, high_res_var2

    def curveOfGrowth(self, sky=0):
        '''
        Produces a curve of growth in magnitudes

        Parameters
        ----------
        sky : float, optional
            Estimate of the average sky flux local to the galaxy.
            The default is 0.

        Returns
        -------
        sma : numpy.ndarray
            Semi-major-axis array, in pixels
        mags : numpy.ndarray
            Total magnitude enclosed within sma
        '''
        sma = self.isoTable['sma']
        tflux = self.isoTable['tflux_e'] - self.isoTable['npix_e']*sky
        mags = -2.5*np.log10(tflux) + self.magZp

        return sma, mags

    def totalMagnitude(self, sma, mags):
        '''
        Computes total magnitude of galaxy using curve of growth

        Parameters
        ----------
        sma : numpy.ndarray
            Semi-major-axis array, in pixels
        mags : numpy.ndarray
            Total magnitude enclosed within sma

        Returns
        -------
        totalMag : float
            Total magnitude extrapolated to infinity

        NOTE: fitting limits currently hard-coded, but proven effective for
        exponential profiles
        '''
        slope = utils.localSlope(sma, mags)
        want = (slope >= -0.008) & (slope <= 0)  # Typical useful range
        fit = np.polyfit(slope[want], mags[want], 1)
        totalMag = fit[1]

        return totalMag

    def fractionalRadius(self, totalMag, fluxFrac=0.5):
        '''
        Computes the radius containing some fraction of the galaxy's total
        light.

        Parameters
        ----------
        totalMag : float
            Galaxy total magnitude
        fluxFrac : float, optional
            The desired fraction of enclosed light. The default is 0.5, i.e.
            by defaul this returns the half-light radius.

        Returns
        -------
        fracRad : float
            The radius containing the desired fraction of the galaxy's total
            light, in pixels
        '''
        totalFlux = 10**(-0.4*(totalMag - self.magZp))
        findFlux = totalFlux * fluxFrac

        # A little crude, but we interpolate to improve the radius resolution
        sma, cog = self.interpolateCurve(self.isoTable['sma'],
                                         self.isoTable['tflux_e'])

        idx = utils.findNearest(cog, findFlux)
        fracRad = sma[idx]

        return fracRad

    def concentration82(self, totalMag):
        '''
        Derives the concentration parameter C discussed by Conselice (2003),
        originally proposed by Kent (1985), ApJS, 59, 115

        Parameters
        ----------
        totalMag : float
            Galaxy total magnitude

        Returns
        -------
        c82 : float
            The concentration parameter C = 5log(R80/R20)
        '''
        r80 = self.fractionalRadius(totalMag, 0.8)
        r20 = self.fractionalRadius(totalMag, 0.2)
        c82 = 5*np.log10(r80/r20)

        return c82

    def concentrationRe(self, totalMag, alpha=0.7):
        '''
        Derived the concentration parameter proposed by Trujillo et al. (2001)
        MNRAS, 326, 869

        Parameters
        ----------
        totalMag : float
            Galaxy total magnitude
        alpha : float
            Factor by which to define outer isophote level, as alpha*rEff

        Returns
        -------
        cRe : float
            The concentration parameter sum(I[<alpha*Reff])/sum(I[<Reff])
        '''
        rEff = self.fractionalRadius(totalMag, 0.5)

        sma, cog = self.interpolateCurve(self.isoTable['sma'],
                                         self.isoTable['tflux_e'])

        idx = utils.findNearest(sma, rEff)
        idy = utils.findNearest(sma, alpha*rEff)

        sum1 = cog[idy]
        sum2 = cog[idx]

        cRe = sum1/sum2

        return cRe

    def petrosianRadius(self, eta=0.2):
        '''
        Derives Petrosian radius (Petrosian 1976).

        Parameters
        ----------
        eta : float, optional
            Value of Petrosian index used to define Petrosian radius.
            The default is 0.2 (Bershady et al. 2000)

        Returns
        -------
        radPetro : float
            Petrosian radius in pixels

        NOTE: quick check, for an exponential profile, R_petrosian ~
        2x R_eff.
        '''
        sma, sb = self.interpolateCurve(self.isoTable['sma'],
                                        self.isoTable['intens'])
        __, cog = self.interpolateCurve(self.isoTable['sma'],
                                        self.isoTable['tflux_e'])
        __, area = self.interpolateCurve(self.isoTable['sma'],
                                         self.isoTable['npix_e'])
        petrosian = sb * (area/cog)
        idx = utils.findNearest(petrosian, 0.2)

        radPetro = sma[idx]
        return radPetro


class Profile():
    '''
    A class containing functions to compute radial profiles and related
    quantities
    '''

    def __init__(self, imageArray):
        '''
        Initialize class instance by giving it an image array, in numpy.ndarray
        format.

        Give it a masked array if you want to ignore pixels
        (numpy.ma.masked_array)
        '''
        if type(imageArray) == np.ma.core.MaskedArray:
            data = np.zeros(imageArray.shape) + imageArray.data
            mask = np.zeros(imageArray.shape, dtype=bool) + imageArray.mask
            self.imageArray = np.ma.masked_array(data, mask=mask)
        else:
            self.imageArray = np.zeros(imageArray.shape) + imageArray

    def initializeCenter(self, halfBoxWid):
        '''
        Makes an initial guess at the galaxy center, to be fed into photutils
        ellipse function later.

        Parameters
        ----------
        halfBoxWid : int
            Half the full width of the central image segment to clip

        Returns
        -------
        newX : float
            Initial estimate of the central x-coordinate
        newY : float
            Initial estimate of central y-coordinate
        '''
        cenx = self.imageArray.shape[1]//2
        ceny = self.imageArray.shape[0]//2
        cenBox = utils.imBox(halfBoxWid, cenx, ceny, self.imageArray)
        box_x, box_y = centroids.centroid_quadratic(cenBox)
        delta_x = box_x - cenBox.shape[1]//2
        delta_y = box_y - cenBox.shape[0]//2

        newX = cenx + delta_x
        newY = ceny + delta_y

        return newX, newY

    def initializeEllipse(self, **kwargs):
        '''
        Initializes the ellipse for fitting.  Give it initial parameter
        estimates.

        Parameters
        ----------
        **kwargs
            See photutils.isophote.EllipseGeometry() docs

        Returns
        -------
        ellipse : photutils.isophote.ellipse.Ellipse
            Initialized Ellipse object, for fitting
        '''
        geom = isophote.EllipseGeometry(**kwargs)
        ellipse = isophote.Ellipse(self.imageArray, geom)

        return ellipse

    def fitEllipses(self, ellipse, **kwargs):
        '''
        Function to run the ellipse fitter and produce a table of best-fit
        ellipse parameters

        Parameters
        ----------
        ellipse : photutils.isophote.ellipse.Ellipse
            Ellipse object initialized with self.initializeEllipse()
        **kwargs
            See photutils.isophote.ellipse.Ellipse() docs

        Returns
        -------
        isoList : photutils.isophote.ellipse.Ellipse.isolist
            Class instance containing isophotal parameters

        To do no-fit mode, use maxrit=1 or some number smaller than your step
        size.  Beware: also give it a maxsma, or it will take a long time to
        run.
        '''
        isoList = ellipse.fit_image(**kwargs)

        return isoList

    def visualizeEllipses(self, isoList, vmin, vmax):
        '''
        For plotting the fitted ellipses on the image, with log scaling

        Parameters
        ----------
        isoList : photutils.isophote.ellipse.Ellipse.isolist
            Class instance containing isophotal parameters
        vmin : float
            Minimum in imshow log image scaling
        vmax : float
            Maximum in imshow log image scaling

        Returns
        -------
        None.  Just makes a plot.
        '''
        logIm = utils.ds9LogScale(self.imageArray)
        plt.imshow(logIm, vmin=vmin, vmax=vmax)
        for i in isoList.sma:
            iso = isoList.get_closest(i)
            x, y, = iso.sampled_coordinates()
            plt.plot(x, y, color='w', markersize=0.5)

    def toTable(self, isoList):
        '''
        Because the isoList.to_table() function adds None values to some of
        these columns, which screws everything up, we replace those with
        some slightly more reasonable values and output a table that way.

        Parameters
        ----------
        isoList : hotutils.isophote.ellipse.Ellipse.isolist
            Class instance containing isophotal parameters

        Returns
        -------
        isoTab : astropy.table.table.Table
            isoList converted to a table format for easier storage
        '''
        isoTab = isoList.to_table(columns='all')
        isoTab['tflux_e'][0] = isoTab['intens'][0]
        isoTab['tflux_e'] = np.array(isoTab['tflux_e'], dtype=float)
        isoTab['tflux_c'][0] = isoTab['intens'][0]
        isoTab['tflux_c'] = np.array(isoTab['tflux_c'], dtype=float)
        isoTab['npix_e'][0] = 1
        isoTab['npix_e'] = np.array(isoTab['npix_e'], dtype=float)
        isoTab['npix_c'][0] = 1
        isoTab['npix_c'] = np.array(isoTab['npix_c'], dtype=float)
        isoTab['rms'][0] = 0
        isoTab['rms'] = np.array(isoTab['rms'], dtype=float)
        isoTab['pix_stddev'][0] = 0
        isoTab['pix_stddev'] = np.array(isoTab['pix_stddev'], dtype=float)
        isoTab['grad_error'][0] = 0
        isoTab['grad_error'] = np.array(isoTab['grad_error'], dtype=float)
        isoTab['grad_rerror'][0] = 0
        isoTab['grad_rerror'] = np.array(isoTab['grad_rerror'], dtype=float)
        isoTab['sarea'][0] = isoTab['sarea'][1]
        isoTab['sarea'] = np.array(isoTab['sarea'], dtype=float)

        return isoTab


def measureImageNoise(maskedImageArray,
                      galX,
                      galY,
                      galRad,
                      halfBoxWidth=10,
                      seed=None):
    '''
    Outputs metrics for the background noise and flux.

    Parameters
    ----------
    maskedImageArray : numpy.ma.core.MaskedArray
        Image with interloping sources masked
    galX : float
        Central x-coordinate of target galaxy
    galY : float
        Central y-coordinate of target galaxy
    galRad : int
        Radius out to which to apply circular mask to target galaxy
    halfBoxWidth : int, optional
        Half the desired width of the boxes used to calculate the noise.
        The default is 10.
    seed : int, optional
        A seed for the random generator, for testing purposes

    Returns
    -------
    avRms : float
        The average root mean square among counts in N=1000 randomly
        distributed boxes, ignoring masked pixels
    sbLim : float
        The standard deviation of the median values within N=1000 randomly
        distributed boxes, ignoring masked pixels
    sky : float
        Median of the median values within N=1000 randomly distributed boxes,
        ignoring masked pixels
    '''
    rng = np.random.default_rng(seed)
    # Need to avoiding actually altering the image and mask used
    if type(maskedImageArray) == np.ma.core.MaskedArray:
        data = np.zeros(maskedImageArray.shape) + maskedImageArray.data
        mask = np.zeros(maskedImageArray.shape, dtype=bool)\
            + maskedImageArray.mask
        maskedImageArray = np.ma.masked_array(data, mask=mask)
    else:
        maskedImageArray = np.zeros(maskedImageArray.shape) + maskedImageArray

    # Boundaries: avoid masked interlopers, the target galaxy, and image edges
    circle_mask = utils.imCircle(galX, galY, galRad, maskedImageArray.data)
    edge_mask = np.zeros(maskedImageArray.data.shape, dtype=bool) + True
    edge_mask[halfBoxWidth: -halfBoxWidth, halfBoxWidth: -halfBoxWidth] = False
    mask = maskedImageArray.mask | circle_mask | edge_mask

    # Ignore only galaxies and other sources
    maskedImageArray.data[maskedImageArray.mask | circle_mask] = np.nan

    n_boxes = 1000
    good_coords = np.where(~mask)
    rms = np.array([])
    medians = np.array([])
    for i in range(n_boxes):
        idx = rng.choice(np.arange(len(good_coords[0])))
        ceny, cenx = good_coords[0][idx], good_coords[1][idx]
        box = utils.imBox(halfBoxWidth, cenx, ceny, maskedImageArray.data)
        rms = np.append(rms, utils.rootMeanSquare(box[np.isfinite(box)]))
        medians = np.append(medians, np.nanmedian(box))

    avRms = np.mean(rms)
    sbLim = np.std(medians)
    sky = np.median(medians)

    return avRms, sbLim, sky


def interpolateAcrossMasks(maskedImageArray,
                           x0, y0, maxRad, pa, ellip,
                           rms, ampCut=0.01):
    '''
    Replaces masked pixels in an image out to a certain radius with an FFT
    recreation of an elliptical aperture within that masked region, plus some
    noise

    Parameters
    ----------
    maskedImageArray : numpy.ma.core.MaskedArray
        Image with masked pixels
    x0 : float
        Central x-coordinate of galaxy in image
    y0 : float
        Central y-coordinate of galaxy in image
    maxRad : int
        Maximum radius (semi-major axis) out to which to replace masks
    pa : float
        Galaxy average isophote shape position angle, in degrees
    ellip : float
        Galaxy average isophote shape ellipticity, 1-b/a where b and a are
        semi-major and major axis lengths
    rms : float
        Amount of noise to add to each replacement flux value.  Added as
        draw from standard normal N(0, rms)
    ampCut : float, optional
        DFT amplitude below which to ignore contribution from spectrum.
        Default is 0.01.

    Returns
    -------
    unmaskedImage : numpy.ndarray
        Image with masked pixels replaced out to maxRad
    '''
    rng = np.random.default_rng()

    mask = maskedImageArray.mask
    image = maskedImageArray.data
    rads = np.arange(0, maxRad+1)

    for i in range(len(rads)):
        if i == 0:
            continue
        __, __, ellRad, theta = utils.reprojectToEllipse(image,
                                                         x0,
                                                         y0,
                                                         pa,
                                                         ellip)
        wantRad = (ellRad <= rads[i]) & (ellRad >= rads[i-1])
        maskRing = mask[wantRad]
        if len(maskRing[maskRing]) == 0:
            continue

        # Flux in a ring, with association position angles
        flux = image[wantRad]
        angle = theta[wantRad]
        flux[maskRing] = np.nan

        # Sort these by angle
        idx = np.argsort(angle)
        sort_flux = flux[idx]
        sort_angle = angle[idx]

        # Need maximum and minimum in masked arrays to be full angle range
        if np.isnan(sort_flux[0]):
            # Replacing with the mean of the first five non-NaN values
            sort_flux[0] = np.mean(sort_flux[~np.isnan(sort_flux)][: 6])
        if np.isnan(sort_flux[-1]):
            # Replacing with the mean of last five non-NaN values
            sort_flux[-1] = np.mean(sort_flux[~np.isnan(sort_flux)][-5:])

        # Now Interpolate across masked pixels
        nans = np.isnan(sort_flux)
        f = interp1d(sort_angle[~nans], sort_flux[~nans], kind='cubic')
        new_flux = f(sort_angle)

        # Derive the DFT
        dft = np.fft.fft(new_flux)
        amp = np.sqrt(dft.real**2 + dft.imag**2)/len(dft)
        # Suppress low-amplitude frequencies
        # Maybe better to cut a specific range...?  This fails in the BG.
        low_amp = amp <= ampCut
        dft[low_amp] *= 0
        smooth_flux = np.fft.ifft(dft)

        # Now replace masked pixels with interpolated ones
        nanIdx = np.array([np.where(angle == i)[0][0]
                           for i in sort_angle[nans]])
        noise = rng.normal(0, rms, size=len(nanIdx))
        flux[nanIdx] = smooth_flux.real[nans] + noise  # Add noise
        image[wantRad] = flux
    unmaskedImage = image

    return unmaskedImage
