#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:20:06 2022

@author: awatkins

Functions for doing surface photometry on image stamps and returning derived
parameters.
"""
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np
from photutils import centroids
from photutils import isophote
from scipy.interpolate import interp1d
from scipy.special import gammaincinv
import warnings
import utils


class DerivedValues():
    '''
    A class with functions for computing galaxy properties derived from
    surface photometry
    '''

    def __init__(self, isoTable, sky=0, magZp=27.0, pxScale=0.167):
        '''
        Parameters
        ----------
        isoTable : `astropy.table.table.Table`
            Table (or dictionary with the proper keys) of isophotal fit params
        sky : `float`, optional
            Estimate of the average sky flux local to the galaxy.
            The default is 0.
        magZp : `float`, optional
            Photometric zeropoint to conver to mags. The default is 27.0.
        pxScale : `float`, optional
            Pixel scale in arcsec/px. The default is 0.167.

        Returns
        -------
        None.

        '''
        if 'tflux_e' not in isoTable.keys():
            warnings.warn("Use isoList.to_table(columns='all'),"
                          + "or failures will ensue.")
        self.isoTable = isoTable
        self.sky = sky
        self.magZp = magZp
        self.pxScale = pxScale

    def interpolateCurve(self, var1, var2, nElements=10000, kind='linear'):
        '''
        Enhances resolution of the curve of growth via interpolation

        Parameters
        ----------
        var1 : `numpy.ndarray`
            Abscissa array
        var2 : `numpy.ndarray`
            Ordinate array
        nElements : `int`
            Number of resolution elements for interpolated curve
        kind : `string`
            Interpolation type.  Valid values are: ‘linear’, ‘nearest’,
            ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’,
            or ‘next’.

        Returns
        -------
        high_res_var1 : `numpy.ndarray`
            Variable 1 with resolution defined by nElements
        high_res_var2 : `numpy.ndarray`
            Variable 2 with resolution defined by nElements
        '''
        f = interp1d(var1, var2, kind=kind)
        high_res_var1 = np.linspace(np.min(var1), np.max(var1), nElements)
        high_res_var2 = f(high_res_var1)

        return high_res_var1, high_res_var2

    def correctPa(self, thresh=130):
        '''
        Reverses large PA swings when PA~0 or ~180, sequentially.
        
        Parameters
        ----------
        thresh: `float`, optional
            Threshold in degrees for swing correction.  The default value is
            130 deg.
        
        Notes
        -----
        Acts on table object attached to class instance.
        '''
        for i in range(len(self.isoTable)-1):
            # If difference from one radius to the next exceeds thresh, apply
            # the appropriate 180 correction
            dpa = self.isoTable["pa"][i+1] - self.isoTable["pa"][i]
            if dpa > np.abs(thresh*u.deg):
                self.isoTable["pa"][i+1] -= 180*u.deg
            elif dpa < -np.abs(thresh*u.deg):
                self.isoTable["pa"][i+1] += 180*u.deg

    def curveOfGrowth(self):
        '''
        Produces a curve of growth in magnitudes

        Returns
        -------
        sma : `numpy.ndarray`
            Semi-major-axis array, in pixels
        mags : `numpy.ndarray`
            Total magnitude enclosed within sma
        '''
        sma = self.isoTable['sma']
        tflux = self.isoTable['tflux_e'] - self.isoTable['npix_e']*self.sky
        mags = -2.5*np.log10(tflux) + self.magZp

        return sma, mags

    def surfaceBrightnessProfile(self):
        '''
        Converts from linear to magnitude surface brightness profile

        Returns
        -------
        mu : `numpy.ndarray`
            Surface brightness in units of magnitude/arcsec^2
        '''
        mu = -2.5*np.log10(self.isoTable["intens"] - self.sky) \
            + self.magZp \
            + 2.5*np.log10(self.pxScale**2)

        return mu

    def totalMagnitude(self, sma, mags, slopeBound=-0.008):
        '''
        Computes total magnitude of galaxy using curve of growth

        Parameters
        ----------
        sma : `numpy.ndarray`
            Semi-major-axis array, in pixels
        mags : `numpy.ndarray`
            Total magnitude enclosed within sma

        Returns
        -------
        totalMag : `float`
            Total magnitude extrapolated to infinity

        Notes
        ------
        -0.008 is typically acceptable for exponential profiles.  Other kinds
        with higher Sersic index may require a different value.
        Reference: Munoz-Mateos et al. (2015), ApJS, 219, 3
        '''
        slope = utils.localSlope(sma, mags)
        want = (slope >= slopeBound) & (slope <= 0)
        fit = np.polyfit(slope[want], mags[want], 1)
        totalMag = fit[1]

        return totalMag

    def fractionalRadius(self, totalMag, fluxFrac=0.5):
        '''
        Computes the radius containing some fraction of the galaxy's total
        light.

        Parameters
        ----------
        totalMag : `float`
            Galaxy total magnitude
        fluxFrac : `float`, optional
            The desired fraction of enclosed light. The default is 0.5, i.e.
            by default this returns the half-light radius.

        Returns
        -------
        fracRad : `float`
            The radius containing the desired fraction of the galaxy's total
            light, in pixels
        '''
        totalFlux = 10**(-0.4*(totalMag - self.magZp))
        findFlux = totalFlux * fluxFrac

        # Interpolate the curve to improve the derived radius resolution
        sma, cog = self.interpolateCurve(self.isoTable['sma'],
                                         self.isoTable['tflux_e'])

        idx = utils.findNearest(cog, findFlux)
        fracRad = sma[idx]

        return fracRad

    def muEffective(self, totalMag):
        '''
        Computes the mean surface brightness WITHIN one effective radius

        Parameters
        ----------
        totalMag : `float`
            Galaxy total magnitude

        Returns
        -------
        muEff : `float`
            Mean surface brightness within one effective radius, as derived
            from the curve of growth, in mag/arcsec^2
        '''
        sma, i = self.interpolateCurve(self.isoTable['sma'],
                                       self.isoTable['intens'])
        rEff = self.fractionalRadius(totalMag)
        iEff = np.nanmean(i[sma <= rEff])
        muEff = -2.5*np.log10(iEff) \
            + self.magZp \
            + 2.5*np.log10(self.pxScale**2)

        return muEff

    def muAtR(self, totalMag, rad):
        '''
        Computes the surface brightness at a specified radius, e.g. AT the
        effective radius

        Parameters
        ----------
        totalMag : `float`
            Galaxy total magnitude
        rad : `float`
            Radius at which to estimate the surface brightness

        Returns
        -------
        muR : `float`
            Surface brightness of the isophotes at radius rad, in mag/arcsec^2
        '''
        sma, i = self.interpolateCurve(self.isoTable['sma'],
                                       self.isoTable['intens'])
        idx = utils.findNearest(sma, rad)
        muR = -2.5*np.log10(i[idx]) \
            + self.magZp \
            + 2.5*np.log10(self.pxScale**2)

        return muR

    def isophotalRadius(self, muIso, tol=0.1):
        '''
        Computes the radius at which the surface brightness profile reaches
        some surface brightness value

        Parameters
        ----------
        muIso : `float`
            The desired surface brightness at which to measure the radius, in
            mag/arcsec^2
        tol : `float`, optional
            Tolerance for acceptance.  If muIso from the curve is within
            +/- tol of desired value, accept, but otherwise reject

        Returns
        -------
        isoRad : `float`
            The radius at which the surface brightness profiles reaches the
            desired surface brightness value.  If not found within tol, returns
            NaN (e.g. if the central surface brightness is fainter than the
            desired isophote surface brightness)
        '''
        # Interpolate same as with other methods
        mu = self.surfaceBrightnessProfile()
        sma, mu = self.interpolateCurve(self.isoTable['sma'],
                                        mu)

        idx = utils.findNearest(mu, muIso)
        good = (mu[idx] >= muIso - tol) & (mu[idx] <= muIso + tol)
        if good:
            isoRad = sma[idx]
        else:
            isoRad = np.nan

        return isoRad

    def isophotalRadiusDeproj(self, muIso, axRat, tol=0.1):
        '''
        As isophotalRadius, but deprojects the profile using the supplied axial
        ratio

        Parameters
        ----------
        muIso : `float`
            The desired surface brightness at which to measure the radius, in
            mag/arcsec^2
        axRat : `float`
            Axial ratio (b/a) of the projected isophote shapes
        tol : `float`, optional
            Tolerance for acceptance.  If muIso from the curve is within
            +/- tol of desired value, accept, but otherwise reject

        Returns
        -------
        isoRad : `float`
            The radius at which the surface brightness profiles reaches the
            desired surface brightness value
        '''
        # Interpolate same as with other methods
        iProj = self.isoTable["intens"]
        iDeproj = iProj * axRat
        mu = -2.5*np.log10(iDeproj)\
            + self.magZp\
            + 2.5*np.log10(self.pxScale**2)
        sma, mu = self.interpolateCurve(self.isoTable['sma'],
                                        mu)

        idx = utils.findNearest(mu, muIso)
        isoRad = sma[idx]
        good = (mu[idx] >= muIso - tol) & (mu[idx] <= muIso + tol)
        if good:
            isoRad = sma[idx]
        else:
            isoRad = np.nan

        return isoRad

    def concentration82(self, totalMag):
        '''
        Derives the concentration parameter C discussed by Conselice (2003),
        originally proposed by Kent (1985), ApJS, 59, 115

        Parameters
        ----------
        totalMag : `float`
            Galaxy total magnitude

        Returns
        -------
        c82 : `float`
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
        totalMag : `float`
            Galaxy total magnitude
        alpha : `float`
            Factor by which to define outer isophote level, as alpha*R_eff

        Returns
        -------
        cRe : `float`
            The concentration parameter sum(I[<alpha*R_eff])/sum(I[<R_eff])
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
        eta : `float`, optional
            Value of Petrosian index used to define Petrosian radius.
            The default is 0.2 (Bershady et al. 2000)

        Returns
        -------
        radPetro : `float`
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

    def lumWeight(self, param):
        '''Estimates the luminosity-weighted mean value of the input parameter,
        taking into account profile ellipticity changes vs. radius

        Parameters
        ----------
        param : `string`
            Table column name of the parameter to measure

        Returns
        -------
        q : `numpy.ndarray`
            Axial ratio array (1 - ellipticity)
        La0 : `numpy.ndarray`
            Total luminosity derived from ellipticity-corrected surface
            brightness profile
        lumWtParam : `numpy.ndarray`
            Mean value of the parameter normalized by the total luminosity
        
        Notes
        -----
        Source: Ryden et al. (1999), ApJ, 517, 650
        '''
        # Estimating total luminosity
        q = 1 - self.isoTable["ellipticity"]
        dqda = np.diff(np.log(q)) / np.diff(np.log(self.isoTable["sma"]))
        dqda = np.append(np.log(q[0]) / np.log(self.isoTable["sma"][0]), dqda)
        dA = 2*np.pi * q * self.isoTable["sma"] * (1 + 0.5*dqda)
        La0 = np.trapezoid(self.isoTable["intens"]*dA,
                           x=self.isoTable["sma"])

        # Estimating luminosity-weighted parameter value
        integral = np.trapezoid(self.isoTable[param]
                                * self.isoTable["intens"] * dA,
                                x=self.isoTable["sma"])
        lumWtParam = integral / La0

        return q, La0, lumWtParam

    def paramLimits(self, param, skyNoise, minSN=2, innerSma=1.2):
        '''
        Derives the maximum, minimum, and median values of a chosen parameter
        outside of some radius boundary and within a surface brightness limit

        Parameters
        ----------
        param : `string`
            Table column name of the parameter to measure
        skyNoise : `float`
            Estimated per-pixel noise in sky background, in counts
        minSN : `float`, optional
            The minimum S/N value in flux out to which to calculate the
            parameter.  The default value is 2.
        innerSma : `float`, optional
            Innermost semi-major axis cutoff, in arcseconds; everything inside
            here is ignored for parameter measurement.  The default value is
            1.2".

        Returns
        -------
        maxParam : `float`
            Maximum value of the parameter within chosen boundaries
        maxParam : `float`
            Minimum value of the parameter within chosen boundaries
        medParam : `float`
            Median value of the parameter within chosen boundaries
        '''
        innerSma /= self.pxScale  # Attached table is in pixels
        noise = np.sqrt(
            self.isoTable["intens_err"]**2
            + (skyNoise / np.sqrt(self.isoTable["npix_e"]))**2
            )
        want = (self.isoTable["sma"] >= innerSma) \
            & (self.isoTable["intens"]/noise >= minSN)

        maxParam = np.nanmax(self.isoTable[param][want])
        minParam = np.nanmin(self.isoTable[param][want])
        medParam = np.nanmedian(self.isoTable[param][want])

        return maxParam, minParam, medParam

    def maxDeltaPa(self, skyNoise, minSN=2, innerSma=1.2, thresh=130):
        '''
        Derives the maximum change in position angle across a radial PA
        profile.
        Avoids PSF-convolved inner isophotes, and only until a provided surface
        brightness limit. Excludes nearly round isophotes as well due to
        uncertainty.

        Parameters
        ----------
        skyNoise : `float`
            Estimated per-pixel noise in sky background, in counts
        minSN : `float`, optional
            The minimum S/N value in flux out to which to calculate the
            parameter.  The default value is 2.
        innerSma : `float`, optional
            Innermost semi-major axis cutoff, in arcseconds; everything inside
            here is ignored for parameter measurement.  The default value is
            1.2".
        thresh: `float`, optional
            Threshold in degrees for swing correction.  The default value is
            130 deg.

        Returns
        -------
        maxDelPa : `float`
            Difference between the maximum and minimum value of position angle
            outside of innerSma and within minSb
        '''
        innerSma /= self.pxScale  # Work in pixels
        noise = np.sqrt(
            self.isoTable["intens_err"]**2
            + (skyNoise / np.sqrt(self.isoTable["npix_e"]))**2
            )
        want = (self.isoTable["sma"] >= innerSma) \
            & (self.isoTable["intens"]/noise >= minSN) \
            & (self.isoTable["ellipticity"] >= 0.1)    # From Busko et al. 1996

        self.correctPa(thresh)  # Ignore 180 flips
        maxDelPa = np.nanmax(self.isoTable["pa"][want]) \
            - np.nanmin(self.isoTable["pa"][want])

        return maxDelPa

    def twistiness(self, skyNoise, minSN=2, innerSma=1.2, flag=False):
        '''
        Derives the twistiness parameter defined by Ryden et al. 1999.
        We define the fiducial radius based on the provided surface brightness
        limit, and ignore the central PSF-convolved regions.

        Parameters
        ----------
        skyNoise : `float`
            Estimated per-pixel noise in sky background, in counts
        minSN : `float`, optional
            The minimum S/N value in flux out to which to calculate the
            parameter.  The default value is 2.
        innerSma : `float`, optional
            Innermost semi-major axis cutoff, in arcseconds; everything inside
            here is ignored for parameter measurement.  The default value is
            1.2".
        flag : `bool`, optional
            If true, returns the integral array, for visualisation and
            debugging purposes.  The default value is False.

        Returns
        -------
        T : `float`
            Twistiness parameter measured within chosen limits

        Notes
        -----
        CAUTION: modifies self.isoTable in-place via boolean selection!
        
        Reference: Ryden et al. (1999), ApJ, 517, 650
        '''
        # Works in sine and cosine; no need to correct for PA swings
        innerSma /= self.pxScale  # Work in pixels
        
        # Reject NaN values in the calculation
        # Calculate noise array for S/N cut
        noise = np.sqrt(
            self.isoTable["intens_err"]**2 
            + (skyNoise / np.sqrt(self.isoTable["npix_e"]))**2
            )
        wantRad = (self.isoTable["sma"] >= innerSma) \
            & (self.isoTable["intens"] / noise >= minSN)
        wantFinite = np.isfinite(self.isoTable["ellipticity"]) \
            & np.isfinite(self.isoTable["pa"]) \
            & np.isfinite(self.isoTable["pa_err"])
        wantPaErr = self.isoTable["ellipticity"] >= 0.1  # From Busko et al. 1996
        want = wantRad & wantFinite & wantPaErr

        # TODO: bad that this currently modifies the isoTable object!
        self.isoTable = self.isoTable[want]
    
        q, La0, phiAv = self.lumWeight("pa")

        # Now calculating T, the twistiness parameter
        gamma = np.radians(self.isoTable["pa"]) - np.radians(phiAv)
        dIda = np.diff(self.isoTable["intens"]) / np.diff(self.isoTable["sma"])
        # My own derivation... original paper has a typo in the LaTeX file.
        quant = np.sqrt((np.sin(gamma)**2/q**2) + np.cos(gamma)**2) - 1
        C = dIda * self.isoTable["sma"][1:] * quant[1:]
        tInt = np.trapezoid(np.abs(C) * self.isoTable["sma"][1:],
                            x=self.isoTable["sma"][1:])
        T = ((2*np.pi)/La0) * tInt
    
        if not flag:
            return T
        else:
            return T, np.abs(C) * self.isoTable["sma"][1:], \
                self.isoTable["sma"][1:]
    
    def iteratedSersicIndex(self, rLow, sbLim, nMin=0.5, nMax=6.5, step=0.1):
        '''
        Identifies ideal Sersic index of a measured profile by finding that
        which minimizes the chi^2 of a linear fit in r^(1/n) space

        Parameters
        ----------
        rLow : `float`
            Lower limit on radius for the fit, arcseconds
        sbLim : `float`
            Faintest surface brightness out to which to fit, in mag/arcsec^2
        nMin : `float`
            Lowest desired Sersic index.  The default value is 0.5.
        nMax : `float`
            Highest desired Sersic index.  The default value is 6.5.
        step : `float`, optional
            Step size in Sersic index array.  The default value is 0.1.

        Returns
        -------
        bestN : `float`
            The Sersic index n which minimizes the fit chi^2
        bestMuEff : `float`
            Effective surface brightness associated with the best fit profile,
            mag/arcsec^2
        bestReff : `float`
            Half-light radius associated with the best fit profile, arcsec
        '''
        ns = np.arange(nMin, nMax+step, step)
        rad = self.isoTable["sma"] * self.pxScale
        sb = self.surfaceBrightnessProfile()
        
        chi2s = []
        for n in ns:
            x = rad**(1/n)
            want = rad > rLow
            want2 = sb <= sbLim
            want = want & want2
            fit = np.polyfit(x[want], sb[want], 1)
            dumY = fit[0]*x + fit[1]
            chi2 = np.sum((sb - dumY)**2/dumY)
            chi2s.append(chi2)
        
        # Interpolate to a higher resolution and find the minimum
        f = interp1d(ns, chi2s, kind="cubic")
        dumN = np.linspace(np.min(ns), np.max(ns), 10000)
        dumChi2 = f(dumN)
        minIdx = np.argmin(dumChi2)
        bestN = dumN[minIdx]
        if minIdx == len(dumN) - 1:
            warnings.warn("Minimum found at final index; chi^2 values likely"
                         + " did not converge")

        # Use that fit to derive the other Sersic parameters
        x = rad**(1/bestN)
        fit = np.polyfit(x, sb, 1)
        bn = gammaincinv(2*bestN, 0.5)
        A = 2.5*bn/np.log(10)
        bestMuEff = fit[1] + A
        bestReff = (A/fit[0])**bestN
        
        return bestN, bestMuEff, bestReff

    def singleSersicResiduals(self, n, muEff, rEff, rLow, sbLim):
        '''
        Derive mean residual sum of squares from best-fit single-Sersic profile

        Parameters
        ----------
        n : `float`
            Sersic index
        muEff : `float`
            Effective surface brightness, mag/arcsec^2
        rEff : `float`
            Half-light radius, arcsec
        rLow : `float`
            Lower limit on radius for the fit, arcseconds
        sbLim : `float`
            Faintest surface brightness out to which to fit, in mag/arcsec^2

        Returns
        -------
        sRes : `float`
            Sum of squares of residuals of surface brightness profile from its
            best-fit single-Sersic model
            
        Notes
        -----
        Reference: Barazza et al. (2003), A&A, 427, 121
        '''
        rad = self.isoTable["sma"] * self.pxScale
        sb = self.surfaceBrightnessProfile()
        
        want = (rad >= rLow) & (sb <= sbLim)
    
        bn = gammaincinv(2*n, 0.5)
        muR = muEff + ((2.5*bn)/np.log(10)) * ((rad/rEff)**(1/n) - 1)
        res = sb - muR
        
        sRes = np.sqrt(np.sum(res[want]**2) / len(res[want]))
        
        return sRes


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
        (numpy.ma.masked_array).
        
        Parameters
        ----------
        imageArray : `numpy.ndarray` or `numpy.ma.core.MaskedArray`
            Array containing image data
        '''
        if type(imageArray) is np.ma.core.MaskedArray:
            data = np.zeros(imageArray.shape) + imageArray.data
            mask = np.zeros(imageArray.shape, dtype=bool) + imageArray.mask
            self.imageArray = np.ma.masked_array(data, mask=mask)
        else:
            self.imageArray = np.zeros(imageArray.shape) + imageArray

    def initializeCenter(self, halfBoxWid, cType="quadratic"):
        '''
        Makes an initial guess at the galaxy center, to be fed into photutils
        ellipse function later.

        Parameters
        ----------
        halfBoxWid : `int`
            Half the full width of the central image segment to clip
        cType : `string`
            Type of centroid to use.  Valid choices are:
                "quadratic", "com", "1dg", "2dg"

        Returns
        -------
        newX : `float`
            Initial estimate of the central x-coordinate
        newY : `float`
            Initial estimate of central y-coordinate
        '''
        cenx = self.imageArray.shape[1]//2
        ceny = self.imageArray.shape[0]//2
        cenBox = utils.imBox(halfBoxWid, cenx, ceny, self.imageArray)
        if cType == "quadratic":
            box_x, box_y = centroids.centroid_quadratic(cenBox)
        elif cType == "com":
            box_x, box_y = centroids.centroid_com(cenBox)
        elif cType == "1dg":
            box_x, box_y = centroids.centroid_1dg(cenBox)
        elif cType == "2dg":
            box_x, box_y = centroids.centroid_2dg(cenBox)
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
        ellipse : `photutils.isophote.ellipse.Ellipse`
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
        ellipse : `photutils.isophote.ellipse.Ellipse`
            Ellipse object initialized with self.initializeEllipse()
        **kwargs
            See photutils.isophote.ellipse.Ellipse() docs

        Returns
        -------
        isoList : `photutils.isophote.ellipse.Ellipse.isolist`
            Class instance containing isophotal parameters

        Notes
        -----
        To do no-fit mode, use maxrit=1 or some number smaller than your step
        size.  Give it a maxsma value as well to shorten runtime.
        '''
        isoList = ellipse.fit_image(**kwargs)

        return isoList

    def visualizeEllipses(self, isoList, vmin, vmax):
        '''
        For plotting the fitted ellipses on the image, with log scaling

        Parameters
        ----------
        isoList : `photutils.isophote.ellipse.Ellipse.isolist`
            Class instance containing isophotal parameters
        vmin : `float`
            Minimum in imshow log image scaling
        vmax : `float`
            Maximum in imshow log image scaling
        '''
        logIm = utils.ds9LogScale(self.imageArray)
        plt.imshow(logIm, vmin=vmin, vmax=vmax)
        for i in isoList.sma:
            iso = isoList.get_closest(i)
            x, y, = iso.sampled_coordinates()
            plt.plot(x, y, color='w', markersize=0.5)

    def toTable(self, isoList):
        '''
        Correction function to make ellipse output more machine-readable.
        Replaces all None values within columns with concrete values.

        Parameters
        ----------
        isoList : `photutils.isophote.ellipse.Ellipse.isolist`
            Class instance containing isophotal parameters

        Returns
        -------
        isoTab : `astropy.table.table.Table`
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
                      nBoxes=1000,
                      halfBoxWidth=10,
                      seed=None):
    '''
    Outputs metrics for the background noise and flux for a given image.

    Parameters
    ----------
    maskedImageArray : `numpy.ma.core.MaskedArray`
        Image with interloping sources masked
    galX : `float`
        Central x-coordinate of target galaxy
    galY : `float`
        Central y-coordinate of target galaxy
    galRad : `int`
        Radius out to which to apply circular mask to target galaxy
    nBoxes : `int`, optional
        Number of boxes to use to measure parameters.  The default is 1000.
    halfBoxWidth : `int`, optional
        Half the desired width of the boxes used to calculate the noise, in px.
        The default is 10 px.
    seed : `int`, optional
        A seed for the random generator, for testing purposes

    Returns
    -------
    avRms : `float`
        The average root mean square among counts in N randomly distributed
        boxes, ignoring masked pixels
    sbLim : `float`
        The standard deviation of the median values within N randomly
        distributed boxes, ignoring masked pixels
    sky : `float`
        Median of the median values within N randomly distributed boxes,
        ignoring masked pixels
    '''
    rng = np.random.default_rng(seed)
    # Needed to avoid actually altering the image and mask used
    if type(maskedImageArray) is np.ma.core.MaskedArray:
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

    good_coords = np.where(~mask)
    rms = np.array([])
    medians = np.array([])
    for i in range(nBoxes):
        idx = rng.choice(np.arange(len(good_coords[0])))
        ceny, cenx = good_coords[0][idx], good_coords[1][idx]
        box = utils.imBox(halfBoxWidth, cenx, ceny, maskedImageArray.data)
        rms = np.append(rms, utils.rootMeanSquare(box[np.isfinite(box)]))
        medians = np.append(medians, np.nanmedian(box))

    avRms = np.nanmean(rms)
    sbLim = np.nanstd(medians)
    sky = np.nanmedian(medians)

    return avRms, sbLim, sky


def interpolateAcrossMasks(maskedImageArray,
                           x0, y0, maxRad, pa, ellip, rms):
    '''
    Replaces masked pixels in an image out to a certain radius with an FFT
    recreation of an elliptical aperture within that masked region, plus some
    noise

    Parameters
    ----------
    maskedImageArray : `numpy.ma.core.MaskedArray`
        Image with masked pixels
    x0 : `float`
        Central x-coordinate of galaxy in image
    y0 : `float`
        Central y-coordinate of galaxy in image
    maxRad : `int`
        Maximum radius (semi-major axis) out to which to replace masks
    pa : `float`
        Galaxy average isophote shape position angle, in degrees
    ellip : `float`
        Galaxy average isophote shape ellipticity, 1-b/a where b and a are
        semi-major and major axis lengths
    rms : `float`
        Amount of noise to add to each replacement flux value.  Added as
        drawn from standard normal N(0, rms)

    Returns
    -------
    unmaskedImage : `numpy.ndarray`
        Image with masked pixels replaced out to maxRad
    
    Notes
    -----
    This is a very crude way to do this.  Use at your own risk.
    '''
    rng = np.random.default_rng()

    mask = maskedImageArray.mask
    image = maskedImageArray.data
    rads = np.arange(0, maxRad+1)

    __, __, ellRad, theta = utils.reprojectToEllipse(image,
                                                     x0,
                                                     y0,
                                                     pa,
                                                     ellip)
    for i in range(len(rads)):
        if i == 0:
            continue
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
        old_nans = np.isnan(sort_flux)

        # Need maximum and minimum in masked arrays to be full angle range
        if np.isnan(sort_flux[0]):
            # Replacing with the mean of the first five non-NaN values
            sort_flux[0] = np.mean(sort_flux[~np.isnan(sort_flux)][: 6])
        if np.isnan(sort_flux[-1]):
            # Replacing with the mean of last five non-NaN values
            sort_flux[-1] = np.mean(sort_flux[~np.isnan(sort_flux)][-5:])

        # Now Interpolate across masked pixels
        nans = np.isnan(sort_flux)
        f = interp1d(sort_angle[~nans], sort_flux[~nans], kind='slinear')
        new_flux = f(sort_angle)

        # Derive the DFT
        dft = np.fft.fft(new_flux)
        dft[5:-5] = 0  # Flattening out high frequencies
        smooth_flux = np.fft.ifft(dft).real

        # Now replace masked pixels with interpolated ones
        nanIdx = np.array([np.where(angle == i)[0][0]
                           for i in sort_angle[old_nans]])
        noise = rng.normal(0, rms, size=len(nanIdx))
        flux[nanIdx] = smooth_flux.real[old_nans] + noise  # Add noise
        image[wantRad] = flux
    unmaskedImage = image

    return unmaskedImage
