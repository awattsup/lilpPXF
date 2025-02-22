"""
#####################################################################

Copyright (C) 2001-2022, Michele Cappellari
E-mail: michele.cappellari_at_physics.ox.ac.uk

Updated versions of the software are available from my web page
http://purl.org/cappellari/software

If you have found this software useful for your
research, we would appreciate an acknowledgment to use of
"the Voronoi binning method by Cappellari & Copin (2003)".

This software is provided as is without any warranty whatsoever.
Permission to use, for non-commercial purposes is granted.
Permission to modify for personal or internal use is granted,
provided this copyright and disclaimer are included unchanged
at the beginning of the file. All other rights are reserved.
In particular, redistribution of the code is not allowed.

#MODIFIED BY A.B.WATTS FOR MULTI-PARAMETER VORONOI BINNING

#####################################################################

"""
from time import perf_counter as clock
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial, ndimage

#----------------------------------------------------------------------------

def _sn_func(index, signal=None, noise=None):
    """
    Default function to calculate the S/N of a bin with spaxels "index".

    The Voronoi binning algorithm does not require this function to have a
    specific form and this default one can be changed by the user if needed
    by passing a different function as::

        voronoi_2d_binning(..., sn_func=sn_func)

    The S/N returned by sn_func() does not need to be an analytic
    function of S and N.

    There is also no need for sn_func() to return the actual S/N.
    Instead sn_func() could return any quantity the user needs to equalize.

    For example sn_func() could be a procedure which uses ppxf to measure
    the velocity dispersion from the coadded spectrum of spaxels "index"
    and returns the relative error in the dispersion.

    Of course an analytic approximation of S/N, like the one below,
    speeds up the calculation.

    :param index: integer vector of length N containing the indices of
        the spaxels for which the combined S/N has to be returned.
        The indices refer to elements of the vectors signal and noise.
    :param signal: vector of length M>N with the signal of all spaxels.
    :param noise: vector of length M>N with the noise of all spaxels.
    :return: scalar S/N or another quantity that needs to be equalized.

    """
    sn = np.sum(signal[index])/np.sqrt(np.sum(noise[index]**2))

    # The following commented line illustrates, as an example, how one
    # would include the effect of spatial covariance using the empirical
    # Eq.(1) from http://adsabs.harvard.edu/abs/2015A%26A...576A.135G
    # Note however that the formula is not accurate for large bins.
    #
    # sn /= 1 + 1.07*np.log10(index.size)

    return  sn

#----------------------------------------------------------------------

def voronoi_tessellation(x, y, xnode, ynode, scale):
    """
    Computes (Weighted) Voronoi Tessellation of the pixels grid

    """
    if scale[0] == 1:  # non-weighted VT
        tree = spatial.cKDTree(np.column_stack([xnode, ynode]))
        classe = tree.query(np.column_stack([x, y]))[1]
    else:
        if x.size < 1e4:
            classe = np.argmin(((x[:, None] - xnode)**2 + (y[:, None] - ynode)**2)/scale**2, axis=1)
        else:  # use for loop to reduce memory usage
            classe = np.zeros(x.size, dtype=int)
            for j, (xj, yj) in enumerate(zip(x, y)):
                classe[j] = np.argmin(((xj - xnode)**2 + (yj - ynode)**2)/scale**2)

    return classe

#----------------------------------------------------------------------

def _roundness(x, y, pixelSize):
    """
    Implements equation (5) of Cappellari & Copin (2003)

    """
    n = x.size
    equivalentRadius = np.sqrt(n/np.pi)*pixelSize
    xBar, yBar = np.mean(x), np.mean(y)  # Geometric centroid here!
    maxDistance = np.sqrt(np.max((x - xBar)**2 + (y - yBar)**2))
    roundness = maxDistance/equivalentRadius - 1.

    return roundness

#----------------------------------------------------------------------

def _accretion(x, y, signal, noise, target_sn, pixelsize, quiet, sn_func):
    """
    Implements steps (i)-(v) in section 5.1 of Cappellari & Copin (2003)

    """
    n = x.size
    classe = np.zeros(n, dtype=int)  # will contain the bin number of each given pixel
    good = np.zeros(n, dtype=bool)   # will contain 1 if the bin has been accepted as good

    # For each point, find the distance to all other points and select the minimum.
    # This is a robust but slow way of determining the pixel size of unbinned data.
    #
    if pixelsize is None:
        if x.size < 1e4:
            pixelsize = np.min(spatial.distance.pdist(np.column_stack([x, y])))
        else:
            raise ValueError("Dataset is large: Provide `pixelsize`")

    if signal.ndim == 2:
        currentBin = np.argmax(signal[:,0]/noise[:,0])  # Start from the pixel with highest S/N
        print('flag')
    else:
        currentBin = np.argmax(signal/noise)  # Start from the pixel with highest S/N
    # print(currentBin)
    SN = sn_func(currentBin, signal, noise)
    # print(SN)
    # exit()


    # Rough estimate of the expected final bins number.
    # This value is only used to give an idea of the expected
    # remaining computation time when binning very big dataset.
    #
    # w = np.min(signal/noise,axis=1) < target_sn
    w = signal/noise < target_sn
    # w = sn_func(np.arange(len(signal),dtype=int),signal,noise) < target_sn
    # print(w)
    # exit()
    maxnum = int(np.sum((signal[w]/noise[w])**2)/target_sn**2 + np.sum(~w))
    # print(maxnum)
    # exit()
    # The first bin will be assigned CLASS = 1
    # With N pixels there will be at most N bins
    #
    for ind in range(1, n+1):

        if not quiet:
            print(ind, ' / ', maxnum)

        classe[currentBin] = ind  # Here currentBin is still made of one pixel
        xBar, yBar = x[currentBin], y[currentBin]    # Centroid of one pixels


        while True:

            if np.all(classe):
                break  # Stops if all pixels are binned

            # Find the unbinned pixel closest to the centroid of the current bin
            #
            unBinned = np.flatnonzero(classe == 0)
            k = np.argmin((x[unBinned] - xBar)**2 + (y[unBinned] - yBar)**2)

            # (1) Find the distance from the closest pixel to the current bin
            #
            minDist = np.min((x[currentBin] - x[unBinned[k]])**2 + (y[currentBin] - y[unBinned[k]])**2)

            # (2) Estimate the `roundness' of the POSSIBLE new bin
            #
            nextBin = np.append(currentBin, unBinned[k])
            roundness = _roundness(x[nextBin], y[nextBin], pixelsize)

            # (3) Compute the S/N one would obtain by adding
            # the CANDIDATE pixel to the current bin
            #
            SNOld = SN
            SN = sn_func(nextBin, signal, noise)

            # Break if any of these tests is true:
            # 1) The CANDIDATE pixel is disconnected from the current bin;
            # 2) The POSSIBLE new bin is too elongated;
            # 3) The distance between new-S/N and target_sn increases;
            # 4) The new-S/N decreases.
            #
            if (np.sqrt(minDist) > 1.2*pixelsize or roundness > 0.3
                or abs(SN - target_sn) > abs(SNOld - target_sn) or SNOld > SN):
                if SNOld > 0.8*target_sn:
                    good[currentBin] = 1
                # print('flag')
                break

            # If all the above 3 tests are negative then accept the CANDIDATE
            # pixel, add it to the current bin, and continue accreting pixels
            #
            classe[unBinned[k]] = ind
            currentBin = nextBin

            # Update the centroid of the current bin
            #
            xBar, yBar = np.mean(x[currentBin]), np.mean(y[currentBin])

        # Get the centroid of all the binned pixels
        #
        binned = classe > 0
        if np.all(binned):
            break  # Stop if all pixels are binned
        xBar, yBar = np.mean(x[binned]), np.mean(y[binned])

        # Find the closest unbinned pixel to the centroid of all
        # the binned pixels, and start a new bin from that pixel.
        #
        unBinned = np.flatnonzero(classe == 0)
        if sn_func(unBinned, signal, noise) < target_sn:
            break  # Stops if the remaining pixels do not have enough capacity
        k = np.argmin((x[unBinned] - xBar)**2 + (y[unBinned] - yBar)**2)
        currentBin = unBinned[k]    # The bin is initially made of one pixel
        SN = sn_func(currentBin, signal, noise)

    classe *= good  # Set to zero all bins that did not reach the target S/N
    # print(classe)ß

    return classe, pixelsize

#----------------------------------------------------------------------------

def _reassign_bad_bins(classe, x, y):
    """
    Implements steps (vi)-(vii) in section 5.1 of Cappellari & Copin (2003)

    """
    # Find the centroid of all successful bins.
    # CLASS = 0 are unbinned pixels which are excluded.
    #
    good = np.unique(classe[classe > 0])
    xnode = ndimage.mean(x, labels=classe, index=good)
    ynode = ndimage.mean(y, labels=classe, index=good)

    # Reassign pixels of bins with S/N < target_sn
    # to the closest centroid of a good bin
    #
    bad = classe == 0
    index = voronoi_tessellation(x[bad], y[bad], xnode, ynode, [1])
    classe[bad] = good[index]

    # Recompute all centroids of the reassigned bins.
    # These will be used as starting points for the CVT.
    #
    good = np.unique(classe)
    xnode = ndimage.mean(x, labels=classe, index=good)
    ynode = ndimage.mean(y, labels=classe, index=good)

    return xnode, ynode

#----------------------------------------------------------------------------

def _cvt_equal_mass(x, y, signal, noise, xnode, ynode, pixelsize, quiet, sn_func, wvt):
    """
    Implements the modified Lloyd algorithm
    in section 4.1 of Cappellari & Copin (2003).

    NB: When the keyword WVT is set this routine includes
    the modification proposed by Diehl & Statler (2006).

    """
    dens2 = (signal/noise)**4     # See beginning of section 4.1 of CC03
    scale = np.ones_like(xnode)   # Start with the same scale length for all bins
    diff = np.zeros_like(xnode)

    for it in range(xnode.size):  # Do at most xnode.size iterations

        xnode_old, ynode_old = xnode.copy(), ynode.copy()
        classe = voronoi_tessellation(x, y, xnode, ynode, scale)

        # Computes centroids of the bins, weighted by dens**2.
        # Exponent 2 on the density produces equal-mass Voronoi bins.
        # The geometric centroids are computed if WVT keyword is set.
        #
        good = np.unique(classe)
        if wvt:
            for k in good:
                index = np.flatnonzero(classe == k)   # Find subscripts of pixels in bin k.
                xnode[k], ynode[k] = np.mean(x[index]), np.mean(y[index])
                sn = sn_func(index, signal, noise)
                scale[k] = np.sqrt(index.size/sn)  # Eq. (4) of Diehl & Statler (2006)
        else:
            mass = ndimage.sum(dens2, labels=classe, index=good)
            xnode = ndimage.sum(x*dens2, labels=classe, index=good)/mass
            ynode = ndimage.sum(y*dens2, labels=classe, index=good)/mass

        diff2 = np.sum((xnode - xnode_old)**2 + (ynode - ynode_old)**2)
        diff[it] = np.sqrt(diff2)/pixelsize

        if not quiet:
            print('Iter: %4i  Diff: %.4g' % (it + 1, diff[it]))

        # Test for convergence, or for repetition to avoid cycling
        if diff[it] < 0.1 or diff[it] in diff[:it]:
            break

    # If coordinates have changed, re-compute (Weighted) Voronoi Tessellation of the pixels grid
    #
    if diff[it] > 0:
        classe = voronoi_tessellation(x, y, xnode, ynode, scale)
        good = np.unique(classe)  # Check for zero-size Voronoi bins

    # Only return the generators and scales of the nonzero Voronoi bins

    return xnode[good], ynode[good], scale[good], it

#-----------------------------------------------------------------------

def _compute_useful_bin_quantities(x, y, signal, noise, xnode, ynode, scale, sn_func):
    """
    Recomputes (Weighted) Voronoi Tessellation of the pixels grid to make sure
    that the class number corresponds to the proper Voronoi generator.
    This is done to take into account possible zero-size Voronoi bins
    in output from the previous CVT (or WVT).

    """
    # classe will contain the bin number of each given pixel
    classe = voronoi_tessellation(x, y, xnode, ynode, scale)

    # At the end of the computation evaluate the bin luminosity-weighted
    # centroids (xbar, ybar) and the corresponding final S/N of each bin.
    #
    good = np.unique(classe)
    xbar = ndimage.mean(x, labels=classe, index=good)
    ybar = ndimage.mean(y, labels=classe, index=good)
    area = np.bincount(classe)
    sn = np.empty_like(xnode)
    for k in good:
        index = np.flatnonzero(classe == k)   # index of pixels in bin k.
        sn[k] = sn_func(index, signal, noise)

    return classe, xbar, ybar, sn, area

#-----------------------------------------------------------------------

def _display_pixels(x, y, counts, pixelsize):
    """
    Display pixels at coordinates (x, y) coloured with "counts".
    This routine is fast but not fully general as it assumes the spaxels
    are on a regular grid. This needs not be the case for Voronoi binning.

    """
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    nx = int(round((xmax - xmin)/pixelsize) + 1)
    ny = int(round((ymax - ymin)/pixelsize) + 1)
    img = np.full((nx, ny), np.nan)  # use nan for missing data
    j = np.round((x - xmin)/pixelsize).astype(int)
    k = np.round((y - ymin)/pixelsize).astype(int)
    img[j, k] = counts

    plt.imshow(np.rot90(img), interpolation='nearest', cmap='prism',
               extent=[xmin - pixelsize/2, xmax + pixelsize/2,
                       ymin - pixelsize/2, ymax + pixelsize/2])

#----------------------------------------------------------------------

def mod_voronoi_2d_binning(x, y, signal, noise, target_sn, cvt=True, pixelsize=None,
                       plot=True, quiet=True, sn_func=None, wvt=True):
    """
    VorBin Purpose
    --------------

    Perform adaptive spatial binning of two-dimensional data
    to reach a chosen constant signal-to-noise ratio per bin.
    This program implements the algorithm described in section 5.1 of
    `Cappellari & Copin (2003) <http://adsabs.harvard.edu/abs/2003MNRAS.342..345C>`_.

    Calling Sequence
    ----------------

    .. code-block:: python

        bin_number, x_gen, y_gen, x_bar, y_bar, sn, nPixels, scale = voronoi_2d_binning(
            x, y, signal, noise, target_sn, cvt=True, pixelsize=None, plot=True,
            quiet=True, sn_func=None, wvt=True)


    Input Parameters
    ----------------

    x: array_like with shape (npix,)
        Vector containing the X coordinate of the pixels to bin.
        Arbitrary units can be used (e.g. arcsec or pixels).
        In what follows the term "pixel" refers to a given
        spatial element of the dataset (sometimes called "spaxel" in
        the integral-field spectroscopy community): it can be an actual pixel
        of a CCD image, or a spectrum position along the slit of a long-slit
        spectrograph or in the field of view of an IFS
        (e.g. a lenslet or a fibre).
        It is assumed here that pixels are arranged in a regular
        grid, so that the pixel size is a well-defined quantity.
        The pixel grid, however, can contain holes (some pixels can be
        excluded from the binning) and can have an irregular boundary.
        See the above reference for an example and details.ß
    y: array_like with shape (npix,)
        Vector containing the Y coordinate of the pixels to bin.
    signal: array_like with shape (npix,)
        Vector containing the signal associated with each pixel, having 
        coordinates ``(x, y)``. If the "pixels" are the apertures of an 
        integral-field spectrograph, then the signal can be defined as the 
        average flux in the spectral range under study, for each aperture.
        If pixels are those of a CCD in a galaxy image, the signal will be 
        simply the counts in each pixel.
    noise: array_like with shape (npix,)
        Vector containing the corresponding noise (1 sigma error) associated 
        with each pixel.
        
        Generally the ``signal`` and ``noise`` are used to comnpute the 
        binning but one may also compute the ``S/N`` on-the-fly using the 
        ``sn_func`` keyword.
    target_sn: float
        The desired signal-to-noise ratio in the final 2D-binned data. 
        E.g. a ``S/N~50`` per spectral pixel may be a reasonable value to 
        extract stellar kinematics information from galaxy spectra.

    Optional Keywords
    -----------------

    cvt: bool, optional
        Set ``cvt=False`` to skip the Centroidal Voronoi Tessellation
        (CVT) step (vii) of the algorithm in Section 5.1 of
        `Cappellari & Copin (2003)`_.
        This may be useful if the noise is strongly non-Poissonian,
        the pixels are not optimally weighted, and the CVT step
        appears to introduce significant gradients in the S/N.
        An alternative consists of setting ``wvt=True`` below.
    plot: bool, optional
        Set ``plot=True`` to produce a plot of the two-dimensional
        bins and of the corresponding S/N at the end of the computation.
    pixsize: float, optional
        Optional pixel scale of the input data.
        This can be the size of a pixel of an image or the size
        of a spaxel or lenslet in an integral-field spectrograph.

        The value is computed automatically by the program, but
        this can take a long time when ``(x, y)`` have many elements.
        In those cases, the ``pixsize`` keyword should be given.
    sn_func: callable, optional
        Generic function to calculate the S/N of a bin with spaxels
        ``index`` with the form:: 
            
            sn = func(index, signal, noise)
            
        If ``sn_func=None``, the binning uses the default ``_sn_func()`` 
        included in the program file, which implements eq.(2) of 
        `Cappellari & Copin (2003)`_. However, another more general 
        function can be adopted if needed. 

        The S/N returned by sn_func() does not need to be an analytic
        function of S and N. There is also no need for sn_func() to 
        return the actual S/N. Instead sn_func() could return any 
        quantity the user needs to equalize.
    
        For example ``sn_func()`` could be a procedure which uses ``ppxf`` 
        to measure the velocity dispersion from the coadded spectrum of 
        spaxels ``index`` and returns the relative error in the dispersion.
    
        Of course an analytic approximation of ``S/N``, like the default,
        speeds up the calculation.

        Read the docstring of ``_sn_func()`` inside the file 
        ``voronoi_2d_binning.py`` for more details.
    quiet: bool, optional
        By default, the program shows the progress while accreting
        pixels and then while iterating the CVT. Set ``quiet=True``
        to avoid printing progress results.
    wvt:
        When ``wvt=True``, the routine ``bin2d_cvt_equal_mass`` is
        modified as proposed by Diehl & Statler (2006, MNRAS, 368, 497).
        In this case the final step of the algorithm, after the bin-accretion
        stage, is not a modified Centroidal Voronoi Tessellation, but it uses
        a Weighted Voronoi Tessellation.
        This may be useful if the noise is strongly non-Poissonian,
        the pixels are not optimally weighted, and the CVT step
        appears to introduce significant gradients in the S/N.
        An alternative consists of setting ``cvt=False`` above.
        If you set ``wvt=True`` you should also include a reference to
        "the WVT modification proposed by Diehl & Statler (2006)."

    Output Parameters
    -----------------

    bin_number: array_like with shape (npix,)
        Vector containing the bin number assigned to each input pixel.
        The index goes from zero to ``nbin - 1``.

        IMPORTANT: This vector alone is all one needs to make any subsequent
        computation on the binned data. everything else is optional and can
        be ignored!
    x_gen: array_like with shape (nbin,)
        Vector of the X coordinates of the bin generators.
        These generators are an alternative way of defining 
        the Voronoi tessellation.

        NOTE: usage of this vector is deprecated as it can be confusing.
    y_gen: array_like with shape (nbin,)
        Vector of Y coordinates of the bin generators.

        NOTE: usage of this vector is deprecated as it can be confusing.
    x_bar: array_like with shape (nbin,)
        Vector of X coordinates of the bins luminosity weighted centroids.
        Useful for plotting interpolated maps.
    y_bar: array_like with shape (nbin,)
        Vector of Y coordinates of the bins luminosity-weighted centroids.
    sn: array_like with shape (nbin,)
        Vector with the final SN of each bin.
    npixels: array_like with shape (nbin,)
        Vector with the number of pixels of each bin.
    scale: array_like with shape (nbin,)
        Vector with the scale length of the Weighted Voronoi Tessellation, when
        ``wvt=True``. In that case ``scale`` is *needed* together with the
        coordinates ``xbin`` and ``ybin`` of the generators, to compute the
        tessellation (but it is safer to simply use the ``binnumber`` vector
        instead).

    When some pixels have no signal
    -------------------------------

    Binning should not be used blindly when some pixels contain significant
    noise but virtually no signal. This situation may happen e.g. when
    extracting the gas kinematics from observed galaxy spectra. One way of
    using ``voronoi_2d_binning`` consists of first selecting the pixels with
    ``S/N`` above a minimum threshold and then binning each set of connected
    pixels *separately*. Alternatively one may optimally weight the pixels
    before binning. For details, see Sec. 2.1 of `Cappellari & Copin (2003)`_.

    Binning X-ray data
    ------------------

    For X-ray data, or other data coming from photon-counting devices the noise
    is generally accurately Poissonian. In the Poissonian case, the S/N in a
    bin can never decrease by adding a pixel [see Sec.2.1 of
    `Cappellari & Copin (2003)`_], and it is preferable to bin the data
    *without* first removing the observed pixels with no signal.

    Binning very big images
    -----------------------

    Computation time in voronoi_2d_binning scales nearly as npixels^1.5, so it
    may become a problem for large images (e.g. at the time of writing
    ``npixels > 1000x1000``). Let's assume that we really need to bin the image
    as a whole and that the S/N in a significant number of pixels is well above
    our target S/N. As for many other computational problems, a way to radically
    decrease the computation time consists of proceeding hierarchically. Suppose
    for example we have a 4000x4000 pixels image, we can do the following:

    1. Rebin the image regularly (e.g. in groups of 8x8 pixels) to a manageable
       size of 500x500 pixels;
    2. Apply the standard Voronoi 2D-binning procedure to the 500x500 image;
    3. Transform all unbinned pixels (which already have enough S/N) of the
       500x500 Voronoi 2D-binned image back into their original individual
       full-resolution pixels;
    4. Now apply Voronoi 2D-binning only to the connected regions of
       full-resolution pixels;
    5. Merge the set of lower resolution bins with the higher resolution ones.

    """
    assert x.size == y.size == signal.size == noise.size, \
        'Input vectors (x, y, signal, noise) must have the same size'
    assert np.all((noise > 0) & np.isfinite(noise)), \
        'NOISE must be positive and finite'


    if x.ndim == 2:
        x = x[:,0]
        y = y[:,0]

    if sn_func is None:
        sn_func = _sn_func

    # Perform basic tests to catch common input errors
    #
    # if sn_func(np.flatnonzero(noise > 0), signal, noise) < target_sn:
    if sn_func(np.flatnonzero(noise > 0), signal.flatten(), noise.flatten()) < target_sn:
        raise ValueError("""Not enough S/N in the whole set of pixels.
            Many pixels may have noise but virtually no signal.
            They should not be included in the set to bin,
            or the pixels should be optimally weighted.
            See Cappellari & Copin (2003, Sec.2.1) and README file.""")
    if np.min(signal/noise) > target_sn:
        raise ValueError('All pixels have enough S/N and binning is not needed')

    t1 = clock()
    if not quiet:
        print('Bin-accretion...')
    classe, pixelsize = _accretion(
        x, y, signal, noise, target_sn, pixelsize, quiet, sn_func)
    if not quiet:
        print(np.max(classe), ' initial bins.')
        print('Reassign bad bins...')
    xnode, ynode = _reassign_bad_bins(classe, x, y)
    if not quiet:
        print(xnode.size, ' good bins.')
    t2 = clock()
    if cvt:
        if not quiet:
            print('Modified Lloyd algorithm...')
        xnode, ynode, scale, it = _cvt_equal_mass(
            x, y, signal, noise, xnode, ynode, pixelsize, quiet, sn_func, wvt)
        if not quiet:
            print(it, ' iterations.')
    else:
        scale = np.ones_like(xnode)
    classe, xBar, yBar, sn, area = _compute_useful_bin_quantities(
        x, y, signal, noise, xnode, ynode, scale, sn_func)
    single = area == 1
    t3 = clock()
    if not quiet:
        print('Unbinned pixels: ', np.sum(single), ' / ', x.size)
        print('Fractional S/N scatter (%):', np.std(sn[~single] - target_sn, ddof=1)/target_sn*100)
        print('Elapsed time accretion: %.2f seconds' % (t2 - t1))
        print('Elapsed time optimization: %.2f seconds' % (t3 - t2))

    if plot:
        plt.clf()
        plt.subplot(211)
        rnd = np.argsort(np.random.random(xnode.size))  # Randomize bin colors
        _display_pixels(x, y, rnd[classe], pixelsize)
        plt.plot(xnode, ynode, '+w', scalex=False, scaley=False) # do not rescale after imshow()
        plt.xlabel('R (arcsec)')
        plt.ylabel('R (arcsec)')
        plt.title('Map of Voronoi bins')

        plt.subplot(212)
        rad = np.sqrt(xBar**2 + yBar**2)  # Use centroids, NOT generators
        plt.plot(np.sqrt(x**2 + y**2), signal/noise, '.k', label='Input S/N')
        if np.any(single):
            plt.plot(rad[single], sn[single], 'xb', label='Not binned')
        plt.plot(rad[~single], sn[~single], 'or', label='Voronoi bins')
        plt.xlabel('R (arcsec)')
        plt.ylabel('Bin S/N')
        plt.axis([np.min(rad), np.max(rad), 0, np.max(sn)*1.05])  # x0, x1, y0, y1
        plt.plot([np.min(rad), np.max(rad)], [target_sn, target_sn], ls='--', label='Target S/N')
        plt.legend()

    return classe, xnode, ynode, xBar, yBar, sn, area, scale

#----------------------------------------------------------------------------
