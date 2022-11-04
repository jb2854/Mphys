#!/opt/anaconda3/envs/astroconda3/bin/python
'''
Update History:
(v0_1) - Correction to the way that the poisson noise is 
handled. Changed incorrect float padding to a sampling of 
a true Poisson distribution with correct implementation 
of the Gain quantity. SkyServer 'sky' and 'skysig' 
quantities are added to the header keywords when using 
real SDSS images. Updated to Python 3.
211026: Mathilda is going to take this apart and put it
        back together...
220126: MSAM working on making a 'cleanish' version of this
        script that is more readable but misses earlier dev
        stages
220322: Adaptation by MSAM to implement realism for HST 
        observations instead of SDSS
221027: Cleanup & commenting for J&E
'''
import time
start_time = time.time()
import numpy as np
import os,sys,string,time
import scipy.interpolate
import scipy.ndimage
import warnings
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from astropy.cosmology import FlatLambdaCDM, Planck15
from astropy.table import Table
from astropy.nddata.utils import Cutout2D
import pdb
import tools
import sep
sep.set_extract_pixstack(9999999)
# EDIT THIS
realsim_dir = ''

# MSAM: leaving this one alone for now
def rebin(array, dimensions=None, scale=None):
    """
    Return the array 'array' to the new 'dimensions'
    conserving the flux in the bins. The sum of the
    array will remain the same as the original array.
    Congrid from the scipy recipies does not generally
    conserve surface brightness with reasonable accuracy.
    As such, even accounting for the ratio of old and new
    image areas, it does not conserve flux. This function
    nicely solves the problem and more accurately
    redistributes the flux in the output. This function
    conserves FLUX so input arrays in surface brightness
    will need to use the ratio of the input and output
    image areas to go back to surface brightness units.
    
    EXAMPLE
    -------
    
    In [0]:
    
    # input (1,4) array (sum of 6)
    y = np.array([0,2,1,3]).reshape(1,4).astype(float)
    # rebin to (1,3) array
    yy = rebin(y,dimensions=(1,3))
    print yy
    print np.sum(yy)
    
    Out [0]:
    
    Rebinning to Dimensions: 1, 3
    [[0.66666667 2.         3.33333333]]
    6.0

    RAISES
    ------
    AssertionError
        If the totals of the input and result array don't
        agree, raise an error because computation may have
        gone wrong.
        
    Copyright: Martyn Bristow (2015) and licensed under GPL v3:
    i.e. free to use/edit but no warranty.
    """
    if dimensions is not None:
        if isinstance(dimensions, float):
            dimensions = [int(dimensions)] * len(array.shape)
        elif isinstance(dimensions, int):
            dimensions = [dimensions] * len(array.shape)
        elif len(dimensions) != len(array.shape):
            raise RuntimeError('')
    elif scale is not None:
        if isinstance(scale, float) or isinstance(scale, int):
            dimensions = map(int, map(round, map(lambda x: x*scale, array.shape)))
        elif len(scale) != len(array.shape):
            raise RuntimeError('')
    else:
        raise RuntimeError('Incorrect parameters to rebin.\n\trebin(array, dimensions=(x,y))\n\trebin(array, scale=a')
    #print "Rebinning to Dimensions: %s, %s" % tuple(dimensions)
    import itertools
    dY, dX = map(divmod, map(float, array.shape), dimensions)

    result = np.zeros(dimensions)
    for j, i in itertools.product(*map(range, array.shape)):
        (J, dj), (I, di) = divmod(j*dimensions[0], array.shape[0]), divmod(i*dimensions[1], array.shape[1])
        (J1, dj1), (I1, di1) = divmod(j+1, array.shape[0]/float(dimensions[0])), divmod(i+1, array.shape[1]/float(dimensions[1]))
        
        # Moving to new bin
        # Is this a discrete bin?
        dx,dy=0,0
        if (I1-I == 0) | ((I1-I == 1) & (di1==0)):
            dx = 1
        else:
            dx=1-di1
        if (J1-J == 0) | ((J1-J == 1) & (dj1==0)):
            dy=1
        else:
            dy=1-dj1
        # Prevent it from allocating outide the array
        I_=min(dimensions[1]-1,I+1)
        J_=min(dimensions[0]-1,J+1)
        result[J, I] += array[j,i]*dx*dy
        result[J_, I] += array[j,i]*(1-dy)*dx
        result[J, I_] += array[j,i]*dy*(1-dx)
        result[J_, I_] += array[j,i]*(1-dx)*(1-dy)
    allowError = 0.1
    assert (array.sum() < result.sum() * (1+allowError)) & (array.sum() >result.sum() * (1-allowError))
    return result

# MSAM: going to rearrange some stuff in this
def ObsRealism(inputName,outputName,
               cosmo=Planck15,
               common_args = {
                   'redshift'      : None,
                   'rebin_to_CCD'  : False, # rebin to CCD angular scale
                   'add_false_psf' : False, # convolve with gaussian psf
                   'add_poisson'   : False  # add poisson noise to galaxy
               },
               hst_args    = {
                   'hst_cam'      : 'wfc3_ir',  # HST camera
                   'hst_band'     : 'f160w',    # HST band
                   'exptime'      : 90          # exposure time in seconds
               },
               hdu_idx=1, # MSAM: my images live in HDUs 1,2,3
               ):
    
    """
    Add realism to idealized unscaled image.
    
    Keyword descriptions pending...
    
    PARAMETERS
    ----------
    inputName : str
        name of FITS file containing input image
    outputName : str
        name of FITS file to put output image into
    common_args : dict
        options for levels of realism to add, see the defaults
    hst_args : dict
        options for HST camera, band, and exposure time. Camera_band are 
        (currently) constrained to 'wfc3_ir_f160w, wfc3_uvis_f438w, 
        wfc3_uvis_f814w, wfc_acs_f814w
    hdu_idx : int, default 1
        index of the HDU in the input file that contains the image

    RETURNS
    -------
    
    """

    band = hst_args['hst_cam'] + '_' + hst_args['hst_band']
    exptime = hst_args['exptime']

    # EDIT THIS
    # properties of the HST filters
    # has to be 'hst_cam' + '_' + 'hst_band' to make them distinct
    # I've just added the filters we looked at for the SNAP proposal but a
    # full list can be found at https://www.tng-project.org/data/docs/api/
    # these are set up as dicts indexed by the camera + band. Add the
    # appropriate values for your bands to the dicts or add items to the
    # hst_args dict as you see fit
    
    # index in the FITS image for each band - update for your images
    band_indices = {'wfc3_ir_f160w'   : 0, 'wfc3_uvis_f438w' : 1,
                    'wfc3_uvis_f814w' : 2, 'wfc_acs_f814w'   : 3}
           
    gain         = {'wfc3_ir_f160w'   : 2.5, 'wfc3_uvis_f438w' : 1.5,
                    'wfc3_uvis_f814w' : 1.5, 'wfc_acs_f814w'   : 2.0} # [e/DN]

    # there are tables for UVIS1 and UVIS2, these are UVIS1
    lambda_eff   = {'wfc3_ir_f160w':15278.47, 'wfc3_uvis_f438w':4323.35,
                    'wfc3_uvis_f814w':7964.25, 'wfc_acs_f814w':7973.39} # [A]

    photflam     = {'wfc3_ir_f160w':1.9429e-20,'wfc3_uvis_f438w':6.7593e-19,
                    'wfc3_uvis_f814w':1.498e-19,'wfc_acs_f814w':7.109e-20}
                   # erg cm**-2 A**-1 e**-1

    # for IR pixels aren't square, they're .135x.121 arcsec. The scale used
    # is the side of a square pixel with the same area
    # for CANDELS you will presumably want to figure out the postprocessed
    # scale
    ccd_scale    = {'wfc3_ir_f160w'   : 0.128, 'wfc3_uvis_f438w' : 0.04,
                    'wfc3_uvis_f814w' : 0.04, 'wfc_acs_f814w'   : 0.05}
                   #[arcsec/px]

    # reasonable for wfc3 filters, the acs wfc seems to have a more complex psf
    fwhm         = {'wfc3_ir_f160w'   : 0.18, # @ 1600 nm
                    'wfc3_uvis_f438w' : 0.070, # @ 400 nm
                    'wfc3_uvis_f814w' : 0.074, # @ 800 nm
                    'wfc_acs_f814w'   : 0.1 }  # don't trust this  #[arcsec] 
    # END HST PARAMS

    # img header and data
    with fits.open(inputName,mode='readonly') as hdul:
        # img header
        header = hdul[0].header
        # img data
        img_data = hdul[hdu_idx].data[band_indices[band]]
        n_pixel = hdul[hdu_idx].header['NAXIS1'] # MSAM: size of my image
        axes = hdul[hdu_idx].header['AXES'] # MSAM: additional image specification

    # mock observation redshift
    # MSAM: think this could be taken from the header
    # redshift = common_args['redshift']
    # MSAM: use common_args to pass a new redshift to change to
    # for HST this shouldn't be an issue, just pass None
    # I've left this in but it's unnecessary
    if common_args['redshift'] is None:
        # leave redshift alone
        redshift = header['redshift']
        arcsec_per_pixel = header['imsize'] / n_pixel # [arcsec/pixel]
        

    else:
        # rescale and adjust SB dimming for new redshift
        redshift = common_args['redshift']
        # pixel scale at new redshift
        arcsec_per_pixel = (header['stellar_half_mass_radius'] /
                            (cosmo.H0.value/100) *
                            cosmo.arcsec_per_kpc_comoving(redshift).value *
                            header['size'] /
                            n_pixel)
        header['imsize'] = arcsec_per_pixel*n_pixel
        
        # cosmological surface brightness dimming
        img_data -= 2.5*4*np.log10((1+header['redshift'])/(1+redshift))

        print(' Original redshift = %g' %header['redshift'])
        print('Placed at redshift = %g' %redshift)
        sys.stdout.flush()
        
    header['HIERARCH new_redshift'] = redshift # useful to have all the same

    speed_of_light = 2.99792458e8 # speed of light [m/s]    
    
    # kiloparsec per arcsecond scale
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z=redshift).value/60. # [kpc/arcsec]
    # luminosity distance in Mpc
    luminosity_distance = cosmo.luminosity_distance(z=redshift) # [Mpc]

    # convert from surface brightness to flux
    img_flux = 10**(-0.4*(img_data + 48.6))  # [erg/(cm**2 s Hz)/arcsec**2]
    img_flux *= (speed_of_light*1e10/lambda_eff[band]**2) 
    img_flux *= arcsec_per_pixel**2 # [erg/(cm**2 s A)]

    # initialize gain in case it is not used
    img_gain = gain[band]
    
    # Add levels of realism

    if common_args['rebin_to_CCD']:
        '''
        Rebin image to a given angular CCD scale
        '''
        # telescope ccd angular scale
        ccd_scale = ccd_scale[band]
        # axes of original image
        nPixelsOld = n_pixel # MSAM
        # axes of regridded image
        nPixelsNew = int(np.floor((arcsec_per_pixel/ccd_scale)*nPixelsOld))
        print(nPixelsNew)
        sys.stdout.flush()
        # rebin to new ccd scale
        if nPixelsNew > nPixelsOld:
            interp = scipy.interpolate.RectBivariateSpline(np.linspace(-1,1,nPixelsOld),
                                                           np.linspace(-1,1,nPixelsOld),
                                                           img_flux, kx=1, ky=1)
            img_flux = interp(np.linspace(-1,1,nPixelsNew),
                              np.linspace(-1,1,nPixelsNew))*(nPixelsOld/nPixelsNew)**2
        else:
            img_flux = rebin(img_flux, dimensions=(nPixelsNew, nPixelsNew))
        n_pixels = nPixelsNew
        # new kpc_per_pixel on ccd
        kpc_per_pixel = kpc_per_arcsec * ccd_scale
        # new arcsec per pixel
        arcsec_per_pixel = ccd_scale
        # header updates
        print('\nAfter CCD scaling:')
        print('kpc_per_arcsec: {}'.format(kpc_per_arcsec)) # MSAM: proper
        print('kpc_per_pixel: {}'.format(kpc_per_pixel)) # MSAM: proper...
        print('arcsec_per_pixel: {}'.format(arcsec_per_pixel))
    print(nPixelsNew)
    sys.stdout.flush()

    
    # convolve with gaussian psf
    if common_args['add_false_psf']:
        '''
        Add Gaussian PSF to image with provided FWHM in
        arcseconds.
        '''
        std = fwhm[band]/arcsec_per_pixel/2.355
        kernel = Gaussian2DKernel(std) # there's an optional y_stddev?!?
        img_flux = convolve(img_flux, kernel)

    # add poisson noise to image
    if common_args['add_poisson']:        
        img_counts = img_flux / photflam[band] * exptime
        img_counts = np.clip(img_counts,a_min=0,a_max=None)
        # add poisson noise to image in electrons
        img_counts = np.random.poisson(lam=img_counts*gain[band])/gain[band]
        # convert back to flux
        img_flux = img_counts * photflam[band] / exptime # double check this

    # convert back to surface brightness
    img_flux /= arcsec_per_pixel**2
    # convert back to AB magnitude
    img_flux *= lambda_eff[band]**2/(speed_of_light*1e10)
    img_mag = -2.5*np.log10(img_flux) - 48.6
    header['BUNIT'] = 'AB mag/arcsec**2'
        
    gimage = outputName
    # if os.access(gimage,0): os.remove(gimage) # or just use overwrite=True

    # header.append(('REDSHIFT',redshift,'Redshift'),end=True)
    header.append(('COSMO','FLAT_LCDM','Cosmology'),end=True)
    header.append(('OMEGA_M',cosmo.Om(0),'Matter density'),end=True)
    header.append(('OMEGA_L',cosmo.Ode(0),'Dark energy density'),end=True)
    header.append(('SCALE_1',arcsec_per_pixel,'[arcsec/pixel]'),end=True)
    header.append(('SCALE_2',kpc_per_pixel,'[kpc/pixel]'),end=True)
    header.append(('SCALE_3',kpc_per_arcsec,'[kpc/arcsec]'),end=True)
    header.append(('LUMDIST',cosmo.luminosity_distance(z=redshift).value,'Luminosity Distance [Mpc]'),end=True)
    header.append(('HIERARCH cam', hst_args['hst_cam']))
    header.append(('HIERARCH band', hst_args['hst_band']))
    header.append(('HIERARCH exptime', exptime, '[s]'))
    header.append(('HIERARCH gain', gain[band], '[e/DN]'))
    header.append(('HIERARCH lambda_eff', lambda_eff[band], '[Angstrom]'))
    header.append(('HIERARCH photflam', photflam[band]), '[erg/cm**2/A/e]')
    header.append(('FWHM', fwhm[band]))

    warnings.simplefilter('ignore', category=AstropyWarning)
    header.extend(zip(common_args.keys(),common_args.values()),unique=True)
    hdu_pri = fits.PrimaryHDU(header=header)
    hdu_pri.header = header
    
    hdu_img = fits.ImageHDU(img_mag)
    hdul = fits.HDUList([hdu_pri, hdu_img])
    hdul.writeto(gimage, overwrite=True) # dangerous

'''
Script executions start here.

The way this is currently set up, it expects input files with the idealised
images in HDUs 1,2,3 (three different angles) with mock observations by 
different cameras in different bands stacked together in each image as defined
in the band_indices dict in the ObsRealism function. It outputs a separate
FITS file for each camera/band, however. It'd be simple enough to edit the loop
to put all the bands together into a single image for each angle. The current
setup also uses temp images created by the ObsRealism function that are deleted
after being combined into the final output, but in principle you could modify
the function to return the realistic images/HDUs and build them up in the main
rather than writing to storage.
'''

# EDIT THIS
img_base_path = ''
output_type = 'FullReal_hst_' # prefix for naming files
output_path = '' # temporary for individual bands
perm_output_path = '' # location of final image

# args for the images
common_args = { 
    'redshift'      : None,    # redshift to place galaxy at--keep as None
    'rebin_to_CCD'  : True,    # rebin to CCD angular scale
    'add_false_psf' : True,    # convolve with gaussian psf
    'add_poisson'   : True,    # add poisson noise to galaxy
}

# EDIT THIS
imgList = [830461612, 830139086, 840341110, 840459489, 840485682]
hst_cams  = ['wfc3_ir', 'wfc3_ir', 'wfc3_uvis', 'wfc3_uvis', 'wfc_acs']
hst_bands = [  'f160w',   'f160w',     'f438w',     'f814w',   'f814w']
exptimes =  [       90,      1730,        1000,         300,       300]


print('%d images to add realism to' %len(imgList))
sys.stdout.flush()

for imgNum in imgList:

    # try:
    
    # EDIT THIS to fit your naming convention
    imgName = '%shst_arcmin_%d.fits' %(img_base_path, imgNum) 

    print('Reading from ' + imgName)
    sys.stdout.flush()

    for cam, band, exp in zip(hst_cams, hst_bands exptimes):
        finName = '%s%s_%s_%s_%d_%d.fits' %(perm_output_path, output_type, cam, band, exp, imgNum)
        print('Writing to ' + finName)
        sys.stdout.flush()
        # loop over the 3 angles
        for hdu_idx in range(1,4):

            outName = '%sFullReal_tmp_%s_%s_%d_%d_%d.fits' %(output_path, cam, band, exp, hdu_idx, imgNum)
            print('HDU idx = %d' %hdu_idx)
            print('%s_%s band' %(cam, band))
            print('Exposure time %d s' %exp)
            sys.stdout.flush()

            # add realism to input and save to output

            ObsRealism(imgName,outName,
                       common_args=common_args,
                       cosmo=Planck15,
                       hst_args = {'hst_cam' : cam,
                                   'hst_band': band,
                                   'exptime' : exp},
                       hdu_idx=hdu_idx)
            print('Time: %s' %tools.time_tostring(time.time() - start_time))
            sys.stdout.flush()


        tmp_hdul = []
        fnames = []
        # primary header, would be nice to put the data about the galaxy here
        pheader = fits.Header()
        for hdu_idx, axes in enumerate(['0,1', '0,2', '1,2'], start=1):
            header = fits.Header()
            # header for each image HDU, would be nice to put only the
            # information about each angle here
            fname = '%sFullReal_tmp_%s_%s_%d_%d_%d.fits' %(output_path, cam, band, exp, hdu_idx, imgNum)
            with fits.open(fname) as hdul:
                img = hdul[1].data
                pheader.extend(hdul[0].header, unique=True)
                header.extend(hdul[1].header, unique=True)
                fnames.append(fname)

            header['HIERARCH axes'] = axes
            tmp_hdul.append(fits.ImageHDU(data=np.array(img), header=header))


            phdu = fits.PrimaryHDU(header=pheader)
            hdul = fits.HDUList([phdu])
            for hdu in tmp_hdul:
                hdul.append(hdu)

            hdul.writeto(finName, overwrite=True)
            print('Saved to ' + finName)

            for fname in fnames:
                if os.access(fname,0):os.remove(fname)


    # except Exception as excep:
    #     print(excep)
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     print (exc_type, exc_tb.tb_lineno)
    #     print("Error generating realism for image %d - Skipping." %imgNum)
    #     sys.stdout.flush()    
    #     with open('realism-bad-gals.dat', 'a+') as logfile:
    #         logfile.write('%d\n' %imgNum)

