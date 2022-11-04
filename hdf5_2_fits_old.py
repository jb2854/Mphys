#!/opt/anaconda3/envs/astroconda3/bin/python
'''
ATTENTION: comments saying 'EDIT THIS' mark places where you need to make
changes to get the script to run. Setting paths, adding your API key, 
optionally adding whatever merger catalogue info you have

30 June 2021
Script that (if necessary) downloads the SDSS ugriz images of the specified
IllustrisTNG subhalo and converts the five HDF5 files to a single FITS file. 
The idea is that you can set the image parameters the way you want and then
write a script that will go through a catalogue and batch download images.
Their server doesn't love it if you request too many downloads in parallel
though.

args:
subhaloID  index of the desired subhalo in the IllustrisTNG Subfind catalogue
snap       snapshot number of the desired image (some subhalos seem to linger
           between multiple snapshots)
NOTES 21 March 2022 have added some options to use this for HST bands
      12 May 2022 rearranging somewhat to make it clear what changes between
                  choosing SDSS and HST bands
      11 October 2022 this is a copy for James and Ethan MPhys
      13 October 2022 Ethan edit
            import os
            base = './' 
            for index in p_test.loc[p_test["ProgIDs"].str.len() > 5].index:
                FITS_path = os.path.join(base+'TNG_FITS_dir/'+str(p_test.loc[index, 'DesSFID']))
                print(FITS_path)
                os.mkdir(FITS_path)
                SFID = p_test.loc[index,'DesSFID']
                SS = p_test.loc[index, 'SnapNum']-1
                %run -i hdf52fits_mphys1.py $SFID $SS $FITS_path
                SS += 1
                for ID in p_test.loc[index, 'ProgSFIDs']:
                    SFID = ID
                    %run -i hdf52fits_mphys1.py $SFID $SS $FITS_path
'''
import time
start_time = time.time()
import os
import sys
import datetime
import glob
'''
Arguments:
subhaloID  ID number of subhalo to image
snap       snapshot of subhalo to image
'''

def grabFITS(subhalo_id, snap, img_dir):
    #if len(sys.argv) < 4:
     #   print('Usage: ./hdf52fits.py subhaloID snap FITS_dir')
      #  sys.exit(0)

    #subhalo_id = int(sys.argv[1])
    #snap = int(sys.argv[2])
    #img_dir = str(sys.argv[3])

    # set specifications for bands here
    # these lines are HST
    bands =  ['wfc3_ir_f160w','wfc3_uvis_f438w','wfc3_uvis_f814w','wfc_acs_f814w']
    n_pix = 1500 # HST is pretty high quality
    # these lines are SDSS
    #bands = ['sdss_' + band for band in 'ugriz']
    #n_pix = 800 # default size

    ## EDIT THIS
    # permanent storage for FITS files
    #img_dir = './DF_dir/FITS_dir/'
    # temporary storage for HDF5 files
    tmp_dir = '/mnt/jb2854-XDrive/Physics/StudentProjects/MPhys/2022-23/IllustrisTNG-Mergers/IMG_dir/HDF5_dir/'

    # default image names
    fitsname = '%s%02d%07d.fits' %(img_dir, snap, subhalo_id)
    
    #fitsname = '%ssfid_{}_snap_{}_axes_{}_band_{}.fits' %(img_dir)
    # but call it something that makes sense for what you're doing

    print('Writing final output to %s' %fitsname)
    print('Time: ' + str(datetime.timedelta(seconds=(time.time() - start_time))))
    sys.stdout.flush()

    # uncomment to prevent overwriting the FITS files
    if os.path.isfile(fitsname):
        print('%s already exists' %fitsname)
        sys.exit(0)

    import numpy as np
    from astropy.table import Table
    from astropy.io import fits
    from astropy.cosmology import Planck15
    import pandas as pd
    from illustris_python import illustris_python as il
    import h5py

    # EDIT THIS
    api_key = '7441604020b774ac494f99f04f80efd7'
    url_prefix = 'https://www.tng-project.org/api/TNG100-1/snapshots/%d/subhalos/%d/' %(snap, subhalo_id)

    basePath = '/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG/TNG100-1/output'

    # EDIT THIS
    # below is the code where I load my merger catalogue information for the
    # subhalo. Assuming you want to add stuff from your own catalogue, uncomment
    # and update as you need
    #commented out stuff to make proper headers involving reading from table
    table_dir = '/mnt/jb2854-XDrive/Physics/StudentProjects/MPhys/2022-23/IllustrisTNG-Mergers/DF_dir/'

    # get merger label info for our subhalo
    object_table_name = 'objectsdf_backup.parquet'
    #merger_table = Table.read(table_dir + merger_table_name,
    #                          format='ascii.csv')
    object_table = pd.read_parquet(table_dir + object_table_name)

    print('Read merger table from %s%s' %(table_dir, object_table_name))
    print('Time: ' + str(datetime.timedelta(seconds=(time.time() - start_time))))
    sys.stdout.flush()


    # index of the galaxy in the table
    idx = np.where(np.logical_and(object_table['SFID'] == subhalo_id,
                                  object_table['SnapNum'] == snap))[0]
    # end getting merger catalogue info

    # start building header for FITS file
    header = fits.Header()
    header['HIERARCH subhaloID'] = subhalo_id
    header['HIERARCH snapshot'] = snap

    # EDIT THIS
    # lookback time & redshift of snap, I have a table of this info
    time_table = pd.read_csv(table_dir + 'time_table_TNG_100.csv')

    header['HIERARCH redshift'] = float(time_table['Redshift'][snap])
    header['HIERARCH lookback_time'] = float(time_table['Lookback'][snap])

    # parameters to set for the image download from the IllustrisTNG visualiser
    # see the very last table at https://www.tng-project.org/data/docs/api/
    # might also need fields for 'rVirFracs', 'fracsType'
    img_params = {'partType' : 'stars',
                  'partField': 'stellarBandObsFrame-', # 'stellarBand-' for RF
                  'size'     : 20, #4, #2,
                  'sizeType' : 'rHalfMassStars', #'arcmin', #'rVirial',
                  'method'   : 'sphMap',
                  'nPixels'  : '%d,%d' %(n_pix, n_pix), 
                  'axes'     : '0,1',  # I think all the arguments below here are
                  'rasterPx' : 1100,   # not necessary for the HDF5 files but
                  'plotStyle': 'edged',# they don't do any HARM
                  'labelZ'   : 'False',
                  'labelSim' : 'False',
                  'labelHalo': 'False',
                  'title'    : 'False',
                  'colorbars': 'False',
                  'scalebar' : 'False'
                 }

    # and add these to our header
    for pname in img_params.keys():
        header['HIERARCH ' + pname] = img_params[pname]

    # add band info to the header
    for i, band in enumerate(bands):
        header['HIERARCH partField_%d' %i] = (header['partField'] + band)

    # EDIT THIS
    # adding merger catalogue information to the header
    keys_to_copy = ['ratio_recent', 'snap_recent', 'dt_recent',
                     'ratio_biggest', 'snap_biggest', 'dt_biggest']
    # relevant keys have different endings depending on if it's a pre
    # or post merger but they are always ratio_, snap_, and dt_

    '''keys_to_copy = np.array(merger_table.keys())[[key[:3] == 'dt_' or
                                                  key[:5] == 'snap_' or
                                                  key[:6] == 'ratio_'
                                                  for key in merger_table.keys()]]'''

    # FITS is moaning about the astropy tables so I have to cast to ints/floats
    types = [float, int, float, float, int, float] 
    # FITS seems to not let me make long enough comments, need to
    # figure out how to paraphrase
    comments = [
        'mass ratio of most recent merger',
        'snapshot of most recent merger', 
        'time since most recent merger', 
        'highest mass ratio up to 5 snapshots before most recent',
        'snapshot of highest-mass-ratio merger up to 5 before most recent',
        'time since highest-mass-ratio merger up to 5 snapshots before most recent'
    ]

    n_keys = len(keys_to_copy)
    for i in range(n_keys):
        key = keys_to_copy[i]
        dtype = types[i]
        header['HIERARCH ' + key] = (dtype(object_table[key][idx]), comments[i])
    # end adding merger catalogue info

    # adding some relevant group catalogue info for our subhalo
    subhalo_data = il.groupcat.loadSubhalos(basePath, snap,
                                            fields=['SubhaloMassType',
                                                    'SubhaloHalfmassRadType'])
    header['HIERARCH stellar_mass'] = (float(subhalo_data['SubhaloMassType'][subhalo_id,
                                                                             il.util.partTypeNum('stellar')]),
                                       '1e10 Msun/h')
    header['HIERARCH stellar_half_mass_radius'] = (float(subhalo_data['SubhaloHalfmassRadType'][subhalo_id,
                                                                                                il.util.partTypeNum('stellar')]),
                                                         'ckpc/h')
    header['HIERARCH stellar_mass_ratio'] = float(subhalo_data['SubhaloMassType'][subhalo_id,
                                                                                  il.util.partTypeNum('stellar')]
                                                  / subhalo_data['SubhaloMassType'][subhalo_id,
                                                                                   il.util.partTypeNum('dm')])


    # snapshot 99 galaxies have their assumed size based on snapshot 98 redshift
    # a source of many, many headaches
    if snap == 99:
        redshift = time_table['Redshift'][98]
    else:
        redshift = header['Redshift']

    # EDIT THIS if you change the sizeType in the img_params
    # scaled by half mass radius
    header['HIERARCH imsize'] = (header['stellar_half_mass_radius'] *
                                 img_params['size']/(Planck15.H(0).value/100) *
                                 Planck15.arcsec_per_kpc_comoving(redshift).value,
                                 'arcsec')



    # storing HDF5 filesnames to remove at the end
    filenames = []

    # this is the loop where we actually get the images
    part_field_prefix = img_params['partField']
    # one ImageHDU per viewing angle
    for axes in ['0,1', '0,2', '1,2']:
        
        img = []
        
        for band in bands:
            
            # start hdu list
            phdu = fits.PrimaryHDU(header=header)
            hdul = fits.HDUList([phdu])
            print('Constructed header')
            print('Time: ' + str(datetime.timedelta(seconds=(time.time() - start_time))))
            
            # img = []
    
            fname = '%s%d-%d-%s-%s.hdf5' %(tmp_dir, snap, subhalo_id, band, axes)
            filenames.append(fname)

            # load the hdf5 file if it exists
            try:
                with h5py.File(fname, 'r') as tmpfile:
                    img.append(np.array(tmpfile['grid']))
                print('Opening %s' %fname)
                sys.stdout.flush()

            # if the file does not exist or is corrupted, download it
            except (OSError, KeyError) as e:
                print('Downloading %s from tng-project.org' %fname)
                sys.stdout.flush()

                # build filename of download with image params and appropriate band
                dname = 'vis.hdf5?'
                img_params['partField'] = part_field_prefix + band
                for pname in img_params.keys():
                    dname = dname + '%s=%s&' %(pname, img_params[pname])
                # dname = dname + 'rotation=%s' %rotation
                dname = dname + 'axes=%s' %axes

                # do the download & open the result
                print('wget --retry-connrefused -O %s --header="API-Key: %s" "%s%s"' 
                      %(fname, api_key, url_prefix, dname))
                osanswer = os.system('wget --retry-connrefused -O %s --header="API-Key: %s" "%s%s"' 
                                     %(fname, api_key, url_prefix, dname))

                try:
                    with h5py.File(fname, 'r') as tmpfile:
                        img.append(np.array(tmpfile['grid']))
                except (OSError, KeyError) as e:
                    pass
                    # I write errors to a logfile
                    # with open('../logs/bad_gal.csv', 'a+') as logfile:
                    #   logfile.write('%d,%d\n' %(subhalo_id, snap))

            hdul.append(fits.ImageHDU(data=np.array(img),
                                  header=fits.Header([('axes', axes)])))
            
            #hdul.writeto(fitsname.format(subhalo_id, snap, axes, band), overwrite = True)

    # finish setting up FITS file and write
    hdul.writeto(fitsname, overwrite=True)
    
    ######
    #ALTERNATIVE FILEPATH
    #sfid_{}_snap_{}_axes_{}_band_{}
    #eg: sfid_19582_snap_89_axes_01_band_g.fits
    #search function to find all images of an SFID and snap:
    #imgList_r = list(glob('./IMG_dir/FITS_dir/sfid_{}_snap_{}*.fits'.format(SFID,snap)))
    #imgList_r = sorted(imgList_r)
    #_imgList = [img.replace('axes_01','axes_{}') for img in imgList_r]
    #_imgList = [img.replace(

    for fname in filenames:
        os.system('rm ' + fname)

    print('Finished')
    print('Time: ' + str(datetime.timedelta(seconds=(time.time() - start_time))))