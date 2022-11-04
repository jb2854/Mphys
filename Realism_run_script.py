import time
import os
import sys
import datetime
from glob import glob
import numpy as np
from joblib import Parallel, delayed
from astropy.table import Table
from astropy.io import fits
from astropy.cosmology import Planck15
sys.path.append('./Images/RealSim/')
from ObsRealism_CANDELS import *
from joblib import Parallel, delayed

start_time = time.time()

def no_realism_yet(real_path, SFID, snap, file_name_format='SFID_{}_SNAP_{}_SemiReal.fits'):
    '''
    checks if an object is already in the realism folder, returns false if it is, true if it isn't
    real_path: directory for realism images
    SFID: the SFID of the object
    snap: the snapnumber of the object
    '''
    value = os.path.exists(real_path+file_name_format.format(SFID,snap))
    if value:
        return False
    else:
        return True

def do_realism_on(raw_path, real_path):
    '''
    takes a raw fits file directory and compares with a real fits directory to identify which raw files should be put through realism script
    '''
    imgList = sorted(list(glob(raw_path + '*.fits')))
    call_list = []
    for raw in imgList:
        raw_end = raw.replace(raw_path,'').replace('SFID_','').replace('SNAP_','').replace('.fits','')
        split = raw_end.split('_')
#         SFIDs.append(split[0])
#         SFIDs.append(split[1])
        if no_realism_yet(real_path,split[0],split[1]):
            call_list.append(raw)
        else:
            print("Realised file already exists: ", raw)
    #return call_list
    return imgList

def parallel_realism(call_list, n_jobs=16):
    Parallel(n_jobs=n_jobs)(delayed(ObsRealism)(img, img.replace('FITS_11_02','REAL_FITS_11_03').replace('.fits', '_SemiReal.fits')) for img in call_list)
                            

                            
raw_path = '/home/AstroPhysics-Shared/PROJECTS/MPhys_Villforth/DATA/IMG_dir/FITS_11_02_dir/'
real_path = '/home/AstroPhysics-Shared/PROJECTS/MPhys_Villforth/DATA/IMG_dir/REAL_FITS_11_03_dir/'
call_list = do_realism_on(raw_path, real_path)
parallel_realism(call_list)

end_time = time.time()

print('Time taken: ', end_time - start_time)