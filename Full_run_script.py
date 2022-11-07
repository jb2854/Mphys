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
from ObsRealism_CANDELS_v2 import *
import hdf52_fits_MPhys_modified_26_10 as grabber
from joblib import Parallel, delayed


REAL_dir = '/home/AstroPhysics-Shared/PROJECTS/MPhys_Villforth/DATA/IMG_dir/REAL_FITS_11_04v2_dir'
FITS_dir = '/home/AstroPhysics-Shared/PROJECTS/MPhys_Villforth/DATA/IMG_dir/FITS_11_04v2_dir'

