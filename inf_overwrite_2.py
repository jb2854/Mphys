import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from glob import glob
import csv
import os, sys
sys.path.append('/mnt/jb2854-XDrive/Physics/StudentProjects/MPhys/2022-23/IllustrisTNG-Mergers')
from ImageGrabScript11_02_Copy1 import *

sfids = []
snaps = []

with open('problem_subhalos_2.csv', 'r', encoding='UTF8') as f:
    reader = csv.reader(f)
    
    for row in reader:
        sfids.append(int(row[1]))
        snaps.append(int(row[2]))

outpath = '/home/AstroPhysics-Shared/PROJECTS/MPhys_Villforth/DATA/IMG_dir/FITS_11_04v2_dir/'
real_path = '/home/AstroPhysics-Shared/PROJECTS/MPhys_Villforth/DATA/IMG_dir/REAL_FITS_11_04v2_dir/'

parallel_grab(sfids, snaps, real_path, bands= ['wfc3_ir_f160w', 'wfc3_ir_f125w', 'wfc_acs_f814w'], n_jobs=8, start=0, end=-1)  
        