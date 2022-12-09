import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from glob import glob
import csv
import os, sys
sys.path.append('/mnt/jb2854-XDrive/Physics/StudentProjects/MPhys/2022-23/IllustrisTNG-Mergers')
from ImageGrabScript11_02_Copy1 import *

imgList = pd.read_csv('inf_check.csv')
print(imgList['IMG'].to_list())
imgList = imgList['IMG'].to_list()
sfids = []
snaps = []
for i, img in enumerate(imgList):
    print(i)

    if i%1000 ==0:
        print('adding image %i of %i\n'%(i, len(imgList)))
    path, file = os.path.split(img)
    SFIDandSNAP = file.replace('SFID_','').replace('_SNAP','').replace('.fits','')
    split = SFIDandSNAP.split('_') #split = [SFID,snap]
    sfids.append(int(split[0]))
    snaps.append(int(split[1]))

outpath = '/home/AstroPhysics-Shared/PROJECTS/MPhys_Villforth/DATA/IMG_dir/FITS_11_04v2_dir/'
real_path = '/home/AstroPhysics-Shared/PROJECTS/MPhys_Villforth/DATA/IMG_dir/REAL_FITS_11_04v2_dir/'

parallel_grab(sfids, snaps, real_path, bands= ['wfc3_ir_f160w', 'wfc3_ir_f125w', 'wfc_acs_f814w'], n_jobs=8, start=0, end=-1)  
        