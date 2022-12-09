import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from glob import glob
import csv 

img_list = list(glob('/home/AstroPhysics-Shared/PROJECTS/MPhys_Villforth/DATA/IMG_dir/REAL_FITS_11_04v2_dir/*'))

with_inf = []

def check_inf(img):
    img = img.flatten()
    for i in range(np.size(img)):
        if np.isnan(img[i]):
            return 1
        if np.isinf(img[i]):
            return 1
    return 0

infs = 0
for j, img in enumerate(img_list):
    if j%1000 == 0:
        print(j)
    try:
        hdul = fits.open(img)
    except:
        print('bad image')
    for i in range(3):
        data = hdul[i+1].data[0].flatten()
        inf = check_inf(data)
    if inf:
        with_inf.append(img)
    infs += inf 

print('n_infs = %d, n_images = %d'%(infs, len(img_list)))

with open('inf_check.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    header = ['IMG']
    writer.writerow(header)
    for i in range(len(with_inf)):
        # write the data
        data = [with_inf[i]]
        writer.writerow(data)


    
