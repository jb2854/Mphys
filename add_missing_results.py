import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy.stats.distributions as dist
from matplotlib.colors import LogNorm
import os

def save_results(run_id):

    CNN_dir = '/home/AstroPhysics-Shared/PROJECTS/MPhys_Villforth/DATA/CNN_dir/CNN_run_%05d/' %(run_id)
    subhalos_csv = 'CNN_run_%05d_subhalo_labels.csv' %(run_id)
    CNN_csv = 'CNN_run_%05d_info.csv' %(run_id)
    results = pd.read_csv(CNN_dir + subhalos_csv)
    df_dir = '/home/AstroPhysics-Shared/PROJECTS/MPhys_Villforth/DATA/DF_dir/'
    objects_df = pd.read_parquet(df_dir + 'objects_master_22_10_28.parquet')
    
    out = []
    for idx, row in results.iterrows():
        
        halo = objects_df.loc[(objects_df['SFID'] == row.SFID) & (objects_df['SnapNum'] == row.SNAP)]
        out.append(halo.values.flatten().tolist())
        gas_frac = (halo['Mgas'].values/halo['Mstellar'].values).flatten()[0]
        
        if gas_frac == float('-inf'):
            gas_frac = 0
            
        out[-1].extend([gas_frac, row.PRED_LABEL, row.LABEL])

    cols = objects_df.columns.values.tolist()
    cols.extend(['GAS_FRAC', 'PRED_LABEL', 'LABEL'])
    results = pd.DataFrame(out,columns = cols)
    
    G = 6.6743e-11 #m^3 kg^-1 s^-2
    c = 3e8 #ms^-1
    mp = 1.673e-27 #kg
    sigma_e = 6.65e-29 #m^2

    L = results['BHacc'] * 0.1 * c**2 / (0.978e9 * 3.154e7) * 0.7 # (1e10 Msol/h) m^2 s^-3
    Ledd = 4*np.pi*G*c*mp/sigma_e * results['MBH'] # (1e10 Msol/h) m^2 s^-3
    results['Edd_ratio'] = L/Ledd
    
    results.to_parquet(CNN_dir + 'full_subhalo_results.parquet')
    
for run_id in range(337):
    print('run id: ', run_id)
    path = '/home/AstroPhysics-Shared/PROJECTS/MPhys_Villforth/DATA/CNN_dir/CNN_run_%05d' %(run_id)
    if os.path.exists(path):
        if not os.path.exists(path + '/full_subhalo_results.parquet'):
            print('Subhalos file does not exist')
            try:
                save_results(run_id)
            except:
                print('Problem with run id ', run_id)
