import numpy as np
import h5py
import pandas as pd
import time
from illustris_python import illustris_python as il

path =  '/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG/TNG100-1/postprocessing/stellar_circs/'
file = 'stellar_circs.hdf5'
basePath = '/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG/TNG100-1/output'

snap_data = {}

with h5py.File(path + file, "r") as f:
    # Print all root level object names (aka keys) 
    # these can be group or dataset names 
    print("Keys: %s" % f.keys())
    print(f['Snapshot_50'].keys())
    
    for snap in range(28,51):
        snap_key = 'Snapshot_%02d'%(snap)
        data = list(f[snap_key])
        shape = np.shape(f[snap_key]['CircAbove07Frac'])
        keys = ['CircAbove07Frac', 'CircAbove07Frac_allstars', 'CircAbove07MinusBelowNeg07Frac', 'CircAbove07MinusBelowNeg07Frac_allstars', 'CircTwiceBelow0Frac', 'CircTwiceBelow0Frac_allstars', 'SubfindID']
        data = {}
        for key in keys:
            data[key] = np.zeros(shape)
            data[key][:] = f[snap_key][key][:]
        snap_data[snap] = data
        
df_dir = '/home/AstroPhysics-Shared/PROJECTS/MPhys_Villforth/DATA/DF_dir/'
objects_df = pd.read_parquet(df_dir + 'objects_master_neighbours_22_12_13.parquet')

start = time.time()
for snap in range(28,51):
    for i in range(len(snap_data[snap]['SubfindID'])):
        print('Snap',snap,' i',i,'/',len(snap_data[snap]['SubfindID']))
        index = objects_df.loc[(objects_df.SnapNum == snap) & (objects_df.SFID == snap_data[snap]['SubfindID'][i])].index
        for key in keys[:-1]:

            objects_df.at[index,key] = snap_data[snap][key][i]

objects_df.to_parquet(df_dir + 'objects_master_neighbours_morph_22_12_13.parquet')