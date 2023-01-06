import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from illustris_python import illustris_python as il
import time
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt

df_dir = '/home/AstroPhysics-Shared/PROJECTS/MPhys_Villforth/DATA/DF_dir/'
objects_df = pd.read_parquet(df_dir + 'objects_master_22_10_28.parquet')
basePath = '/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG/TNG100-1/output'

objects_df['r1'] = np.NaN
objects_df['r1_id'] = np.NaN
objects_df['r2'] = np.NaN
objects_df['r2_id'] = np.NaN

# For one snapshot...

    
def loadSnapNeighbours(snap):
    
    print('Loading header')
    header = il.groupcat.loadHeader(basePath, snap)
    box_size = header['BoxSize']

    # Load snapshot data
    print('Loading snapshot data')
    snapshot_data = il.groupcat.loadSubhalos(basePath, snap, fields=['SubhaloPos', 'SubhaloFlag', 'SubhaloMassType', 'SubhaloMassInHalfRadType'])
    
    print('creating dfs')
    gals = pd.DataFrame({'Flag': snapshot_data['SubhaloFlag'], 'Mstellar' :snapshot_data['SubhaloMassType'][:, 4], 'HMRad' : snapshot_data['SubhaloMassInHalfRadType'][:, 4]})
    gals['pos_x'] = ''
    gals['pos_y'] = ''
    gals['pos_z'] = ''
    gals['pos_x'] = snapshot_data['SubhaloPos'][:,0]
    gals['pos_y'] = snapshot_data['SubhaloPos'][:,1]
    gals['pos_z'] = snapshot_data['SubhaloPos'][:,2]
    gals = gals.loc[(gals['Flag'] == True) & (gals['Mstellar'] > 0.001) & (gals['HMRad'] > 0)]
    
    positions = np.array([gals['pos_x'].values, gals['pos_y'].values, gals['pos_z'].values])
    positions = np.transpose(positions)

    
    # For each subhalo in the object df in that snapshot
    subhalos = objects_df.loc[(objects_df['SnapNum'] == snap) & (objects_df['Mstellar'] > 0.1)]['SFID'].values

    
    for i, subhalo in enumerate(subhalos):
        
        if (i%100 == 0):
            print('Snap ', snap, ' Subhalo ',i,'/',len(subhalos))
    
        # Find the nearest and second nearest neighbour and the subhalo id
        subhalo_data = il.groupcat.loadSingle(basePath, snap, subhaloID=subhalo)
        r = subhalo_data['SubhaloPos']
        mass = subhalo_data['Mstellar']
        
        new_positions = positions - (r - np.array([box_size/2,box_size/2,box_size/2]))
        r = r - (r - np.array([box_size/2,box_size/2,box_size/2]))
        new_positions[new_positions>box_size] -= box_size
        new_positions[new_positions<0] += box_size
        sep = new_positions - r
        dist = np.sqrt((sep*sep).sum(axis=1))
        
        gals['dist'] = dist
        #gals = gals.loc[gals['Mstellar'] > (mass * 0.1)]
        
        neighbours = gals.loc[(gals['Flag'] == True) & (gals['Mstellar'] > (mass * 0.1)) & (gals['dist'] > 0)].nsmallest(2,'dist')
        r1 = [np.min(neighbours['dist']), neighbours['dist'].idxmin()]
        r2 = [np.max(neighbours['dist']), neighbours['dist'].idxmax()]
        
        objects_df.loc[(objects_df['SnapNum'] == snap) & (objects_df['SFID'] == subhalo), 'r1'] = r1[0]
        objects_df.loc[(objects_df['SnapNum'] == snap) & (objects_df['SFID'] == subhalo), 'r1_id'] = r1[1]
        objects_df.loc[(objects_df['SnapNum'] == snap) & (objects_df['SFID'] == subhalo), 'r2'] = r2[0]
        objects_df.loc[(objects_df['SnapNum'] == snap) & (objects_df['SFID'] == subhalo), 'r2_id'] = r2[1]
        
    objects_df.loc[(objects_df['SnapNum'] == snap)].to_parquet(df_dir + 'objects_master_neighbours_snap_%02d_v2.parquet'%(snap))
        
        
#Parallel(n_jobs = 12)(delayed(loadSnapNeighbours)(snap) for snap in range(28,51))
#loadSnapNeighbours(28)

dfs = []
for snap in range(28,51):
    df = pd.read_parquet(df_dir + 'objects_master_neighbours_snap_%02d_v2.parquet'%(snap))
    print(df)
    dfs.append(df)

master = pd.concat(dfs)
    
master.to_parquet(df_dir + 'objects_master_neighbours_23_1_3.parquet')

