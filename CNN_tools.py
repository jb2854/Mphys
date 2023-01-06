import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy.stats.distributions as dist
from matplotlib.colors import LogNorm
import sys
sys.path.append('/mnt/jb2854-XDrive/Physics/StudentProjects/MPhys/2022-23/IllustrisTNG-Mergers/')
from CNN_execute_script_v2 import *
from tensorflow import keras
from sklearn.metrics import auc
import os

def save_avg_results(grab_list_ids, min_acc):
    
    if (type(grab_list_ids) == int):
        grab_list_ids = [grab_list_ids]
    
    CNN_csv = 'CNN_info_V4.parquet'
    CNN_dir = '/home/AstroPhysics-Shared/PROJECTS/MPhys_Villforth/DATA/CNN_dir/'
    cnn_run_table = pd.read_parquet(CNN_dir + CNN_csv)
    
    for idx, grab_list_id in enumerate(grab_list_ids):

        filtered = cnn_run_table.loc[(cnn_run_table.boost_massive_gals == 'True') & (cnn_run_table.ratio_merger == 0.1) & 
                         (cnn_run_table.dt_merger == 0.2) & (cnn_run_table.ratio_nonmerger == 0.1) & (cnn_run_table.dt_nonmerger == 1)
                                     & (cnn_run_table.grab_list_name == 'grablist_run_%05d.parquet' %(grab_list_id))& (cnn_run_table.PIX_SCALE == 50)]

        # Select the best runs
        
        if (type(min_acc) == list):
            min_acc_value = min_acc[i]
        else:
            min_acc_value = min_acc
        
        run_ids = filtered[filtered.accuracy > min_acc_value].RUN_ID.values
        results = pd.read_csv('/home/AstroPhysics-Shared/PROJECTS/MPhys_Villforth/DATA/CNN_dir/CNN_run_%05d/CNN_run_%05d_subhalo_labels.csv' %(grab_list_id, grab_list_id))
        print('Valid cnn runs: ', run_ids)
        # Load the test image set from that grab list

        print('Loading test images...')
        imgs = []
        img_dir = '/home/AstroPhysics-Shared/PROJECTS/MPhys_Villforth/DATA/IMG_dir/REAL_FITS_11_04v2_dir/'
        img_fmt = 'SFID_%07d_SNAP_%02d.fits'

        img_rs = []
        for i, row in results.iterrows():
            SFID = int(row.SFID)
            SNAP = int(row.SNAP)
            AXES = int(row.AXES)

            this_img = []
            with fits.open(img_dir + img_fmt%(SFID,SNAP)) as hdul:
                hdr = hdul[0].header
                for band_idx in [1,2,3]:
                    band = hdul[band_idx]
                    this_img.append(rescale_img_and_size(band.data[AXES], hdr, band_idx, 0, new_size=50))
            img_rs.append(this_img)
        img_list = np.moveaxis(np.array(img_rs),1,3)[:, :, :, :, 0]


        # Dictionaries to store models and their predictions
        models = {}
        p_tst = {}

        # Load models and get predictions
        print('Loading models...')
        for run_id in run_ids:
            print(list(glob(CNN_dir+'CNN_run_%05d/'%(run_id)+'*.h5')))
            model_name = list(glob(CNN_dir+'CNN_run_%05d/'%(run_id)+'*.h5'))[0]
            model = keras.models.load_model(model_name)
            p_tst[run_id] = model.predict(img_list)[:,1]

        # Create 'predicted label' columns in df for each run and input values
        for run_id in run_ids:
            results['PRED_LABEL_' + str(run_id)] = ''
            for i, row in results.iterrows():
                results['PRED_LABEL_' + str(run_id)][i] = p_tst[run_id][i]

        # Find the accuracy of each run
        accuracy = {}
        for run_id in run_ids:
            TP = results.loc[(results['PRED_LABEL_' + str(run_id)] > 0.5) & (results['LABEL'] == 1)]
            TN = results.loc[(results['PRED_LABEL_' + str(run_id)] < 0.5) & (results['LABEL'] == 0)]
            accuracy[run_id] = (len(TP) + len(TN))/len(results)
        print('Individual run accuracies: ')
        print(accuracy)

        # Find the median label for each subhalo
        results['PRED_MED'] = ''
        results['STD_DEV'] = ''
        for i, row in results.iterrows():
            preds = []
            for run_id in run_ids:
                preds.append(row['PRED_LABEL_' + str(run_id)])
            med = np.median(preds)
            results['PRED_MED'][i] = med
            results['STD_DEV'][i] = np.std(preds)

        # Find the optimal threshold and AUC score using the median labels
        pred = results['PRED_MED'].values
        lab = results['LABEL'].values
        thresholds = np.linspace(0,1,num=1000)
        tpr = np.array([np.sum(np.logical_and(pred > th, lab == 1)) for th in thresholds])/np.sum(lab)
        fpr = np.array([np.sum(np.logical_and(pred > th, lab == 0)) for th in thresholds])/(len(lab) - np.sum(lab))

        dist = np.sqrt(fpr**2 + (tpr-1)**2)
        th_opt = thresholds[np.argmin(dist)]
        print('Optimal threshold: ', th_opt)

        auc_score = auc(fpr, tpr)
        print('AUC score', auc_score)

        # Find accuracy using optimal threshold
        TP = results.loc[(results['PRED_MED'] > th_opt) & (results['LABEL'] == 1)]
        TN = results.loc[(results['PRED_MED'] < th_opt) & (results['LABEL'] == 0)]
        acc = (len(TP) + len(TN))/len(results)
        print('Accuracy: ', acc)

        results.to_parquet('/home/AstroPhysics-Shared/PROJECTS/MPhys_Villforth/DATA/CNN_dir/CNN_averages/%04d_average_results.parquet'%(grab_list_id))
        
def load_avg_results(grab_list_ids):
    if type(grab_list_ids) == int:
        return pd.read_parquet('/home/AstroPhysics-Shared/PROJECTS/MPhys_Villforth/DATA/CNN_dir/CNN_averages/%04d_average_results.parquet'%(grab_list_ids))
    else:
        results_list = []
        for grab_list_id in grab_list_ids:
            results = pd.read_parquet('/home/AstroPhysics-Shared/PROJECTS/MPhys_Villforth/DATA/CNN_dir/CNN_averages/%04d_average_results.parquet'%(grab_list_id))
            results_list.append(results)
        return results_list