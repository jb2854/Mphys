import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
import sys
from copy import deepcopy
from illustris_python import illustris_python as il #this is in the same directory
basePath = '/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG/TNG100-1/output'
import time

def maxPastMassSnapNum(tree, index, partType='stars'):
    """ Get maximum past mass (of the given partType) along the main branch of a subhalo
        specified by index within this tree. No time/snap cutoff applied """
    ptNum = il.sublink.partTypeNum(partType)


    branchSize = tree['MainLeafProgenitorID'][index] - tree['SubhaloID'][index] + 1
    masses = tree['SubhaloMassType'][index: index + branchSize, ptNum]
    MaxIndex = index + np.argmax(masses) 
    return np.max(masses), tree['SnapNum'][MaxIndex], MaxIndex

def mergerInfoDF(basePath,snapNum=99,ratio=1.0/5.0,start=0,count=10,massPartTypeList=['stars'],columns=['DesSFID', 'DesSnap', 'DesMstellar', 'DesMDM', 'DesMBH','DesMgas', 'DesBHacc','DesSFR','ProgSFIDlate','ProgSnaplate','ProgSFID','ProgSnap', 'ProgMstellar','M_ratio','ProgMDM','ProgMBH','ProgMgas','ProgBHacc','ProgSFR'], good_idx_path='DF_dir/good_idx_df.parquet', output='master_out.parquet'):
                                                                                                        #output=[desSFID,desSnap,desMstellar,desMDM,desMBH,desMgas,desBHacc,desSFR,[fpSFIDlate,npSFIDlate],[fpSnap,npSnap],[fpSFID,npSFID],npMPMSnap,[fpSLMass,npMPM],ratio,[fpMDM,npMDM],[fpMBH,npMBH],[fpMgas,npMgas],[fpBHacc,npBHacc],[fpSFR,npSFR]]
                       #Above comment is just to compare the rows which are being created with the columns that they need to match
    '''
    Function
    Calls mergerInfo function to pull data on all mergers in a tree 
    IMPORTANT: make sure arguments massPartTypeList and columns are matching ie:
    *** asterisks highlight changes, remove if using this code
    massPartTypeList=['stars'],columns=['DesID', 'DesSFID', 'ProgIDs', 'ProgSFIDs', 'ProgM*s','SnapNum'] is acceptable (will return a column for stellar mass)
    massPartTypeList=['stars',***'dm'***],columns=['DesID', 'DesSFID', 'ProgIDs', 'ProgSFIDs', 'ProgM*s', ***'ProgM_DM'***, 'SnapNum'] is acceptable (will return a column for stellar mass and dark matter)
    massPartTypeList=['stars',***'dm'***],columns=['DesID', 'DesSFID', 'ProgIDs', 'ProgSFIDs', 'ProgM*s', 'SnapNum'] is NOT acceptable (will crash as it tries to return data for stellar mass and dark matter without )
    Returns:
    Saves a parquet of the dataframe with name specified by user
    returns a master copy of the data frame as an object alongside a copy of the data frame intended for editing
    
    '''
    
    ####good_idx_df = pd.read_parquet(good_idx_path)
    
    # the following fields are required for the walk and the mass ratio analysis
    groupFirstSub = il.groupcat.loadSubhalos(basePath,snapNum,fields=['SubhaloFlag'])
    
    if start+count > len(groupFirstSub):
        raise Exception('Error: There aren\'t enough trees to process this request')
    
    #if output is ID, SFID, ID, SFID, (Masses), SnapNum then there should be 5 non mass fields, anything else suggests an error will occur during output
#     if len(columns)-len(massPartTypeList) != 6:
#         if len(columns)-len(massPartTypeList) > 6:
#             raise Exception('Error: Too many columns for not enough mass types')
#         if len(columns)-len(massPartTypeList) < 6:
#             raise Exception('Error: Not enough columns for too many mass types')
        
    fields = ['SubfindID','SubhaloID','NextProgenitorID','MainLeafProgenitorID','FirstProgenitorID','SubhaloMassType','SnapNum','DescendantID','SubhaloBHMdot','SubhaloSFR']
    #basePath = '/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG/TNG100-1/output'
    mergers_list = []
    ratio = 1.0/5.0#mass ratio of interest
    for i in range(start,start+count):
        ####if good_idx_df.loc[i,str(99)]:
        tree = il.sublink.loadTree(basePath,99,i,fields=fields)#call the tree of this subhalo
            #if tree doesn't exist, don't run the merger grabbing function, there are no mergers
        if tree is not None:
            mergerInfoListAppend(tree, mergers_list, 'good_idx_df', minMassRatio=ratio)#pull merger info, storing it in mergers
                #### Removerd good_idx_df from args
                
    #mergersmaster = deepcopy(mergers_list)
    masterdf = pd.DataFrame(mergers_list, columns=columns)
    masterdf.to_parquet(output)
    return masterdf
    
    #mergersdf = pd.DataFrame(mergers, columns=columns)


def mergerInfoListAppend(tree, mergers_list, good_idx_df, minMassRatio=1e-10, minDesMstellar=1e-2, index=0):
    """ Looks through one merger tree and identifies all mergers, 
    Based off numMergers function to identify the mergers,
    massPartType is a list of the mass types that are to be output
    Mass is taken using sublink method
    No return, takes a list and appends mergers to the end of it one by one
    """
    # verify the input sub-tree has the required fields
    reqFields = ['SubfindID','SubhaloID', 'NextProgenitorID',
                 'FirstProgenitorID', 'SubhaloMassType','SnapNum','DescendantID']

    if not set(reqFields).issubset(tree.keys()):
        raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqFields))
    
    #inv mass ratio is used instead of checking minMassRatio in both directions
    invMassRatio = 1.0 / minMassRatio 

    # walk back main progenitor branch
    rootID = tree['SubhaloID'][index]
    fpID   = tree['FirstProgenitorID'][index]
    
    #while there is a first progenitor
    while fpID != -1:
        
        fpIndex = index + (fpID - rootID)
        
        # get descendant properties
        desID = tree['DescendantID'][fpIndex]
        desIndex = index + (desID - rootID)
        desSFID = tree['SubfindID'][desIndex]
        desSnap = tree['SnapNum'][desIndex]
        desMstellar = tree['SubhaloMassType'][desIndex,il.util.partTypeNum('stars')]
        
        ### desFlag = good_idx_df.loc[desSFID,str(desSnap)]
        desFlag = 1
        
        
        # ignore this merger if not above minDesMstellar and desFlag not good
        if desMstellar > minDesMstellar and desFlag == 1:
            # explore breadth
            npID = tree['NextProgenitorID'][fpIndex]

            #while there is a next progenitor, ie a merger of any mass ratio has occurred
            while npID != -1:
                npIndex = index + (npID - rootID)
                # mass and snap of next progenitor max past mass
                npMPM, npMPMSnap, npMPMindex  = maxPastMassSnapNum(tree, npIndex, 'stars')
                fpMassIndex = fpIndex
                while (npMPMSnap != tree['SnapNum'][fpMassIndex]) and (npMPMSnap > tree['SnapNum'][fpMassIndex]):
                    fpMassIndex += 1
                fpSLMass = tree['SubhaloMassType'][fpMassIndex,il.util.partTypeNum('stars')]
                
        

                # include if both masses are non-zero
                if fpSLMass > 0.0 and npMPM > 0.0:
                    ratio = npMPM / fpSLMass
                    if ratio > 1:
                        ratio = 1/ratio
                    # if ratio exceeds threshold
                    if ratio >= minMassRatio and ratio <= invMassRatio:
                        #all output values not already found are found below
                        desMDM = tree['SubhaloMassType'][desIndex,il.util.partTypeNum('dm')]
                        desMBH = tree['SubhaloMassType'][desIndex,il.util.partTypeNum('bh')]
                        desMgas = tree['SubhaloMassType'][desIndex,il.util.partTypeNum('gas')]
                        desBHacc = tree['SubhaloBHMdot'][desIndex]
                        desSFR = tree['SubhaloSFR'][desIndex]
                        
                        #progenitor properties are found at same snap as mass
                        fpSnap = tree['SnapNum'][fpIndex]
                        fpSFIDlate = tree['SubfindID'][fpIndex]#SubFind ID at the last snapshot before merger
                        #3 mass types @ np MPM snap
                        fpMDM = tree['SubhaloMassType'][fpMassIndex,il.util.partTypeNum('dm')]
                        fpMBH = tree['SubhaloMassType'][fpMassIndex,il.util.partTypeNum('bh')]
                        fpMgas = tree['SubhaloMassType'][fpMassIndex,il.util.partTypeNum('gas')]
                        fpSFID = tree['SubfindID'][fpMassIndex]#SubFind ID at snap of MPM of np
                        fpBHacc = tree['SubhaloBHMdot'][fpMassIndex]
                        fpSFR = tree['SubhaloSFR'][fpMassIndex]
                        
                        npSnap = tree['SnapNum'][npIndex]
                        npSFIDlate = tree['SubfindID'][npIndex]#SubFind ID at the last snapshot before merger
                        #3 mass types @ np MPM snap
                        npMDM = tree['SubhaloMassType'][npMPMindex,il.util.partTypeNum('dm')]
                        npMBH = tree['SubhaloMassType'][npMPMindex,il.util.partTypeNum('bh')]
                        npMgas = tree['SubhaloMassType'][npMPMindex,il.util.partTypeNum('gas')]
                        npSFID = tree['SubfindID'][npMPMindex]#SubFind ID at snap of MPM of np
                        npBHacc = tree['SubhaloBHMdot'][npMPMindex]
                        npSFR = tree['SubhaloSFR'][npMPMindex]
                        #prepare list for output
                        #include both progenitor snaps to spot difference
                        output=[desSFID,desSnap,desMstellar,desMDM,desMBH,desMgas,desBHacc,desSFR,[fpSFIDlate,npSFIDlate],[fpSnap,npSnap],[fpSFID,npSFID],npMPMSnap,[fpSLMass,npMPM],ratio,[fpMDM,npMDM],[fpMBH,npMBH],[fpMgas,npMgas],[fpBHacc,npBHacc],[fpSFR,npSFR]]
                        #append output to merger_list
                        mergers_list.append(output)

                        
                npID = tree['NextProgenitorID'][npIndex]

        fpID = tree['FirstProgenitorID'][fpIndex]

def collateMultiProgMergersv2(df,collate_columns=['ProgIDs','ProgSFIDs','ProgMs']):
    '''
    Takes a df and combines merger events with the same descendant, showing only one instance of the repeat progenitors. 
    This allows for easier statistical analysis of merger events but harder mass ratio calculations
    
    Inputs:
    df - the dataframe of merger events from mergerInfoDF
    collate_columns - the columns where data is being combinedb THIS MUST MATCH THE COLUMNS YOU WISH TO COMBINE
    Returns:
    The collated df
    '''
    reqFields = collate_columns

#     if not set(reqFields).issubset(df.columns.values.tolist()):
#         raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqFields))

#     mergersdf[mergersdf.duplicated(subset='DesID',keep=False)]
    
    caught = []
    for index in df[df.duplicated(subset=['DesSFID','DesSnap'],keep=False)].index:
        for index2 in df[df.duplicated(subset=['DesSFID','DesSnap'],keep='first')].index:
            if index2 != index and index2 > index and index2 not in caught:
                if df.loc[index,'DesSFID'] == df.loc[index2,'DesSFID'] and df.loc[index,'DesSnap'] == df.loc[index2,'DesSnap'] :
                    for column in collate_columns:
                        df.at[index,column] = np.append(df.at[index,column],(df.at[index2,column][1]))
                    caught.append(index2)
                    
    df.drop(df.index[caught], inplace=True)
    return df

'''
Usage: 

mergers = mergerInfoDF(basePath,99,count=200000,output='temp2df.parquet')
testdf = pd.read_parquet('temp2df.parquet')
'''