#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from copy import deepcopy
from illustris_python import illustris_python as il #this is in the same directory
basePath = '/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG/TNG100-1/output'

def mergerInfoListAppend(tree, mergers, minMassRatio=1e-10, massPartType=['stars'], index=0):
    """ Looks through one merger tree and identifies all mergers, 
    Based off numMergers function to identify the mergers,
    massPartType is a list of the mass types that are to be output, the first entry in the list defines the mass ratio calculation
    Mass is taken to be maxPastMass (which is subject to change), as a result required fields are only the ID
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
        fpMass  = il.sublink.maxPastMass(tree, fpIndex, massPartType[0])
        fpSnap = tree['SnapNum'][fpIndex]

        # explore breadth
        npID = tree['NextProgenitorID'][fpIndex]
        
        #while there is a next progenitor, ie a merger of any mass ratio has occurred
        while npID != -1: 
            npIndex = index + (npID - rootID)
            npMass  = il.sublink.maxPastMass(tree, npIndex, massPartType[0])

            # include if both masses are non-zero, and ratio exceeds threshold
            if fpMass > 0.0 and npMass > 0.0:
                ratio = npMass / fpMass

                if ratio >= minMassRatio and ratio <= invMassRatio:
                    #all output values not already found are done here
                    desID = tree['DescendantID'][fpIndex]
                    desIndex = index + (desID - rootID)
                    desSFID = tree['SubfindID'][desIndex]
                    fpSFID = tree['SubfindID'][fpIndex]
                    npSFID = tree['SubfindID'][npIndex]
                    output = []
                    output=[desID,desSFID,[fpID,npID],[fpSFID,npSFID],[fpMass,npMass]]
#                     output.append()
#                     output.append()
#                     output.append() #first masstype of interest
#                     output.append()
                    #if we are interested in more than one mass type ie stellar and black hole
                    if len(massPartType) > 1:
                        for mass_type in massPartType[1:]:
                            #new mass variables, no affect on mass ratio calculations in later loops
                            fpOutMass = il.sublink.maxPastMass(tree, fpIndex, mass_type)
                            npOutMass = il.sublink.maxPastMass(tree, npIndex, mass_type)
                            output.append([fpOutMass,npOutMass])
                    output.append(fpSnap)
                    mergers.append(output)
                        
                    #mergerSet.append([desID,fpMass,npMass,fpSnap])
                    #mergers.append([desID,desSFID,[fpID,npID],[fpSFID,npSFID],[fpMass,npMass],fpSnap]) #data appended to the series in rows already orgnaised into df columns

            npID = tree['NextProgenitorID'][npIndex]

        fpID = tree['FirstProgenitorID'][fpIndex]
        
        #### Dont return anything?
        
def collateMultiProgMergers(df,collate_columns=['ProgIDs','ProgSFIDs','ProgM*s']):
    '''
    Takes a df and combines merger events with the same descendant, showing only one instance of the repeat progenitors. This allows for easier statistical analysis of merger events but harder mass ratio calculations
    Inputs:
    df - the dataframe of merger events from mergerInfoDF
    collate_columns - the columns where data is being combinedb THIS MUST BE CHANGED IF DIFFERENT MASSES HAVE BEEN INCLUDED, WILL THROW AN ERROR IF NOT
    Returns:
    The collated df as a csv and dataframe return 
    '''
    reqFields = collate_columns

    if not set(reqFields).issubset(df.columns.values.tolist()):
        raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqFields))
    
    caught = []
    for index in df.index:
        for index2 in df.index:
            if index2 != index and index2 > index and index2 not in caught:
                if df.loc[index,'DesID'] == df.loc[index2,'DesID']:
                    for column in collate_columns:
                        df.at[index,column].append(df.at[index2,column][1])
                    caught.append(index2)
                    
    df.drop(df.index[caught], inplace=True)
    return df
        
def mergerInfoDF(basePath,snapNum = 99,ratio=1.0/5.0,start=0,count=10,massPartTypeList=['stars'],columns=['DesID', 'DesSFID', 'ProgIDs', 'ProgSFIDs', 'ProgM*s','SnapNum'],output='master_out.csv'):
    '''
    Function
    Calls mergerInfo function to pull data on all mergers in a tree, as many times as 
    IMPORTANT: make sure arguments massPartTypeList and columns are matching ie:
    *** asterisks highlight changes, remove if using this code
    massPartTypeList=['stars'],columns=['DesID', 'DesSFID', 'ProgIDs', 'ProgSFIDs', 'ProgM*s','SnapNum'] is acceptable (will return a column for stellar mass)
    massPartTypeList=['stars',***'dm'***],columns=['DesID', 'DesSFID', 'ProgIDs', 'ProgSFIDs', 'ProgM*s', ***'ProgM_DM'***, 'SnapNum'] is acceptable (will return a column for stellar mass and dark matter)
    massPartTypeList=['stars',***'dm'***],columns=['DesID', 'DesSFID', 'ProgIDs', 'ProgSFIDs', 'ProgM*s', 'SnapNum'] is NOT acceptable (will crash as it tries to return data for stellar mass and dark matter without )
    Returns:
    Saves a csv of the dataframe with name specified by user
    returns a master copy of the data frame as an object alongside a copy of the data frame intended for editing
    
    '''
    # the following fields are required for the walk and the mass ratio analysis
    groupFirstSub = il.groupcat.loadHalos(basePath,snapNum,fields=['GroupFirstSub'])
    #### Should we be pulling satellites as well or just centrals?
    
    if start+count > len(groupFirstSub):
        raise Exception('Error: There aren\'t enough trees to process this request')
    
    #if output is ID, SFID, ID, SFID, (Masses), SnapNum then there should be 5 non mass fields, anything else suggests an error will occur during output
    if len(columns)-len(massPartTypeList) != 5:
        if len(columns)-len(massPartTypeList) > 5:
            raise Exception('Error: Too many columns for not enough mass types')
        if len(columns)-len(massPartTypeList) < 5:
            raise Exception('Error: Not enough columns for too many mass types')
        
    fields = ['SubfindID','SubhaloID','NextProgenitorID','MainLeafProgenitorID','FirstProgenitorID','SubhaloMassType','SnapNum','DescendantID']
    basePath = '/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG/TNG100-1/output'
    mergers = []
    ratio = 1.0/5.0#mass ratio of interest
    for i in range(start,start+count):
        tree = il.sublink.loadTree(basePath,99,groupFirstSub[i],fields=fields)#call the tree of this subhalo
        mergerInfoListAppend(tree, mergers, minMassRatio=ratio)#pull merger info, storing it in mergers

    mergersmaster = deepcopy(mergers)
    masterdf = pd.DataFrame(mergersmaster, columns=columns)
    masterdf.to_csv(output)
    mergersdf = pd.DataFrame(mergers, columns=columns)
    collatedf = collateMultiProgMergers(mergersdf)

    
    return masterdf, collatedf

'''
TYPE THIS

masterdf, collated = mergerInfoDF('/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG/TNG100-1/output',99,start=0,count=100)'''