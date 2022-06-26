import urllib.request, os
import zipfile
import pandas as pd
import lazypredict
import matplotlib.pyplot as plt
import seaborn as sns


#Define functions for calculating the different features:
# Amino acid composition (AAC)
from Pfeature.pfeature import aac_wp

def aac(input):
    a = input.rstrip('txt')
    output = a + 'aac.csv'
    df_out = aac_wp(input, output)
    df_in = pd.read_csv(output)
    return df_in

aac('train_po_cdhit.txt')
aac_out = pd.read_csv('train_po_cdhit.aac.csv')

# Dipeptide composition (DPC)

from Pfeature.pfeature import dpc_wp

def dpc(input):
    a = input.rstrip('txt')
    output = a + 'dpc.csv'
    df_out = dpc_wp(input, output, 1)
    df_in = pd.read_csv(output)
    return df_in

dpc('train_po_cdhit.txt')
dpc_out = pd.read_csv('train_po_cdhit.dpc.csv')


#Calculate feature for both positive and negative classes + combines the two classes + merge with class labels
pos = 'train_po_cdhit.txt'
neg = 'train_ne_cdhit.txt'


def feature_calc(po, ne, feature_name, prfx):
    # Calculate feature
    po_feature = feature_name(po)
    ne_feature = feature_name(ne)
    # Create class labels
    po_class = pd.Series(['positive' for i in range(len(po_feature))])
    ne_class = pd.Series(['negative' for i in range(len(ne_feature))])
    # Combine po and ne
    po_ne_class = pd.concat([po_class, ne_class], axis=0)
    po_ne_class.name = 'class'
    po_ne_feature = pd.concat([po_feature, ne_feature], axis=0)
    # Combine feature and class
    df = pd.concat([po_ne_feature, po_ne_class], axis=1)
    df.to_csv(f"{prfx}_{po.replace('.txt','')}_{ne.replace('.txt','')}.csv", index=False)
    return df

feature_calc(pos, neg, aac, 'aac') # AAC
feature_calc(pos, neg, dpc, 'dpc') # DPC

