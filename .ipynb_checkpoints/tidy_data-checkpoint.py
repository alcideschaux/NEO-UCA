import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('Data/NEO-UCA-DATA.csv')

"""
Data wrangling
"""
# Recode sex
df['sex'] = df['sex'].replace({'M':'Male','m':'Male','F':'Female'})

# Recode dichotomous variables
df['variant_histology'] = pd.Categorical(df['variant_histology'], categories=['No','Yes'], ordered=True)
df['recurrence'] = pd.Categorical(df['recurrence'], categories=['No','Yes'], ordered=True)

df['dod'] = df['dod'].replace({0: 'No', 1: 'Yes'})
df['dod'] = pd.Categorical(df['dod'], categories=['No','Yes'], ordered=True)

df['doc'] = df['doc'].replace({0:'No', 1:'Yes'})
df['doc'] = pd.Categorical(df['doc'], categories=['No','Yes'], ordered=True)

# Recode ypT_stage column
df['ypT_stage'] = df['ypT_stage'].replace({'0':'pT0','is':'pTis','a':'pTa','2':'pT2'}) # Recode to pT
df['ypT_stage'] = df['ypT_stage'].replace(['2a','2b'], 'pT2') # Merge 2a and 2b to pT2
df['ypT_stage'] = df['ypT_stage'].replace(['3a','3b'], 'pT3') # Merge 3a and 3b to pT3
df['ypT_stage'] = pd.Categorical(df['ypT_stage'], categories=['pT0','pTis','pTa','pT2','pT3'], ordered=True) # Order the categories

# Recode ypT_stage by groups
# Group 1: pT0-pTa-pTis vs pT2-pT3
df['ypT_group1'] = df['ypT_stage'].replace(['pT0','pTa','pTis'], 'pT0-pTa-pTis')
df['ypT_group1'] = df['ypT_group1'].replace(['pT2','pT3'], 'pT2-pT3')
df['ypT_group1'] = pd.Categorical(df['ypT_group1'], categories=['pT0-pTa-pTis','pT2-pT3'], ordered=True)
# Group 2: pT0-pTa-pTis vs pT2 vs pT3
df['ypT_group2'] = df['ypT_stage']
df['ypT_group2'] = df['ypT_stage'].replace(['pT0','pTa','pTis'], 'pT0-pTa-pTis')
df['ypT_group2'] = pd.Categorical(df['ypT_group2'], categories=['pT0-pTa-pTis','pT2','pT3'], ordered=True)

# Recode ypN_stage column
df['ypN_stage'] = df['ypN_stage'].replace({0:'pN0',1:'pN1',2:'pN2',3:'pN3'}) # Recode to pN
df['ypN_stage'] = pd.Categorical(df['ypN_stage'], categories=['pN0','pN1','pN2','pN3'], ordered=True) # Order the categories

# Recode ypN_stage by groups
# Group 1: pN0 vs pN1-pN2-pN3
df['ypN_group1'] = df['ypN_stage']
df['ypN_group1'] = df['ypN_group1'].replace(['pN1','pN2','pN3'], 'pN1-pN2-pN3')
df['ypN_group1'] = pd.Categorical(df['ypN_group1'], categories=['pN0','pN1-pN2-pN3'], ordered=True)
# Group 2: pN0 vs pN1 vs pN2-pN3
df['ypN_group2'] = df['ypN_stage']
df['ypN_group2'] = df['ypN_group2'].replace(['pN2','pN3'], 'pN2-pN3')
df['ypN_group2'] = pd.Categorical(df['ypN_group2'], categories=['pN0','pN1','pN2-pN3'], ordered=True)

# Create dre
# Disease-related event:
# 'No', patients who died from other causes (unrelated to cancer) and didn't have tumor recurrence
# 'Yes', otherwise (patients with tumor recurrence, either who were alive or not, and patients who died from cancer, either with or without tumor recurrence)
df['dre'] = np.where((df['doc'] == 'No') & (df['recurrence'] == 'No'), 'No', 'Yes')
df['dre'] = pd.Categorical(df['dre'], categories=['No','Yes'], ordered=True)