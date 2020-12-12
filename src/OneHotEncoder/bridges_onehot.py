#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# creating initial dataframe
bridge_types = ('Arch','Beam','Truss','Cantilever','Tied Arch','Suspension','Cable')
bridge_df = pd.DataFrame(bridge_types, columns=['Bridge_Types'])

# creating instance of labelencoder
labelencoder = LabelEncoder()

# Assigning numerical values and storing in another column
bridge_df['Bridge_Types_Cat'] = labelencoder.fit_transform(bridge_df['Bridge_Types'])

print("BRIDGE DF after LabelEncoder fit_transform")
print(bridge_df)

print("BRIDGE DF TYPES CAT after LabelEncoder fit_transform")
print(bridge_df['Bridge_Types_Cat'])

# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')

# passing bridge-types-cat column (label encoded values of bridge_types)
enc_df = pd.DataFrame(enc.fit_transform(bridge_df[['Bridge_Types_Cat']]).toarray())
print("ENC DF:")
print(enc_df)

# merge with main df bridge_df on key values
bridge_df = bridge_df.join(enc_df)

print("BRIDGE DF after joining with ENC DF")
print(bridge_df)