# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 09:22:08 2018

@author: 狮子尾啾啾
"""

import numpy as np
import pandas as pd
# load dataset
dataframe = pd.read_csv("D:\working\\part1\Fe1.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0].astype(float)
X=X.reshape(1, 250, 283,1)


autoencoder.fit(X,X,
                epochs=600,
                batch_size=128
                )
decoded_result= autoencoder.predict(X);

# decoded_result=decoded_result.tolist()
decoded_result=decoded_result.reshape(70750,1)

import csv
f=open('D:\working\\p1\Fe1.csv','w')
for i in decoded_result:
    k=' '.join([str(j) for j in i])
    f.write(k+"\n")
f.close()