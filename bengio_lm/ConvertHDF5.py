# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 15:50:45 2018

@author: akyng
"""
import numpy as np
import h5py

f = open('trn_dat.csv','r')
trn_data = f.readlines()
f.close()
trn_x = [None]*len(trn_data)
for n in range(len(trn_data)):
    trn_x[n] = trn_data[n].split('\n')[0].split(',')
trn_x = np.asarray(trn_x, dtype='float32')

f = open('trn_lab.csv','r')
trn_lab = f.readlines()
f.close()
trn_t = np.asarray(trn_lab, dtype='float32')

f = open('tst_dat.csv','r')
tst_data = f.readlines()
f.close()
tst_x = [None]*len(tst_data)
for n in range(len(tst_data)):
    tst_x[n] = tst_data[n].split('\n')[0].split(',')
tst_x = np.asarray(tst_x, dtype='float32')

f = open('tst_lab.csv','r')
tst_lab = f.readlines()
f.close()
tst_t = np.asarray(tst_lab, dtype='float32')

f = open('val_dat.csv','r')
val_data = f.readlines()
f.close()
val_x = [None]*len(val_data)
for n in range(len(val_data)):
    val_x[n] = val_data[n].split('\n')[0].split(',')
val_x = np.asarray(val_x, dtype='float32')

f = open('val_lab.csv','r')
val_lab = f.readlines()
f.close()
val_t = np.asarray(val_lab, dtype='float32')

hdf_trn_file = "train_bengio.hdf5"
hdf_list_trn_file = "train_bengio.txt"

with h5py.File(hdf_trn_file, "w") as f:
    #Create dataset
    f.create_dataset("data", data=trn_x)
    f.create_dataset("label", data=trn_t)
    f.close()

with open(hdf_list_trn_file, "w") as f:
    f.write(hdf_trn_file)
    f.close()

hdf_tst_file = "test_bengio.hdf5"
hdf_list_tst_file = "test_bengio.txt"

with h5py.File(hdf_tst_file, "w") as f:
    #Create dataset
    f.create_dataset("data", data=tst_x)
    f.create_dataset("label", data=tst_t)
    f.close()

with open(hdf_list_tst_file, "w") as f:
    f.write(hdf_tst_file)
    f.close()

hdf_val_file = "val_bengio.hdf5"
hdf_list_val_file = "val_bengio.txt"

with h5py.File(hdf_val_file, "w") as f:
    #Create dataset
    f.create_dataset("data", data=val_x)
    f.create_dataset("label", data=val_t)
    f.close()

with open(hdf_list_val_file, "w") as f:
    f.write(hdf_val_file)
    f.close()

