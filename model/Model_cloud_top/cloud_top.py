#Libraries
import os
import sys
import dask
import warnings
import netCDF4 as nc
import numpy as np
import xarray as xr
import datetime
import cftime
from dask.diagnostics import ProgressBar

#--------------------------------------------------------------
#Helper functions 
#--------------------------------------------------------------

def time_filter(data):
    winter_month = list([1,2,11,12])
    time_months = data['time'].dt.month
    time_mask = time_months.isin(winter_month)
    data = data.where(time_mask, drop=True)
    return data

def lat_lon_filter(data):
    lat_mask = (data['lat'] >= 55) & (data['lat'] <= 80)
    lon_mask = (data['lon'] >= 300) | (data['lon'] <= 60)
    data = data.where(lat_mask & lon_mask, drop=True)
    return data



#--------------------------------------------------------------
#Input data 
#--------------------------------------------------------------
#Folder path
folder_path = "/nfs/n2o/wolke_scratch/pfarron/data/1st_paper_data/"
array_path = "/nfs/n2o/wolke_scratch/pfarron/data/arrays/1st_paper/CTT/"
file = str(sys.argv[1])
file_path = folder_path + str(file)

#---------------------------------------------------------------
#Data setup
#---------------------------------------------------------------
#Load data with Dask
data = xr.open_dataset(
    file_path,
    chunks={'time': 1}
)

#Apply filters
data = time_filter(data)
data = lat_lon_filter(data)

#Load variables
CLDLIQ = data['CLDLIQ']
CLDICE = data['CLDICE']
TWC = CLDLIQ + CLDICE
T = data['T']

SWCF = data['SWCF']
LWCF = data['LWCF']
CRE = SWCF + LWCF


#---------------------------------------------------------------
#setup arrays
#---------------------------------------------------------------
CRE_T_loop = []
CC_T_loop = []

#---------------------------------------------------------------
#Cloud masking
#---------------------------------------------------------------
#Cloud cover masking
TWC_threshold = 0.00001
CC_3d_mask = TWC > TWC_threshold
CC_3d_mask_index = CC_3d_mask.argmax(dim='lev')
#Remove values with no cloud 
CC_3d_check = ~CC_3d_mask.any(dim='lev')
CC_3d_final = xr.where(CC_3d_check, -999, CC_3d_mask_index)
#TWC mask is 2d, it has either index of CT or -999 for no cloud



#---------------------------------------------------------------
#Temperature range loop
#---------------------------------------------------------------
T_ranges = [f"{i}-{i+4}" for i in range(183, 303, 4)]

for T_range in T_ranges:
    #Temperature bounds
    T_min, T_max = map(int, T_range.split('-'))
    #Mask cloudy gridboxes within T range
    CTT_mask = (TWC > TWC_threshold) & (T >= T_min) & (T < T_max)
    #Find lowest level where condition is met (highest since reversed)
    CTT_index = CTT_mask.argmax(dim='lev')
    #If gridbox has a cloud + highest cloudy point is within temperature range -> cloud has right CTT
    CTT_filter = (CTT_index == CC_3d_final)

    #Apply mask
    CRE_CTT = xr.where(CTT_filter, CRE, np.nan)

    #Compute averages
    CRE_mean = CRE_CTT.mean(dim=('time', 'lat', 'lon'), skipna=True)
    CRE_T_loop.append(CRE_mean)

    #Compute cloud cover lazily
    cloudy_gp = CRE_CTT.notnull().sum(dim=('time', 'lat', 'lon'))
    total_gp = CRE.notnull().sum(dim=('time', 'lat', 'lon'))
    cloud_fraction = cloudy_gp / total_gp

    CC_T_loop.append(cloud_fraction)


#---------------------------------------------------------------
#Dask lazy computation
#---------------------------------------------------------------
# Compute all cloud cover values in parallel
with ProgressBar(dt=15.0):
    (
        CRE_mean_output,
        CC_output

    ) = dask.compute(
        CRE_T_loop,
        CC_T_loop
    )


#---------------------------------------------------------------
#Save arrays
#---------------------------------------------------------------
name = '_' + file.split('.')[0]

np.save(array_path + 'CRE' + name, CRE_mean_output)
np.save(array_path + 'CC' + name, CC_output)