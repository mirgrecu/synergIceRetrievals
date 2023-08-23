import glob
from netCDF4 import Dataset
from pyhdf.SD import SD, SDC
from pyhdf import HDF, VS, V
import numpy as np


def read2C_ICE(f):
    fh=Dataset(f)
    iwc=fh["IWC"][:]
    re=fh["re"][:]
    dbz=fh["dBZe_simulation"][:]
    temp=fh["Temperature"][:]
    height=fh["Height"][:]

    # Open HDF4 file.
    FILE_NAME = f
    hdf = SD(FILE_NAME, SDC.READ)

    #print(hdf.datasets())

    #h = HDF.HDF(FILE_NAME)
    #vs = h.vstart()

    h = HDF.HDF(FILE_NAME)
    vs = h.vstart()
    
    xid = vs.find('Latitude')
    latid = vs.attach(xid)
    latid.setfields('Latitude')
    nrecs, _, _, _, _ = latid.inquire()
    latitude = latid.read(nRec=nrecs)
    latid.detach()
    
    lonid = vs.attach(vs.find('Longitude'))
    lonid.setfields('Longitude')
    nrecs, _, _, _, _ = lonid.inquire()
    longitude = lonid.read(nRec=nrecs)
    lonid.detach()
    

    h = HDF.HDF(FILE_NAME)
    vs = h.vstart()
    
    xid = vs.find('Latitude')
    latid = vs.attach(xid)
    latid.setfields('Latitude')
    nrecs, _, _, _, _ = latid.inquire()
    latitude = latid.read(nRec=nrecs)
    latid.detach()
    
    lonid = vs.attach(vs.find('Longitude'))
    lonid.setfields('Longitude')
    nrecs, _, _, _, _ = lonid.inquire()
    longitude = lonid.read(nRec=nrecs)
    lonid.detach()
    
    
    iwp_id = vs.attach(vs.find('ice_water_path'))
    iwp_id.setfields('ice_water_path')
    nrecs, _, _, _, _ = iwp_id.inquire()
    iwp = iwp_id.read(nRec=nrecs)
    iwp_id.detach()

    return iwc,dbz,temp,re,height,longitude,latitude,np.array(iwp)[:,0]
