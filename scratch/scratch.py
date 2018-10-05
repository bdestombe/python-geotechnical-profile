# coding=utf-8
import os

import numpy as np
from scipy import stats
import dask.array as da

# from dtscalibration.datastore import DataStore
from dtscalibration.datastore import read_xml_dir
from dtscalibration.plot import plot_sigma_report
import matplotlib.pyplot as plt
import time
from geotechnicalprofile.gp_datastore import read_gef

t0 = time.time()
# wd = os.path.dirname(os.path.abspath(__file__))
fp = os.path.join('..', 'tests', 'data', 'GEF', '67059_DKM006.GEF')

ds = read_gef(fp)
