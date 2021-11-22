#### Bachelor preprocessing - test

# importing packages
import mne
import h5py

# Import common libraries
import numpy as np
import pandas as pd
import os

# Import MNE processing
from mne.preprocessing.nirs import optical_density, beer_lambert_law

# Import MNE-NIRS processing
from mne_nirs.statistics import run_glm
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import statsmodels_to_results
from mne_nirs.channels import get_short_channels, get_long_channels
from mne_nirs.channels import picks_pair_to_idx
from mne_nirs.visualisation import plot_glm_group_topo
from mne_nirs.datasets import fnirs_motor_group
from mne_nirs.visualisation import plot_glm_surface_projection
from mne_nirs.io.fold import fold_landmark_specificity, fold_channel_specificity

# Import MNE-BIDS processing
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals

# Import StatsModels
import statsmodels.formula.api as smf

# Import Plotting Library
import matplotlib.pyplot as plt
import matplotlib as mpl
from lets_plot import *
LetsPlot.setup_html()


raw_path = "/Users/sigridagersnapbomnielsen/Documents/Python/PyCharm_projects/Bachelor_project_F21/phase_2_data/NCPE-2021-11-01/NP-Ph2-303-2021-11-01/2021-11-01_001.snirf"

raw_intensity = mne.io.read_raw_snirf(raw_path, preload=True)
raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)

#sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od, l_freq=0.7, h_freq=1.5)
#raw_od.info['bads'] = list(compress(raw_od.ch_names, sci < 0.5))

if verbose:
 ic("Apply short channel regression.")
 od_corrected = mne_nirs.signal_enhancement.short_channel_regression(raw_od)

if verbose:
   ic("Do temporal derivative distribution repair on:", od_corrected)
tddr_od = mne.preprocessing.nirs.tddr(od_corrected)

if verbose:
   ic("Convert to haemoglobin with the modified beer-lambert law.")
raw_haemo = beer_lambert_law(tddr_od, ppf=0.1)

if verbose:
   ic("Apply further data cleaning techniques and extract epochs.")
raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

if verbose:
   ic("Separate the long channels and short channels.")
sht_chans = get_short_channels(raw_haemo)
raw_haemo = get_long_channels(raw_haemo)

if verbose:
   ic("Bandpass filter on:", raw_haemo)
filter_haemo = raw_haemo.filter(
    0.01, 0.7, h_trans_bandwidth=0.3, l_trans_bandwidth=0.005)

# Create a design matrix
design_matrix = make_first_level_design_matrix(raw_haemo)