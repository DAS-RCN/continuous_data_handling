import os

# --- Reading data --- #

reading_mode = 'Disk'
# Where are the data stored - Disk or Cloud. For cloud, only Google Cloud Platform is implemented
flex_nt = False
# Flexible number of samples per file. Not fully tested
data_path = 'Data'
# Path (relative or absolute) to the folder in which data are stored. Files are read in order after sorting by name
first_file_index = 0
# Serial number of first file to read (useful for visualizing different portions of the dataset)

# --- Data and sampling parameters ###
n_chan = 1280
# number of channels in raw DAS files
last_live_ch = 1160
# last live channel (beyond which we see optical noise). Data are usable up to this channel
first_live_ch = 199
# first live channel for analysis
nt = 30000
# number of time samples in raw DAS files.
dt = 0.0005
# time between adjacent samples (sec)
d_chan = 1.02
# spacing between adjacent channels, after correction for stretching factors
gauge_length = 10.0
# DAS gauge length

# --- Processing parameters  --- #

proc_steps = ['median', 'clip', 'bandpass', 'normalize']
#proc_steps = ['median', 'clip', 'normalize']
# Processing steps applied to loaded data. Can be: 'median', 'clip', 'bandpass', 'normalize', 'downsample'
# median - median filter (sample-by-sample)
# clip - order filter, percentage chosen by user
# bandpass - bandpass filter, lower and upper bounds chosen by user
# lowpass - lowpass filter, chosen by user
# normalize - trace by trace normalization, type chosen by user
# downsample - downsampling, integer decimation factor chosen by user

clip_perc = 99.0
# Percentage for clipping
bp_low = 5.0
# Lower boundary of bandpass applied (Hz)
bp_high = 100.0
# Higher boundary of bandpass applied (Hz).
lp_cutoff = 100.0
# Cutoff frequency for low-pass filter
norm_type = 'std'
# Type of trace-by-trace normalization (see signal_processing for details)
dt_decim_fac = 1
# Integer decimation factor, Anti-alias filter applied beforehand.
overlap_samples = 0
# number of samples (original sampling rate) in the buffer around each data file to be returned.
n_ch_stack = 1
# Number of nearby channels to stack and downsample in space

# --- Consistency checks - do not modify --- #

if data_path.endswith('/'):
    data_path = data_path[:-1]

batch_process_folders = []
events_db_name = []
