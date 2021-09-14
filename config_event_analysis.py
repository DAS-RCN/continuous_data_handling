""" Configuration for event-based processing (location) """

import numpy as np

# --- Data --- #
dx = 1.015
dt = 0.001
nch = 2000
nt = 500
folder_name = 'gs://das-small/predictions/location_windows'

dt = 0.002
nch = 712
nt = 196
folder_name = 'gs://das-small/picked_events/processed_windows'


nt_live = 200
time_bef_inf_win = 0.2  # Time (sec) to save before picks
time_aft_inf_win = 0.4  # Time (sec) to save after picks

# --- Event location --- #
clip_for_apex = 0.3
length_for_apex = 7
smooth_for_apex = 6
trough_width = 30
data_width = 200
time_win_apex = 0.1
coh_thres = 0.2
shift_for_apex = 0.015
scan_distances = np.linspace(0, 500, 51)
freqs = np.linspace(5.0, 300.0, 296)
vels = np.linspace(-1500.0, -6000.0, 451)
n_dist = scan_distances.size
save_scan = True
d_off = 0.5
max_off = 600.0
save_scan = True
mask_file = 'gs://das-small/picked_events/Auxiliary/curved_polygon_NE11_v2.npy'
mask_modes = [0, 1, 2, 3, 4]
nfft_k = 2048
nfft_f = 1024

# --- Display --- #
nominal_dist = 250.0 # Used for nominal correction of offset
disp_ch_win = 600
apex_only = False # Only show events with estimated apex

# --- Databases --- #
label_db = 'gs://das-small/picked_events/Windowed_events_labeled.db' # Database that also includes labels. Purged events will be removed.
# TODO - reading from bucket directly does not work - not sure why
#origin_db = 'gs://das-small/picked_events/Windowed_events.db' # Database from which windows were cut - needs to be consistent. Is copied and appended.
#origin_db = 'Databases/inference.db'
origin_db = 'Databases/Windowed_events.db'

# --- Checks --- #
# folder_name =folder_name.rstrip('/')
