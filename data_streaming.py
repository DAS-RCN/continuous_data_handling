"""Module for streaming continuous data / separate events """

import copy
import numpy as np
import config_general as cfg
import config_event_analysis as cfg_event
import signal_processing as proc
import databases
import threading
if cfg.reading_mode.lower() == 'cloud':
    import tensorflow as tf
    import gcp_io
else:
    import glob
from numba import njit, prange

class DataFiles:

    def __init__(self, remove_proc_files=True, folder_name=cfg.data_path, db_name=cfg.events_db_name,
                 filename_list=None):
        self.curr = 0
        self.next = 1
        self.prev = -1
        if filename_list:
            self.all_files = filename_list
            self.all_files.sort()
            self.n_files = len(self.all_files)
            self.n_tot_files = self.n_files
        else:
            if cfg.reading_mode.lower() == 'cloud':
                self.all_files = tf.io.gfile.glob(
                    '{}/*.np[z|y]'.format(folder_name))

            else:
                self.all_files = glob.glob(folder_name + '/*.npz')
                self.all_files.extend(glob.glob(folder_name + '/*.npy'))
                self.all_files.extend(glob.glob(folder_name + '/*.sgy'))
                self.all_files.extend(glob.glob(folder_name + '/*.segy'))
                self.all_files.extend(glob.glob(folder_name + '/*.h5'))

            self.all_files.sort()
            self.all_files = self.all_files[cfg.first_file_index:]
            self.n_tot_files = len(self.all_files)
            self.n_files = self.n_tot_files

        if remove_proc_files:
            self.remove_proc_files(db_name)
            self.n_proc_files = self.n_tot_files - self.n_files

    def remove_proc_files(self, db_name):
        self.all_files = [
            f for f in self.all_files
            if f not in databases.find_processed_files(db_name)
        ]
        if self.all_files:
            self.n_files = len(self.all_files)
            self.all_files.sort()
        else:
            raise RuntimeError(
                'All files in this folder have already been processed'
            )

    def get_filename(self, mode):
        if mode == 'prev':
            return self.all_files[self.prev]
        if mode == 'curr':
            return self.all_files[self.curr]
        if mode == 'next':
            return self.all_files[self.next]

    def move_filename(self, mode):
        if mode == 'next' and self.curr < self.n_files - 1:
            self.curr += 1
            self.next += 1
            self.prev += 1
        elif mode == 'prev' and self.curr > 0:
            self.curr -= 1
            self.next -= 1
            self.prev -= 1
        return 0


class DataChunk:
    # Todo  - add UTC time support (absolute times)
    def __init__(self, filename, nt=cfg.nt, dt=cfg.dt, n_chan=cfg.n_chan,
                 d_chan=cfg.d_chan, gauge_length=cfg.gauge_length, pad_size=0):
        self.filename = filename
        self.orig_filename = filename
        self.nt_file = nt
        self.nt = nt
        self.dt = dt
        self.n_chan = n_chan
        self.d_chan = d_chan
        self.gauge_length = gauge_length
        self.data = np.empty(shape=(n_chan, nt), dtype=np.float32)
        self.valid_nt = True
        self.pad_size = pad_size

    def load(self):
        if cfg.reading_mode.lower() == 'cloud':
            with tf.io.gfile.GFile(self.filename, 'rb') as f:
                if self.filename.endswith('.npz'):
                    data = np.load(f)
                    self.data = data['dataSamples'].T

                elif self.filename.endswith('.npy'):
                    self.data = np.load(f).astype(np.float32)
                elif self.filename.endswith('.sgy') or self.filename.endswith('.segy'):
                    raise TypeError('Not supported for now')
                else:
                    raise ValueError(
                        'Unsuppported file extension. Received file: {}. '
                        'Expected extentions .npz or .npy'.format(self.filename)
                    )
        elif cfg.reading_mode.lower() == 'disk':
            with open(self.filename, 'rb') as f:
                if self.filename.endswith('.npz'):
                    data = np.load(f)
                    self.data = data['dataSamples'].T
                elif self.filename.endswith('.npy'):
                    self.data = np.load(f).astype(np.float32)
                elif self.filename.endswith('.sgy') or self.filename.endswith('.segy'):
                    cfg.flex_nt = False
                    # cannot handle varying nt for now, even though it could be deducted from the size of the file and
                    # the number of channels
                    self.data = load_segy_direct(self.filename, cfg.nt, 0, cfg.n_chan)
                    # For .npy/.npz we cannot do partial reading. For consistency, we read the full dataset from SEG-Y and
                    # crop it later, as we do for .npy/.npz files
                elif self.filename.endswith('.h5'):
                    self.data = None # TODO bin
                else:
                    raise ValueError(
                        'Unsuppported file extension. Received file: {}. '
                        'Expected extentions .npz, .npy, .sgy'.format(self.filename)
                    )

        else:
            raise NameError('Reading mode incorrect')

        n_chan, nt = np.shape(self.data)
        if not n_chan or n_chan != self.n_chan:
            self.valid_nt = False

        if cfg.flex_nt:
            self.nt = nt
        elif not nt or nt != self.nt:
            self.valid_nt = False
            print('NT does not match information in config file, this will be skipped')

    def add_stream(self, stream_to_add):
        self.nt += stream_to_add.nt
        self.data = np.concatenate((self.data, stream_to_add.data), axis=-1)

    def pad_chans(self, l_chan, r_chan):
        self.data = np.pad(self.data, ((l_chan, r_chan), (0, 0)),
                           'constant', constant_values=((0, 0), (0, 0)))
        self.n_chan += (l_chan+r_chan)

    def downsample(self, dt_decim_fac, chan_decim_fac, overwrite_stream, extra_lowpass_freq=99999999.9):

        orig_stream = None
        orig_freq = 1.0 / self.dt
        decim_nyfreq = orig_freq / dt_decim_fac / 2.0
        decim_n_chan = int(np.floor(self.n_chan / chan_decim_fac))

        if overwrite_stream:
            if chan_decim_fac >= 2:
                self.data = spatial_stack_downsample(self.data, self.n_chan, self.nt, chan_decim_fac)
                assert self.data.shape[0] == int(self.n_chan / chan_decim_fac)

                self.n_chan = int(self.n_chan / chan_decim_fac)
                self.d_chan = self.d_chan * chan_decim_fac

            self.data = proc.lpfilter(self.data, self.dt, min(
                extra_lowpass_freq, 0.8 * decim_nyfreq))
            self.data = self.data[:, 0::dt_decim_fac]
            self.nt = self.data.shape[1]
            self.dt *= dt_decim_fac
        else:
            orig_stream = copy.deepcopy(self)
            if chan_decim_fac >= 2:
                orig_stream.data = spatial_stack_downsample(orig_stream.data, self.n_chan, self.nt, chan_decim_fac)
                assert orig_stream.data.shape[0] == int(self.n_chan / chan_decim_fac)

                orig_stream.n_chan = int(self.n_chan / chan_decim_fac)
                orig_stream.d_chan = self.d_chan * chan_decim_fac

            orig_stream.data = proc.lpfilter(orig_stream.data, self.dt, min(
                extra_lowpass_freq, 0.8 * decim_nyfreq))
            orig_stream.data = orig_stream.data[:, 0::dt_decim_fac]
            orig_stream.nt = orig_stream.data.shape[1]
            orig_stream.dt *= dt_decim_fac

        return orig_stream

    def cut(self, chans, samples, modify_stream=False):

        if (chans[0] < 0 or samples[0] < 0 or chans[1] > self.n_chan or
                samples[1] > self.nt):
            raise RuntimeError(
                'Boundaries for cutting are out of range '
            )

        if modify_stream:

            self.nt = samples[1] - samples[0]
            self.n_chan = chans[1] - chans[0]
            self.data = self.data[chans[0]:chans[1], samples[0]:samples[1]]

            return self

        else:

            cut_stream = copy.deepcopy(self)
            cut_stream.nt = samples[1] - samples[0]
            cut_stream.n_chan = chans[1] - chans[0]
            cut_stream.data = cut_stream.data[chans[0]:chans[1], samples[0]:samples[1]]
            return cut_stream

    def write(self, mode='overwrite'):
        if mode == 'overwrite':
            out_f = open(self.filename, 'wb')
        elif mode == 'append':
            out_f = open(self.filename, 'ab')
        else:
            raise TypeError('write mode unrecognized')
        self.data.tofile(out_f)
        out_f.close()
        return 0

    def populate(self, proc_steps=('median')):
        self.load()
        if self.valid_nt:
            self.cut(chans=(cfg.first_live_ch, cfg.last_live_ch),
                     samples=(0, self.nt), modify_stream=True)
            if 'median' in proc_steps:
                self.data = proc.remove_median(self.data)
            if 'bandpass' in proc_steps:
                self.data = proc.bpfilter(
                    self.data, dt=cfg.dt, bp_low=cfg.bp_low, bp_high=cfg.bp_high)
            if 'clip' in proc_steps:
                self.data = proc.clip(self.data, cfg.clip_perc)
            if 'normalize' in proc_steps:
                self.data = proc.normalization(self.data, cfg.norm_type)
            if 'downsample' in proc_steps:
                self.downsample(cfg.dt_decim_fac,
                                chan_decim_fac=1, overwrite_stream=True)
            return True
        else:
            return False

    def check_validity(self):
        return self.valid_nt

class DataStream:

    def __init__(self, remove_proc_files=True, proc_steps='median', folder_name=cfg.data_path,
                 db_name=cfg.events_db_name, filename_list=None):
        self.proc_steps = proc_steps
        self.file_stream = DataFiles(remove_proc_files=remove_proc_files, folder_name=folder_name, db_name=db_name,
                                     filename_list=filename_list)
        self.curr_chunk = DataChunk(
            filename=self.file_stream.all_files[self.file_stream.curr])

        self.curr_chunk.populate(proc_steps=self.proc_steps)
        self.next_chunk = DataChunk(
            filename=self.file_stream.all_files[self.file_stream.next])
        self.next_chunk.populate(proc_steps=self.proc_steps)
        self.prev_chunk = None
        self.buffered_chunk = None

    def load_buffered(self):
        self.buffered_chunk = DataChunk(
            filename=self.file_stream.all_files[self.file_stream.next + 1])
        self.buffered_chunk.populate(proc_steps=self.proc_steps)
        return 0

    def cut_stream(self, chans, samples, pad_zeros=True):
        nt = self.curr_chunk.nt
        n_chan = self.curr_chunk.n_chan

        if samples[0] < -nt or samples[1] > 2*nt:
            raise RuntimeError(
                'Time window in cutting stream is out of range despite padding')

        if not pad_zeros and (samples[0] < 0 or samples[1] > nt):
            raise RuntimeError('Time window in cutting stream is out of range')

        c_stream = copy.deepcopy(self)
        if chans[0] < 0 or chans[1] > n_chan:
            if pad_zeros:
                c_stream.curr_chunk.pad_chans(
                    l_chan=max(-chans[0], 0), r_chan=max(chans[1] - n_chan, 0))
                c_stream.prev_chunk.pad_chans(
                    l_chan=max(-chans[0], 0), r_chan=max(chans[1] - n_chan, 0))
                c_stream.next_chunk.pad_chans(
                    l_chan=max(-chans[0], 0), r_chan=max(chans[1] - n_chan, 0))
            else:
                raise RuntimeError(
                    'Channel window in cutting stream is out of range')

        if samples[0] > -1 and samples[1] < nt:
            return c_stream.curr_chunk.cut(chans=chans, samples=samples, modify_stream=True)

        elif samples[0] < 0 and samples[1] < nt:
            overlap_chunk = c_stream.prev_chunk.cut(chans=(0, c_stream.curr_chunk.n_chan), samples=(nt + samples[0], nt),
                                                    modify_stream=True)
            overlap_chunk.add_stream(c_stream.curr_chunk)
            return overlap_chunk.cut(chans=chans, samples=(0, samples[1]-samples[0]), modify_stream=True)

        elif samples[0] > 0 and samples[1] > nt:
            overlap_chunk = c_stream.curr_chunk.cut(chans=(0, c_stream.curr_chunk.n_chan), samples=(samples[0], nt),
                                                    modify_stream=True)
            overlap_chunk.add_stream(c_stream.next_chunk)
            return overlap_chunk.cut(chans=chans, samples=(0, samples[1]-samples[0]), modify_stream=True)

        else:
            raise RuntimeError(
                'Time window for cutting stream is larger than a full file, not supported')

    def bandpass_stream(self, bp_low, bp_high):
        # Assumes buffer loading is finished
        if self.prev_chunk and self.curr_chunk and self.next_chunk and self.prev_chunk.valid_nt \
                and self.curr_chunk.valid_nt and self.next_chunk.valid_nt:
            nt = self.curr_chunk.nt
            nch = self.curr_chunk.n_chan
            sample_overlap = int(0.1 * nt)

            overlap_chunk = self.prev_chunk.cut(chans=(0, nch), samples=(nt - sample_overlap, nt),
                                                modify_stream=False)
            overlap_chunk.add_stream(self.curr_chunk)
            overlap_chunk.add_stream(self.next_chunk.cut(chans=(0, nch), samples=(0, sample_overlap),
                                                         modify_stream=False))
            filt_data = proc.bpfilter(
                overlap_chunk.data, overlap_chunk.dt, bp_low, bp_high)
            return filt_data[:, sample_overlap:-sample_overlap]

        else:
            raise RuntimeError(
                'One of data chunks in stream are inexistent or have different number of sample')

    def roll_stream(self, direction, use_buffered_chunk=False):
        # Assumes buffer loading is finished
        self.file_stream.move_filename(direction)
        if direction == 'next':
            self.prev_chunk = self.curr_chunk
            self.curr_chunk = self.next_chunk
            if use_buffered_chunk and self.buffered_chunk is not None:
                self.next_chunk = self.buffered_chunk
            else:
                if self.file_stream.next < self.file_stream.n_files:
                    self.next_chunk = DataChunk(
                        filename=self.file_stream.all_files[self.file_stream.next])
                    self.next_chunk.populate(proc_steps=self.proc_steps)
                else:
                    self.next_chunk = None
        elif direction == 'prev':
            self.next_chunk = self.curr_chunk
            self.curr_chunk = self.prev_chunk
            if use_buffered_chunk and self.buffered_chunk is not None:
                self.prev_chunk = self.buffered_chunk
            else:
                self.prev_chunk = DataChunk(
                    filename=self.file_stream.all_files[self.file_stream.prev])
                self.prev_chunk.populate(proc_steps=self.proc_steps)
        return 0

    def check_validity(self):
        if self.next_chunk and self.prev_chunk and self.curr_chunk and self.curr_chunk.valid_nt and \
                self.next_chunk.valid_nt and self.prev_chunk.valid_nt:
            return True
        else:
            return False


class AsyncEventLoad(threading.Thread):
    def __init__(self, threadid, name, counter, buffer, filename):
        threading.Thread.__init__(self)
        self.threadID = threadid
        self.name = name
        self.counter = counter
        self.filename = filename
        self.buffer = buffer

    def run(self,):
        self.buffer[:, :] = gcp_io.read(self.filename)[:, 0:cfg_event.nt]
        return 0


class AsyncLoad(threading.Thread):

    def __init__(self, threadid, name, counter, stream, proc_steps=('median', 'downsample', 'preprocess')):
        threading.Thread.__init__(self)
        self.threadID = threadid
        self.name = name
        self.counter = counter
        self.stream = stream
        self.proc_steps = proc_steps

    def run(self, ):
        if self.stream.file_stream.next + 1 < self.stream.file_stream.n_files:
            self.stream.buffered_chunk = DataChunk(
                filename=self.stream.file_stream.all_files[self.stream.file_stream.next + 1])
            self.stream.buffered_chunk.populate(proc_steps=self.proc_steps)
        else:
            self.stream.buffered_chunk = None
        return 0


class AsyncWrite(threading.Thread):

    def __init__(self, threadid, name, out_filename, out_data, format='binary'):
        threading.Thread.__init__(self)
        self.threadID = threadid
        self.name = name
        self.out_filename = out_filename
        self.out_data = out_data
        self.format = format

    def run(self, ):
        gcp_io.write(self.out_filename, self.out_data, format=self.format)
        del self.out_data
        return 0


class DataIterator:
    """ This is used to serially go through all files in folder without interacting with the user"""

    def __init__(self, folder_name=cfg.data_path, db_name=cfg.events_db_name, remove_proc_files=False,
                 proc_steps=[], filenames=None, output_overlap=-999):

        self.proc_steps = proc_steps
        self.data = DataStream(remove_proc_files=remove_proc_files, proc_steps=[], folder_name=folder_name,
                               db_name=db_name, filename_list=filenames)
        if self.data.file_stream.n_files < 3:
            raise RuntimeError('Not enough files in folder')

        self.buf_id = 1
        self.threads = []
        self.threads.append(AsyncLoad(threadid=self.buf_id, name='Dataloading', counter=self.buf_id,
                                      stream=self.data, proc_steps=[]))
        self.buf_id += 1
        self.internal_overlap = int(self.data.curr_chunk.nt/10)
        self.output_overlap = output_overlap
        if self.output_overlap < 0:
            self.output_overlap = 0

        self.threads[-1].start()

    def __iter__(self):
        return self

    def __next__(self):
        if self.data.check_validity():
            nt = self.data.curr_chunk.nt
            nch = self.data.curr_chunk.n_chan
            overlap_chunk = self.data.prev_chunk.cut(chans=(0, nch), samples=(nt - self.internal_overlap, nt),
                                                     modify_stream=False)
            overlap_chunk.add_stream(self.data.curr_chunk)
            overlap_chunk.add_stream(self.data.next_chunk.cut(chans=(0, nch), samples=(0, self.internal_overlap),
                                                              modify_stream=False))
            overlap_chunk.filename = self.data.curr_chunk.filename
            overlap_chunk.pad_size = self.internal_overlap
        else:
            if self.data.curr_chunk:
                overlap_chunk = DataChunk(filename=self.data.curr_chunk.filename, nt=0, n_chan=0, dt=0.0)
                overlap_chunk.valid_nt = False
            else:
                raise StopIteration

        if self.buf_id == self.data.file_stream.n_files+1:
            raise StopIteration
        elif self.buf_id < self.data.file_stream.n_files:
            self.threads[-1].join()
            self.data.roll_stream(direction='next', use_buffered_chunk=True)
            self.buf_id += 1
            self.threads.append(AsyncLoad(threadid=self.buf_id, name='Dataloading', counter=self.buf_id,
                                          stream=self.data, proc_steps=[]))
            self.threads[-1].start()
        else:
            self.data.prev_chunk = self.data.curr_chunk
            self.data.curr_chunk = self.data.next_chunk
            self.data.next_chunk = None

        if overlap_chunk.check_validity():

            if 'median' in self.proc_steps:
                overlap_chunk.data = proc.remove_median(overlap_chunk.data)
            if 'clip' in self.proc_steps:
                overlap_chunk.data = proc.clip(overlap_chunk.data, cfg.clip_perc)
            if 'bandpass' in self.proc_steps:
                overlap_chunk.data = proc.bpfilter(overlap_chunk.data, dt=cfg.dt, bp_low=cfg.bp_low, bp_high=cfg.bp_high)
            if 'lowpass' in self.proc_steps:
                overlap_chunk.data = proc.lpfilter(overlap_chunk.data, dt=cfg.dt, cutoff=cfg.lp_cutoff)
            if 'normalize' in self.proc_steps:
                overlap_chunk.data = proc.normalization(overlap_chunk.data, cfg.norm_type)
            if 'downsample' in self.proc_steps:
                overlap_chunk.downsample(cfg.dt_decim_fac, chan_decim_fac=cfg.n_ch_stack, overwrite_stream=True)
                self.internal_overlap = int(self.internal_overlap/cfg.dt_decim_fac)
                self.output_overlap = int(self.output_overlap/cfg.dt_decim_fac)

            overlap_chunk.cut(chans=(0, overlap_chunk.n_chan),
                              samples=(int(self.internal_overlap-self.output_overlap),
                                       int(int(self.data.curr_chunk.nt/cfg.dt_decim_fac)+self.internal_overlap+self.output_overlap)),
                              modify_stream=True)

        return overlap_chunk


class EventIterator:
    """ This is used to serially go through all events in a database"""

    def __init__(self, folder_name, f_file=-1, l_file=-1):

        file_names = gcp_io.get_filenames(folder_name)
        self.file_names = [i for i in file_names if '.npy' in i]
        self.n_evs = len(self.file_names)

        if l_file == -1:
            l_file = self.n_evs
        if f_file == -1:
            f_file = 0

        self.file_names = self.file_names[f_file:l_file]
        self.n_evs = len(self.file_names)

        self.threads=[]
        self.curr = np.zeros(shape=(cfg_event.nch, cfg_event.nt))
        self.next = np.zeros(shape=(cfg_event.nch, cfg_event.nt))
        self.buff = np.zeros(shape=(cfg_event.nch, cfg_event.nt))

        self.buf_id = 1
        self.threads.append(AsyncEventLoad(threadid=self.buf_id, name='Eventloading', counter=self.buf_id,
                                           buffer=self.buff, filename=self.file_names[self.buf_id]))
        self.threads[-1].start()
        self.threads[-1].join()
        self.curr = np.copy(self.buff)

        self.buf_id += 1
        self.threads.append(AsyncEventLoad(threadid=self.buf_id, name='Eventloading', counter=self.buf_id,
                                           buffer=self.buff, filename=self.file_names[self.buf_id]))
        self.threads[-1].start()

    def __iter__(self):
        return self

    def __next__(self):
        if self.buf_id > self.n_evs:
            raise StopIteration

        self.threads[-1].join()
        self.next = np.copy(self.buff)
        out_data = np.copy(self.curr)
        self.buf_id += 1
        self.curr = np.copy(self.next)
        if self.buf_id < self.n_evs:
            self.threads.append(AsyncEventLoad(threadid=self.buf_id, name='Eventloading', counter=self.buf_id,
                                           buffer=self.buff, filename=self.file_names[self.buf_id]))
            self.threads[-1].start()
        return out_data


def get_files_in_range(filerange):
    all_files = []
    for fold in cfg.batch_process_folders:
        fold_files = tf.io.gfile.glob('{}/*.np[z|y]'.format(fold))
        fold_files.sort()
        all_files.extend(fold_files)

    f_file = all_files.index(filerange[0])-1
    l_file = all_files.index(filerange[1])+2
    return all_files[f_file:l_file]


def load_segy_direct(filename, nt, f_ch, l_ch):
    """ Reads binary content of SEG-Y file
    Inputs:
    filename - name (including path) of the SEG-Y file
    nt - number of samples
    f_ch - first channel number to read (python notation - starts at 0)
    l_ch - last channel number to read (python notation - that )
    """
    file_to_read = open(filename, 'rb')
    file_to_read.seek(3600, 1)
    file_to_read.seek(f_ch*(nt*4+240), 1)
    data = np.zeros(shape=(l_ch-f_ch, nt), dtype='float32')
    for ch in range(l_ch-f_ch):
        file_to_read.seek(240, 1)
        data[ch, :] = np.fromfile(file_to_read, dtype='>f', count=nt)

    return data


@njit(parallel=True)
def spatial_stack_downsample(data, nch, nt, decim_fac):
    new_nch = int(nch/decim_fac)
    decim_data = np.zeros(shape=(new_nch, nt), dtype=np.float32)
    for ch_num in prange(new_nch):
        for ind in range(decim_fac):
            decim_data[ch_num, :] += data[ch_num*decim_fac+ind, :]
    decim_data = 1.0/decim_fac*decim_data
    return decim_data