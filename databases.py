''' Database handling functions, including merging DBs '''

import config_general as cfg
import os
import numpy as np
import sqlite3
import config_event_analysis as cfg_ev
from typing import List, Text, Tuple

#TODO - add time support!

class Event:

    def __init__(self, ev_id, time, das_location=(0, 0, 0), das_magnitude=-999.9, geo_location=(0, 0, 0),
                 geo_magnitude=-999.99, das_channel_apex=-999, das_classif='none', geo_classif='none', out_file='none'):
        self.ev_id = ev_id
        self.time = time
        self.das_location = das_location
        self.das_magnitude = das_magnitude
        self.geo_location = geo_location
        self.geo_magnitude = geo_magnitude
        self.das_classif = das_classif
        self.geo_classif = geo_classif
        self.das_channel_apex = das_channel_apex
        self.out_file = out_file

def init_events_db():
    if os.path.isfile(cfg.events_db_name):
        build_db_backup(cfg.events_db_name)
    else:
        new_events_db(cfg.events_db_name)


def init_processing_db():
    if os.path.isfile(cfg.processing_db_name):
        build_db_backup(cfg.processing_db_name)
    else:
        new_processing_db(cfg.processing_db_name)


def init_labeling_db():
    if os.path.isfile(cfg_ev.label_db):
        build_db_backup(cfg_ev.label_db)
    else:
        new_labeling_db(cfg_ev.label_db)


def build_db_backup(orig_db_name):
    copy_ord = 'cp ' + orig_db_name + ' ' + orig_db_name+'_backup'
    os.system(copy_ord)


def new_events_db(db_name):

    if os.path.isfile(db_name):
        print('This DB file already exists. To prevent erasing, the program will stop')
        raise OSError

    sql_conn = sqlite3.connect(db_name)
    sql_curs = sql_conn.cursor()
    sql_curs.execute('''CREATE TABLE events (origin_filename text, pick_chan integer, pick_samp integer, cut_ev_filename text, 
    cut_fchan integer, cut_nchan integer, d_chan real, cut_nt integer, dt real, gauge_length real)''')
    sql_conn.commit()
    sql_conn.close()


def new_labeling_db(db_name):

    if os.path.isfile(db_name):
        print('This DB file already exists. To prevent erasing, the program will stop')
        raise OSError

    sql_conn = sqlite3.connect(db_name)
    sql_curs = sql_conn.cursor()
    sql_curs.execute('''CREATE TABLE events (origin_filename text, pick_chan integer, pick_samp integer, cut_ev_filename text, 
    cut_fchan integer, cut_nchan integer, d_chan real, cut_nt integer, dt real, gauge_length real, int est_apex_ch, 
    float est_dist, reservoir_classification text)''')
    sql_conn.commit()
    sql_conn.close()


def new_processing_db(db_name):

    if os.path.isfile(db_name):
        print('This DB file already exists. To prevent erasing, the program will stop')
        raise OSError

    sql_conn = sqlite3.connect(db_name)
    sql_curs = sql_conn.cursor()
    sql_curs.execute('''CREATE TABLE events (file_index integer, origin_filename text, downsampled_filename text, 
    n_chan integer, first_chan integer, nt_downsampled integer, dt_downsampled real, templ_ind integer, max_cc real, max_cc_samp integer, 
    max_cc_channel integer)''')
    sql_conn.commit()
    sql_conn.close()


def order_by_filename(events):

    filenames = []
    for ev in events:
        curr_file = raw_filename(ev[0])
        curr_file.replace('_', '')
        filenames.append(curr_file)

    return [x for _, x in sorted(zip(filenames, events), key=lambda pair: pair[0])]


def cut_event_database(db_file, lines_to_use, cut_db_name):
    db_cursor = sqlite3.connect(db_file).cursor()
    new_events_db(cut_db_name)
    out_conn = sqlite3.connect(cut_db_name)
    out_cursor = out_conn.cursor()

    db_cursor.execute('SELECT * FROM events')
    events = db_cursor.fetchall()
    events = order_by_filename(events)

    for ind in lines_to_use:
        event = events[ind]
        print(event)
        out_cursor.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)', event)

    out_conn.commit()
    out_cursor.close()
    db_cursor.close()


def save_events(stream, picks, mode='sec'):
    dt = stream.curr_chunk.dt
    nt = stream.curr_chunk.nt

    picks.sort(key=lambda x: x[1])

    count = -1
    for count, apex in enumerate(picks):

        apex[0] = int(round(apex[0]))
        if mode == 'sec':
            apex[1] = int(round(apex[1]/dt))
        elif mode == 'samp':
            pass
        else:
            raise TypeError

        if cfg.save_cut_events:
            first_samp = apex[1] - int(round(cfg.pick_output_bef/dt))
            last_samp = apex[1] + int(round(cfg.pick_output_aft/dt)) + 1
            first_chan = max(0, int(round(apex[0] - cfg.pick_output_left)))
            last_chan = min(cfg.last_live_ch, int(round(apex[0] + cfg.pick_output_right)) + 1)

            if first_samp < 0:
                cut_event = stream.curr_chunk.cut(chans=(first_chan, last_chan), samples=(0, last_samp),
                                                  modify_stream=False)
                if stream.prev_chunk:
                    prev_event = stream.prev_chunk.cut(chans=(first_chan, last_chan), samples=(nt+first_samp, nt), modify_stream=False)
                    cut_event.data = np.concatenate((prev_event.data, cut_event.data), axis=-1)
                    cut_event.nt = last_samp-first_samp
                else:
                    cut_event.nt = last_samp

            elif last_samp > nt:
                print(stream.curr_chunk.data.shape)
                print(nt)
                print(first_samp)
                print(first_chan)
                print(last_chan)
                cut_event = stream.curr_chunk.cut(chans=(first_chan, last_chan), samples=(first_samp, nt),
                                                  modify_stream=False)
                if stream.next_chunk:
                    next_event = stream.next_chunk.cut(chans=(first_chan, last_chan), samples=(0, last_samp-nt),
                                                   modify_stream=False)
                    cut_event.data = np.concatenate((cut_event.data, next_event.data), axis=-1)
                    cut_event.nt = last_samp-first_samp
                else:
                    cut_event.nt = nt-first_samp
            else:
                print(first_chan, last_chan, first_samp, last_samp)
                print(stream.curr_chunk.data.shape)
                cut_event = stream.curr_chunk.cut(chans=(first_chan, last_chan), samples=(first_samp, last_samp),
                                                  modify_stream=False)

            filename_to_save = stream.curr_chunk.filename[:-4].split('/')
            filename_to_save = filename_to_save[-1]
            cut_event.filename = cfg.saved_ev_path + '/' + filename_to_save + '_ev' + str(count) + '.bin'
            cut_event.write()
            picked_ev_to_db(cut_event, apex, cfg.events_db_name)

        else:
            stream.curr_chunk.filename = 'DAS data not saved'
            stream.curr_chunk.first_chan = -999
            stream.curr_chunk.n_chan = -999
            stream.curr_chunk.nt = -999
            picked_ev_to_db(stream.curr_chunk, apex, cfg.events_db_name)

    return count+1


def picked_ev_to_db(event, apex, db_name):
    sql_conn = sqlite3.connect(db_name)
    sql_curs = sql_conn.cursor()

    entry = (event.orig_filename, apex[0]+cfg.first_live_ch, apex[1], event.filename, int(event.first_chan),
             int(event.n_chan), event.d_chan, int(event.nt), event.dt, event.gauge_length)

    sql_curs.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)', entry)
    sql_conn.commit()
    sql_conn.close()

def labeled_ev_to_db(event, apex_ch, label_db_name):
    sql_conn = sqlite3.connect(label_db_name)
    sql_curs = sql_conn.cursor()

    entry = (event.orig_filename, apex[0]+cfg.first_live_ch, apex[1], event.filename, int(event.first_chan),
             int(event.n_chan), event.d_chan, int(event.nt), event.dt, event.gauge_length, apex, classif)

    sql_curs.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', entry)
    sql_conn.commit()
    sql_conn.close()


def empty_ev_to_db(filename, db_name):
    sql_conn = sqlite3.connect(db_name)
    sql_curs = sql_conn.cursor()

    entry = (filename, -999, -999,  'No picks', -999, -999, -999.99, -999, -999.99, -999.99)

    sql_curs.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)', entry)
    sql_conn.commit()
    sql_conn.close()


def proc_data_to_db(data, file_index, templ_ind, cc_val, cc_sample, cc_chan, db_name):
    sql_conn = sqlite3.connect(db_name)
    sql_curs = sql_conn.cursor()
    entry = (file_index, data.orig_filename, data.filename, data.n_chan, cfg.first_live_ch, data.nt, data.dt,
             str(templ_ind), str(cc_val), str(cc_sample), str(cc_chan))
    sql_curs.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?)', entry)
    sql_conn.commit()
    sql_conn.close()

def empty_proc_data_to_db(data, file_index, templ_ind, cc_val, cc_sample, cc_chan, db_name):
    sql_conn = sqlite3.connect(db_name)
    sql_curs = sql_conn.cursor()
    entry = (file_index, data.orig_filename, data.filename, data.n_chan, cfg.first_live_ch, data.nt, data.dt,
             str(templ_ind), str(cc_val), str(cc_sample), str(cc_chan))
    sql_curs.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?)', entry)
    sql_conn.commit()
    sql_conn.close()


def find_processed_files(db_name):
    sql_conn = sqlite3.connect(db_name)
    sql_curs = sql_conn.cursor()
    sql_curs.execute('SELECT origin_filename FROM events')
    all_files = sql_curs.fetchall()
    unique_files = list(set(all_files))
    list_files = [x[0] for x in unique_files]
    return list_files


def get_all_filenames(folds=cfg.batch_process_folders):
    all_files = []
    for folder in folds:
        numpy_files = list(tf.io.gfile.glob('{}/*.np[z|y]'.format(folder)))
        if not numpy_files:
            all_files.extend(list(tf.io.gfile.glob('{}/*.bin'.format(folder))))
        else:
            all_files.extend(numpy_files)

    return all_files


def raw_filename(filename):
    raw_name = filename.split('/')
    raw_name = raw_name[-1]
    raw_name = raw_name.partition('.')
    raw_name = raw_name[0]
    raw_name = raw_name.partition('_downsamp_')
    raw_name = raw_name[0]
    return raw_name


def get_events_by_filename(events, filename):

    matching_evs = []
    for ev in events:
        if raw_filename(ev[0]) == raw_filename(filename):
            matching_evs.append(ev)
    return matching_evs


def get_files_to_read(events, all_filenames):

    ev_list = []
    filename_list = []

    for ev in events:
        curr_file = raw_filename(ev[0])
        curr_file.replace('_', '')
        ev_list.append(curr_file)

    for file in all_filenames:
        curr_file = raw_filename(file)
        curr_file.replace('_', '')
        filename_list.append(curr_file)

    ev_list.sort()
    order = sorted(range(len(filename_list)), key=filename_list.__getitem__)
    all_files_ordered = []
    for ind in order:
        all_files_ordered.append(all_filenames[ind])
    filename_list.sort()

    # Detect gaps between events, and skip reading for these

    file_diff = []
    for i in range(len(ev_list)-1):
        diff = int(ev_list[i+1][-2:])-int(ev_list[i][-2:]) + 60 * (int(ev_list[i+1][-4:-2])-int(ev_list[i][-4:-2]))\
            + 3600 * (int(ev_list[i+1][-6:-4])-int(ev_list[i][-6:-4])) + 86400 * (int(ev_list[i+1][-9:-7]) - int(ev_list[i][-9:-7])) \
            + 86400*30*(int(ev_list[i+1][-11:-9]) - int(ev_list[i][-11:-9]))
        file_diff.append(diff)

    file_diff = np.array(file_diff)
    gap_inds = np.empty(shape=(1,), dtype=np.int32)
    gap_inds[0] = -1
    gap_inds = np.append(gap_inds, np.nonzero(file_diff > 150))
    gap_inds = np.append(gap_inds, len(ev_list)-1)

    files_to_read = []
    for i in range(len(gap_inds)-1):
        first_file = ev_list[1+gap_inds[i]]
        last_file = ev_list[gap_inds[i+1]]
        f_ind = filename_list.index(first_file)
        l_ind = filename_list.index(last_file)
        files_to_read.append(all_files_ordered[f_ind - 1:l_ind + 2])

    return [item for sublist in files_to_read for item in sublist]

def get_entries(
        database_file: Text
) -> List[Tuple[Text, int, int, Text, int, int, float, int, float, float]]:
    """Retrieves a list of entries from a database file.

    The database should be a database of picked events, and have the columns
    origin_filename, pick_chan, pick_samp, cut_ev_filename, cut_fchan,
    cut_nchan, d_chan, cut_nt, dt, gauge_length.

    Args:
      database_file: Name of the file containing the database.

    Returns:
      A list of database entries.
    """
    db_cursor = sqlite3.connect(database_file).cursor()
    db_cursor.execute('SELECT * FROM events')
    return db_cursor.fetchall()


def write_to_event_database(
        entries: List[Tuple[
            Text, int, int, Text, int, int, float, int, float, float]],
        filename: Text
):
    """Writes entries to a database.

    Creates a database with columns origin_filename, pick_chan, pick_samp,
    cut_ev_filename, cut_fchan, cut_nchan, d_chan, cut_nt, dt, gauge_length,
    and fills it with the values from `entries`, and saves it to `filename`.

    Args:
        entries: List of entries to the database.
        filename: File to which to write the database.
    """
    db_connect = sqlite3.connect(filename)
    db_cursor = db_connect.cursor()
    db_cursor.execute(
        'CREATE TABLE events (origin_filename text, pick_chan integer, '
        'pick_samp integer, cut_ev_filename text, cut_fchan integer, '
        'cut_nchan integer, d_chan real, cut_nt integer, dt real, '
        'gauge_length real)'
    )
    db_cursor.executemany(
        'INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', entries)
    db_connect.commit()
    db_connect.close()


def merge(database_files: List[Text], output_database: Text):
    """Merges databases together and writes to `output_database`.

    Args:
        database_files: List of file names of databases to merge.
        output_database: Name of the file to which to write the merged
            database.
    """
    entries = []
    for database_file in database_files:
        entries += get_entries(database_file)
    write_to_event_database(entries, output_database)


def main():
    """Merges the databases."""
    merge(['ArielStage12.db', 'PaigeStage11.db'], 'Merged.db')


if __name__ == "__main__":
    main()