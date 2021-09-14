import data_streaming as stream
import config_general as cfg
import matplotlib.pyplot as plt
import numpy as np
import signal_processing as proc

def main():
    test_forge()
    if 2<1:
        template = np.zeros(shape=(400, 400))
        for i in range(400):
            template[i, i] = 1.0
        #TODO Bin - load template. If multiple templates are needed, we can parallelize the template matching call.
        #TODO Needs to undergo same processing as the continuous data. dimensions are (channel, time)
        data_iter = stream.DataIterator(proc_steps=cfg.proc_steps, folder_name=cfg.data_path,
                                        db_name=cfg.events_db_name)
        for curr_chunk in data_iter:
            if curr_chunk.check_validity():
                cc_res = proc.template_matching(curr_chunk.data, template, 0.0)
                print(cc_res)
                if cc_res:
                    with open('results.txt', 'a') as f:
                        f.write(curr_chunk.filename + '\t')
                        f.write(str(cc_res[0]))
                        f.write('\t')
                        f.write(str(cc_res[1]))
                        f.write('\t')
                        f.write(str(cc_res[2]))
                        f.write('\t')
                        f.write('\n')

    return 0


def test_forge():
    data_iter = stream.DataIterator(proc_steps=cfg.proc_steps, folder_name=cfg.data_path,
                                        db_name=cfg.events_db_name, output_overlap=cfg.overlap_samples)
    plt.figure(figsize=(17, 14))
    for curr_chunk in data_iter:
        if curr_chunk.check_validity():
            plt.imshow(curr_chunk.data, aspect='auto', cmap='seismic')
            plt.title(curr_chunk.filename)
            plt.show()


def test_cc():
    data = np.zeros(shape=(4926, 2400))
    for i in range(1800, 2200):
        data[i, i-1000] = 1.0

    template = np.zeros(shape=(400, 400))
    for i in range(400):
        template[i, i] = 1.0

    data = data + np.random.normal(loc=0.0, scale=0.5, size=data.shape)

    data = proc.normalization(data, 'std')

    cc_res = proc.template_matching(data, template, 0.1)
    print(cc_res)

if __name__ == '__main__':
    main()