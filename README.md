# continuous_data_handling

Most DAS acquisitions yield continuous data, which are cumbersome to handle. This repository includes methods for handling continuous DAS data (loading, buffering, data processing, space/time downsampling...). For now, I implemented a data iterator looping over all files in a folder. It automatically handles I/O, pre-processing, etc in a relatively fast manner and, most importantly, with minimal hassle. 

Later versions will include a data visualizer, database interface to save events of interest, and viewing of such events. 

The most important part is an iterator that will allow you to serially access all files in a certain folder. Various processing steps can be applied directly during loading, and work on buffered versions of the data so there are no edge effects because of filtering. 

Maintained by Ariel Lellouch, ariellel@tauex.tau.ac.il

Please let me know if you found bugs or need added functionality.
