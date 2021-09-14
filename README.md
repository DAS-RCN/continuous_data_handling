# continuous_data_handling

Most DAS acquisitions yield continuous data, which are cumbersome to handle. This repository includes methods for handling continuous DAS data (loading, buffering, data processing...)

The most important part is an iterator that will allow you to serially access all files in a certain folder. Various processing steps can be applied directly during loading, and work on buffered versions of the data so there are no edge effects because of filtering. 

Maintained by Ariel Lellouch, ariellel@tauex.tau.ac.il
Please let me know if you found bugs or need something else!
