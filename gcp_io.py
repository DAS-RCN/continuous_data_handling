"""Module for reading/writing files from/to Google Cloud or local disk."""

import enum
import os
from typing import List, Sequence, Text, Tuple

import numpy as np
import tensorflow as tf


DEFAULT_DATATYPE = np.float32
DEFAULT_SHAPE = (-1,)


class FileExtension(enum.Enum):
    """Class for file extensions."""
    BINARY = '.bin'
    NUMPY = '.npy'


def get_filenames(datapath: Text) -> List[Text]:
    """Gets a list of files located at `datapath`.

    Datapath can refer to either local disk or Google Cloud storage.

    Args:
        datapath: Directory from which to list the files.
            For Google Cloud, the datapath should be of the form
            `gs://bucket_name/path_to_directory`.

    Returns:
        A sorted list of files located at `datapath`.
    """
    return sorted(tf.io.gfile.glob(os.path.join(datapath, '*')))


def _read_numpy(gfile: tf.io.gfile.GFile) -> Sequence[float]:
    """Reads data from a Numpy file.

    Args:
        gfile: TensorFlow file I/O wrapper.

    Returns:
        A Numpy array.
    """
    return np.load(gfile)


def _read_binary(gfile: tf.io.gfile.GFile, shape: Tuple[int],
                 data_type: np.dtype) -> Sequence[float]:
    """Reads data from a binary file.

    Args:
        gfile: TensorFlow file I/O wrapper.
        shape: Shape of the data array.
        data_type: Data type of the data stored in the file.

    Returns:
        A Numpy array.
    """
    print('Reading data as data type {} and shape {}'.format(data_type, shape))
    data = np.frombuffer(gfile.read(), dtype=data_type)
    return data.reshape(shape)


def read(filename: Text, shape: Tuple[int] = DEFAULT_SHAPE,
         data_type: np.dtype = DEFAULT_DATATYPE) -> Sequence[float]:
    """Reads data from `filename`.

    The file can be located on local disk or Google Cloud storage.

    Args:
        filename: Name of the file from which to read the data.
            For Google Cloud, the filename should be of the form
            `gs://bucket_name/path_to_directory/file_to_read`.
        shape: Shape of the data array. Only used if the file is a binary file.
        data_type: Data type of the data stored in the file. Only used if the
            file is a binary file.

    Raises:
        NotImplementedError when the file extension is not supported.

    Returns:
        A Numpy array.
    """
    allowed_extensions = [ext.value for ext in FileExtension]
    extension = os.path.splitext(filename)[1]
    if extension not in allowed_extensions:
        raise NotImplementedError(
            'Unsupported file extension. Expected: '
            '{}, Received: {}.'.format(allowed_extensions, extension)
        )
    extension = FileExtension(extension)
    with tf.io.gfile.GFile(filename, 'rb') as f:
        if extension == FileExtension.NUMPY:
            return _read_numpy(f)
        if extension == FileExtension.BINARY:
            return _read_binary(f, shape, data_type)


def _write_numpy(gfile: tf.io.gfile.GFile, data: Sequence[float]):
    """Writes data to a Numpy file.

    Args:
        gfile: TensorFlow file I/O wrapper.
        data: Data to write to file.
    """
    np.save(gfile, data)


def _write_binary(gfile: tf.io.gfile.GFile, data: Sequence[float]):
    """Writes data from a binary file.

    Args:
        gfile: TensorFlow file I/O wrapper.

    Returns:
        A Numpy array.
    """
    gfile.write(data.tobytes())


def write(filename: Text, data: Sequence[float], append=False):
    """Writes data to `filename`.

    The file can be located on local disk or Google Cloud storage.
    The type of file depends on the file extension of `filename`.

    Args:
        filename: Name of the file to which to write the data.
            For Google Cloud, the filename should be of the form
            `gs://bucket_name/path_to_directory/file_to_write`.
        data:
        append: boolean flag for appending data (binary only)

    Raises:
        NotImplementedError when the file extension is not supported.
    """
    allowed_extensions = [ext.value for ext in FileExtension]
    extension = os.path.splitext(filename)[1]
    if extension not in allowed_extensions:
        raise NotImplementedError(
            'Unsupported file extension. Expected: '
            '{}, Received: {}.'.format(allowed_extensions, extension)
        )
    extension = FileExtension(extension)
    if extension == FileExtension.BINARY and append:
        return _write_binary(tf.io.gfile.GFile(filename, 'ab'), data)
    else:
        with tf.io.gfile.GFile(filename, 'wb') as f:
            if extension == FileExtension.NUMPY:
                return _write_numpy(f, data)
            if extension == FileExtension.BINARY:
                return _write_binary(f, data)


